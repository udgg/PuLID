# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import torch
import numpy as np
from PIL import Image
from typing import List
from einops import rearrange
import time

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long
# from huggingface_hub import login

MODEL_CACHE = "models"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

torch.set_grad_enabled(False)

DEFAULT_NEGATIVE_PROMPT = (
    'bad quality, worst quality, text, signature, watermark, extra limbs, '
    'low resolution, partially rendered objects, deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed, blurry'
)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start_time = time.time()
        self.device = torch.device("cuda")
        self.model_name = "flux-dev"
        self.offload = False

        self.model, self.ae, self.t5, self.clip = self.get_models(
            self.model_name,
            device=self.device,
            offload=self.offload,
            cache_dir=MODEL_CACHE,
        )
        self.pulid_model = PuLIDPipeline(self.model, self.device, weight_dtype=torch.bfloat16)
        self.pulid_model.load_pretrain(f"{MODEL_CACHE}/pulid_flux_v0.9.0.safetensors")  # Update this path
        end_time = time.time()
        print(f"Setup completed in {end_time - start_time:.2f} seconds")

    def get_models(self, name: str, device: torch.device, offload: bool, cache_dir: str):
        t5 = load_t5(device, max_length=128)
        clip = load_clip(device)  # Remove cache_dir argument
        model = load_flow_model(name, device=device)
        model.eval()
        ae = load_ae(name, device=device)
        return model, ae, t5, clip

    @torch.inference_mode()
    def predict(
        self,
        id_image: Path = Input(description="ID image"),
        prompt: str = Input(description="Prompt", default="portrait, color, cinematic"),
        negative_prompt: str = Input(description="Negative Prompt", default=DEFAULT_NEGATIVE_PROMPT),
        width: int = Input(description="Width", ge=256, le=1536, default=896),
        height: int = Input(description="Height", ge=256, le=1536, default=1152),
        num_steps: int = Input(description="Number of steps", ge=1, le=20, default=20),
        start_step: int = Input(description="Timestep to start inserting ID", ge=0, le=10, default=0),
        guidance: float = Input(description="Guidance scale", ge=1.0, le=10.0, default=4.0),
        id_weight: float = Input(description="ID weight", ge=0.0, le=3.0, default=1.0),
        seed: int = Input(description="Seed (-1 for random)", default=-1),
        true_cfg: float = Input(description="True CFG scale", ge=1.0, le=10.0, default=1.0),
        max_sequence_length: int = Input(description="Max sequence length for prompt (T5)", ge=128, le=512, default=128),
        output_format: str = Input(description="Output format", choices=["png", "jpg", "webp"], default="png"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        start_time = time.time()

        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        # Load and preprocess the ID image
        id_image_np = np.array(Image.open(str(id_image))) if id_image else None

        # Generate the image
        generated_image, used_seed, _ = self.generate_image(
            width=width,
            height=height,
            num_steps=num_steps,
            start_step=start_step,
            guidance=guidance,
            seed=seed,
            prompt=prompt,
            id_image=id_image_np,
            id_weight=id_weight,
            neg_prompt=negative_prompt,
            true_cfg=true_cfg,
            timestep_to_start_cfg=start_step,
            max_sequence_length=max_sequence_length,
        )

        # Save the generated image
        output_path = f"output.{output_format}"
        generated_image.save(output_path)

        end_time = time.time()
        print(f"Image generated with seed: {used_seed}")
        print(f"Total prediction time: {end_time - start_time:.2f} seconds")
        return [Path(output_path)]

    def generate_image(self, width, height, num_steps, start_step, guidance, seed, prompt, id_image=None, id_weight=1.0, neg_prompt="", true_cfg=1.0, timestep_to_start_cfg=1, max_sequence_length=128):
        generate_start_time = time.time()
        self.t5.max_length = max_sequence_length

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image, cal_uncond=use_true_cfg)
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

        # denoise initial noise
        denoise_start_time = time.time()
        x = denoise(
            self.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
            start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
        )
        denoise_end_time = time.time()
        print(f"Denoising time: {denoise_end_time - denoise_start_time:.2f} seconds")

        # decode latents to pixel space
        decode_start_time = time.time()
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)
        decode_end_time = time.time()
        print(f"Decoding time: {decode_end_time - decode_start_time:.2f} seconds")

        # bring into PIL format
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        generate_end_time = time.time()
        print(f"Total generate_image time: {generate_end_time - generate_start_time:.2f} seconds")
        return img, str(opts.seed), self.pulid_model.debug_img_list