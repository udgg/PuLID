# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import subprocess
from cog import BasePredictor, Input, Path
import os
import torch
import numpy as np
from PIL import Image
from typing import List
from einops import rearrange
import time

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, get_noise_batch
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
    "bad quality, worst quality, text, signature, watermark, extra limbs, "
    "low resolution, partially rendered objects, deformed or partially rendered eyes, "
    "deformed, deformed eyeballs, cross-eyed, blurry"
)

BASE_URL = f"https://weights.replicate.delivery/default/PuLID-FLUX/{MODEL_CACHE}/"


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start_time = time.time()

        model_files = [
            "antelopev2.tar",
            "models--black-forest-labs--FLUX.1-dev.tar",
            "models--DIAMONIK7777--antelopev2.tar",
            "models--openai--clip-vit-large-patch14.tar",
            "models--QuanSun--EVA-CLIP.tar",
            "pulid_flux_v0.9.0.safetensors",
            "models--XLabs-AI--xflux_text_encoders.tar",
        ]

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for model_file in model_files:
            url = BASE_URL + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

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
        main_face_image: Path = Input(description="ID image for face generation"),
        prompt: str = Input(description="Text prompt for image generation", default="portrait, color, cinematic"),
        negative_prompt: str = Input(
            description="Negative prompt to guide what to avoid in the image", default=DEFAULT_NEGATIVE_PROMPT
        ),
        width: int = Input(description="Width of the generated image", ge=256, le=1536, default=896),
        height: int = Input(description="Height of the generated image", ge=256, le=1536, default=1152),
        num_steps: int = Input(description="Number of denoising steps", ge=1, le=20, default=20),
        start_step: int = Input(
            description="Timestep to start inserting ID (0-4 recommended, lower for higher fidelity, higher for more editability)",
            ge=0,
            le=10,
            default=0,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for text prompt influence", ge=1.0, le=10.0, default=4.0
        ),
        id_weight: float = Input(description="Weight of the ID image influence", ge=0.0, le=3.0, default=1.0),
        seed: int = Input(description="Random seed for generation (leave blank or -1 for random)", default=None),
        true_cfg: float = Input(
            description="True CFG scale (1.0 for fake CFG, >1.0 for true CFG)", ge=1.0, le=10.0, default=1.0
        ),
        max_sequence_length: int = Input(
            description="Max sequence length for prompt (T5), smaller is faster", ge=128, le=512, default=128
        ),
        output_format: str = Input(
            description="Format of the output image", choices=["png", "jpg", "webp"], default="webp"
        ),
        output_quality: int = Input(
            description="Quality of the output image (for jpg and webp)", ge=1, le=100, default=80
        ),
        num_outputs: int = Input(description="Number of images to generate", ge=1, le=4, default=1),
    ) -> List[Path]:
        """Run a single prediction on the model to generate multiple outputs"""
        start_time = time.time()

        # Generate a list of seeds for each output to ensure uniqueness
        if seed is None or seed == -1:
            seeds = [int.from_bytes(os.urandom(4), "big") for _ in range(num_outputs)]
        else:
            seeds = [seed + i for i in range(num_outputs)]
        print(f"Using seeds: {seeds}")

        # Load and preprocess the ID image
        id_image_np = np.array(Image.open(str(main_face_image))) if main_face_image else None

        # Generate the images
        generated_images, used_seeds, _ = self.generate_image(
            width=width,
            height=height,
            num_steps=num_steps,
            start_step=start_step,
            guidance=guidance_scale,
            seeds=seeds,
            prompt=prompt,
            id_image=id_image_np,
            id_weight=id_weight,
            neg_prompt=negative_prompt,
            true_cfg=true_cfg,
            timestep_to_start_cfg=start_step,
            max_sequence_length=max_sequence_length,
            num_outputs=num_outputs,
        )

        output_paths = []
        for i, (generated_image, used_seed) in enumerate(zip(generated_images, used_seeds)):
            # Save the generated image
            output_path = f"output_{i+1}.{output_format}"
            save_params = {"format": output_format.upper()}
            if output_format in ["jpg", "webp"]:
                save_params["quality"] = output_quality
                if output_format == "jpg":
                    save_params["optimize"] = True

            generated_image.save(output_path, **save_params)
            output_paths.append(Path(output_path))

            print(f"Image {i+1} generated with seed: {used_seed}")

        end_time = time.time()
        print(f"Total prediction time: {end_time - start_time:.2f} seconds")
        return output_paths

    def generate_image(
        self,
        width,
        height,
        num_steps,
        start_step,
        guidance,
        seeds,
        prompt,
        id_image=None,
        id_weight=1.0,
        neg_prompt="",
        true_cfg=1.0,
        timestep_to_start_cfg=1,
        max_sequence_length=128,
        num_outputs=1,
    ):
        generate_start_time = time.time()
        self.t5.max_length = max_sequence_length

        print(f"Generating '{prompt}' with seeds {seeds}")

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        # Process ID image embeddings
        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image, cal_uncond=use_true_cfg)
            # Expand embeddings to match batch size
            id_embeddings = id_embeddings.expand(num_outputs, *id_embeddings.shape[1:])
            if uncond_id_embeddings is not None:
                uncond_id_embeddings = uncond_id_embeddings.expand(num_outputs, *uncond_id_embeddings.shape[1:])
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # Generate batched noise
        x = get_noise_batch(
            seeds=seeds,
            height=height,
            width=width,
            device=self.device,
            dtype=torch.bfloat16,
        )

        timesteps = get_schedule(
            num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        # Prepare inputs with batched prompts
        inp = prepare(
            t5=self.t5,
            clip=self.clip,
            img=x,
            prompt=[prompt] * num_outputs,
            device=self.device,
        )
        inp_neg = (
            prepare(
                t5=self.t5,
                clip=self.clip,
                img=x,
                prompt=[neg_prompt] * num_outputs,
                device=self.device,
            )
            if use_true_cfg
            else None
        )

        # Denoise initial noise
        denoise_start_time = time.time()
        x = denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            guidance=guidance,
            id=id_embeddings,
            id_weight=id_weight,
            start_step=start_step,
            uncond_id=uncond_id_embeddings,
            true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
        )
        denoise_end_time = time.time()
        print(f"Denoising time: {denoise_end_time - denoise_start_time:.2f} seconds")

        # Decode latents to pixel space
        decode_start_time = time.time()
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)
        decode_end_time = time.time()
        print(f"Decoding time: {decode_end_time - decode_start_time:.2f} seconds")

        # Convert to PIL images
        x = x.clamp(-1, 1)
        images = []
        for i in range(num_outputs):
            img = rearrange(x[i], "c h w -> h w c")
            img = Image.fromarray((127.5 * (img + 1.0)).cpu().byte().numpy())
            images.append(img)

        generate_end_time = time.time()
        print(f"Total generate_image time: {generate_end_time - generate_start_time:.2f} seconds")
        return images, seeds, self.pulid_model.debug_img_list
