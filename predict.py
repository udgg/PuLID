# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import subprocess
import os
import torch
import numpy as np
from PIL import Image
from typing import List
from einops import rearrange
import time
import requests
import base64

from io import BytesIO
from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, get_noise_batch
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long

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


class Predictor():
    def __init__(self) -> None:
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

    def predict(
        self,
        main_face_image,
        prompt,
        negative_prompt,
        width,
        height,
        num_steps,
        start_step,
        guidance_scale,
        id_weight,
        seed,
        true_cfg,
        max_sequence_length,
        output_format,
        output_quality,
        num_outputs,
    ):
        """Run a single prediction on the model to generate multiple outputs"""
        start_time = time.time()

        # Generate a list of seeds for each output to ensure uniqueness
        if seed is None or seed == -1:
            seeds = [int.from_bytes(os.urandom(4), "big") for _ in range(num_outputs)]
        else:
            seeds = [seed + i for i in range(num_outputs)]
        print(f"Using seeds: {seeds}")

        # Load and preprocess the ID image
        response = requests.get(main_face_image)
        id_image_np = np.array(Image.open(BytesIO(response.content))) if main_face_image else None

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
            
            buffer = BytesIO()
            generated_image.save(buffer, **save_params)
            image_bytes = buffer.getvalue()
           
            output_paths.append(base64.b64encode(image_bytes).decode('utf-8'))

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

predictor = Predictor()

def handler(job):
    job_input = job['input']
    prompt = job_input['prompt']

    time_start = time.time()
    image = predictor.predict(
        main_face_image=job_input['main_face_image'],
        prompt=job_input['prompt'],
        negative_prompt=job_input.get('negative_prompt', 'bad quality, worst quality, text, signature, watermark, extra limbs'),
        width=int(job_input.get("width", 896)),
        heigh=int(job_input.get("heigh", 1152)),
        num_steps=int(job_input.get("num_steps", 20)),
        start_step=int(job_input.get("start_step", 4)),
        guidance_scale=float(job_input.get("guidance_scale", 4)),
        id_weight=int(job_input.get("id_weight", 1)),
        seed=None,
        true_cfg=int(job_input.get("true_cfg", 1)),
        max_sequence_length=int(job_input.get('max_sequence_length', 128)),
        output_format="png,
        output_quality=100,
        num_outputs=1,
    )
    print(f"Time taken: {time.time() - time_start}")

    return image[0]
    
runpod.serverless.start({"handler": handler})
