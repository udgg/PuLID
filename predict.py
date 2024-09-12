# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import torch
import numpy as np
from PIL import Image
from typing import List

# Import FluxGenerator from app_flux.py
from app_flux import FluxGenerator, get_models

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "flux-dev"
        self.offload = False

        class Args:
            pretrained_model = "path/to/pretrained/model"  # Update this path

        self.args = Args()
        self.generator = FluxGenerator(self.model_name, self.device, self.offload, self.args)

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
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        # Load and preprocess the ID image
        id_image_np = np.array(Image.open(str(id_image))) if id_image else None

        # Generate the image using FluxGenerator
        generated_image, used_seed, _ = self.generator.generate_image(
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

        return [Path(output_path)]
