import math
from typing import Callable, List

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def get_noise_batch(
    seeds: List[int],
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
):
    num_samples = len(seeds)
    c = 16
    h = 2 * math.ceil(height / 16)
    w = 2 * math.ceil(width / 16)

    # Preallocate the output tensor
    x = torch.empty(num_samples, c, h, w, device=device, dtype=dtype)

    # Create a generator once
    generator = torch.Generator(device=device)

    for i, seed in enumerate(seeds):
        generator.manual_seed(seed)
        x[i] = torch.randn(c, h, w, generator=generator, device=device, dtype=dtype)

    return x


def prepare(
    t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: List[str], device: torch.device
) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    ph, pw = 2, 2
    h_patches = h // ph
    w_patches = w // pw
    hw_patches = h_patches * w_patches

    # Reshape img
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=ph, pw=pw)

    # Generate img_ids efficiently
    y_coords = torch.arange(h_patches, device=device, dtype=img.dtype)
    x_coords = torch.arange(w_patches, device=device, dtype=img.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    img_ids = torch.stack(
        (
            torch.zeros_like(grid_y),  # Assuming the first channel is zeros
            grid_y,
            grid_x,
        ),
        dim=-1,
    )
    img_ids = img_ids.reshape(1, hw_patches, 3).expand(bs, hw_patches, 3)

    # Prepare text embeddings
    txt = t5(prompt)

    # Prepare txt_ids as zeros (if required)
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=device, dtype=txt.dtype)

    # Prepare CLIP embeddings
    vec = clip(prompt)

    return {
        "img": img.to(device),
        "img_ids": img_ids.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    id_weight=1.0,
    id=None,
    start_step=0,
    uncond_id=None,
    true_cfg=1.0,
    timestep_to_start_cfg=1,
    neg_txt=None,
    neg_txt_ids=None,
    neg_vec=None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    use_true_cfg = abs(true_cfg - 1.0) > 1e-2
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            id=id if i >= start_step else None,
            id_weight=id_weight,
        )

        if use_true_cfg and i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                id=uncond_id if i >= start_step else None,
                id_weight=id_weight,
            )
            pred = neg_pred + true_cfg * (pred - neg_pred)

        img = img + (t_prev - t_curr) * pred

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
