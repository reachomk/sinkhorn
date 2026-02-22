"""
SD-VAE tokenizer utilities used by the ImageNet drifting pipeline.

The paper *Generative Modeling via Drifting* trains generators either in pixel space
or in a tokenizer latent space. In our ImageNet256 setup we use the Stable Diffusion
VAE tokenizer ('stabilityai/sd-vae-ft-ema') so that:
  RGB image:  '[B,3,256,256]'  ↔  latent: '[B,4,32,32]'

The 'scaling_factor=0.18215' matches the standard diffusers convention for this VAE.

Scope note:
- Tokenization/decoding is shared by baseline Drift and follow-up Sinkhorn-flow
  experiments; algorithmic differences are in drifting-loss construction only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class VaeConfig:
    vae_id: str = "stabilityai/sd-vae-ft-ema"
    scaling_factor: float = 0.18215
    dtype: Literal["fp16", "fp32"] = "fp16"
    encode_mode: Literal["mean", "sample"] = "mean"


def load_vae(cfg: VaeConfig, device: torch.device):
    """
    Load SD VAE via diffusers.

    Note: we keep this function local to avoid importing diffusers in modules
    that don't need it (faster CLI startup for non-VAE commands).
    """
    from diffusers import AutoencoderKL

    torch_dtype = torch.float16 if cfg.dtype == "fp16" else torch.float32
    vae = AutoencoderKL.from_pretrained(cfg.vae_id, torch_dtype=torch_dtype)
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_images_to_latents(
    vae,
    images: torch.Tensor,
    *,
    scaling_factor: float,
    encode_mode: Literal["mean", "sample"],
) -> torch.Tensor:
    """Encode images in [-1, 1] to scaled latents (4x32x32).

    Returns latents multiplied by 'scaling_factor' (diffusers convention).
    """
    enc = vae.encode(images)
    if encode_mode == "mean":
        latents = enc.latent_dist.mean
    elif encode_mode == "sample":
        latents = enc.latent_dist.sample()
    else:
        raise ValueError(f"Unknown encode_mode: {encode_mode}")
    return latents * float(scaling_factor)


@torch.no_grad()
def decode_latents_to_images(
    vae,
    latents: torch.Tensor,
    *,
    scaling_factor: float,
) -> torch.Tensor:
    """
    Decode scaled latents to images in [-1, 1].
    """
    latents = latents / float(scaling_factor)
    dec = vae.decode(latents)
    # diffusers AutoencoderKL returns a DiagonalGaussianDistribution-like output object;
    # '.sample' contains decoded images.
    return dec.sample
