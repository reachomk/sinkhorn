"""
DiT-B/2-like generator for SD-VAE latents (Appendix A.2 / Table 8).

This implements the latent-space generator f_ω used in the ImageNet experiments of
*Generative Modeling via Drifting*:
  noise z ~ N(0,I) in R^{4×32×32} + conditioning (class c, CFG strength ω, style tokens)
    → output latent x in R^{4×32×32}.

Key paper features:
- RoPE, RMSNorm, SwiGLU, QK-Norm (Appendix A.2).
- In-context conditioning tokens: prepend 16 learnable tokens formed by summing the
  projected conditioning vector with per-token embeddings (Appendix A.2, “In-context tokens”).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


def _rotary_freqs(seq_len: int, dim: int, base: float = 10000.0, device=None, dtype=None):
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, dim]
    return emb.cos(), emb.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to last dim. x: [B,H,S,D], cos/sin: [S,D].
    """
    # Split into even/odd and rotate.
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos1 = cos[:, ::2]
    sin1 = sin[:, ::2]
    # Broadcast cos/sin to [1,1,S,D/2]
    cos1 = cos1.unsqueeze(0).unsqueeze(0)
    sin1 = sin1.unsqueeze(0).unsqueeze(0)
    y1 = x1 * cos1 - x2 * sin1
    y2 = x1 * sin1 + x2 * cos1
    y = torch.stack([y1, y2], dim=-1).flatten(-2)
    return y


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, qk_norm_eps: float = 1e-6):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim must be divisible by n_heads, got {dim} and {n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = dim // n_heads
        self.qk_norm_eps = float(qk_norm_eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, *, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(x)  # [B,S,3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,S,hd]
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        # QK-Norm (paper uses QK-Norm; we use RMS normalization per head).
        q = q * torch.rsqrt(torch.mean(q * q, dim=-1, keepdim=True) + self.qk_norm_eps)
        k = k * torch.rsqrt(torch.mean(k * k, dim=-1, keepdim=True) + self.qk_norm_eps)

        # RoPE
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Attention
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(b, s, d)
        return self.proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 4.0):
        super().__init__()
        inner = int(dim * hidden_mult)
        self.fc = nn.Linear(dim, inner * 2, bias=False)
        self.proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.silu(a) * b)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_mult: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiheadSelfAttention(dim, n_heads=n_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_mult=mlp_mult)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        # AdaLN-Zero style init for stability.
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, *, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)

        h = self.norm1(x)
        h = h * (1.0 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + gate1.unsqueeze(1) * self.attn(h, cos=cos, sin=sin)

        h2 = self.norm2(x)
        h2 = h2 * (1.0 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h2)
        return x


@dataclass(frozen=True)
class DiTB2Config:
    num_classes: int = 1000
    in_ch: int = 4
    out_ch: int = 4
    input_size: int = 32
    patch_size: int = 2
    hidden_dim: int = 768
    depth: int = 12
    n_heads: int = 12
    ctx_mode: Literal["register", "in_context"] = "in_context"
    register_tokens: int = 16
    style_codebook: int = 64
    style_tokens: int = 32
    mlp_mult: float = 4.0


class DiTLatentB2(nn.Module):
    """
    A minimal DiT-B/2-style latent generator (Appendix A.2 / Table 8).

    Output is a single-step latent prediction in R^{4x32x32}.
    """

    def __init__(self, cfg: DiTB2Config):
        super().__init__()
        self.cfg = cfg
        p = int(cfg.patch_size)
        if cfg.input_size % p != 0:
            raise ValueError("input_size must be divisible by patch_size")
        self.grid = cfg.input_size // p
        self.n_patches = self.grid * self.grid
        self.patch_dim = cfg.in_ch * p * p

        self.in_proj = nn.Linear(self.patch_dim, cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, self.patch_dim, bias=False)

        # Conditioning
        self.class_emb = nn.Embedding(cfg.num_classes + 1, cfg.hidden_dim)  # +1 for unconditional (-1)
        self.omega_mlp = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.style_emb = nn.Embedding(cfg.style_codebook, cfg.hidden_dim)

        self.register = nn.Parameter(torch.randn(cfg.register_tokens, cfg.hidden_dim) * 0.02)

        self.blocks = nn.ModuleList([DiTBlock(cfg.hidden_dim, n_heads=cfg.n_heads, mlp_mult=cfg.mlp_mult) for _ in range(cfg.depth)])
        self.final_norm = RMSNorm(cfg.hidden_dim)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = int(self.cfg.patch_size)
        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(b, self.n_patches, self.patch_dim)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = tokens.shape
        p = int(self.cfg.patch_size)
        g = self.grid
        x = tokens.view(b, g, g, self.cfg.out_ch, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(b, self.cfg.out_ch, g * p, g * p)

    def forward(self, noise: torch.Tensor, class_labels: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Forward:
          noise: [B,4,32,32] Gaussian noise.
          class_labels: [B] int64, in [0,num_classes) or -1 for unconditional.
          omega: [B] float, CFG scale conditioning.
        """
        b = noise.shape[0]

        # Conditioning vector.
        cls = class_labels.clone()
        cls = torch.where(cls < 0, torch.full_like(cls, self.cfg.num_classes), cls)
        cond = self.class_emb(cls)

        omega_in = omega.view(b, 1).to(dtype=cond.dtype)
        cond = cond + self.omega_mlp(omega_in)

        # Random style embeddings: sum 32 codebook vectors.
        style_idx = torch.randint(0, self.cfg.style_codebook, (b, self.cfg.style_tokens), device=noise.device)
        cond = cond + self.style_emb(style_idx).sum(dim=1)

        # Patch tokens.
        x = self._patchify(noise)
        x = self.in_proj(x)

        # In-context conditioning tokens (Appendix A.2):
        # prepend 16 tokens formed by summing the conditioning vector with per-token
        # learnable positional embeddings.
        if str(self.cfg.ctx_mode) == "in_context":
            ctx = cond.unsqueeze(1) + self.register.unsqueeze(0)  # [B,register_tokens,dim]
        elif str(self.cfg.ctx_mode) == "register":
            ctx = self.register.unsqueeze(0).expand(b, -1, -1)
        else:
            raise ValueError(f"Unknown ctx_mode: {self.cfg.ctx_mode}")
        x = torch.cat([ctx, x], dim=1)

        # RoPE cache (1D positions).
        seq_len = x.shape[1]
        cos, sin = _rotary_freqs(seq_len, self.blocks[0].attn.head_dim, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            x = block(x, cond, cos=cos, sin=sin)

        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        x = self.final_norm(x)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = x[:, self.cfg.register_tokens :, :]
        x = self.out_proj(x)
        return self._unpatchify(x)
