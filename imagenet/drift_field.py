from __future__ import annotations

import math
from typing import Literal

import torch


def compute_drift_alg2(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    temp: float,
    mask_self_in_y_neg: bool = False,
    y_neg_self_cols: int | None = None,
    uncond_cols: slice | None = None,
    uncond_weight: float | torch.Tensor | None = None,
    eps: float = 1e-12,
    impl: Literal["logspace", "kernel"] = "logspace",
) -> torch.Tensor:
    """
    Algorithm 2 drift computation (appendix) in a batched form.

    This matches the toy reference in 'original jupyter notebook from Kaiming's paper':
      kernel = exp(-cdist(x, targets)/temp)
      normalized_kernel = kernel / sqrt(row_sum * col_sum)
      pos_coeff = K_pos * sum(K_neg)
      neg_coeff = K_neg * sum(K_pos)
      V = pos_coeff @ y_pos - neg_coeff @ y_neg

    Shapes:
      x:     [..., N_x, D]
      y_pos: [..., N_pos, D]  (broadcastable over leading dims)
      y_neg: [..., N_neg, D]  (broadcastable over leading dims)

    Args:
      temp: temperature T (must be > 0).
      mask_self_in_y_neg: if True, assumes the first 'y_neg_self_cols' columns of y_neg
        correspond to x itself, and masks the diagonal self-coupling by setting distance to +1e6.
      y_neg_self_cols: number of "self" columns in y_neg (defaults to N_x when masking is enabled).
      uncond_cols: optional slice into the last dimension of the y_neg columns to treat as
        unconditional negatives (CFG). If provided with 'uncond_weight', adds 'log(w)' to those
        logits (equivalent to multiplying kernel by w).
      uncond_weight: scalar or tensor broadcastable to [..., 1, 1] that supplies w.
        If w <= 0, unconditional columns are effectively removed.
      impl:
        - "logspace": stable 'logsumexp' implementation.
        - "kernel":   matches the notebook's explicit kernel sums (may underflow for small T).
    """
    if not (temp > 0):
        raise ValueError(f"temp must be > 0, got {temp}")

    # Pairwise distances: Euclidean (not squared), as in the appendix + notebook.
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)

    if mask_self_in_y_neg:
        n_x = x.shape[-2]
        n_self = n_x if y_neg_self_cols is None else int(y_neg_self_cols)
        if n_self != n_x:
            raise ValueError(f"Expected y_neg_self_cols == N_x when masking. Got {n_self} vs {n_x}.")
        idx = torch.arange(n_x, device=x.device)
        dist_neg[..., idx, idx] = 1e6

    logit_pos = -dist_pos / float(temp)
    logit_neg = -dist_neg / float(temp)

    if uncond_cols is not None and uncond_weight is not None:
        w = torch.as_tensor(uncond_weight, device=x.device, dtype=logit_neg.dtype)
        # Broadcast to [..., 1, 1] then to the slice.
        while w.ndim < logit_neg.ndim:
            w = w.unsqueeze(-1)
        w = w[..., :1, :1]
        # If w <= 0, treat as -inf bias (remove unconditional negatives).
        bias = torch.where(w > 0, w.log(), torch.tensor(-math.inf, device=x.device, dtype=logit_neg.dtype))
        logit_neg[..., uncond_cols] = logit_neg[..., uncond_cols] + bias

    # Concatenate positive and negative targets for the two-axis normalization.
    logits = torch.cat([logit_pos, logit_neg], dim=-1)  # [..., N_x, N_pos + N_neg]

    if impl == "kernel":
        kernel = torch.exp(logits)
        # Avoid 0/0; clamping is only used for stability and does not change the math when sums are > 0.
        row_sum = kernel.sum(dim=-1, keepdim=True).clamp_min(eps)
        col_sum = kernel.sum(dim=-2, keepdim=True).clamp_min(eps)
        A = kernel / (row_sum * col_sum).sqrt()
    elif impl == "logspace":
        log_row_sum = torch.logsumexp(logits, dim=-1, keepdim=True)
        log_col_sum = torch.logsumexp(logits, dim=-2, keepdim=True)
        A = torch.exp(logits - 0.5 * (log_row_sum + log_col_sum))
    else:
        raise ValueError(f"Unknown impl: {impl}")

    n_pos = y_pos.shape[-2]
    A_pos = A[..., :n_pos]
    A_neg = A[..., n_pos:]

    # Coefficients / weights (appendix Alg.2).
    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)

    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    return drift_pos - drift_neg

