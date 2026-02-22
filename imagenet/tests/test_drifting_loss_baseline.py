from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from imagenet.drifting_loss import drifting_loss_for_feature_set


def _ref_compute_uncond_weight(omega: torch.Tensor, *, nneg: int, nuncond: int) -> torch.Tensor:
    if nneg <= 1:
        raise ValueError("nneg must be > 1 for CFG weighting")
    if nuncond <= 0:
        return torch.zeros_like(omega)
    return (float(nneg - 1) * (omega - 1.0)) / float(nuncond)


def _ref_alg2_from_distances(
    dist_pos: torch.Tensor,
    dist_neg: torch.Tensor,
    *,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temp: float,
    mask_self_in_y_neg: bool,
    nuncond: int,
    uncond_weight: torch.Tensor | None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not (temp > 0):
        raise ValueError(f"temp must be > 0, got {temp}")

    if mask_self_in_y_neg:
        n_x = dist_neg.shape[-2]
        idx = torch.arange(n_x, device=dist_neg.device)
        dist_neg[..., idx, idx] = 1e6

    logit_pos = -dist_pos / float(temp)
    logit_neg = -dist_neg / float(temp)

    if nuncond > 0 and uncond_weight is not None:
        w = uncond_weight.to(device=logit_neg.device, dtype=logit_neg.dtype)
        while w.ndim < logit_neg.ndim:
            w = w.unsqueeze(-1)
        w = w[..., :1, :1]
        bias = torch.where(w > 0, w.log(), torch.tensor(-math.inf, device=logit_neg.device, dtype=logit_neg.dtype))
        logit_neg[..., -nuncond:] = logit_neg[..., -nuncond:] + bias

    logits = torch.cat([logit_pos, logit_neg], dim=-1)
    log_row_sum = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_col_sum = torch.logsumexp(logits, dim=-2, keepdim=True)
    A = torch.exp(logits - 0.5 * (log_row_sum + log_col_sum))

    n_pos = dist_pos.shape[-1]
    A_pos = A[..., :n_pos]
    A_neg = A[..., n_pos:]

    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)

    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    return drift_pos - drift_neg


def _ref_drifting_loss_for_feature_set_baseline(
    x_feat: torch.Tensor,
    y_pos_feat: torch.Tensor,
    y_uncond_feat: torch.Tensor,
    *,
    omega: torch.Tensor,
    temps: list[float],
) -> torch.Tensor:
    x_feat = x_feat if x_feat.ndim == 3 else (_ for _ in ()).throw(ValueError("x_feat must be [N,L,C]"))
    y_pos_feat = y_pos_feat if y_pos_feat.ndim == 3 else (_ for _ in ()).throw(ValueError("y_pos_feat must be [N,L,C]"))
    y_uncond_feat = y_uncond_feat if y_uncond_feat.ndim == 3 else (_ for _ in ()).throw(ValueError("y_uncond_feat must be [N,L,C]"))

    nneg, l, c = x_feat.shape
    npos = y_pos_feat.shape[0]
    nuncond = y_uncond_feat.shape[0]

    x_lnc = x_feat.permute(1, 0, 2).contiguous()
    y_pos_lpc = y_pos_feat.permute(1, 0, 2).contiguous()
    y_uncond_luc = y_uncond_feat.permute(1, 0, 2).contiguous() if nuncond > 0 else None

    with torch.no_grad():
        x_det = x_lnc.detach().float()
        y_pos_det = y_pos_lpc.detach().float()
        omega_s = omega.detach().to(device=x_feat.device, dtype=torch.float32).reshape(())
        if nuncond > 0:
            assert y_uncond_luc is not None
            y_uncond_det = y_uncond_luc.detach().float()
            y_neg_det = torch.cat([x_det, y_uncond_det], dim=1)
            w = _ref_compute_uncond_weight(omega_s, nneg=nneg, nuncond=nuncond).to(dtype=torch.float32)
        else:
            y_neg_det = x_det
            w = None

        dist_pos_raw = torch.cdist(x_det, y_pos_det)
        dist_neg_raw = torch.cdist(x_det, y_neg_det)

        # Feature normalization scale S_j (paper Eq. 21; baseline behavior).
        sum_pos = dist_pos_raw.sum()
        denom_pos = float(dist_pos_raw.numel())
        dist_neg_gen = dist_neg_raw[..., :nneg]
        sum_neg_gen = dist_neg_gen.sum()
        denom_neg_gen = float(dist_neg_gen.numel() - (l * nneg))

        if nuncond > 0:
            dist_neg_unc = dist_neg_raw[..., nneg:]
            sum_neg_unc = dist_neg_unc.sum()
            denom_neg_unc = float(dist_neg_unc.numel())

            w_eff = torch.clamp(w if w is not None else torch.tensor(0.0, device=x_feat.device), min=0.0).to(
                device=x_feat.device, dtype=dist_pos_raw.dtype
            )
            sum_dist = sum_pos + sum_neg_gen + w_eff * sum_neg_unc
            denom_t = torch.tensor(denom_pos + denom_neg_gen, device=x_feat.device, dtype=dist_pos_raw.dtype) + w_eff * denom_neg_unc
        else:
            sum_dist = sum_pos + sum_neg_gen
            denom_t = torch.tensor(denom_pos + denom_neg_gen, device=x_feat.device, dtype=dist_pos_raw.dtype)

        mean_dist = sum_dist / denom_t.clamp_min(1e-12)
        s = (mean_dist / math.sqrt(float(c))).clamp_min(1e-6)

        dist_pos = dist_pos_raw / s
        dist_neg = dist_neg_raw / s

        omega_s = omega_s.to(device=x_feat.device, dtype=dist_pos.dtype)
        w = w.to(device=x_feat.device, dtype=dist_pos.dtype) if w is not None else None

        y_pos_norm = y_pos_det / s
        if nuncond > 0:
            drift_y_neg_raw = torch.cat([x_det, y_uncond_det], dim=1)
        else:
            drift_y_neg_raw = x_det

        v_agg = torch.zeros((l, nneg, c), device=x_feat.device, dtype=x_det.dtype)
        for rho in temps:
            temp_eff = float(rho) * float(c)
            v_raw = _ref_alg2_from_distances(
                dist_pos,
                dist_neg,
                y_pos=y_pos_norm,
                y_neg=drift_y_neg_raw / s,
                temp=temp_eff,
                mask_self_in_y_neg=True,
                nuncond=nuncond,
                uncond_weight=w,
            )
            theta = torch.sqrt((v_raw * v_raw).mean().clamp_min(1e-12))
            v_agg = v_agg + (v_raw / theta)
        v_agg_nlc = v_agg.permute(1, 0, 2).contiguous()

    x_norm = x_feat.float() / s.to(dtype=torch.float32)
    target = (x_norm + v_agg_nlc).detach()
    return F.mse_loss(x_norm, target, reduction="mean")


@torch.no_grad()
def test_drifting_loss_default_matches_reference_cpu() -> None:
    torch.manual_seed(0)
    nneg, npos, nuncond = 4, 3, 2
    l, c = 2, 5
    x = torch.randn(nneg, l, c, dtype=torch.float32, requires_grad=True)
    y_pos = torch.randn(npos, l, c, dtype=torch.float32)
    y_unc = torch.randn(nuncond, l, c, dtype=torch.float32)
    omega = torch.tensor(2.5, dtype=torch.float32)
    temps = [0.02, 0.05, 0.2]

    ref = _ref_drifting_loss_for_feature_set_baseline(x, y_pos, y_unc, omega=omega, temps=temps)
    ours = drifting_loss_for_feature_set(x, y_pos, y_unc, omega=omega, temps=temps, impl="logspace")
    torch.testing.assert_close(ours, ref, rtol=0, atol=1e-6)


def _assert_raises(exc_type, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:  # pragma: no cover
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


@torch.no_grad()
def test_invalid_sinkhorn_combinations_raise() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 1, 3)
    y = torch.randn(2, 1, 3)
    omega = torch.tensor(2.0)
    temps = [0.2]

    _assert_raises(
        ValueError,
        drifting_loss_for_feature_set,
        x,
        y,
        y[:0],
        omega=omega,
        temps=temps,
        coupling="sinkhorn",
        impl="kernel",
    )

    _assert_raises(
        ValueError,
        drifting_loss_for_feature_set,
        x,
        y,
        y[:0],
        omega=omega,
        temps=temps,
        coupling="row",
        sinkhorn_marginal="weighted_cols",
    )

    # Why the config (coupling="sinkhorn", sinkhorn_marginal="none", omega=1.0) must raise:
    # 1) CFG weight is w = (Nneg-1)*(omega-1)/Nuncond, so omega=1 => w=0.
    # 2) With sinkhorn_marginal="none", the implementation still applies log(w) to
    #    unconditional negative logits; w<=0 means log(w) -> -inf, i.e. those columns
    #    get zero kernel mass.
    # 3) Sinkhorn simultaneously enforces strictly positive column marginals, so the
    #    constraints become infeasible. We fail fast with ValueError instead of NaNs.
    _assert_raises(
        ValueError,
        drifting_loss_for_feature_set,
        x,
        y,
        y[:1],  # Nuncond=1
        omega=torch.tensor(1.0),
        temps=temps,
        coupling="sinkhorn",
        sinkhorn_marginal="none",
        impl="logspace",
        sinkhorn_iters=5,
    )


@torch.no_grad()
def test_sinkhorn_none_rejects_nonpositive_cfg_weight() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 2, 4)
    y_pos = torch.randn(4, 2, 4)
    y_unc = torch.randn(2, 2, 4)
    omega = torch.tensor(1.0)  # -> w = 0

    _assert_raises(
        ValueError,
        drifting_loss_for_feature_set,
        x,
        y_pos,
        y_unc,
        omega=omega,
        temps=[0.05],
        drift_form="alg2_joint",
        coupling="sinkhorn",
        sinkhorn_marginal="none",
        impl="logspace",
    )


@torch.no_grad()
def test_l2_sq_distance_mode_runs() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 2, 6)
    y_pos = torch.randn(5, 2, 6)
    y_unc = torch.randn(3, 2, 6)
    omega = torch.tensor(2.0)
    temps = [0.02, 0.05]

    loss_l2 = drifting_loss_for_feature_set(
        x,
        y_pos,
        y_unc,
        omega=omega,
        temps=temps,
        dist_metric="l2",
    )
    loss_l2_sq = drifting_loss_for_feature_set(
        x,
        y_pos,
        y_unc,
        omega=omega,
        temps=temps,
        dist_metric="l2_sq",
    )
    assert torch.isfinite(loss_l2_sq)
    assert not torch.isclose(loss_l2, loss_l2_sq, rtol=0, atol=1e-9)


@torch.no_grad()
def test_invalid_dist_metric_raises() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 1, 3)
    y = torch.randn(2, 1, 3)
    _assert_raises(
        ValueError,
        drifting_loss_for_feature_set,
        x,
        y,
        y[:0],
        omega=torch.tensor(2.0),
        temps=[0.2],
        dist_metric="bad_metric",  # type: ignore[arg-type]
    )
