import torch

from imagenet.drifting_loss import feature_sets_from_feature_map


def test_feature_std_backward_is_finite_for_constant_maps() -> None:
    # Constant feature maps can have exactly-zero variance, which makes the
    # derivative of sqrt(var) ill-defined at var=0. We want this code path to
    # be numerically stable (no NaN/Inf gradients), since these std features are
    # part of the drifting feature sets (Appendix A.5).
    fmap = torch.zeros((2, 8, 4, 4), dtype=torch.float32, requires_grad=True)

    sets = feature_sets_from_feature_map(fmap, prefix="enc00")
    by_name = {fs.name: fs.x for fs in sets}

    loss = torch.zeros((), dtype=torch.float32)
    global_ms = by_name["enc00.global_ms"]  # [N,2,C] => mean/std
    loss = loss + (global_ms[:, 1, :] ** 2).sum()

    for p in (2, 4):
        k = f"enc00.patch{p}_ms"
        if k not in by_name:
            continue
        patch_ms = by_name[k]  # [N,2*hp*wp,C] => mean vectors then std vectors
        half = patch_ms.shape[1] // 2
        loss = loss + (patch_ms[:, half:, :] ** 2).sum()

    loss.backward()

    assert fmap.grad is not None
    assert torch.isfinite(fmap.grad).all()

