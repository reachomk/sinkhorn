from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from imagenet.data.latents_memmap import final_paths
from imagenet.drifting_loss import extract_feature_sets
from imagenet.models.ema import EMA
from imagenet.models.resnet_mae import ResNetMAE, ResNetMAEConfig
from imagenet.vae_sd import VaeConfig, decode_latents_to_images, load_vae


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick latent-MAE sanity check: reconstruct masked latents and decode to images for visual inspection."
    )
    p.add_argument("--mae-ckpt", type=str, required=True, help="Path to latent-MAE checkpoint (.pt).")
    p.add_argument("--mae-use-ema", action="store_true", help="Use EMA weights from checkpoint (recommended).")

    p.add_argument("--latents-dir", type=str, default="data/imagenet256_latents")
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--out-dir", type=str, default="", help="Output dir (default: outputs/inspect_mae/...).")

    p.add_argument("--num", type=int, default=32, help="Number of random samples to inspect.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mask-ratio", type=float, default=0.5)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", help="Use automatic mixed precision for MAE forward (CUDA only).")
    p.add_argument("--amp-dtype", type=str, choices=["fp16", "bf16"], default="fp16")

    p.add_argument(
        "--decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode latents to RGB with SD-VAE and save PNG triplets (slower, but useful for visual inspection).",
    )
    p.add_argument("--save-grid", action="store_true", help="If decoding, also save grid montages for easier comparison.")
    p.add_argument("--grid-cols", type=int, default=0, help="Grid columns for --save-grid (0=auto).")
    p.add_argument("--compare-jpeg", action="store_true", help="Also save original ImageNet JPEGs for side-by-side comparison.")
    p.add_argument("--imagenet-root", type=str, default="/home/public/imagenet", help="ImageNet root with train/val folders.")
    p.add_argument(
        "--check-feature-sets",
        action="store_true",
        help="Also extract multi-scale feature sets from the MAE encoder as used by the drifting loss (Appendix A.5).",
    )
    p.add_argument("--feature-every-n-blocks", type=int, default=2, help="every_n_blocks for encoder.forward_feature_maps.")
    p.add_argument(
        "--include-input-x2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the extra input feature E[x^2] used by the drifting loss (Appendix A.5).",
    )

    p.add_argument("--vae-id", type=str, default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--vae-scale", type=float, default=0.18215)
    p.add_argument("--vae-dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    return p.parse_args()


def _mask_latents(lat: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    if lat.shape[-2:] != (32, 32):
        raise ValueError(f"Expected latents spatial size 32x32, got {tuple(lat.shape)}")
    b = lat.shape[0]
    device = lat.device
    mask_p = (torch.rand(b, 1, 16, 16, device=device) < float(mask_ratio)).to(lat.dtype)
    mask_px = mask_p.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # [B,1,32,32]
    lat_masked = lat * (1.0 - mask_px)
    return lat_masked, mask_px


def _to_uint8(images: torch.Tensor) -> np.ndarray:
    # images: [B,3,256,256] in [-1,1] -> uint8 HWC
    img = (images.clamp(-1, 1) + 1.0) * 127.5
    return img.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _save_triplet(
    *,
    idx: int,
    cls: int,
    out_dir: str,
    orig: np.ndarray,
    masked: np.ndarray,
    recon: np.ndarray,
) -> None:
    Image.fromarray(orig).save(os.path.join(out_dir, f"{idx:07d}_cls{cls:04d}_orig.png"))
    Image.fromarray(masked).save(os.path.join(out_dir, f"{idx:07d}_cls{cls:04d}_masked.png"))
    Image.fromarray(recon).save(os.path.join(out_dir, f"{idx:07d}_cls{cls:04d}_recon.png"))


def _write_grid(images_u8: np.ndarray, *, out_path: str, cols: int, pad: int = 2) -> None:
    if images_u8.ndim != 4 or images_u8.shape[-1] != 3:
        raise ValueError(f"Expected images_u8 [N,H,W,3], got {images_u8.shape}")
    n, h, w, _ = images_u8.shape
    if n == 0:
        return
    cols_eff = int(cols) if int(cols) > 0 else int(math.ceil(math.sqrt(n)))
    cols_eff = max(1, cols_eff)
    rows_eff = int(math.ceil(n / cols_eff))
    canvas = np.full(
        (rows_eff * h + pad * (rows_eff - 1), cols_eff * w + pad * (cols_eff - 1), 3),
        255,
        dtype=np.uint8,
    )
    for i in range(n):
        r = i // cols_eff
        c = i % cols_eff
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0 : y0 + h, x0 : x0 + w] = images_u8[i]
    Image.fromarray(canvas).save(out_path)


def _concat_triplets(orig: np.ndarray, masked: np.ndarray, recon: np.ndarray, *, pad: int = 2) -> np.ndarray:
    if orig.shape != masked.shape or orig.shape != recon.shape:
        raise ValueError(f"Triplet image shapes must match, got {orig.shape}, {masked.shape}, {recon.shape}")
    if orig.ndim != 4 or orig.shape[-1] != 3:
        raise ValueError(f"Expected orig/masked/recon [N,H,W,3], got {orig.shape}")
    n, h, _, c = orig.shape
    spacer = np.full((n, h, int(pad), c), 255, dtype=np.uint8)
    return np.concatenate([orig, spacer, masked, spacer, recon], axis=2)


def _concat_quads(jpeg: np.ndarray, orig: np.ndarray, masked: np.ndarray, recon: np.ndarray, *, pad: int = 2) -> np.ndarray:
    if jpeg.shape != orig.shape or orig.shape != masked.shape or orig.shape != recon.shape:
        raise ValueError(f"Quad image shapes must match, got {jpeg.shape}, {orig.shape}, {masked.shape}, {recon.shape}")
    if orig.ndim != 4 or orig.shape[-1] != 3:
        raise ValueError(f"Expected jpeg/orig/masked/recon [N,H,W,3], got {orig.shape}")
    n, h, _, c = orig.shape
    spacer = np.full((n, h, int(pad), c), 255, dtype=np.uint8)
    return np.concatenate([jpeg, spacer, orig, spacer, masked, spacer, recon], axis=2)


def _tensor_stats(x: torch.Tensor) -> dict:
    """Small helper for sanity-check stats (computed in float32)."""
    x_f = x.detach().float()
    return {
        "shape": list(map(int, x.shape)),
        "finite": bool(torch.isfinite(x_f).all().item()),
        "mean": float(x_f.mean().item()),
        "std": float(x_f.std(unbiased=False).item()),
        "min": float(x_f.min().item()),
        "max": float(x_f.max().item()),
    }


@torch.no_grad()
def main() -> None:
    args = _parse_args()

    lat_path, lab_path = final_paths(args.latents_dir, args.split)
    if not os.path.exists(lat_path) or not os.path.exists(lab_path):
        raise FileNotFoundError(
            "Missing latent files. Expected:\n"
            f"  {lat_path}\n"
            f"  {lab_path}\n"
            "Run something like:\n"
            f"  torchrun --standalone --nproc_per_node=NUM_GPUS -m imagenet.encode_latents --split {args.split} --out-dir {args.latents_dir} --merge ..."
        )
    lat = np.load(lat_path, mmap_mode="r")
    lab = np.load(lab_path, mmap_mode="r")

    n = int(lat.shape[0])
    k = min(int(args.num), n)
    rng = np.random.RandomState(int(args.seed))
    idx = rng.choice(n, size=k, replace=False).astype(np.int64)

    out_dir = args.out_dir or os.path.join(
        "outputs",
        "inspect_mae",
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.split}",
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(str(args.device))
    use_amp = bool(args.amp) and device.type == "cuda"
    amp_dtype = torch.float16 if str(args.amp_dtype) == "fp16" else torch.bfloat16
    decode = bool(args.decode)
    save_grid = bool(args.save_grid)
    compare_jpeg = bool(args.compare_jpeg)
    if save_grid and not decode:
        raise ValueError("--save-grid requires decoding; remove --no-decode.")
    if compare_jpeg and not decode:
        raise ValueError("--compare-jpeg requires decoding; remove --no-decode.")

    # Reproducible masking (torch.rand) for easier comparisons across runs.
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    # Load MAE checkpoint.
    ckpt = torch.load(args.mae_ckpt, map_location="cpu")
    mae_cfg = ResNetMAEConfig(**ckpt["mae_cfg"]) if isinstance(ckpt.get("mae_cfg"), dict) else ResNetMAEConfig()
    mae = ResNetMAE(mae_cfg)
    mae.load_state_dict(ckpt["model"], strict=True)
    if bool(args.mae_use_ema) and "ema" in ckpt:
        ema_tmp = EMA(mae, decay=float(ckpt["ema"].get("decay", 0.9995)))
        ema_tmp.load_state_dict(ckpt["ema"])
        ema_tmp.copy_to(mae)
    mae.to(device)
    mae.eval()

    # Load VAE only when needed for visualization.
    vae_cfg = None
    vae = None
    if decode:
        vae_cfg = VaeConfig(
            vae_id=str(args.vae_id),
            scaling_factor=float(args.vae_scale),
            dtype=str(args.vae_dtype),
            encode_mode="mean",
        )
        vae = load_vae(vae_cfg, device=device)

    jpeg_imgs: list[np.ndarray] | None = None
    jpeg_label_mismatches: list[dict] = []
    if compare_jpeg:
        # JPEG comparison uses a deterministic Resize+CenterCrop(256). If your latents were
        # encoded with RandomResizedCrop / hflip augmentation, the JPEG view may not match
        # the decoded latent pixel-by-pixel, but should still be semantically consistent.
        from torchvision import transforms

        from imagenet.data.imagenet_folders import build_imagenet_dataset

        jpeg_tf = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor(),  # [0,1]
                transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
            ]
        )
        ds, _ = build_imagenet_dataset(args.imagenet_root, args.split, transform=jpeg_tf)
        jpeg_imgs = []
        for ii in idx.tolist():
            img_t, lab_jpeg, _ = ds[int(ii)]
            u8_j = _to_uint8(img_t.unsqueeze(0))[0]
            jpeg_imgs.append(u8_j)

            lab_lat = int(lab[int(ii)])
            if int(lab_jpeg) != lab_lat:
                jpeg_label_mismatches.append({"idx": int(ii), "label_latents": int(lab_lat), "label_jpeg": int(lab_jpeg)})

            out_path = os.path.join(out_dir, f"{int(ii):07d}_cls{lab_lat:04d}_jpeg.png")
            Image.fromarray(u8_j).save(out_path)

    bs = int(args.batch_size)
    losses: list[float] = []
    full_mse: list[float] = []
    feature_check: dict | None = None
    grid_orig: list[np.ndarray] = []
    grid_masked: list[np.ndarray] = []
    grid_recon: list[np.ndarray] = []
    pbar = tqdm(range(0, k, bs), desc="recon", leave=False)
    for off in pbar:
        j = idx[off : off + bs]
        x = torch.from_numpy(lat[j]).to(device).float()

        x_masked, mask_px = _mask_latents(x, mask_ratio=float(args.mask_ratio))
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            recon = mae(x_masked)

        mask = mask_px.expand_as(x)
        denom = mask.sum().clamp_min(1.0)
        loss = ((recon.float() - x.float()) ** 2 * mask.float()).sum() / denom.float()
        losses.append(float(loss.detach().item()))
        full_mse.append(float(((recon.float() - x.float()) ** 2).mean().detach().item()))

        if bool(args.check_feature_sets) and feature_check is None:
            # Stage-3 uses the encoder ϖ on *unmasked* latents, so we validate on x (not x_masked).
            encoder = mae.encoder
            every_n = int(args.feature_every_n_blocks)
            fs = extract_feature_sets(
                encoder,
                x,
                every_n_blocks=every_n,
                include_input_x2=bool(args.include_input_x2),
            )
            feature_check = {
                "note": "Computed on the first inspected batch only.",
                "every_n_blocks": int(every_n),
                "include_input_x2": bool(args.include_input_x2),
                "latents_stats": _tensor_stats(x),
                "feature_sets": [{"name": f.name, **_tensor_stats(f.x)} for f in fs],
            }

        if decode:
            assert vae is not None and vae_cfg is not None
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda" and vae_cfg.dtype == "fp16"),
            ):
                img_orig = decode_latents_to_images(vae, x, scaling_factor=vae_cfg.scaling_factor)
                img_masked = decode_latents_to_images(vae, x_masked, scaling_factor=vae_cfg.scaling_factor)
                img_recon = decode_latents_to_images(vae, recon.float(), scaling_factor=vae_cfg.scaling_factor)

            u8_orig = _to_uint8(img_orig)
            u8_masked = _to_uint8(img_masked)
            u8_recon = _to_uint8(img_recon)

            if save_grid:
                grid_orig.extend([u8_orig[i] for i in range(u8_orig.shape[0])])
                grid_masked.extend([u8_masked[i] for i in range(u8_masked.shape[0])])
                grid_recon.extend([u8_recon[i] for i in range(u8_recon.shape[0])])

            labels = lab[j].astype(np.int64, copy=False)
            for bi in range(u8_orig.shape[0]):
                _save_triplet(
                    idx=int(j[bi]),
                    cls=int(labels[bi]),
                    out_dir=out_dir,
                    orig=u8_orig[bi],
                    masked=u8_masked[bi],
                    recon=u8_recon[bi],
                )

    masked_p = None
    full_p = None
    if losses:
        masked_p = np.percentile(np.asarray(losses, dtype=np.float64), [50, 90, 99]).tolist()
    if full_mse:
        full_p = np.percentile(np.asarray(full_mse, dtype=np.float64), [50, 90, 99]).tolist()

    grid_paths = None
    if decode and save_grid and grid_orig:
        cols = int(args.grid_cols)
        arr_o = np.stack(grid_orig, axis=0)
        arr_m = np.stack(grid_masked, axis=0)
        arr_r = np.stack(grid_recon, axis=0)
        arr_t = _concat_triplets(arr_o, arr_m, arr_r, pad=2)

        p_trip = os.path.join(out_dir, "grid_triplets.png")
        p_orig = os.path.join(out_dir, "grid_orig.png")
        p_mask = os.path.join(out_dir, "grid_masked.png")
        p_recon = os.path.join(out_dir, "grid_recon.png")
        _write_grid(arr_t, out_path=p_trip, cols=cols)
        _write_grid(arr_o, out_path=p_orig, cols=cols)
        _write_grid(arr_m, out_path=p_mask, cols=cols)
        _write_grid(arr_r, out_path=p_recon, cols=cols)

        p_jpeg = None
        p_quads = None
        if compare_jpeg and jpeg_imgs is not None and len(jpeg_imgs) == arr_o.shape[0]:
            arr_j = np.stack(jpeg_imgs, axis=0)
            arr_q = _concat_quads(arr_j, arr_o, arr_m, arr_r, pad=2)
            p_jpeg = os.path.join(out_dir, "grid_jpeg.png")
            p_quads = os.path.join(out_dir, "grid_quads.png")
            _write_grid(arr_j, out_path=p_jpeg, cols=cols)
            _write_grid(arr_q, out_path=p_quads, cols=cols)

        grid_paths = {
            "triplets": str(Path(p_trip).resolve()),
            "orig": str(Path(p_orig).resolve()),
            "masked": str(Path(p_mask).resolve()),
            "recon": str(Path(p_recon).resolve()),
            "jpeg": str(Path(p_jpeg).resolve()) if p_jpeg is not None else None,
            "quads": str(Path(p_quads).resolve()) if p_quads is not None else None,
            "grid_cols": int(cols),
        }

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "latents_path": str(Path(lat_path).resolve()),
        "labels_path": str(Path(lab_path).resolve()),
        "mae_ckpt": str(Path(args.mae_ckpt).resolve()),
        "mae_use_ema": bool(args.mae_use_ema),
        "split": str(args.split),
        "seed": int(args.seed),
        "indices": idx.tolist(),
        "mask_ratio": float(args.mask_ratio),
        "masked_mse_mean": float(np.mean(losses)) if losses else None,
        "masked_mse_std": float(np.std(losses)) if losses else None,
        "masked_mse_p50_p90_p99": masked_p,
        "full_mse_mean": float(np.mean(full_mse)) if full_mse else None,
        "full_mse_std": float(np.std(full_mse)) if full_mse else None,
        "full_mse_p50_p90_p99": full_p,
        "decode": bool(decode),
        "save_grid": bool(save_grid),
        "grid_cols": int(args.grid_cols),
        "grid_paths": grid_paths,
        "compare_jpeg": bool(compare_jpeg),
        "imagenet_root": str(args.imagenet_root) if bool(compare_jpeg) else None,
        "jpeg_label_mismatch_count": int(len(jpeg_label_mismatches)) if bool(compare_jpeg) else None,
        "jpeg_label_mismatches": jpeg_label_mismatches if bool(compare_jpeg) and jpeg_label_mismatches else None,
        "vae": asdict(vae_cfg) if vae_cfg is not None else None,
        "mae_cfg": asdict(mae_cfg),
        "feature_check": feature_check,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"Wrote recon triplets to: {out_dir}")


if __name__ == "__main__":
    main()
