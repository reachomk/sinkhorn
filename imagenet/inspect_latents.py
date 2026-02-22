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
from imagenet.vae_sd import VaeConfig, decode_latents_to_images, load_vae


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick sanity check for pre-encoded ImageNet latents: decode a random subset back to images."
    )
    p.add_argument("--latents-dir", type=str, default="data/imagenet256_latents")
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--out-dir", type=str, default="", help="Output dir (default: outputs/inspect_latents/...).")

    p.add_argument("--num", type=int, default=64, help="Number of random samples to decode.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--save-grid", action="store_true", help="Also write a grid.png montage for quick viewing.")
    p.add_argument("--grid-cols", type=int, default=0, help="Grid columns for --save-grid (0=auto).")

    p.add_argument(
        "--compare-jpeg",
        action="store_true",
        help="Also load original ImageNet JPEGs and save side-by-side comparisons (JPEG vs decoded latent).",
    )
    p.add_argument("--imagenet-root", type=str, default="/home/public/imagenet", help="ImageNet root with train/val folders.")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--vae-id", type=str, default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--vae-scale", type=float, default=0.18215)
    p.add_argument("--vae-dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    return p.parse_args()


def _to_uint8(images: torch.Tensor) -> np.ndarray:
    # images: [B,3,256,256] in [-1,1] -> uint8 HWC
    img = (images.clamp(-1, 1) + 1.0) * 127.5
    return img.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _save_images(images: torch.Tensor, labels: np.ndarray, out_dir: str, indices: np.ndarray) -> np.ndarray:
    # images: [B,3,256,256] in [-1,1]
    img = _to_uint8(images)
    for i in range(img.shape[0]):
        idx = int(indices[i])
        cls = int(labels[i])
        path = os.path.join(out_dir, f"{idx:07d}_cls{cls:04d}.png")
        Image.fromarray(img[i]).save(path)
    return img


def _concat_pairs(left: np.ndarray, right: np.ndarray, *, pad: int = 2) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"Pair shapes must match, got {left.shape} vs {right.shape}")
    if left.ndim != 4 or left.shape[-1] != 3:
        raise ValueError(f"Expected [N,H,W,3], got {left.shape}")
    n, h, _, c = left.shape
    spacer = np.full((n, h, int(pad), c), 255, dtype=np.uint8)
    return np.concatenate([left, spacer, right], axis=2)


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
    if lat.shape[0] != lab.shape[0]:
        raise ValueError(f"Latents/labels length mismatch: lat={lat.shape} lab={lab.shape}")
    if lat.ndim != 4 or lat.shape[1:] != (4, 32, 32):
        raise ValueError(f"Unexpected latent shape: {lat.shape} (expected [N,4,32,32])")

    n = int(lat.shape[0])
    k = min(int(args.num), n)
    rng = np.random.RandomState(int(args.seed))
    idx = rng.choice(n, size=k, replace=False).astype(np.int64)

    out_dir = args.out_dir or os.path.join(
        "outputs",
        "inspect_latents",
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.split}",
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Basic stats on the sampled latents (detect NaNs/inf and gross scaling issues).
    lat_s = lat[idx].astype(np.float32, copy=False)
    stats = {
        "shape": list(map(int, lat.shape)),
        "dtype": str(lat.dtype),
        "sample_n": int(k),
        "finite": bool(np.isfinite(lat_s).all()),
        "mean": float(lat_s.mean()),
        "std": float(lat_s.std()),
        "min": float(lat_s.min()),
        "max": float(lat_s.max()),
    }

    device = torch.device(str(args.device))
    vae_cfg = VaeConfig(
        vae_id=str(args.vae_id),
        scaling_factor=float(args.vae_scale),
        dtype=str(args.vae_dtype),
        encode_mode="mean",
    )
    vae = load_vae(vae_cfg, device=device)

    bs = int(args.batch_size)
    decoded_imgs: list[np.ndarray] = []
    pbar = tqdm(range(0, k, bs), desc="decode", leave=False)
    for off in pbar:
        j = idx[off : off + bs]
        x = torch.from_numpy(lat[j]).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and vae_cfg.dtype == "fp16")):
            imgs = decode_latents_to_images(vae, x, scaling_factor=vae_cfg.scaling_factor)
        u8 = _save_images(imgs, labels=lab[j], out_dir=out_dir, indices=j)
        decoded_imgs.extend([u8[i] for i in range(u8.shape[0])])

    grid_path = None
    if bool(args.save_grid) and decoded_imgs:
        grid_path = os.path.join(out_dir, "grid.png")
        _write_grid(np.stack(decoded_imgs, axis=0), out_path=grid_path, cols=int(args.grid_cols))

    jpeg_grid_path = None
    pairs_grid_path = None
    label_mismatches: list[dict] = []
    if bool(args.compare_jpeg):
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
        jpeg_imgs: list[np.ndarray] = []
        for ii in idx.tolist():
            img_t, lab_jpeg, _ = ds[int(ii)]
            u8_j = _to_uint8(img_t.unsqueeze(0))[0]
            jpeg_imgs.append(u8_j)

            lab_lat = int(lab[int(ii)])
            if int(lab_jpeg) != lab_lat:
                label_mismatches.append({"idx": int(ii), "label_latents": int(lab_lat), "label_jpeg": int(lab_jpeg)})

            out_path = os.path.join(out_dir, f"{int(ii):07d}_cls{lab_lat:04d}_jpeg.png")
            Image.fromarray(u8_j).save(out_path)

        if bool(args.save_grid) and jpeg_imgs:
            jpeg_grid_path = os.path.join(out_dir, "grid_jpeg.png")
            pairs_grid_path = os.path.join(out_dir, "grid_pairs.png")
            arr_j = np.stack(jpeg_imgs, axis=0)
            arr_d = np.stack(decoded_imgs, axis=0)
            _write_grid(arr_j, out_path=jpeg_grid_path, cols=int(args.grid_cols))
            _write_grid(_concat_pairs(arr_j, arr_d, pad=2), out_path=pairs_grid_path, cols=int(args.grid_cols))

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "latents_path": str(Path(lat_path).resolve()),
        "labels_path": str(Path(lab_path).resolve()),
        "out_dir": str(Path(out_dir).resolve()),
        "grid_path": str(Path(grid_path).resolve()) if grid_path is not None else None,
        "jpeg_grid_path": str(Path(jpeg_grid_path).resolve()) if jpeg_grid_path is not None else None,
        "pairs_grid_path": str(Path(pairs_grid_path).resolve()) if pairs_grid_path is not None else None,
        "save_grid": bool(args.save_grid),
        "grid_cols": int(args.grid_cols),
        "compare_jpeg": bool(args.compare_jpeg),
        "imagenet_root": str(args.imagenet_root) if bool(args.compare_jpeg) else None,
        "jpeg_label_mismatch_count": int(len(label_mismatches)) if bool(args.compare_jpeg) else None,
        "jpeg_label_mismatches": label_mismatches if bool(args.compare_jpeg) and label_mismatches else None,
        "seed": int(args.seed),
        "indices": idx.tolist(),
        "stats": stats,
        "vae": asdict(vae_cfg),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"Wrote {k} decoded images to: {out_dir}")


if __name__ == "__main__":
    main()
