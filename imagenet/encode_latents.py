"""
Encode ImageNet RGB images to SD-VAE latents (ImageNet256 → 4×32×32).

This is Stage 1 of the ImageNet pipeline in Generative Modeling via Drifting:
- Stage 1: encode RGB images to SD-VAE latents (this script)
- Stage 2: pretrain a latent-space feature encoder via ResNet-style MAE (Appendix A.3; 'train_mae.py')
- Stage 3: train the drifting generator on latents (Kaiming'spaper §5.2; 'train_drifting.py')

Paper alignment
-------------------------
- The latent space matches the SD-VAE tokenizer used in the paper
  ('stabilityai/sd-vae-ft-ema', scaling factor '0.18215').
- The RandomResizedCrop + hflip options implement the paper-style augmentation
  used before VAE encoding for latent-MAE pretraining (Appendix A.3).

Note:
- In this study, we keep the same data preparation,
  and only compare different drift/coupling algorithms.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from imagenet.data.imagenet_folders import build_imagenet_dataset
from imagenet.data.latents_memmap import (
    LatentsShardPaths,
    final_paths,
    merge_shards_to_final,
    open_latents_memmap,
    shard_paths,
    write_meta,
)
from imagenet.utils.dist import DistInfo, barrier, init_distributed, is_main_process
from imagenet.utils.misc import seed_all
from imagenet.vae_sd import VaeConfig, encode_images_to_latents, load_vae


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode ImageNet images to SD-VAE latents (32x32x4).")
    p.add_argument("--imagenet-root", type=str, default="/home/public/imagenet")
    p.add_argument("--split", type=str, choices=["train", "val"], required=True)
    p.add_argument("--out-dir", type=str, default="data/imagenet256_latents")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Resize+CenterCrop to 256x256 (recommended for 32x32 latents).",
    )
    p.add_argument(
        "--random-resized-crop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use RandomResizedCrop to 256x256 (paper-style augmentation for latent-MAE pretrain).",
    )
    p.add_argument("--rrc-scale-min", type=float, default=0.2, help="Min scale for RandomResizedCrop (if enabled).")
    p.add_argument("--rrc-scale-max", type=float, default=1.0, help="Max scale for RandomResizedCrop (if enabled).")
    p.add_argument("--hflip-prob", type=float, default=0.0, help="Horizontal flip probability (0=disable).")

    p.add_argument("--vae-id", type=str, default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--vae-scale", type=float, default=0.18215)
    p.add_argument("--vae-dtype", type=str, choices=["fp16", "fp32"], default="fp32")
    p.add_argument("--vae-encode", type=str, choices=["mean", "sample"], default="mean")

    p.add_argument("--merge", action="store_true", help="If DDP, merge shards into a single file on rank0.")
    p.add_argument("--max-items", type=int, default=0, help="Debug: encode only first N items (0=all).")
    return p.parse_args()


def _image_transform(
    *,
    center_crop: bool,
    random_resized_crop: bool,
    rrc_scale_min: float,
    rrc_scale_max: float,
    hflip_prob: float,
):
    t = []
    if random_resized_crop:
        t.append(
            transforms.RandomResizedCrop(
                256,
                scale=(float(rrc_scale_min), float(rrc_scale_max)),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )
    else:
        t.append(transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR))
        if center_crop:
            t.append(transforms.CenterCrop(256))
    if hflip_prob and hflip_prob > 0:
        t.append(transforms.RandomHorizontalFlip(p=float(hflip_prob)))
    t.append(transforms.ToTensor())  # [0,1]
    t.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))  # [-1,1]
    return transforms.Compose(t)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.rrc_scale_min > args.rrc_scale_max:
        raise ValueError(f"--rrc-scale-min ({args.rrc_scale_min}) must be <= --rrc-scale-max ({args.rrc_scale_max})")
    if args.hflip_prob < 0.0 or args.hflip_prob > 1.0:
        raise ValueError(f"--hflip-prob must be in [0,1], got {args.hflip_prob}")
    dist_info: DistInfo = init_distributed(device=args.device)
    seed_all(args.seed + dist_info.rank)

    transform = _image_transform(
        center_crop=bool(args.center_crop),
        random_resized_crop=bool(args.random_resized_crop),
        rrc_scale_min=float(args.rrc_scale_min),
        rrc_scale_max=float(args.rrc_scale_max),
        hflip_prob=float(args.hflip_prob),
    )
    ds, _ = build_imagenet_dataset(args.imagenet_root, args.split, transform=transform)
    total_n = len(ds)
    if args.max_items and args.max_items > 0:
        total_n = min(total_n, int(args.max_items))
        ds.samples = ds.samples[:total_n]  # type: ignore[attr-defined]
        ds.targets = ds.targets[:total_n]  # type: ignore[attr-defined]

    # IMPORTANT: torch DistributedSampler pads when dataset size is not divisible by world_size,
    # causing duplicates. For offline encoding we want each index exactly once, so we split indices
    # ourselves without padding.
    shard_indices = list(range(dist_info.rank, total_n, dist_info.world_size))
    ds_shard = Subset(ds, shard_indices)

    g = torch.Generator()
    g.manual_seed(int(args.seed) + int(dist_info.rank))

    def _worker_init_fn(worker_id: int) -> None:
        # Torch seeds each DataLoader worker; propagate it to numpy/python for transforms that may use them.
        seed = int(torch.initial_seed()) % (2**32)
        np.random.seed(seed)
        random.seed(seed)

    dl = DataLoader(
        ds_shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
        generator=g,
    )

    vae_cfg = VaeConfig(
        vae_id=args.vae_id,
        scaling_factor=float(args.vae_scale),
        dtype=args.vae_dtype,
        encode_mode=args.vae_encode,
    )
    vae = load_vae(vae_cfg, device=dist_info.device)

    # Shard memmaps
    shard_n = len(shard_indices)
    if dist_info.world_size == 1:
        lat_path, lab_path = final_paths(args.out_dir, args.split)
        lat_mm = open_latents_memmap(lat_path, shape=(shard_n, 4, 32, 32), dtype=np.float16, mode="w+")
        lab_mm = open_latents_memmap(lab_path, shape=(shard_n,), dtype=np.int64, mode="w+")
        idx_mm = None
    else:
        sp: LatentsShardPaths = shard_paths(args.out_dir, args.split, dist_info.rank)
        lat_mm = open_latents_memmap(sp.latents_path, shape=(shard_n, 4, 32, 32), dtype=np.float16, mode="w+")
        lab_mm = open_latents_memmap(sp.labels_path, shape=(shard_n,), dtype=np.int64, mode="w+")
        idx_mm = open_latents_memmap(sp.indices_path, shape=(shard_n,), dtype=np.int64, mode="w+")

    pbar = tqdm(dl, disable=not is_main_process(), desc=f"encode[{args.split}]")
    write_ptr = 0
    for images, labels, indices in pbar:
        images = images.to(dist_info.device, non_blocking=True)
        with torch.autocast(device_type=dist_info.device.type, enabled=(dist_info.device.type == "cuda" and args.vae_dtype == "fp16")):
            latents = encode_images_to_latents(
                vae,
                images,
                scaling_factor=vae_cfg.scaling_factor,
                encode_mode=vae_cfg.encode_mode,
            )
        lat_np = latents.detach().cpu().to(torch.float16).numpy()
        lab_np = labels.numpy().astype(np.int64, copy=False)
        idx_np = indices.numpy().astype(np.int64, copy=False)

        bsz = lat_np.shape[0]
        lat_mm[write_ptr : write_ptr + bsz] = lat_np
        lab_mm[write_ptr : write_ptr + bsz] = lab_np
        if idx_mm is not None:
            idx_mm[write_ptr : write_ptr + bsz] = idx_np
        write_ptr += bsz

    if write_ptr != shard_n:
        raise RuntimeError(f"Shard write count mismatch: wrote {write_ptr} but expected {shard_n}")
    lat_mm.flush()
    lab_mm.flush()
    if idx_mm is not None:
        idx_mm.flush()

    barrier()

    if dist_info.distributed and args.merge and is_main_process():
        merge_shards_to_final(args.out_dir, args.split, dist_info.world_size, total_n=total_n)

    if is_main_process():
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "imagenet_root": args.imagenet_root,
            "split": args.split,
            "total_n": total_n,
            "seed": int(args.seed),
            "world_size": int(dist_info.world_size),
            "preprocess": {
                "image_size": 256,
                "mode": "random_resized_crop" if bool(args.random_resized_crop) else ("resize_center_crop" if bool(args.center_crop) else "resize"),
                "center_crop": bool(args.center_crop),
                "random_resized_crop": bool(args.random_resized_crop),
                "rrc_scale": [float(args.rrc_scale_min), float(args.rrc_scale_max)] if bool(args.random_resized_crop) else None,
                "hflip_prob": float(args.hflip_prob),
            },
            "vae": asdict(vae_cfg),
        }
        meta_path = write_meta(args.out_dir, args.split, meta)
        print(f"Wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
