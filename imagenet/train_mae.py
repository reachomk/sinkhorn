"""
Pretrain the ResNet-style latent-MAE feature encoder ϖ for drifting (Appendix A.3).

This script corresponds to **Appendix A.3 (Implementation of ResNet-style MAE)**
in *Generative Modeling via Drifting* and produces the feature encoder used in the
ImageNet drifting experiments.

Scope note:
- This stage is shared by baseline Drift training and follow-up Sinkhorn-flow
  experiments; only Stage 3 (`train_drifting.py` / `drifting_loss.py`) changes
  algorithmic coupling behavior.

Pipeline context (ImageNet256)
------------------------------
Stage 1 (data prep): 'imagenet.encode_latents'
  ImageNet RGB images -> SD-VAE latents '[B,4,32,32]'.

Stage 2 (this script): latent-MAE pretraining
  Latents -> mask -> ResNetMAE -> reconstruct -> masked MSE.

Stage 3 (main training): 'imagenet.train_drifting'
  Train the generator with drifting loss computed in the feature space of ϖ
  (Appendix A.5–A.7). Only the **encoder feature maps** are used in Stage 3;
  the MAE decoder is only for Stage 2 pretraining.

Data paths
----------
Two equivalent input paths are supported:
- **Latents mode** (recommended): read pre-encoded SD-VAE latents from '--latents-dir'
  (produced by 'imagenet.encode_latents --merge').
- **Images mode**: decode JPEGs, apply augmentation, and encode with SD-VAE on-the-fly.

Tensor shapes
-------------
- images:  '[B, 3, 256, 256]' float in '[-1, 1]'
- latents: '[B, 4, 32, 32]' SD-VAE latents, scaled by '--vae-scale' (default '0.18215')

MAE objective (Appendix A.3, “Masking”)
--------------------------------------
We mask **2×2 blocks** on the 32×32 latent grid:
- 'mask_p':  '[B,1,16,16]' with 'P(mask)=mask_ratio'
- 'mask_px': '[B,1,32,32]' by expanding each cell into a 2×2 block
- 'lat_masked = lat * (1 - mask_px)'

The training loss is a masked MSE over masked positions only:
  'loss = sum(((recon - lat)^2) * mask) / sum(mask)'.

Output
------
Writes checkpoints ('ckpt_*.pt', 'ckpt_final.pt') containing model + EMA weights.
Later, 'imagenet.train_drifting' loads this checkpoint via '--mae-ckpt' and uses
'encoder.forward_feature_maps(...)' to build multi-scale feature sets (Appendix A.5).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import transforms
from tqdm import tqdm

from imagenet.data.imagenet_folders import build_imagenet_dataset
from imagenet.data.latents_memmap import LatentsDataset, final_paths
from imagenet.models.ema import EMA
from imagenet.models.resnet_mae import ResNetMAE, ResNetMAEConfig
from imagenet.utils.dist import barrier, broadcast_object, init_distributed, is_main_process
from imagenet.utils.misc import seed_all
from imagenet.utils.runs import RunPaths, create_run_dir
from imagenet.vae_sd import VaeConfig, encode_images_to_latents, load_vae


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain latent-MAE ResNet encoder (Appendix A.3).")
    p.add_argument("--imagenet-root", type=str, default="/home/public/imagenet")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    # Data
    p.add_argument(
        "--latents-dir",
        type=str,
        default="",
        help="Optional: train directly from pre-encoded SD-VAE latents in this directory (skips image decode + VAE encode).",
    )
    p.add_argument("--latents-split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--batch-size", type=int, default=32, help="Per-process microbatch (images).")
    p.add_argument("--global-batch", type=int, default=8192, help="Target effective global batch size.")
    p.add_argument("--grad-accum", type=int, default=0, help="Override gradient accumulation steps (0=auto).")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--max-items", type=int, default=0, help="Debug: use only first N train images (0=all).")

    # Performance
    p.add_argument("--amp", action="store_true", help="Use automatic mixed precision for MAE forward/backward (CUDA only).")
    p.add_argument("--amp-dtype", type=str, choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32 matmul/conv on NVIDIA Ampere+.")
    p.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cuDNN benchmark (often faster for fixed shapes; may reduce determinism).",
    )
    p.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Try fused AdamW if available (PyTorch>=2.0 CUDA).",
    )

    # Augmentation (random resized crop before VAE encode; Appendix A.3)
    p.add_argument("--rrc-scale-min", type=float, default=0.2)
    p.add_argument("--rrc-scale-max", type=float, default=1.0)

    # MAE
    p.add_argument("--in-ch", type=int, default=4)
    p.add_argument("--base-width", type=int, default=256)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=192)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--grad-clip", type=float, default=0.0)

    # VAE
    p.add_argument("--vae-id", type=str, default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--vae-scale", type=float, default=0.18215)
    p.add_argument("--vae-dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--vae-encode", type=str, choices=["mean", "sample"], default="mean")

    # Logging / checkpoints
    p.add_argument("--run-root", type=str, default="runs/imagenet_mae")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--resume", type=str, default="", help="Resume from a checkpoint path (.pt).")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=0, help="Debug: stop after N optimizer steps (0=all).")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _build_transform(args: argparse.Namespace):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                256,
                scale=(args.rrc_scale_min, args.rrc_scale_max),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # [0,1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
        ]
    )


def _mask_latents(lat: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mask 2x2 patches on a 32x32 latent grid by zeroing (Appendix A.3).

    Returns:
      lat_masked: [B,4,32,32]
      mask_px:    [B,1,32,32] with 1 where masked
    """
    if lat.shape[-2:] != (32, 32):
        raise ValueError(f"Expected latents spatial size 32x32, got {tuple(lat.shape)}")
    b = lat.shape[0]
    device = lat.device
    mask_p = (torch.rand(b, 1, 16, 16, device=device) < float(mask_ratio)).to(lat.dtype)
    mask_px = mask_p.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # [B,1,32,32]
    lat_masked = lat * (1.0 - mask_px)
    return lat_masked, mask_px


def _infer_run_paths_from_ckpt(ckpt_path: str) -> RunPaths:
    p = Path(ckpt_path).resolve()
    if p.parent.name == "checkpoints":
        run_dir = p.parent.parent
    else:
        run_dir = p.parent
    ckpt_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    return RunPaths(run_dir=str(run_dir), ckpt_dir=str(ckpt_dir), samples_dir=str(samples_dir))


def _append_resume_cmd(run_dir: str, resume_path: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    line = f"[resume {ts}] {resume_path}\n"
    path = os.path.join(run_dir, "cmd.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    args = _parse_args()
    use_latents = bool(args.latents_dir)
    if args.debug:
        args.epochs = min(args.epochs, 1)
        args.max_items = args.max_items or 2048
        args.max_steps = args.max_steps or 50
        args.num_workers = 0

    dist_info = init_distributed(device=args.device)
    seed_all(args.seed + dist_info.rank)

    if dist_info.device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    use_amp = bool(args.amp) and dist_info.device.type == "cuda"
    amp_dtype = torch.float16 if str(args.amp_dtype) == "fp16" else torch.bfloat16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if dist_info.device.type == "cuda" else None

    if use_latents:
        lat_path, lab_path = final_paths(args.latents_dir, args.latents_split)
        if not os.path.exists(lat_path) or not os.path.exists(lab_path):
            raise FileNotFoundError(
                "Missing latent files. Expected:\n"
                f"  {lat_path}\n"
                f"  {lab_path}\n"
                "Run something like:\n"
                f"  torchrun --standalone --nproc_per_node=NUM_GPUS -m imagenet.encode_latents --split {args.latents_split} --out-dir {args.latents_dir} --merge ..."
            )
        ds = LatentsDataset(lat_path, lab_path)
        if args.max_items and args.max_items > 0:
            n = min(len(ds), int(args.max_items))
            ds = Subset(ds, list(range(n)))
    else:
        transform = _build_transform(args)
        ds, _ = build_imagenet_dataset(args.imagenet_root, "train", transform=transform)
        if args.max_items and args.max_items > 0:
            n = min(len(ds), int(args.max_items))
            ds = Subset(ds, list(range(n)))

    sampler = DistributedSampler(ds, shuffle=True, drop_last=False) if dist_info.distributed else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )

    # Auto grad-accum to match target global batch.
    per_step = args.batch_size * dist_info.world_size
    grad_accum = int(args.grad_accum) if args.grad_accum and args.grad_accum > 0 else math.ceil(args.global_batch / per_step)
    grad_accum = max(1, grad_accum)
    eff_global = per_step * grad_accum
    if is_main_process():
        print(f"data={'latents' if use_latents else 'images'}")
        print(f"DDP={dist_info.distributed} world={dist_info.world_size} per_step={per_step} grad_accum={grad_accum} eff_global={eff_global}")
        print(f"batches/epoch={len(dl)} opt_steps/epoch≈{len(dl) // grad_accum}")

    vae_cfg = None
    vae = None
    if not use_latents:
        vae_cfg = VaeConfig(
            vae_id=args.vae_id,
            scaling_factor=float(args.vae_scale),
            dtype=args.vae_dtype,
            encode_mode=args.vae_encode,
        )
        vae = load_vae(vae_cfg, device=dist_info.device)

    mae_cfg = ResNetMAEConfig(in_ch=args.in_ch, base_width=args.base_width)
    model = ResNetMAE(mae_cfg).to(dist_info.device)
    ddp_model = DDP(model, device_ids=[dist_info.local_rank]) if dist_info.distributed and dist_info.device.type == "cuda" else model

    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if bool(args.fused_adamw) and dist_info.device.type == "cuda":
        try:
            opt = torch.optim.AdamW(ddp_model.parameters(), fused=True, **opt_kwargs)
        except TypeError:
            if is_main_process():
                print("Warning: fused AdamW not available; falling back to standard AdamW.")
            opt = torch.optim.AdamW(ddp_model.parameters(), **opt_kwargs)
    else:
        opt = torch.optim.AdamW(ddp_model.parameters(), **opt_kwargs)
    ema = EMA(model, decay=args.ema_decay)

    run_name = args.run_name or f"latent_mae_w{args.base_width}_ep{args.epochs}"
    if args.resume:
        if is_main_process():
            run = _infer_run_paths_from_ckpt(args.resume)
            os.makedirs(run.ckpt_dir, exist_ok=True)
            os.makedirs(run.samples_dir, exist_ok=True)
            _append_resume_cmd(run.run_dir, args.resume)
        else:
            run = RunPaths(run_dir="", ckpt_dir="", samples_dir="")
        run_dir = broadcast_object(run.run_dir, src=0)
        ckpt_dir = broadcast_object(run.ckpt_dir, src=0)
        samples_dir = broadcast_object(run.samples_dir, src=0)
        run = RunPaths(run_dir=str(run_dir), ckpt_dir=str(ckpt_dir), samples_dir=str(samples_dir))
    else:
        run = create_run_dir(dist_info, run_root=args.run_root, name=run_name, config={"args": vars(args), "mae_cfg": asdict(mae_cfg)})

    step = 0
    opt_step = 0
    start_epoch = 0
    start_batch_in_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        else:
            if is_main_process():
                print(f"Warning: resume checkpoint has no optimizer state: {args.resume} (optimizer will restart)")
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])

        opt_step = int(ckpt.get("opt_step", 0))
        step = int(ckpt.get("step", opt_step * grad_accum))
        start_epoch = int(ckpt.get("epoch", 0))
        start_batch_in_epoch = int(ckpt.get("batch_in_epoch", 0))
        saved_grad_accum = ckpt.get("grad_accum", None)
        if saved_grad_accum is not None and int(saved_grad_accum) != int(grad_accum) and is_main_process():
            print(f"Warning: grad_accum mismatch (ckpt {saved_grad_accum} vs current {grad_accum}); resume will be approximate.")
        if is_main_process():
            print(f"Resumed: {args.resume} (epoch={start_epoch}, batch={start_batch_in_epoch}, opt_step={opt_step})")

    ddp_model.train(True)
    is_ddp = isinstance(ddp_model, DDP)
    micro_bsz_global = args.batch_size * dist_info.world_size
    opt_total = int(args.max_steps) if args.max_steps and args.max_steps > 0 else (args.epochs * len(dl)) // grad_accum

    if start_batch_in_epoch >= len(dl):
        start_epoch += 1
        start_batch_in_epoch = 0
    if start_epoch >= args.epochs or (args.max_steps and opt_step >= args.max_steps):
        if is_main_process():
            print("Nothing to do: resume checkpoint already meets the requested training length.")
        barrier()
        return

    last_epoch = start_epoch - 1
    last_batch_in_epoch = start_batch_in_epoch

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_t0 = time.perf_counter()
        epoch_imgs = 0
        epoch_pbar = tqdm(
            dl,
            total=len(dl),
            disable=not is_main_process(),
            desc=f"epoch {epoch+1}/{args.epochs}",
            leave=False,
        )
        for batch_i, batch in enumerate(epoch_pbar):
            if epoch == start_epoch and batch_i < start_batch_in_epoch:
                continue
            # batch is:
            # - images mode: (img, label, idx) from IndexedImageFolder, or a tuple from Subset.
            # - latents mode: (lat, label) from LatentsDataset, or a tuple from Subset.
            epoch_imgs += micro_bsz_global

            if use_latents:
                lat = batch[0].to(dist_info.device, non_blocking=True)
                if not use_amp:
                    lat = lat.float()
                elif amp_dtype == torch.bfloat16:
                    lat = lat.to(dtype=torch.bfloat16)
            else:
                images = batch[0].to(dist_info.device, non_blocking=True)
                assert vae is not None and vae_cfg is not None
                with torch.no_grad():
                    with torch.autocast(
                        device_type=dist_info.device.type,
                        enabled=(dist_info.device.type == "cuda" and args.vae_dtype == "fp16"),
                    ):
                        lat = encode_images_to_latents(
                            vae,
                            images,
                            scaling_factor=vae_cfg.scaling_factor,
                            encode_mode=vae_cfg.encode_mode,
                        )
                if not use_amp:
                    lat = lat.float()
                elif amp_dtype == torch.bfloat16:
                    lat = lat.to(dtype=torch.bfloat16)

            will_sync = (not is_ddp) or (grad_accum == 1) or ((step + 1) % grad_accum == 0)
            sync_ctx = nullcontext() if will_sync else ddp_model.no_sync()
            with sync_ctx:
                lat_masked, mask_px = _mask_latents(lat, mask_ratio=args.mask_ratio)
                with torch.autocast(device_type=dist_info.device.type, dtype=amp_dtype, enabled=use_amp):
                    recon = ddp_model(lat_masked)

                    mask = mask_px.expand_as(lat)
                    denom = mask.sum().clamp_min(1.0)
                    if use_amp:
                        loss = ((recon.float() - lat.float()) ** 2 * mask.float()).sum() / denom.float()
                    else:
                        loss = ((recon - lat) ** 2 * mask).sum() / denom
                    loss = loss / grad_accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            step += 1
            last_epoch = epoch
            last_batch_in_epoch = batch_i + 1

            if is_main_process() and (step % 20 == 0):
                elapsed = time.perf_counter() - epoch_t0
                img_s = epoch_imgs / max(1e-6, elapsed)
                accum_k = (step - 1) % grad_accum + 1
                epoch_pbar.set_postfix(
                    loss=float(loss.detach().item()) * grad_accum,
                    opt_step=f"{opt_step}/{opt_total}",
                    accum=f"{accum_k}/{grad_accum}",
                    img_s=f"{img_s:.1f}",
                    lr=f"{opt.param_groups[0]['lr']:.1e}",
                )

            if step % grad_accum == 0:
                if args.grad_clip and args.grad_clip > 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=float(args.grad_clip))
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
                opt_step += 1

                if is_main_process() and (opt_step % 10 == 0):
                    elapsed = time.perf_counter() - epoch_t0
                    img_s = epoch_imgs / max(1e-6, elapsed)
                    epoch_pbar.set_postfix(
                        loss=float(loss.detach().item()) * grad_accum,
                        opt_step=f"{opt_step}/{opt_total}",
                        accum=f"{grad_accum}/{grad_accum}",
                        img_s=f"{img_s:.1f}",
                        lr=f"{opt.param_groups[0]['lr']:.1e}",
                    )

                if is_main_process() and args.save_every > 0 and opt_step % args.save_every == 0:
                    ckpt = {
                        "opt_step": opt_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_i + 1,
                        "step": step,
                        "grad_accum": grad_accum,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "ema": ema.state_dict(),
                        "args": vars(args),
                        "mae_cfg": asdict(mae_cfg),
                    }
                    path = os.path.join(run.ckpt_dir, f"ckpt_optstep_{opt_step}.pt")
                    torch.save(ckpt, path)
                    print(f"Saved: {path}")

                if args.max_steps and opt_step >= args.max_steps:
                    break

        if args.max_steps and opt_step >= args.max_steps:
            break

    barrier()
    if is_main_process():
        # Save final (EMA + raw) for convenience.
        ckpt = {
            "opt_step": opt_step,
            "epoch": last_epoch,
            "batch_in_epoch": last_batch_in_epoch,
            "step": step,
            "grad_accum": grad_accum,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "ema": ema.state_dict(),
            "args": vars(args),
            "mae_cfg": asdict(mae_cfg),
        }
        path = os.path.join(run.ckpt_dir, "ckpt_final.pt")
        torch.save(ckpt, path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
