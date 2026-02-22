"""
Generate samples from a trained drifting generator and compute ImageNet256 FID.

Paper alignment
-------------------------
- Kaiming's paper §5.2 evaluation protocol:
  generate images from the trained latent generator and report FID.
- Appendix A.2:
  generator outputs SD-VAE latents '[B,4,32,32]', decoded to RGB before FID.
- Appendix A.7:
  inference-time CFG strength 'omega' is explicit at sampling.


-------------------
For our paper, this script
supports omega sweeps and logs per-omega FID plus best-omega summary.
omega: guidance strength of CFG
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from imagenet.models.dit_b2 import DiTB2Config, DiTLatentB2
from imagenet.models.ema import EMA
from imagenet.utils.dist import DistInfo, barrier, init_distributed, is_main_process
from imagenet.vae_sd import VaeConfig, decode_latents_to_images, load_vae


def _parse_floats(csv: str) -> list[float]:
    out: list[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ImageNet256 samples and compute FID (paper §5.2).")
    p.add_argument("--ckpt", type=str, required=True, help="Path to drifting generator checkpoint (.pt).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-ema", action="store_true", help="Use EMA weights from checkpoint.")

    p.add_argument("--num-gen", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--omegas", type=str, default="1.0", help="Comma-separated omega values to evaluate (ignored if --omega-num > 0).")
    p.add_argument("--omega-min", type=float, default=1.0, help="Omega sweep min (used when --omega-num > 0).")
    p.add_argument("--omega-max", type=float, default=4.0, help="Omega sweep max (used when --omega-num > 0).")
    p.add_argument("--omega-num", type=int, default=0, help="If > 0, evaluate a linspace of this many omegas on [omega-min, omega-max].")

    p.add_argument("--out-dir", type=str, default="", help="Output directory (default: alongside ckpt).")
    p.add_argument("--real-dir", type=str, default="/home/public/imagenet/val", help="Real images root for FID.")
    p.add_argument("--real-stats", type=str, default="", help="Cache file for real mu/sigma (.npz).")

    # VAE decode
    p.add_argument("--vae-id", type=str, default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--vae-scale", type=float, default=0.18215)
    p.add_argument("--vae-dtype", type=str, choices=["fp16", "fp32"], default="fp16")

    # clean-fid
    p.add_argument("--fid-mode", type=str, choices=["clean", "legacy_pytorch", "legacy_tensorflow"], default="clean")
    p.add_argument("--fid-batch-size", type=int, default=128)
    p.add_argument("--fid-workers", type=int, default=12)
    return p.parse_args()


def _rank_range(total: int, rank: int, world: int) -> tuple[int, int]:
    base = total // world
    rem = total % world
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


@torch.no_grad()
def _save_images(images: torch.Tensor, out_dir: str, start_idx: int) -> None:
    # images: [B,3,256,256] in [-1,1]
    img = (images.clamp(-1, 1) + 1.0) * 127.5
    img = img.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
    for i in range(img.shape[0]):
        path = os.path.join(out_dir, f"{start_idx + i:06d}.png")
        Image.fromarray(img[i]).save(path)


def _compute_folder_stats(fdir: str, *, mode: str, batch_size: int, num_workers: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    from cleanfid.fid import get_folder_features
    from cleanfid.features import build_feature_extractor

    feat_model = build_feature_extractor(mode, device=device, use_dataparallel=True)
    feats = get_folder_features(
        fdir,
        model=feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        verbose=True,
    )
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    from cleanfid.fid import frechet_distance

    return float(frechet_distance(mu1, sigma1, mu2, sigma2))


def _load_generator(ckpt_path: str, device: torch.device, *, use_ema: bool) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt.get("dit_cfg"), dict):
        cfg_dict = dict(ckpt["dit_cfg"])
        # Backward compat: older checkpoints used constant "register" tokens and did not store ctx_mode.
        cfg_dict.setdefault("ctx_mode", "register")
        dit_cfg = DiTB2Config(**cfg_dict)
    else:
        dit_cfg = DiTB2Config()
    gen = DiTLatentB2(dit_cfg)
    gen.load_state_dict(ckpt["gen"], strict=True)
    if use_ema and "ema" in ckpt:
        ema_tmp = EMA(gen, decay=float(ckpt["ema"].get("decay", 0.999)))
        ema_tmp.load_state_dict(ckpt["ema"])
        ema_tmp.copy_to(gen)
    gen.to(device)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    meta = {"dit_cfg": asdict(dit_cfg)}
    return gen, meta


def main() -> None:
    args = _parse_args()
    dist_info: DistInfo = init_distributed(device=args.device)
    torch.manual_seed(args.seed + dist_info.rank)

    if int(args.omega_num) > 0:
        omegas = torch.linspace(float(args.omega_min), float(args.omega_max), int(args.omega_num)).tolist()
    else:
        omegas = _parse_floats(args.omegas)
    if not omegas:
        raise ValueError("--omegas must be non-empty")

    ckpt_path = args.ckpt
    out_root = args.out_dir or str(Path(ckpt_path).resolve().parent / "eval_fid")
    if is_main_process():
        os.makedirs(out_root, exist_ok=True)

    gen, gen_meta = _load_generator(ckpt_path, dist_info.device, use_ema=bool(args.use_ema))

    vae_cfg = VaeConfig(vae_id=args.vae_id, scaling_factor=float(args.vae_scale), dtype=args.vae_dtype, encode_mode="mean")
    vae = load_vae(vae_cfg, device=dist_info.device)

    # Real stats cache.
    real_stats_path = args.real_stats or os.path.join(out_root, "real_stats_imagenet_val_256.npz")
    real_mu: np.ndarray
    real_sigma: np.ndarray

    if is_main_process():
        if os.path.exists(real_stats_path):
            stats = np.load(real_stats_path)
            real_mu = stats["mu"]
            real_sigma = stats["sigma"]
        else:
            print(f"Computing real stats for: {args.real_dir}")
            real_mu, real_sigma = _compute_folder_stats(
                args.real_dir,
                mode=str(args.fid_mode),
                batch_size=int(args.fid_batch_size),
                num_workers=int(args.fid_workers),
                device=dist_info.device,
            )
            Path(real_stats_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(real_stats_path, mu=real_mu, sigma=real_sigma)
            print(f"Saved real stats: {real_stats_path}")
    barrier()

    # Broadcast real stats by reloading (simple and robust).
    if not is_main_process():
        stats = np.load(real_stats_path)
        real_mu = stats["mu"]
        real_sigma = stats["sigma"]

    start, end = _rank_range(int(args.num_gen), dist_info.rank, dist_info.world_size)
    n_rank = end - start
    if is_main_process():
        print(f"Generating {args.num_gen} images total; rank0 range [{start},{end})")

    sweep: list[dict] = []
    for omega in omegas:
        tag = str(omega).replace(".", "p")
        out_dir = os.path.join(out_root, f"gen_omega_{tag}")
        if is_main_process():
            os.makedirs(out_dir, exist_ok=True)
            # Save meta for reproducibility.
            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ckpt": ckpt_path,
                        "use_ema": bool(args.use_ema),
                        "omega": float(omega),
                        "num_gen": int(args.num_gen),
                        "batch_size": int(args.batch_size),
                        "real_dir": args.real_dir,
                        "real_stats": real_stats_path,
                        "fid": {"mode": args.fid_mode, "batch_size": int(args.fid_batch_size), "workers": int(args.fid_workers)},
                        "vae": asdict(vae_cfg),
                        **gen_meta,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
        barrier()

        # Generate and save images for this rank.
        bs = int(args.batch_size)
        pbar = tqdm(range(0, n_rank, bs), disable=not is_main_process(), desc=f"gen ω={omega}")
        for off in pbar:
            b = min(bs, n_rank - off)
            cls = torch.randint(0, gen.cfg.num_classes, (b,), device=dist_info.device, dtype=torch.long)
            z = torch.randn(b, 4, 32, 32, device=dist_info.device)
            omega_t = torch.full((b,), float(omega), device=dist_info.device, dtype=torch.float32)
            lat = gen(z, cls, omega_t)
            with torch.autocast(device_type=dist_info.device.type, enabled=(dist_info.device.type == "cuda" and vae_cfg.dtype == "fp16")):
                imgs = decode_latents_to_images(vae, lat, scaling_factor=vae_cfg.scaling_factor)
            _save_images(imgs, out_dir=out_dir, start_idx=start + off)
        barrier()

        # FID on rank0.
        if is_main_process():
            mu_g, sig_g = _compute_folder_stats(
                out_dir,
                mode=str(args.fid_mode),
                batch_size=int(args.fid_batch_size),
                num_workers=int(args.fid_workers),
                device=dist_info.device,
            )
            fid = _fid(mu_g, sig_g, real_mu, real_sigma)
            print(f"FID (ω={omega}): {fid:.4f}")
            with open(os.path.join(out_dir, "fid.json"), "w", encoding="utf-8") as f:
                json.dump({"omega": float(omega), "fid": fid}, f, indent=2, sort_keys=True)
            sweep.append({"omega": float(omega), "fid": float(fid), "dir": out_dir})
        barrier()

    if is_main_process() and sweep:
        best = min(sweep, key=lambda r: r["fid"])
        summary = {
            "ckpt": ckpt_path,
            "use_ema": bool(args.use_ema),
            "num_gen": int(args.num_gen),
            "batch_size": int(args.batch_size),
            "fid": {"mode": str(args.fid_mode), "batch_size": int(args.fid_batch_size), "workers": int(args.fid_workers)},
            "real_dir": str(args.real_dir),
            "real_stats": str(real_stats_path),
            "omegas": [r["omega"] for r in sweep],
            "results": sweep,
            "best": best,
        }
        with open(os.path.join(out_root, "fid_sweep.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        with open(os.path.join(out_root, "fid_sweep.csv"), "w", encoding="utf-8") as f:
            f.write("omega,fid,dir\n")
            for r in sweep:
                f.write(f"{r['omega']},{r['fid']},{r['dir']}\n")


if __name__ == "__main__":
    main()
