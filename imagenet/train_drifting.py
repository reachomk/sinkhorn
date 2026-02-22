"""
Train the drifting generator on ImageNet256 SD-VAE latents (paper §5.2).

Kaiming's paper alignment
-------------------------
- Algorithm 1 + §3.5 "Batching":
  class-wise microbatching with generator optimization per step.
- Appendix A.2:
  generator architecture ('imagenet.models.dit_b2.DiTLatentB2').
- Appendix A.3:
  MAE encoder checkpoint loaded by '--mae-ckpt' as drifting feature extractor.
- Appendix A.5/A.6:
  feature sets, drift construction, and normalized loss in 'imagenet.drifting_loss'.
- Appendix A.7:
  CFG weighting through unconditional negatives and omega-conditioned generation.
- Appendix A.8:
  optional per-class/global real-sample queue ('SampleQueue').


---------------------------
Important: the Kaiming's paper baseline is:
'--drift-form alg2_joint --coupling partial_two_sided --alg2-impl logspace'.

-----------------------------------
For our paper,
this script should be '--drift-form', '--coupling', and '--sinkhorn-*' to compare
row-only / partial-two-sided / full-Sinkhorn couplings and split vs joint drift
construction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict
from typing import Iterable, Literal

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from imagenet.data.latents_memmap import final_paths
from imagenet.drifting_loss import (
    drifting_loss_for_feature_set,
    feature_sets_from_encoder_input,
    feature_sets_from_feature_map,
    flatten_latents_as_feature_set,
    sample_power_law_omega,
)
from imagenet.models.dit_b2 import DiTB2Config, DiTLatentB2
from imagenet.models.ema import EMA
from imagenet.models.resnet_mae import ResNetMAE, ResNetMAEConfig
from imagenet.utils.dist import all_reduce_mean, barrier, broadcast_object, init_distributed, is_main_process
from imagenet.utils.misc import seed_all
from imagenet.utils.runs import RunPaths, create_run_dir


def _parse_floats(csv: str) -> list[float]:
    out: list[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train drifting generator on ImageNet256 latents (paper §5.2).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    # Data (pre-encoded SD-VAE latents)
    p.add_argument("--latents-dir", type=str, default="data/imagenet256_latents")
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--max-items", type=int, default=0, help="Debug: use only first N items (0=all).")
    p.add_argument(
        "--sample-queue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the sample queue implementation described in Appendix A.8 (default: True).",
    )
    p.add_argument("--queue-class-size", type=int, default=128, help="Per-class queue size (Appendix A.8).")
    p.add_argument("--queue-global-size", type=int, default=1000, help="Global unconditional queue size (Appendix A.8).")
    p.add_argument("--queue-push", type=int, default=64, help="Number of new real samples pushed per step (Appendix A.8).")

    # Feature encoder (latent-MAE)
    p.add_argument("--mae-ckpt", type=str, required=True)
    p.add_argument("--mae-use-ema", action="store_true", help="Use EMA weights from MAE checkpoint.")
    p.add_argument("--mae-every-n-blocks", type=int, default=2)

    # Generator (DiT-B/2; Table 8)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument(
        "--ctx-mode",
        type=str,
        choices=["register", "in_context"],
        default="in_context",
        help="In-context conditioning tokens mode (Appendix A.2).",
    )
    p.add_argument("--register-tokens", type=int, default=16)
    p.add_argument("--style-codebook", type=int, default=64)
    p.add_argument("--style-tokens", type=int, default=32)

    # Drifting loss (Table 8 + Appendix A.6/A.7).
    # Defaults in this block are paper-faithful baseline settings.
    p.add_argument("--nc", type=int, default=64, help="Global number of class labels per step.")
    p.add_argument("--nneg", type=int, default=64, help="Generated samples per class.")
    p.add_argument("--npos", type=int, default=64, help="Positive real samples per class.")
    p.add_argument("--nuncond", type=int, default=16, help="Unconditional real samples per class (CFG negatives).")
    p.add_argument("--temps", type=str, default="0.02,0.05,0.2")
    p.add_argument("--alg2-impl", type=str, choices=["logspace", "kernel"], default="logspace")
    p.add_argument(
        "--drift-form",
        type=str,
        choices=["alg2_joint", "split"],
        default="alg2_joint",
        help="Drift construction form. 'alg2_joint' matches the paper's Algorithm 2 joint normalization (default). "
        "'split' constructs separate couplings for pos/neg and uses V=Pxy@y_pos - Pxneg@y_neg (follow-up ablation).",
    )
    p.add_argument(
        "--coupling",
        type=str,
        choices=["row", "partial_two_sided", "sinkhorn"],
        default="partial_two_sided",
        help="Coupling normalization. 'partial_two_sided' matches the paper (default).",
    )
    p.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations (only used when --coupling=sinkhorn).",
    )
    p.add_argument(
        "--sinkhorn-marginal",
        type=str,
        choices=["none", "weighted_cols"],
        default="none",
        help="Sinkhorn marginal mode for CFG(w). 'none' keeps log(w) bias semantics (requires omega>1 when nuncond>0); "
        "'weighted_cols' encodes w in column marginals (recommended for sinkhorn fairness).",
    )
    p.add_argument(
        "--mask-self-neg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask diagonal self-coupling inside the generated negative set (Alg.2).",
    )
    p.add_argument(
        "--dist-metric",
        type=str,
        choices=["l2", "l2_sq"],
        default="l2",
        help="Pairwise distance metric for drift construction: l2 uses ||x-y|| (baseline), l2_sq uses ||x-y||^2.",
    )

    # CFG omega sampling (Table 8)
    p.add_argument("--omega-min", type=float, default=1.0)
    p.add_argument("--omega-max", type=float, default=4.0)
    p.add_argument("--omega-exp", type=float, default=3.0, help="Power-law exponent for p(omega) ∝ omega^{-exp}.")

    # Optimizer (Table 8)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--amp", action="store_true", help="Use torch AMP for generator+encoder forward/backward.")

    # Logging / checkpoints
    p.add_argument("--run-root", type=str, default="runs/imagenet_drift")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--resume", type=str, default="", help="Resume from a checkpoint path (.pt).")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument(
        "--log-drift-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log lightweight drift normalization stats (S_j, theta, and optional Sinkhorn marginal error) for one feature set.",
    )
    p.add_argument(
        "--drift-stats-name",
        type=str,
        default="vanilla.latent",
        help="FeatureSet name to attach drift stats to when --log-drift-stats is enabled.",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


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


def _resume_steps_completed(ckpt_path: str, ckpt: dict) -> int:
    """Return number of optimizer steps already completed (start step index)."""
    step = int(ckpt.get("step", 0))
    m = re.search(r"ckpt_step_(\d+)\.pt$", os.path.basename(ckpt_path))
    if m:
        n = int(m.group(1))
        # Backward compat: older checkpoints stored last step index (n-1) instead of completed steps (n).
        if step in {n - 1, n}:
            return n
    return step


def _load_latents(latents_dir: str, split: str) -> tuple[np.memmap, np.memmap]:
    lat_path, lab_path = final_paths(latents_dir, split)
    if not os.path.exists(lat_path) or not os.path.exists(lab_path):
        raise FileNotFoundError(
            f"Missing latent files. Expected:\n  {lat_path}\n  {lab_path}\nRun: python -m imagenet.encode_latents --split {split} --out-dir {latents_dir} ..."
        )
    lat = np.load(lat_path, mmap_mode="r")
    lab = np.load(lab_path, mmap_mode="r")
    return lat, lab


def _build_indices_by_class(labels: np.ndarray, num_classes: int) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(num_classes)]
    for i, c in enumerate(labels):
        if 0 <= int(c) < num_classes:
            buckets[int(c)].append(i)
    return [np.asarray(b, dtype=np.int64) for b in buckets]


def _load_mae_encoder(
    ckpt_path: str,
    device: torch.device,
    *,
    use_ema: bool,
) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mae_cfg = ResNetMAEConfig(**ckpt["mae_cfg"]) if isinstance(ckpt.get("mae_cfg"), dict) else ResNetMAEConfig()
    mae = ResNetMAE(mae_cfg)
    mae.load_state_dict(ckpt["model"], strict=True)
    if use_ema and "ema" in ckpt:
        ema_tmp = EMA(mae, decay=float(ckpt["ema"].get("decay", 0.9995)))
        ema_tmp.load_state_dict(ckpt["ema"])
        ema_tmp.copy_to(mae)
    mae.to(device)
    mae.eval()
    for p in mae.parameters():
        p.requires_grad_(False)
    return mae.encoder, {"mae_cfg": asdict(mae_cfg)}


def _lr_for_step(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(base_lr)
    return float(base_lr) * min(1.0, (step + 1) / float(warmup_steps))


def _as_empty_like(x_feat: torch.Tensor) -> torch.Tensor:
    # x_feat is [N,L,C] -> return [0,L,C]
    return x_feat[:0].detach()


class SampleQueue:
    """
    Per-class + global sample queues for real latents (Appendix A.8).
    """

    def __init__(
        self,
        labels: np.ndarray,
        idx_by_class: list[np.ndarray],
        *,
        rng: np.random.Generator,
        class_size: int,
        global_size: int,
    ) -> None:
        self.labels = labels
        self.rng = rng
        self.class_size = int(class_size)
        self.global_size = int(global_size)
        self.class_q = [deque(maxlen=self.class_size) for _ in range(len(idx_by_class))]
        self.global_q = deque(maxlen=self.global_size)
        self.stream_order = np.arange(int(labels.shape[0]), dtype=np.int64)
        self.stream_pos = 0

        # Initialize queue contents from a full stream pass so that each class queue
        # contains the latest seen samples in a data-stream sense.
        n = int(labels.shape[0])
        if n > 0:
            self.rng.shuffle(self.stream_order)
            for ii in self.stream_order:
                self._append_index(int(ii))
            # Start a fresh stream for training-time "latest" updates.
            self.rng.shuffle(self.stream_order)
            self.stream_pos = 0

    def _append_index(self, ii: int) -> None:
        c = int(self.labels[ii])
        if 0 <= c < len(self.class_q):
            self.class_q[c].append(ii)
        self.global_q.append(ii)

    def _next_stream_indices(self, n_new: int) -> np.ndarray:
        n = int(self.labels.shape[0])
        if n_new <= 0 or n <= 0:
            return np.empty((0,), dtype=np.int64)
        out = np.empty((int(n_new),), dtype=np.int64)
        filled = 0
        while filled < int(n_new):
            if self.stream_pos >= n:
                self.rng.shuffle(self.stream_order)
                self.stream_pos = 0
            take = min(int(n_new) - filled, n - self.stream_pos)
            out[filled : filled + take] = self.stream_order[self.stream_pos : self.stream_pos + take]
            self.stream_pos += take
            filled += take
        return out

    def push_latest(self, n_new: int) -> None:
        """
        Push 'n_new' new real samples into queues (and pop oldest by maxlen).
        """
        idx = self._next_stream_indices(int(n_new))
        for ii in idx:
            self._append_index(int(ii))

    def sample_pos(self, c: int, npos: int) -> np.ndarray:
        q = self.class_q[int(c)]
        if len(q) < int(npos):
            raise RuntimeError(f"Class queue too small for c={c}: have {len(q)} need {npos}.")
        arr = np.fromiter(q, dtype=np.int64)
        return self.rng.choice(arr, size=int(npos), replace=False)

    def sample_uncond(self, nuncond: int) -> np.ndarray:
        if int(nuncond) <= 0:
            return np.empty((0,), dtype=np.int64)
        if len(self.global_q) < int(nuncond):
            raise RuntimeError(f"Global queue too small: have {len(self.global_q)} need {nuncond}.")
        arr = np.fromiter(self.global_q, dtype=np.int64)
        return self.rng.choice(arr, size=int(nuncond), replace=False)

    def state_dict(self) -> dict:
        return {
            "class_size": int(self.class_size),
            "global_size": int(self.global_size),
            "class_q": [list(q) for q in self.class_q],
            "global_q": list(self.global_q),
            "stream_pos": int(self.stream_pos),
            "stream_order": self.stream_order.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("SampleQueue state must be a dict")
        class_q_state = state.get("class_q")
        global_q_state = state.get("global_q")
        if not isinstance(class_q_state, list) or not isinstance(global_q_state, list):
            raise ValueError("Invalid SampleQueue state: missing class_q/global_q")
        if len(class_q_state) != len(self.class_q):
            raise ValueError(f"SampleQueue class_q length mismatch: {len(class_q_state)} vs {len(self.class_q)}")
        for q, vals in zip(self.class_q, class_q_state):
            q.clear()
            q.extend(int(v) for v in vals)
        self.global_q.clear()
        self.global_q.extend(int(v) for v in global_q_state)
        stream_pos = state.get("stream_pos")
        stream_order = state.get("stream_order")
        if isinstance(stream_pos, int):
            self.stream_pos = int(stream_pos)
        if isinstance(stream_order, list):
            arr = np.asarray(stream_order, dtype=np.int64)
            if arr.shape == self.stream_order.shape:
                self.stream_order = arr


def _pack_local_rng_queue_state(rng: np.random.Generator, queue: SampleQueue | None) -> dict:
    return {
        "rng_state": rng.bit_generator.state,
        "queue_state": (queue.state_dict() if queue is not None else None),
    }


def _collect_rng_queue_state(
    rng: np.random.Generator,
    queue: SampleQueue | None,
    *,
    dist_info,
) -> dict:
    local = _pack_local_rng_queue_state(rng, queue)
    if dist_info.distributed:
        import torch.distributed as dist

        gathered = [None for _ in range(dist_info.world_size)]
        dist.all_gather_object(gathered, local)
        return {"type": "per_rank", "states": gathered}
    return {"type": "single", "state": local}


def _extract_local_rng_queue_state(blob, *, rank: int) -> dict | None:
    if not isinstance(blob, dict):
        return None
    kind = blob.get("type")
    if kind == "single":
        state = blob.get("state")
        return state if isinstance(state, dict) else None
    if kind == "per_rank":
        states = blob.get("states")
        if isinstance(states, list) and 0 <= int(rank) < len(states) and isinstance(states[int(rank)], dict):
            return states[int(rank)]
        return None
    # Backward compatibility (older flat checkpoint keys).
    if "rng_state" in blob or "queue_state" in blob:
        return blob
    return None


def main() -> None:
    args = _parse_args()
    if args.debug:
        args.steps = min(args.steps, 20)
        args.nc = min(args.nc, 8)
        args.nneg = min(args.nneg, 8)
        args.npos = min(args.npos, 8)
        args.nuncond = min(args.nuncond, 4)
        args.max_items = args.max_items or 8192
        args.log_every = 1
        args.save_every = 0

    # Early validation (keep defaults paper-equivalent).
    if int(args.nuncond) > 0 and int(args.nneg) <= 1:
        raise ValueError("--nneg must be > 1 when --nuncond > 0 (CFG)")
    if str(args.coupling) == "sinkhorn":
        if int(args.sinkhorn_iters) <= 0:
            raise ValueError("--sinkhorn-iters must be > 0 when --coupling=sinkhorn")
        if str(args.alg2_impl) != "logspace":
            raise ValueError("--coupling=sinkhorn requires --alg2-impl=logspace")
        if str(args.sinkhorn_marginal) == "none" and int(args.nuncond) > 0 and float(args.omega_min) <= 1.0:
            raise ValueError(
                "Incompatible settings: --coupling=sinkhorn with --sinkhorn-marginal=none and "
                "--omega-min<=1 can produce non-positive CFG weights (w<=0) and infeasible Sinkhorn constraints. "
                "Use --sinkhorn-marginal weighted_cols or set --omega-min>1."
            )
    if str(args.sinkhorn_marginal) != "none" and str(args.coupling) != "sinkhorn":
        raise ValueError("--sinkhorn-marginal is only valid when --coupling=sinkhorn")

    dist_info = init_distributed(device=args.device)
    seed_all(args.seed + dist_info.rank)

    if is_main_process():
        print(
            "drift_cfg="
            + json.dumps(
                {
                    "drift_form": str(args.drift_form),
                    "coupling": str(args.coupling),
                    "alg2_impl": str(args.alg2_impl),
                    "mask_self_neg": bool(args.mask_self_neg),
                    "dist_metric": str(args.dist_metric),
                    "sinkhorn_iters": int(args.sinkhorn_iters),
                    "sinkhorn_marginal": str(args.sinkhorn_marginal),
                    "temps": str(args.temps),
                },
                sort_keys=True,
            )
        )

    resume_ckpt = None
    start_step = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        start_step = _resume_steps_completed(args.resume, resume_ckpt)

    # Global->local split (paper defines Nc as global; see main text §3.5 "Batching").
    if args.nc % dist_info.world_size != 0:
        raise ValueError(f"--nc ({args.nc}) must be divisible by world_size ({dist_info.world_size})")
    nc_local = args.nc // dist_info.world_size

    temps: list[float] = _parse_floats(args.temps)
    if not temps:
        raise ValueError("--temps must be non-empty")

    # Load latents + labels.
    lat_mm, lab_mm = _load_latents(args.latents_dir, args.split)
    n_data = int(lab_mm.shape[0])
    if args.max_items and args.max_items > 0:
        n_data = min(n_data, int(args.max_items))
        lat_mm = lat_mm[:n_data]
        lab_mm = lab_mm[:n_data]

    labels_np = np.asarray(lab_mm, dtype=np.int64)
    idx_by_class = _build_indices_by_class(labels_np, num_classes=int(args.num_classes))
    available_classes = [c for c in range(int(args.num_classes)) if idx_by_class[c].shape[0] >= int(args.npos)]
    if len(available_classes) < int(args.nc):
        raise RuntimeError(
            f"Not enough classes with >= Npos samples. available={len(available_classes)} need Nc={int(args.nc)}. "
            f"Try reducing --npos/--nc or increasing --max-items."
        )
    available_classes_arr = np.asarray(available_classes, dtype=np.int64)

    rng = np.random.default_rng(args.seed + dist_info.rank)

    use_queue = bool(args.sample_queue)
    if use_queue:
        if int(args.queue_class_size) < int(args.npos):
            raise ValueError(f"--queue-class-size ({args.queue_class_size}) must be >= --npos ({args.npos})")
        if int(args.queue_global_size) < int(args.nuncond):
            raise ValueError(f"--queue-global-size ({args.queue_global_size}) must be >= --nuncond ({args.nuncond})")
        queue = SampleQueue(
            labels_np,
            idx_by_class,
            rng=rng,
            class_size=int(args.queue_class_size),
            global_size=int(args.queue_global_size),
        )
    else:
        queue = None

    if resume_ckpt is not None:
        local_state = _extract_local_rng_queue_state(resume_ckpt.get("rng_queue_state"), rank=dist_info.rank)
        if local_state is None and ("rng_state" in resume_ckpt or "queue_state" in resume_ckpt):
            local_state = {
                "rng_state": resume_ckpt.get("rng_state"),
                "queue_state": resume_ckpt.get("queue_state"),
            }
        if isinstance(local_state, dict):
            if local_state.get("rng_state") is not None:
                try:
                    rng.bit_generator.state = local_state["rng_state"]
                except Exception as exc:
                    if is_main_process():
                        print(f"Warning: failed to restore rng state from checkpoint: {exc}")
            qstate = local_state.get("queue_state")
            if queue is not None and qstate is not None:
                try:
                    queue.load_state_dict(qstate)
                except Exception as exc:
                    if is_main_process():
                        print(f"Warning: failed to restore sample queue state from checkpoint: {exc}")
            elif queue is None and qstate is not None and is_main_process():
                print("Warning: checkpoint contains sample queue state but --sample-queue is disabled.")

    # Feature encoder (MAE encoder, frozen).
    encoder, mae_meta = _load_mae_encoder(args.mae_ckpt, dist_info.device, use_ema=bool(args.mae_use_ema))

    # Generator.
    if resume_ckpt is not None and isinstance(resume_ckpt.get("dit_cfg"), dict):
        cfg_dict = dict(resume_ckpt["dit_cfg"])
        # Backward compat: older checkpoints used constant "register" tokens and did not store ctx_mode.
        cfg_dict.setdefault("ctx_mode", "register")
        dit_cfg = DiTB2Config(**cfg_dict)
    else:
        dit_cfg = DiTB2Config(
            num_classes=int(args.num_classes),
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            n_heads=int(args.n_heads),
            ctx_mode=str(args.ctx_mode),
            register_tokens=int(args.register_tokens),
            style_codebook=int(args.style_codebook),
            style_tokens=int(args.style_tokens),
        )
    gen = DiTLatentB2(dit_cfg).to(dist_info.device)
    ddp_gen = (
        DDP(gen, device_ids=[dist_info.local_rank], broadcast_buffers=False)
        if dist_info.distributed and dist_info.device.type == "cuda"
        else gen
    )

    opt = torch.optim.AdamW(
        ddp_gen.parameters(),
        lr=args.lr,
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and dist_info.device.type == "cuda")
    ema = EMA(gen, decay=float(args.ema_decay))

    if resume_ckpt is not None:
        gen.load_state_dict(resume_ckpt["gen"], strict=True)
        if "opt" in resume_ckpt:
            opt.load_state_dict(resume_ckpt["opt"])
        else:
            if is_main_process():
                print(f"Warning: resume checkpoint has no optimizer state: {args.resume} (optimizer will restart)")
        if "ema" in resume_ckpt:
            ema.load_state_dict(resume_ckpt["ema"])
        if "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])

    run_name = args.run_name or f"drift_dit_b2_steps{args.steps}_nc{args.nc}_nneg{args.nneg}_npos{args.npos}"
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
        run = create_run_dir(
            dist_info,
            run_root=args.run_root,
            name=run_name,
            config={
                "args": vars(args),
                "dit_cfg": asdict(dit_cfg),
                **mae_meta,
                "nc_local": nc_local,
                "n_data": n_data,
            },
        )
    log_path = os.path.join(run.run_dir, "logs.jsonl")

    if is_main_process():
        print(f"Run dir: {run.run_dir}")
        print(f"DDP={dist_info.distributed} world={dist_info.world_size} nc_local={nc_local}")
        if args.resume:
            print(f"Resuming from: {args.resume} (start_step={start_step} -> steps={args.steps})")

    if start_step >= int(args.steps):
        if is_main_process():
            print("Nothing to do: resume checkpoint already meets the requested training length.")
        barrier()
        return

    ddp_gen.train(True)

    pbar = tqdm(
        range(start_step, int(args.steps)),
        disable=not is_main_process(),
        desc="train",
        initial=start_step,
        total=int(args.steps),
    )
    for step in pbar:
        step_done = step + 1
        lr = _lr_for_step(step, base_lr=float(args.lr), warmup_steps=int(args.warmup_steps))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)

        # Update sample queue with new real samples (Appendix A.8).
        if queue is not None:
            queue.push_latest(int(args.queue_push))

        # Sample class labels and CFG scales globally, then shard to each rank.
        if dist_info.distributed:
            if is_main_process():
                cls_global = rng.choice(available_classes_arr, size=int(args.nc), replace=False).astype(np.int64).tolist()
                omega_global = sample_power_law_omega(
                    int(args.nc),
                    omega_min=float(args.omega_min),
                    omega_max=float(args.omega_max),
                    exponent=float(args.omega_exp),
                    device=dist_info.device,
                    dtype=torch.float32,
                ).detach().cpu().tolist()
            else:
                cls_global = None
                omega_global = None
            cls_global = broadcast_object(cls_global, src=0)
            omega_global = broadcast_object(omega_global, src=0)

            cls_all = np.asarray(cls_global, dtype=np.int64)
            omega_all = torch.tensor(omega_global, device=dist_info.device, dtype=torch.float32)
            c0 = dist_info.rank * nc_local
            c1 = c0 + nc_local
            cls = cls_all[c0:c1]
            omega_t = omega_all[c0:c1]
        else:
            cls = rng.choice(available_classes_arr, size=nc_local, replace=False)
            omega_t = sample_power_law_omega(
                nc_local,
                omega_min=float(args.omega_min),
                omega_max=float(args.omega_max),
                exponent=float(args.omega_exp),
                device=dist_info.device,
                dtype=torch.float32,
            )

        do_log = is_main_process() and (step % int(args.log_every) == 0 or step == int(args.steps) - 1)
        collect_drift_stats = bool(args.log_drift_stats) and do_log
        drift_stats = {} if collect_drift_stats else None
        drift_stats_done = False
        drift_stats_name = str(args.drift_stats_name)

        loss_accum = 0.0
        nan_count = 0
        # Microbatch over classes to keep memory bounded (paper §3.5: "perform Alg. 1 independently").
        for i in range(nc_local):
            c = int(cls[i])
            omega_c = omega_t[i]
            if drift_stats is not None and i == 0:
                drift_stats["drift_class"] = float(c)
                drift_stats["drift_omega"] = float(omega_c.detach().float().cpu().item())

            # Noise -> generated latents
            z = torch.randn(int(args.nneg), 4, 32, 32, device=dist_info.device)
            c_lab = torch.full((int(args.nneg),), c, device=dist_info.device, dtype=torch.long)
            omega_rep = omega_c.expand(int(args.nneg))

            # Positives + unconditionals (real latents)
            if queue is not None:
                pos_idx = queue.sample_pos(c, int(args.npos))
                uncond_idx = queue.sample_uncond(int(args.nuncond))
            else:
                pos_pool = idx_by_class[c]
                pos_idx = rng.choice(pos_pool, size=int(args.npos), replace=(pos_pool.shape[0] < int(args.npos)))
                uncond_idx = (
                    rng.choice(n_data, size=int(args.nuncond), replace=(n_data < int(args.nuncond)))
                    if args.nuncond > 0
                    else np.empty((0,), dtype=np.int64)
                )

            pos_lat = torch.from_numpy(lat_mm[pos_idx]).to(dist_info.device).float()
            uncond_lat = torch.from_numpy(lat_mm[uncond_idx]).to(dist_info.device).float() if args.nuncond > 0 else torch.empty((0, 4, 32, 32), device=dist_info.device)

            sync_ctx = (
                ddp_gen.no_sync()
                if isinstance(ddp_gen, DDP) and dist_info.distributed and i < (nc_local - 1)
                else nullcontext()
            )
            with sync_ctx:
                amp_ctx = torch.autocast(device_type=dist_info.device.type, enabled=bool(args.amp) and dist_info.device.type == "cuda")
                with amp_ctx:
                    x_lat = ddp_gen(z, c_lab, omega_rep)  # [Nneg,4,32,32]

                    # Generated features WITH grad.
                    maps_x = encoder.forward_feature_maps(x_lat, every_n_blocks=int(args.mae_every_n_blocks))
                    x_sets = []
                    for mi, fmap in enumerate(maps_x):
                        x_sets.extend(feature_sets_from_feature_map(fmap, prefix=f"enc{mi:02d}"))
                    x_sets.extend(feature_sets_from_encoder_input(x_lat))
                    x_sets.append(flatten_latents_as_feature_set(x_lat))

                    # Real features under no_grad (stop-grad in drift construction).
                    with torch.no_grad():
                        maps_pos = encoder.forward_feature_maps(pos_lat, every_n_blocks=int(args.mae_every_n_blocks))
                        pos_sets = []
                        for mi, fmap in enumerate(maps_pos):
                            pos_sets.extend(feature_sets_from_feature_map(fmap, prefix=f"enc{mi:02d}"))
                        pos_sets.extend(feature_sets_from_encoder_input(pos_lat))
                        pos_sets.append(flatten_latents_as_feature_set(pos_lat))

                        if int(args.nuncond) > 0:
                            maps_unc = encoder.forward_feature_maps(uncond_lat, every_n_blocks=int(args.mae_every_n_blocks))
                            unc_sets = []
                            for mi, fmap in enumerate(maps_unc):
                                unc_sets.extend(feature_sets_from_feature_map(fmap, prefix=f"enc{mi:02d}"))
                            unc_sets.extend(feature_sets_from_encoder_input(uncond_lat))
                            unc_sets.append(flatten_latents_as_feature_set(uncond_lat))
                        else:
                            unc_sets = []

                    pos_dict = {fs.name: fs.x for fs in pos_sets}
                    unc_dict = {fs.name: fs.x for fs in unc_sets}

                    class_loss = torch.zeros((), device=dist_info.device, dtype=torch.float32)
                    for fs in x_sets:
                        y_pos = pos_dict[fs.name]
                        y_unc = unc_dict.get(fs.name, _as_empty_like(y_pos))
                        stats_arg = None
                        if drift_stats is not None and (not drift_stats_done) and fs.name == drift_stats_name:
                            stats_arg = drift_stats
                            drift_stats_done = True
                        class_loss = class_loss + drifting_loss_for_feature_set(
                            fs.x,
                            y_pos,
                            y_unc,
                            omega=omega_c,
                            temps=temps,
                            impl=str(args.alg2_impl),
                            drift_form=str(args.drift_form),
                            coupling=str(args.coupling),
                            sinkhorn_iters=int(args.sinkhorn_iters),
                            sinkhorn_marginal=str(args.sinkhorn_marginal),
                            mask_self_neg=bool(args.mask_self_neg),
                            dist_metric=str(args.dist_metric),
                            stats=stats_arg,
                        )

                    # Normalize by number of classes on this rank to keep step loss stable.
                    class_loss = class_loss / float(nc_local)

                scaler.scale(class_loss).backward()

            class_loss_v = float(class_loss.detach().item())
            if not math.isfinite(class_loss_v):
                nan_count += 1
            loss_accum += class_loss_v

        # Step optimizer.
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ddp_gen.parameters(), max_norm=float(args.grad_clip))

        scaler.step(opt)
        scaler.update()
        ema.update(gen)

        # Logging (reduce mean across ranks).
        loss_t = torch.tensor(loss_accum, device=dist_info.device, dtype=torch.float32)
        loss_mean = float(all_reduce_mean(loss_t).item())

        nan_t = torch.tensor(float(nan_count), device=dist_info.device, dtype=torch.float32)
        nan_mean = float(all_reduce_mean(nan_t).item())
        nan_global = int(round(nan_mean * float(dist_info.world_size)))

        if do_log:
            # For professor-facing logs we reserve "drift_norm" for an actual drift magnitude.
            # When enabled, 'drift_stats' records 'drift_v_rms' computed from Ṽ for one FeatureSet
            # (default: 'vanilla.latent'). Otherwise we log NaN.
            drift_norm = float("nan")
            if drift_stats is not None and "drift_v_rms" in drift_stats:
                drift_norm = float(drift_stats["drift_v_rms"])
            rec = {
                "step": step_done,
                "step_idx": step,
                "lr": lr,
                "loss": loss_mean,
                "loss_isfinite": bool(math.isfinite(loss_mean)),
                "drift_norm": drift_norm,
                "nan_count": nan_global,
            }
            if drift_stats is not None:
                rec["drift_stats_name"] = drift_stats_name
                rec.update(drift_stats)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            pbar.set_postfix(loss=f"{loss_mean:.4f}", lr=f"{lr:.2e}")

        should_save = int(args.save_every) > 0 and step_done % int(args.save_every) == 0
        if should_save:
            rng_queue_state = _collect_rng_queue_state(rng, queue, dist_info=dist_info)
        if is_main_process() and should_save:
            ckpt = {
                "step": step_done,
                "gen": gen.state_dict(),
                "opt": opt.state_dict(),
                "ema": ema.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_queue_state": rng_queue_state,
                "drift_cfg": {
                    "drift_form": str(args.drift_form),
                    "coupling": str(args.coupling),
                    "alg2_impl": str(args.alg2_impl),
                    "mask_self_neg": bool(args.mask_self_neg),
                    "dist_metric": str(args.dist_metric),
                    "sinkhorn_iters": int(args.sinkhorn_iters),
                    "sinkhorn_marginal": str(args.sinkhorn_marginal),
                    "temps": str(args.temps),
                },
                "args": vars(args),
                "dit_cfg": asdict(dit_cfg),
            }
            path = os.path.join(run.ckpt_dir, f"ckpt_step_{step_done}.pt")
            torch.save(ckpt, path)
            print(f"Saved: {path}")

    barrier()
    rng_queue_state = _collect_rng_queue_state(rng, queue, dist_info=dist_info)
    if is_main_process():
        ckpt = {
            "step": int(args.steps),
            "gen": gen.state_dict(),
            "opt": opt.state_dict(),
            "ema": ema.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_queue_state": rng_queue_state,
            "drift_cfg": {
                "drift_form": str(args.drift_form),
                "coupling": str(args.coupling),
                "alg2_impl": str(args.alg2_impl),
                "mask_self_neg": bool(args.mask_self_neg),
                "dist_metric": str(args.dist_metric),
                "sinkhorn_iters": int(args.sinkhorn_iters),
                "sinkhorn_marginal": str(args.sinkhorn_marginal),
                "temps": str(args.temps),
            },
            "args": vars(args),
            "dit_cfg": asdict(dit_cfg),
        }
        path = os.path.join(run.ckpt_dir, "ckpt_final.pt")
        torch.save(ckpt, path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
