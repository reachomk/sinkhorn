from __future__ import annotations

import json
import os
import shlex
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from imagenet.utils.dist import DistInfo, broadcast_object, is_main_process
from imagenet.utils.misc import jsonable


@dataclass(frozen=True)
class RunPaths:
    run_dir: str
    ckpt_dir: str
    samples_dir: str


def _sanitize(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-")


def create_run_dir(
    dist_info: DistInfo,
    run_root: str,
    name: str,
    config: dict[str, Any],
) -> RunPaths:
    """
    Create a unique run directory and write 'config.json' + 'cmd.txt'.

    DDP: only rank0 creates and writes; the path is broadcast to all ranks.
    """
    if is_main_process():
        os.makedirs(run_root, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        run_slug = _sanitize(name)
        run_dir = os.path.join(run_root, f"{ts}_{run_slug}_{uid}")
        os.makedirs(run_dir, exist_ok=False)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        samples_dir = os.path.join(run_dir, "samples")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)

        with open(os.path.join(run_dir, "cmd.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(shlex.quote(a) for a in sys.argv) + "\n")

        payload = {
            "created_at": ts,
            "run_dir": run_dir,
            "name": name,
            "dist": {
                "distributed": dist_info.distributed,
                "rank": dist_info.rank,
                "world_size": dist_info.world_size,
                "local_rank": dist_info.local_rank,
                "device": str(dist_info.device),
            },
            "config": jsonable(config),
        }
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    else:
        run_dir = ckpt_dir = samples_dir = None  # type: ignore[assignment]

    run_dir = broadcast_object(run_dir, src=0)
    ckpt_dir = broadcast_object(ckpt_dir, src=0)
    samples_dir = broadcast_object(samples_dir, src=0)

    assert isinstance(run_dir, str)
    assert isinstance(ckpt_dir, str)
    assert isinstance(samples_dir, str)
    return RunPaths(run_dir=run_dir, ckpt_dir=ckpt_dir, samples_dir=samples_dir)

