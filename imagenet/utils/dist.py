from __future__ import annotations

import atexit
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


_DESTROY_REGISTERED = False


def init_distributed(device: str | None = None) -> DistInfo:
    """
    Initialize torch.distributed from 'torchrun' env vars (RANK/WORLD_SIZE/LOCAL_RANK).

    Returns a 'DistInfo' describing the process group and selected device.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        global _DESTROY_REGISTERED
        if not _DESTROY_REGISTERED:
            _DESTROY_REGISTERED = True

            def _destroy_process_group() -> None:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()

            atexit.register(_destroy_process_group)

    # Device selection:
    # - if 'device' is None: use cuda:{local_rank} under DDP, else "cuda" if available, else cpu
    # - if 'device' is like "cuda": use cuda:{local_rank} under DDP
    # - if 'device' is like "cuda:3": use it as-is (single-process debug)
    if device is None:
        if torch.cuda.is_available():
            dev = torch.device(f"cuda:{local_rank}" if distributed else "cuda:0")
        else:
            dev = torch.device("cpu")
    else:
        if device == "cuda" and torch.cuda.is_available() and distributed:
            dev = torch.device(f"cuda:{local_rank}")
        elif device == "cuda" and torch.cuda.is_available() and not distributed:
            dev = torch.device("cuda:0")
        else:
            dev = torch.device(device)

    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    return DistInfo(
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=dev,
    )


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def broadcast_object(obj, src: int = 0):
    if dist.is_available() and dist.is_initialized():
        obj_list = [obj] if dist.get_rank() == src else [None]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]
    return obj
