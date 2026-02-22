from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from torchvision.datasets import ImageFolder


@dataclass(frozen=True)
class ImageNetPaths:
    root: str
    train_dir: str
    val_dir: str


def resolve_imagenet(root: str) -> ImageNetPaths:
    root_p = Path(root)
    train_dir = root_p / "train"
    val_dir = root_p / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"ImageNet train dir not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet val dir not found: {val_dir}")
    return ImageNetPaths(root=str(root_p), train_dir=str(train_dir), val_dir=str(val_dir))


class IndexedImageFolder(ImageFolder):
    """torchvision ImageFolder that returns (img, label, index)."""

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return img, target, index


def build_imagenet_dataset(root: str, split: str, transform) -> Tuple[IndexedImageFolder, ImageNetPaths]:
    paths = resolve_imagenet(root)
    if split == "train":
        ds = IndexedImageFolder(paths.train_dir, transform=transform)
    elif split == "val":
        ds = IndexedImageFolder(paths.val_dir, transform=transform)
    else:
        raise ValueError(f"Unknown split: {split} (expected 'train' or 'val')")
    return ds, paths

