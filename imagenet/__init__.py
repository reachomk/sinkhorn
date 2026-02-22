"""ImageNet (paper §5.2) reproduction pipeline.

This package is intentionally separate from the toy 2D scripts in this repo.
Run modules from the repo root folder `/home/hep3/drift`, e.g.:

  torchrun --nproc_per_node=4 -m imagenet.train_drifting ...
"""

