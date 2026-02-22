"""
ResNet-style masked autoencoder (latent-MAE) used as the feature encoder ϖ(\pi) on ImageNet.

Paper alignment
-------------------------
- Appendix A.3:Implementation of ResNet-style MAE" describes the ResNet+GN encoder
  and the light decoder used for masked reconstruction pretraining on SD-VAE latents.
- Appendix A.5: the drifting loss uses multi-scale feature maps extracted from
  the encoder; 'ResNetEncoderGN.forward_feature_maps(...)' exposes those intermediate maps.

In the drifting generator training ('train_drifting.py'), the MAE decoder is not used;
only the encoder feature maps are used to build the feature sets for the drifting loss.

Notes:
- In this study, we keep the same generator and the same feature extractor ϖ(\pi),
  and only compare different drift/coupling algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = _gn(out_ch)
        self.act = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _gn(out_ch),
            )
        else:
            self.down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.act(out + identity)
        return out


@dataclass(frozen=True)
class ResNetMAEConfig:
    in_ch: int = 4
    base_width: int = 256
    blocks_per_stage: Tuple[int, int, int, int] = (3, 4, 6, 3)
    conv1_stride: int = 1


class ResNetEncoderGN(nn.Module):
    """
    ResNet-style encoder with GroupNorm (Appendix A.3).

    Returns a list of stage outputs [f1, f2, f3, f4] for decoder skip connections.
    """

    def __init__(self, cfg: ResNetMAEConfig):
        super().__init__()
        c = int(cfg.base_width)
        self.conv1 = nn.Conv2d(cfg.in_ch, c, kernel_size=3, stride=int(cfg.conv1_stride), padding=1, bias=False)
        self.gn1 = _gn(c)
        self.act = nn.ReLU(inplace=True)

        self.stage1 = self._make_stage(c, c, n_blocks=cfg.blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(c, 2 * c, n_blocks=cfg.blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(2 * c, 4 * c, n_blocks=cfg.blocks_per_stage[2], stride=2)
        self.stage4 = self._make_stage(4 * c, 8 * c, n_blocks=cfg.blocks_per_stage[3], stride=2)

        self.out_channels = (c, 2 * c, 4 * c, 8 * c)

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks: List[nn.Module] = []
        blocks.append(BasicBlockGN(in_ch, out_ch, stride=stride))
        for _ in range(1, n_blocks):
            blocks.append(BasicBlockGN(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.act(self.gn1(self.conv1(x)))
        f1 = self.stage1(x)  # 32x32
        f2 = self.stage2(f1)  # 16x16
        f3 = self.stage3(f2)  # 8x8
        f4 = self.stage4(f3)  # 4x4
        return [f1, f2, f3, f4]

    def forward_feature_maps(self, x: torch.Tensor, *, every_n_blocks: int = 2) -> List[torch.Tensor]:
        """
        Return intermediate feature maps for drifting loss (Appendix A.5).

        We extract the output after every 'every_n_blocks' residual blocks within each stage,
        plus the final output of the stage.

        Returns a list of feature maps at multiple scales.
        """
        if every_n_blocks <= 0:
            raise ValueError(f"every_n_blocks must be > 0, got {every_n_blocks}")

        def run_stage(inp: torch.Tensor, stage: nn.Sequential) -> tuple[torch.Tensor, List[torch.Tensor]]:
            outs: List[torch.Tensor] = []
            h = inp
            n_blocks = len(stage)
            for i, block in enumerate(stage, start=1):
                h = block(h)
                if i % every_n_blocks == 0:
                    outs.append(h)
            if n_blocks % every_n_blocks != 0:
                outs.append(h)
            return h, outs

        h = self.act(self.gn1(self.conv1(x)))
        maps: List[torch.Tensor] = []
        h, out1 = run_stage(h, self.stage1)
        maps.extend(out1)
        h, out2 = run_stage(h, self.stage2)
        maps.extend(out2)
        h, out3 = run_stage(h, self.stage3)
        maps.extend(out3)
        h, out4 = run_stage(h, self.stage4)
        maps.extend(out4)
        return maps


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = _gn(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        return x


class ResNetMAEDecoder(nn.Module):
    """
    U-Net-style decoder (Appendix A.3).
    """

    def __init__(self, out_ch: int, enc_channels: Tuple[int, int, int, int]):
        super().__init__()
        c1, c2, c3, c4 = enc_channels
        self.init = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            _gn(c4),
            nn.ReLU(inplace=True),
        )
        self.up3 = _UpBlock(c4, c3, c3)
        self.up2 = _UpBlock(c3, c2, c2)
        self.up1 = _UpBlock(c2, c1, c1)
        self.out = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = feats
        x = self.init(f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        return self.out(x)


class ResNetMAE(nn.Module):
    def __init__(self, cfg: ResNetMAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ResNetEncoderGN(cfg)
        self.decoder = ResNetMAEDecoder(out_ch=cfg.in_ch, enc_channels=self.encoder.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        recon = self.decoder(feats)
        return recon
