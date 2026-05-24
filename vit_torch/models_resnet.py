# models_resnet_torch.py

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return (x[0], x[1])


class StdConv2d(nn.Conv2d):
    """
    Conv2d with weight standardization.

    Flax kernel shape:
        [H, W, in_channels, out_channels]

    PyTorch kernel shape:
        [out_channels, in_channels, H, W]

    So in PyTorch we standardize over:
        in_channels, H, W
    for each output channel.
    """

    def __init__(self, *args, eps: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def standardized_weight(self) -> torch.Tensor:
        w = self.weight

        mean = w.mean(dim=(1, 2, 3), keepdim=True)
        std = w.std(dim=(1, 2, 3), keepdim=True, unbiased=False)

        return (w - mean) / (std + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.standardized_weight()

        return F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def group_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """
    Flax GroupNorm defaults to 32 groups.
    PyTorch requires num_channels to be divisible by num_groups.
    """
    while num_channels % num_groups != 0:
        num_groups //= 2

    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ResidualUnit(nn.Module):
    """
    PyTorch version of the Flax ResidualUnit.

    Original Flax block:
        conv1: 1x1, features
        gn1
        relu
        conv2: 3x3, features, stride
        gn2
        relu
        conv3: 1x1, features * 4
        gn3 with zero scale init
        residual + y
        relu
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        features: int,
        stride: int | Sequence[int] = 1,
    ):
        super().__init__()

        stride = to_2tuple(stride)
        out_channels = features * self.expansion

        self.needs_projection = (
            in_channels != out_channels or stride != (1, 1)
        )

        if self.needs_projection:
            self.conv_proj = StdConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.gn_proj = group_norm(out_channels)

        self.conv1 = StdConv2d(
            in_channels,
            features,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.gn1 = group_norm(features)

        self.conv2 = StdConv2d(
            features,
            features,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn2 = group_norm(features)

        self.conv3 = StdConv2d(
            features,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.gn3 = group_norm(out_channels)

        # Flax code uses:
        # nn.GroupNorm(name='gn3', scale_init=nn.initializers.zeros)
        # In PyTorch, GroupNorm's affine weight is the scale gamma.
        nn.init.zeros_(self.gn3.weight)
        nn.init.zeros_(self.gn3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.needs_projection:
            residual = self.conv_proj(residual)
            residual = self.gn_proj(residual)

        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y, inplace=True)

        y = self.conv2(y)
        y = self.gn2(y)
        y = F.relu(y, inplace=True)

        y = self.conv3(y)
        y = self.gn3(y)

        y = F.relu(residual + y, inplace=True)
        return y


class ResNetStage(nn.Module):
    """
    PyTorch version of ResNetStage.

    The first block can downsample with first_stride.
    The remaining blocks use stride 1.
    """

    def __init__(
        self,
        in_channels: int,
        block_size: int,
        features: int,
        first_stride: int | Sequence[int] = 1,
    ):
        super().__init__()

        blocks = []

        blocks.append(
            ResidualUnit(
                in_channels=in_channels,
                features=features,
                stride=first_stride,
            )
        )

        out_channels = features * ResidualUnit.expansion

        for _ in range(1, block_size):
            blocks.append(
                ResidualUnit(
                    in_channels=out_channels,
                    features=features,
                    stride=1,
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)