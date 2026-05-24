
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


def to_2tuple(x: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return (x[0], x[1])


class MlpBlock(nn.Module):
    def __init__(self, input_dim: int, mlp_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(
        self,
        num_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
    ):
        super().__init__()

        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mixing = MlpBlock(
            input_dim=num_patches,
            mlp_dim=tokens_mlp_dim,
        )

        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mixing = MlpBlock(
            input_dim=hidden_dim,
            mlp_dim=channels_mlp_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y
        y = self.channel_norm(x)
        y = self.channel_mixing(y)
        x = x + y
        return x


class MlpMixer(nn.Module):
    def __init__(
        self,
        image_size: int | Sequence[int],
        patch_size: int | Sequence[int],
        num_classes: int,
        num_blocks: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
    ):
        super().__init__()

        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)

        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0
        assert image_width % patch_width == 0

        num_patches_h = image_height // patch_height
        num_patches_w = image_width // patch_width
        num_patches = num_patches_h * num_patches_w

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.stem = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_patches=num_patches,
                    hidden_dim=hidden_dim,
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                )
                for _ in range(num_blocks)
            ]
        )

        self.pre_head_layer_norm = nn.LayerNorm(hidden_dim)

        if num_classes:
            self.head = nn.Linear(hidden_dim, num_classes)

            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.blocks(x)
        x = self.pre_head_layer_norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
