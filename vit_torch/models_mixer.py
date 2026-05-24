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
        self, num_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channel_mlp_dim: int,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
