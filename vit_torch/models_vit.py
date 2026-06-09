from __future__ import annotations

# PyTorch adaptation of google-research/vision_transformer/vit_jax/models_vit.py.

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_torch.models_resnet import ResNetStage, StdConv2d, group_norm


@dataclass(frozen=True)
class Patches:
    size: int | Sequence[int]


@dataclass(frozen=True)
class TransformerConfig:
    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = True


@dataclass(frozen=True)
class ResNetConfig:
    num_layers: Sequence[int]
    width_factor: float = 1.0


def to_2tuple(x: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return (x[0], x[1])


def _get_config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _patch_size(patches: Any) -> Tuple[int, int]:
    if isinstance(patches, (int, tuple, list)):
        return to_2tuple(patches)
    return to_2tuple(_get_config_value(patches, "size"))


def _training(module_training: bool, train: Optional[bool]) -> bool:
    return module_training if train is None else train


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AddPositionEmbs(nn.Module):
    def __init__(
        self,
        seq_len: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if seq_len is not None and hidden_dim is not None:
            self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, hidden_dim))
            nn.init.normal_(self.pos_embedding, std=0.02)
        else:
            self.register_parameter("pos_embedding", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.ndim == 3, (
            "Expected inputs with shape (batch, sequence, hidden), "
            f"got {tuple(inputs.shape)}"
        )

        _, seq_len, hidden_dim = inputs.shape
        if self.pos_embedding is None:
            pos_embedding = torch.empty(
                1,
                seq_len,
                hidden_dim,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            nn.init.normal_(pos_embedding, std=0.02)
            self.pos_embedding = nn.Parameter(pos_embedding)
        elif self.pos_embedding.shape[1:] != (seq_len, hidden_dim):
            raise ValueError(
                "Position embedding shape does not match input shape: "
                f"{tuple(self.pos_embedding.shape)} vs {(1, seq_len, hidden_dim)}"
            )

        return inputs + self.pos_embedding.to(dtype=inputs.dtype)


class MlpBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dim: int,
        out_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        out_dim = input_dim if out_dim is None else out_dim

        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.dropout_rate = dropout_rate

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        train: Optional[bool] = None,
    ) -> torch.Tensor:
        training = _training(self.training, train)

        x = self.fc1(inputs)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout_rate, training=training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout_rate, training=training)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads: "
                f"{hidden_dim} vs {num_heads}"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_rate = dropout_rate

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        train: Optional[bool] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = inputs.shape

        qkv = self.qkv(inputs)
        qkv = qkv.reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        dropout_p = self.dropout_rate if _training(self.training, train) else 0.0
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
        )
        x = x.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        return self.out(x)


class Encoder1DBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=attention_dropout_rate,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MlpBlock(
            input_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
        )
        self.dropout_rate = dropout_rate

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        train: Optional[bool] = None,
    ) -> torch.Tensor:
        assert inputs.ndim == 3, (
            "Expected inputs with shape (batch, sequence, hidden), "
            f"got {tuple(inputs.shape)}"
        )
        training = _training(self.training, train)

        x = self.attention_norm(inputs)
        x = self.attention(x, train=train)
        x = F.dropout(x, p=self.dropout_rate, training=training)
        x = x + inputs

        y = self.mlp_norm(x)
        y = self.mlp(y, train=train)
        return x + y


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        add_position_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.position_embedding = (
            AddPositionEmbs() if add_position_embedding else IdentityLayer()
        )
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList(
            [
                Encoder1DBlock(
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        train: Optional[bool] = None,
    ) -> torch.Tensor:
        assert x.ndim == 3
        training = _training(self.training, train)

        x = self.position_embedding(x)
        x = F.dropout(x, p=self.dropout_rate, training=training)

        for layer in self.layers:
            x = layer(x, train=train)

        return self.encoder_norm(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        patches: Any,
        transformer: Optional[Any],
        hidden_size: int,
        resnet: Optional[Any] = None,
        representation_size: Optional[int] = None,
        classifier: str = "token",
        head_bias_init: float = 0.0,
        encoder: Type[nn.Module] = Encoder,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if classifier not in {"token", "gap", "unpooled", "token_unpooled"}:
            raise ValueError(f"Invalid classifier={classifier}")

        patch_size = _patch_size(patches)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.classifier = classifier

        embedding_in_channels = in_channels
        if resnet is None:
            self.resnet_root = IdentityLayer()
            self.resnet_stages = IdentityLayer()
        else:
            width = int(64 * _get_config_value(resnet, "width_factor", 1.0))
            num_layers = tuple(_get_config_value(resnet, "num_layers", ()))

            self.resnet_root = nn.Sequential(
                StdConv2d(
                    in_channels,
                    width,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                group_norm(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

            stages = []
            stage_in_channels = width
            if num_layers:
                stage = ResNetStage(
                    in_channels=stage_in_channels,
                    block_size=num_layers[0],
                    features=width,
                    first_stride=1,
                )
                stages.append(stage)
                stage_in_channels = stage.out_channels

                for i, block_size in enumerate(num_layers[1:], start=1):
                    stage = ResNetStage(
                        in_channels=stage_in_channels,
                        block_size=block_size,
                        features=width * 2**i,
                        first_stride=2,
                    )
                    stages.append(stage)
                    stage_in_channels = stage.out_channels

            self.resnet_stages = nn.Sequential(*stages)
            embedding_in_channels = stage_in_channels

        self.embedding = nn.Conv2d(
            in_channels=embedding_in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if classifier in {"token", "token_unpooled"}:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.register_parameter("cls_token", None)

        if transformer is None:
            self.encoder = IdentityLayer()
        else:
            self.encoder = encoder(
                hidden_dim=hidden_size,
                num_layers=_get_config_value(transformer, "num_layers"),
                mlp_dim=_get_config_value(transformer, "mlp_dim"),
                num_heads=_get_config_value(transformer, "num_heads"),
                dropout_rate=_get_config_value(
                    transformer,
                    "dropout_rate",
                    0.1,
                ),
                attention_dropout_rate=_get_config_value(
                    transformer,
                    "attention_dropout_rate",
                    0.1,
                ),
                add_position_embedding=_get_config_value(
                    transformer,
                    "add_position_embedding",
                    True,
                ),
            )

        if representation_size is None:
            self.pre_logits = IdentityLayer()
        else:
            self.pre_logits = nn.Sequential(
                nn.Linear(hidden_size, representation_size),
                nn.Tanh(),
            )
            hidden_size = representation_size

        if num_classes:
            self.head = nn.Linear(hidden_size, num_classes)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, head_bias_init)
        else:
            self.head = IdentityLayer()

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        train: Optional[bool] = None,
    ) -> torch.Tensor:
        x = self.resnet_root(inputs)
        x = self.resnet_stages(x)

        x = self.embedding(x)
        x = x.flatten(2).transpose(1, 2)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        if isinstance(self.encoder, Encoder):
            x = self.encoder(x, train=train)
        else:
            x = self.encoder(x)

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "gap":
            x = x.mean(dim=1)
        elif self.classifier in {"unpooled", "token_unpooled"}:
            pass
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")

        x = self.pre_logits(x)
        return self.head(x)
