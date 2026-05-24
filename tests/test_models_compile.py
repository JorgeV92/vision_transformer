import os

import pytest
import torch

from vit_torch.models_mixer import MixerBlock, MlpBlock
from vit_torch.models_resnet import ResidualUnit, ResNetStage, StdConv2d


def compiled_forward(model: torch.nn.Module, sample: torch.Tensor) -> torch.Tensor:
    compile_model = getattr(torch, "compile", None)
    if compile_model is None:
        pytest.skip("torch.compile is not available in this PyTorch version")

    model.eval()
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "eager")
    compiled = compile_model(model, backend=backend)

    with torch.no_grad():
        eager_output = model(sample)
        compiled_output = compiled(sample)

    torch.testing.assert_close(compiled_output, eager_output)
    return compiled_output


def test_mlp_block_compiles() -> None:
    model = MlpBlock(input_dim=8, mlp_dim=16)
    sample = torch.randn(2, 4, 8)

    output = compiled_forward(model, sample)

    assert output.shape == (2, 4, 8)


def test_mixer_block_compiles() -> None:
    model = MixerBlock(
        num_patches=4,
        hidden_dim=8,
        tokens_mlp_dim=16,
        channels_mlp_dim=32,
    )
    sample = torch.randn(2, 4, 8)

    output = compiled_forward(model, sample)

    assert output.shape == (2, 4, 8)


def test_std_conv2d_compiles() -> None:
    model = StdConv2d(3, 8, kernel_size=3, padding=1, bias=False)
    sample = torch.randn(2, 3, 16, 16)

    output = compiled_forward(model, sample)

    assert output.shape == (2, 8, 16, 16)


def test_residual_unit_compiles() -> None:
    model = ResidualUnit(in_channels=3, features=4, stride=2)
    sample = torch.randn(2, 3, 16, 16)

    output = compiled_forward(model, sample)

    assert output.shape == (2, 16, 8, 8)


def test_resnet_stage_compiles() -> None:
    model = ResNetStage(
        in_channels=3,
        block_size=2,
        features=4,
        first_stride=2,
    )
    sample = torch.randn(2, 3, 16, 16)

    output = compiled_forward(model, sample)

    assert output.shape == (2, 16, 8, 8)
