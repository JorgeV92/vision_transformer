# Vision Transformer With PyTorch

This repository contains a PyTorch implementation of Vision Transformer models.

## About Vision Transformers

A Vision Transformer, usually shortened to ViT, treats an image as a sequence
rather than as a dense grid processed only by convolutions. The image is split
into fixed-size patches. Each patch is projected into an embedding vector, and
the resulting patch sequence is passed through a Transformer encoder.

The usual flow is:

1. Input image tensor with shape `NCHW`.
2. Patch embedding with a strided convolution.
3. Optional class token prepended to the patch sequence.
4. Positional embedding added to preserve patch order.
5. Transformer encoder blocks with self-attention and MLP layers.
6. A classifier head using either the class token, global average pooling, or
   the unpooled sequence.

This is useful because self-attention lets every patch interact with every
other patch early in the model. In exchange, ViT models often need careful
training, enough data, or pretrained weights to perform well.

## What Is Included

### Vision Transformer model

The main model lives in `vit_torch/models_vit.py`.

Included pieces:

- `VisionTransformer`: image-to-logits model.
- `Patches`: patch-size configuration.
- `TransformerConfig`: number of layers, MLP size, attention heads, dropout,
  attention dropout, and positional embedding toggle.
- `ResNetConfig`: optional hybrid ResNet stem configuration.
- `AddPositionEmbs`: learnable positional embeddings.
- `Encoder`: Transformer encoder stack.
- `Encoder1DBlock`: LayerNorm, self-attention, MLP, dropout, and residual paths.
- `MultiHeadSelfAttention`: PyTorch scaled dot-product attention backend.
- `MlpBlock`: feed-forward block used by the encoder.

Supported classifier modes:

- `token`: prepend a class token and classify from it.
- `gap`: classify from global average pooled sequence features.
- `unpooled`: return predictions for the unpooled sequence.
- `token_unpooled`: include a class token but keep the sequence unpooled.

### Hybrid ResNet components

The ResNet support lives in `vit_torch/models_resnet.py`.

Included pieces:

- `StdConv2d`: convolution with standardized weights.
- `group_norm`: helper that chooses a valid group count.
- `ResidualUnit`: bottleneck residual block.
- `ResNetStage`: stack of residual units.

These components can be used as a convolutional stem before patch projection.

### MLP-Mixer model

The MLP-Mixer implementation lives in `vit_torch/models_mixer.py`.

Included pieces:

- `MlpMixer`: image classifier using patch embeddings and mixer blocks.
- `MixerBlock`: token-mixing and channel-mixing MLPs.
- `MlpBlock`: simple GELU MLP block.

### Preprocessing

The preprocessing utilities live in `vit_torch/preprocess.py`.

Included pieces:

- `PreprocessImages`: resizes images and maps values from uint8 `[0, 255]` to
  float `[-1, 1]`.
- Center-crop mode that resizes the short side first, then crops to a square.
- HWC, CHW, NHWC, and NCHW image input handling.
- `BertTokenizer`: small local WordPiece tokenizer with fixed sequence length,
  prepended `[CLS]`, truncation, and zero padding.
- `SentencepieceTokenizer`: fixed-length SentencePiece tokenizer with sticky
  EOS padding. This requires the optional `sentencepiece` package.
- `get_tokenizer`: tokenizer lookup by name.
- `get_pp`: feature-mapping helper that preprocesses optional `image` and
  `text` keys.

PyTorch does not ship a SentencePiece tokenizer. The BERT tokenizer is
implemented locally; SentencePiece support is delegated to the standard
`sentencepiece` Python package when installed.

## Quick Start

Create a small Vision Transformer:

```python
import torch

from vit_torch.models_vit import Patches, TransformerConfig, VisionTransformer

model = VisionTransformer(
    num_classes=10,
    patches=Patches(size=(16, 16)),
    transformer=TransformerConfig(
        num_layers=6,
        mlp_dim=2048,
        num_heads=8,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    ),
    hidden_size=512,
)

images = torch.randn(2, 3, 224, 224)
logits = model(images)

print(logits.shape)  # torch.Size([2, 10])
```

Use image preprocessing:

```python
import numpy as np

from vit_torch.preprocess import PreprocessImages

preprocess = PreprocessImages(size=224, crop=True)
image = np.zeros((256, 320, 3), dtype=np.uint8)

tensor = preprocess(image)
print(tensor.shape)  # torch.Size([3, 224, 224])
print(tensor.min(), tensor.max())  # values in [-1, 1]
```

Use BERT-style text preprocessing:

```python
from vit_torch.preprocess import BertTokenizer

tokenizer = BertTokenizer(vocab_path="path/to/vocab.txt", max_len=32)
tokens = tokenizer(["a photo of a cat", "a photo of a car"])

print(tokens.shape)  # torch.Size([2, 32])
```

Use the feature mapper:

```python
from vit_torch.preprocess import get_pp

pp = get_pp(
    tokenizer_name="bert",
    vocab_path="path/to/vocab.txt",
    max_len=32,
    size=224,
    crop=True,
)

features = {
    "image": image,
    "text": "a photo of a cat",
    "label": 0,
}

processed = pp(features)
print(processed["image"].shape)   # torch.Size([3, 224, 224])
print(processed["tokens"].shape)  # torch.Size([32])
```

## Tests

Run the tests from the repository root:

```bash
python -m pytest tests/test_models_compile.py tests/test_preprocess.py -q
```

The current tests cover:

- PyTorch compilation compatibility for the core model blocks.
- Vision Transformer and hybrid ResNet Vision Transformer forward passes.
- BERT tokenization behavior.
- Image resizing, center cropping, batch handling, and value range conversion.
- Feature mapping through `get_pp`.
- Optional dependency handling for SentencePiece.

## Current Status

This repository currently provides the core PyTorch modules needed to build and
run small ViT, hybrid ViT, and MLP-Mixer models. It also includes preprocessing
for image tensors and basic text tokenization utilities that are useful for
image-text pipelines.


## What Is Missing

The following pieces are not implemented yet:

- End-to-end training scripts.
- Dataset download and input pipeline code.
- Evaluation scripts and benchmark reporting.
- Pretrained checkpoint loading.
- Checkpoint conversion utilities.
- Model variant factory helpers such as named `B/16`, `B/32`, `L/16`, or hybrid
  presets.
- Distributed training support.
- Mixed precision training utilities.
- Optimizer and learning-rate schedule helpers.
- Data augmentation policies.
- Full SentencePiece test fixture with a trained test model.
- Package metadata such as `pyproject.toml` or install instructions.
- CI configuration.

## Repository Layout

```text
vit_torch/
  models_vit.py       # Vision Transformer and Transformer encoder modules
  models_resnet.py    # ResNet stem and residual blocks
  models_mixer.py     # MLP-Mixer model
  preprocess.py       # Image and text preprocessing utilities

tests/
  test_models_compile.py
  test_preprocess.py
```

## Notes

- Model inputs should be PyTorch tensors in `NCHW` format.
- Image preprocessing outputs `CHW` for single images and `NCHW` for batches.
- Preprocessed images are floating point tensors in `[-1, 1]`.
- Token outputs are `torch.long` tensors.
- The model code is small by design, so it should be straightforward to extend
  with training loops, named configs, checkpoint loading, or dataset-specific
  preprocessing.
