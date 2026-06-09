import builtins

import numpy as np
import pytest
import torch

from vit_torch.preprocess import (
    BertTokenizer,
    PreprocessImages,
    SentencepieceTokenizer,
    get_pp,
    get_tokenizer,
)


def write_vocab(tmp_path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "\n".join(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "hello",
                ",",
                "world",
                "un",
                "##aff",
                "##able",
            ]
        ),
        encoding="utf-8",
    )
    return vocab_path


def test_get_tokenizer_returns_preprocess_classes() -> None:
    assert get_tokenizer("bert") is BertTokenizer
    assert get_tokenizer("sentencepiece") is SentencepieceTokenizer


def test_bert_tokenizer_prepends_cls_wordpieces_and_pads(tmp_path) -> None:
    tokenizer = BertTokenizer(vocab_path=str(write_vocab(tmp_path)), max_len=8)

    tokens = tokenizer("Hello, unaffable")

    assert tokens.dtype == torch.long
    assert tokens.tolist() == [2, 3, 4, 6, 7, 8, 0, 0]


def test_bert_tokenizer_batches_and_truncates(tmp_path) -> None:
    tokenizer = BertTokenizer(vocab_path=str(write_vocab(tmp_path)), max_len=4)

    tokens = tokenizer(["hello world", "missing world"])

    assert tokens.shape == (2, 4)
    assert tokens.tolist() == [
        [2, 3, 5, 0],
        [2, 1, 5, 0],
    ]


def test_preprocess_images_resizes_hwc_to_chw_value_range() -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, 2:, :] = 255
    preprocess = PreprocessImages(size=2)

    output = preprocess(image)

    assert output.shape == (3, 2, 2)
    assert output.dtype == torch.float32
    assert output.min().item() == -1.0
    assert output.max().item() == 1.0


def test_preprocess_images_center_crops_to_square() -> None:
    image = torch.zeros(3, 2, 4, dtype=torch.uint8)
    image[:, :, 1:3] = 255
    preprocess = PreprocessImages(size=2, crop=True)

    output = preprocess(image)

    assert output.shape == (3, 2, 2)
    assert torch.all(output == 1.0)


def test_preprocess_images_batches_to_nchw() -> None:
    images = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    preprocess = PreprocessImages(size=2)

    output = preprocess(images)

    assert output.shape == (2, 3, 2, 2)


def test_get_pp_copies_features_and_adds_tokens(tmp_path) -> None:
    pp = get_pp(
        tokenizer_name="bert",
        vocab_path=str(write_vocab(tmp_path)),
        max_len=4,
        size=2,
    )
    features = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "text": "hello world",
        "label": 7,
    }

    output = pp(features)

    assert output is not features
    assert output["image"].shape == (3, 2, 2)
    assert output["tokens"].tolist() == [2, 3, 5, 0]
    assert output["label"] == 7


def test_sentencepiece_tokenizer_reports_optional_dependency(
    tmp_path,
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentencepiece":
            raise ImportError("blocked by test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="requires the optional"):
        SentencepieceTokenizer(vocab_path=str(tmp_path / "model.spm"), max_len=4)
