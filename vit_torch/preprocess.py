from __future__ import annotations

import dataclasses
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def get_tokenizer(
    tokenizer_name: str,
) -> type[BertTokenizer] | type[SentencepieceTokenizer]:
    """Returns a tokenizer specified by name ("bert" or "sentencepiece")."""
    return {
        "bert": BertTokenizer,
        "sentencepiece": SentencepieceTokenizer,
    }[tokenizer_name]


def _load_vocab(vocab_path: str) -> list[str]:
    with open(vocab_path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _is_control(char: str) -> bool:
    return unicodedata.category(char) in ("Cc", "Cf") and char not in "\t\n\r"


def _is_punctuation(char: str) -> bool:
    codepoint = ord(char)
    if (33 <= codepoint <= 47) or (58 <= codepoint <= 64):
        return True
    if (91 <= codepoint <= 96) or (123 <= codepoint <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def _basic_tokenize(text: str, *, lower_case: bool = True) -> list[str]:
    text = "".join(" " if _is_control(char) else char for char in text)
    tokens: list[str] = []

    for token in text.strip().split():
        if lower_case:
            token = _strip_accents(token.lower())

        current = []
        for char in token:
            if _is_punctuation(char):
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)
            else:
                current.append(char)

        if current:
            tokens.append("".join(current))

    return tokens


@dataclasses.dataclass(frozen=True)
class BertTokenizer:
    """BERT WordPiece tokenizer with prepended CLS token and fixed length.

    This is a small PyTorch-friendly equivalent of the Google ViT preprocessing
    tokenizer. It reads a standard BERT vocabulary file, lowercases by default,
    prepends ``[CLS]``, truncates to ``max_len``, and pads with token id 0.
    """

    vocab_path: str
    max_len: int
    lower_case: bool = True
    cls_token: int = dataclasses.field(init=False)
    unk_token: int = dataclasses.field(init=False)
    vocab: dict[str, int] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        vocab_tokens = _load_vocab(self.vocab_path)
        vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

        try:
            cls_token = vocab["[CLS]"]
            unk_token = vocab["[UNK]"]
        except KeyError as exc:
            raise ValueError(
                "BERT vocabulary must contain [CLS] and [UNK] tokens"
            ) from exc

        object.__setattr__(self, "cls_token", cls_token)
        object.__setattr__(self, "unk_token", unk_token)
        object.__setattr__(self, "vocab", vocab)

    def _wordpiece_tokenize(self, token: str) -> list[int]:
        if len(token) > 100:
            return [self.unk_token]

        token_ids: list[int] = []
        start = 0

        while start < len(token):
            end = len(token)
            subtoken_id = None
            while start < end:
                subtoken = token[start:end]
                if start > 0:
                    subtoken = f"##{subtoken}"
                if subtoken in self.vocab:
                    subtoken_id = self.vocab[subtoken]
                    break
                end -= 1

            if subtoken_id is None:
                return [self.unk_token]

            token_ids.append(subtoken_id)
            start = end

        return token_ids

    def preprocess(self, text: str) -> torch.Tensor:
        """Tokenizes a single text to a ``torch.long`` tensor."""
        token_ids = [self.cls_token]
        for token in _basic_tokenize(text, lower_case=self.lower_case):
            token_ids.extend(self._wordpiece_tokenize(token))

        token_ids = token_ids[: self.max_len]
        token_ids.extend([0] * (self.max_len - len(token_ids)))
        return torch.tensor(token_ids, dtype=torch.long)

    def __call__(self, texts: str | Iterable[str]) -> torch.Tensor:
        """Tokenizes one text or a batch of texts to PyTorch tensors."""
        if isinstance(texts, str):
            return self.preprocess(texts)
        return torch.stack([self.preprocess(text) for text in texts])


@dataclasses.dataclass(frozen=True)
class SentencepieceTokenizer:
    """SentencePiece tokenizer with sticky EOS and fixed sequence length.

    PyTorch does not include a SentencePiece tokenizer, so this class uses the
    optional ``sentencepiece`` package when available.
    """

    vocab_path: str
    max_len: int
    eos_token: int = dataclasses.field(init=False)
    _processor: Any = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "SentencepieceTokenizer requires the optional "
                "`sentencepiece` package."
            ) from exc

        processor = spm.SentencePieceProcessor()
        processor.Load(self.vocab_path)
        eos_token = processor.eos_id()
        if eos_token < 0:
            eos_token = processor.PieceToId("</s>")
        if eos_token < 0:
            raise ValueError("SentencePiece model must define an EOS token")

        object.__setattr__(self, "eos_token", eos_token)
        object.__setattr__(self, "_processor", processor)

    def preprocess(self, text: str) -> torch.Tensor:
        """Tokenizes a single text with EOS guaranteed as the last token."""
        token_ids = self._processor.EncodeAsIds(text)
        token_ids = token_ids[: self.max_len - 1]
        token_ids.append(self.eos_token)
        token_ids.extend([self.eos_token] * (self.max_len - len(token_ids)))
        return torch.tensor(token_ids, dtype=torch.long)

    def __call__(self, texts: str | Iterable[str]) -> torch.Tensor:
        """Tokenizes one text or a batch of texts to PyTorch tensors."""
        if isinstance(texts, str):
            return self.preprocess(texts)
        return torch.stack([self.preprocess(text) for text in texts])


@dataclasses.dataclass(frozen=True)
class PreprocessImages:
    """Resizes images and sets value range to ``[-1, 1]``.

    Inputs can be uint8 numpy arrays or tensors in HWC/CHW format, with optional
    batch dimensions. Outputs are PyTorch tensors in CHW/NCHW format.
    """

    size: int
    crop: bool = False

    def _as_chw_tensor(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected numpy array or torch.Tensor, got {type(image)}")
        if image.ndim != 3:
            raise ValueError(
                "Expected a single image with shape HWC or CHW, "
                f"got {tuple(image.shape)}"
            )

        if image.shape[0] in (1, 3):
            return image
        if image.shape[0] == 4 and image.shape[-1] not in (1, 3, 4):
            return image
        if image.shape[-1] in (1, 3, 4):
            return image.permute(2, 0, 1)
        raise ValueError(
            "Could not infer channel dimension from image shape "
            f"{tuple(image.shape)}"
        )

    def _resize(self, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        image = image.unsqueeze(0).float()
        image = F.interpolate(
            image,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return image.squeeze(0)

    def _resize_small(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        ratio = self.size / min(height, width)
        resized_height = round(height * ratio)
        resized_width = round(width * ratio)
        return self._resize(image, (resized_height, resized_width))

    def _crop(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        top = (height - self.size) // 2
        left = (width - self.size) // 2
        return image[:, top : top + self.size, left : left + self.size]

    def _value_range(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(dtype=torch.float32) / 255.0
        return -1.0 + image * 2.0

    def preprocess(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Resizes a single image to a CHW float tensor in ``[-1, 1]``."""
        image = self._as_chw_tensor(image)
        if image.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 image, got {image.dtype}")

        if self.crop:
            image = self._resize_small(image)
            image = self._crop(image)
        else:
            image = self._resize(image, (self.size, self.size))

        return self._value_range(image.to(dtype=torch.uint8))

    def __call__(
        self,
        images: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        """Resizes one image or a batch of images to CHW/NCHW tensors."""
        if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 3:
            return self.preprocess(images)

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        if isinstance(images, torch.Tensor):
            if images.ndim != 4:
                raise ValueError(
                    "Expected a batch with shape NHWC or NCHW, "
                    f"got {tuple(images.shape)}"
                )
            return torch.stack([self.preprocess(image) for image in images])

        return torch.stack([self.preprocess(image) for image in images])


def get_pp(
    *,
    tokenizer_name: str,
    vocab_path: str,
    max_len: int,
    size: int,
    crop: bool = False,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Returns a preprocessing function for optional "image" and "text" keys.

    The returned function copies the input mapping. If ``image`` is present it
    is replaced with a resized PyTorch tensor. If ``text`` is present, token ids
    are written to a new ``tokens`` key.
    """
    tokenizer_class = get_tokenizer(tokenizer_name)
    tokenizer = tokenizer_class(vocab_path=vocab_path, max_len=max_len)
    preprocess_images = PreprocessImages(size=size, crop=crop)

    def pp(features: Mapping[str, Any]) -> dict[str, Any]:
        output = {**features}

        if "image" in output:
            output["image"] = preprocess_images.preprocess(output["image"])

        if "text" in output:
            output["tokens"] = tokenizer.preprocess(output["text"])

        return output

    return pp
