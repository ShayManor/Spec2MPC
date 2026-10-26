import hashlib
from typing import Protocol, Sequence

import torch
from omegaconf import DictConfig
from torch import Tensor

from spec2mpc.text.slots import slot_matrix


class TextEncoder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> Tensor:
        ...  # returns [N, dim] float32, L2-normalized, CPU, no grad


def _hash(feature: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")
    return h % dim, 1.0 if (h >> 63) & 1 else -1.0


class HashedNgramEncoder:
    def __init__(self, dim: int, word_ngrams: int, char_ngrams: int) -> None:
        self.dim = dim
        self.word_ngrams = word_ngrams
        self.char_ngrams = char_ngrams
        self.name = f"hashed{dim}"

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        words = text.split()
        feats = []
        for n in range(1, self.word_ngrams + 1):
            feats += ["w" + " ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        n = self.char_ngrams
        feats += ["c" + text[i:i + n] for i in range(len(text) - n + 1)]
        return feats

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> Tensor:
        # returns [N, dim]
        out = torch.zeros(len(texts), self.dim, dtype=torch.float32)
        for i, text in enumerate(texts):
            for f in self._features(text):
                idx, sign = _hash(f, self.dim)
                out[i, idx] += sign
        norm = out.norm(dim=1, keepdim=True)
        return out / norm.clamp_min(1e-12)


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_embedding_dimension())
        self.name = model_name

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> Tensor:
        # returns [N, dim]
        e = self.model.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True)
        return e.detach().to("cpu", torch.float32)


def make_text_encoder(text_cfg: DictConfig) -> TextEncoder:
    cfg = text_cfg.encoder
    if cfg.backend == "hashed":
        return HashedNgramEncoder(cfg.hashed.dim, cfg.hashed.word_ngrams, cfg.hashed.char_ngrams)
    if cfg.backend == "sentence_transformers":
        return SentenceTransformerEncoder(cfg.model, cfg.device)
    raise ValueError(f"unknown text encoder backend: {cfg.backend}")


def text_features(encoder: TextEncoder, texts: Sequence[str], slot_cfg: DictConfig) -> tuple[Tensor, Tensor]:
    # returns ([N, dim] embeddings, [N, D_SLOT] slots)
    return encoder.encode(texts), slot_matrix(texts, slot_cfg)
