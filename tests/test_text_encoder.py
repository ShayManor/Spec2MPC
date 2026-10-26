import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

from spec2mpc.text.encoder import HashedNgramEncoder, make_text_encoder, text_features
from spec2mpc.text.slots import D_SLOT

TEXT_CFG = OmegaConf.load("configs/text/default.yaml")
TEXT = "Kart that sips power"


def fmt(v: torch.Tensor) -> str:
    return " ".join(f"{x:.6f}" for x in v.tolist())


def cfg_with(**kw):
    cfg = OmegaConf.create(OmegaConf.to_container(TEXT_CFG))
    cfg.encoder.update(kw)
    return cfg


def test_hashed_deterministic_across_processes():
    code = (
        "from spec2mpc.text.encoder import HashedNgramEncoder\n"
        f"e = HashedNgramEncoder(512, 2, 3).encode([{TEXT!r}])[0, :8]\n"
        "print(' '.join(f'{x:.6f}' for x in e.tolist()))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == fmt(HashedNgramEncoder(512, 2, 3).encode([TEXT])[0, :8])


def test_hashed_shape_norm_and_empty():
    e = HashedNgramEncoder(512, 2, 3).encode(["a", "", "Kart for a novice driver"])
    assert e.shape == (3, 512) and e.dtype == torch.float32
    assert torch.allclose(e[[0, 2]].norm(dim=1), torch.ones(2))
    assert e[1].abs().sum() == 0


def test_hashed_paraphrase_closer_than_unrelated():
    a, b, c = HashedNgramEncoder(512, 2, 3).encode(
        ["Kart driving on a wet track", "Kart on a wet slippery track", "Deliver the package"]
    )
    assert a @ b > a @ c


def test_factory_hashed_and_unknown_backend():
    enc = make_text_encoder(cfg_with(backend="hashed"))
    assert enc.dim == 512 and enc.name == "hashed512"
    with pytest.raises(ValueError):
        make_text_encoder(cfg_with(backend="nope"))


def test_text_features_shapes():
    enc = make_text_encoder(cfg_with(backend="hashed"))
    emb, slots = text_features(enc, ["Finish in under 5 minutes", "Gentle ride"], TEXT_CFG.slots)
    assert emb.shape == (2, 512) and slots.shape == (2, D_SLOT)


@pytest.mark.slow
def test_sentence_transformer_backend():
    try:
        enc = make_text_encoder(cfg_with(backend="sentence_transformers"))
    except Exception as exc:  # model not in the local HF cache
        pytest.skip(str(exc))
    e = enc.encode(["Racing kart, speed is what matters", "Kart that goes fast", "Gentle ride for kids"])
    assert e.shape == (3, 384) and e[0] @ e[1] > e[0] @ e[2]
