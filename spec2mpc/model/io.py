from pathlib import Path
from typing import Callable, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from spec2mpc.sim.storage import code_digest, git_hash


def save_checkpoint(path: Path, model: nn.Module, cfg: DictConfig, extra: dict) -> None:
    # state dict, resolved config, git hash, code digest and caller metadata in one torch file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # temp file then rename, so a job killed mid-save never leaves a truncated model.pt that later stages skip on
    tmp = path.with_name(f".tmp_{path.name}")
    torch.save({"state_dict": model.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True),
                "git_hash": git_hash(), "code_digest": code_digest(), "extra": extra}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, build: Callable[[DictConfig], nn.Module]) -> Tuple[nn.Module, DictConfig, dict]:
    # -> model rebuilt from the stored config with weights loaded, config, extra (plus "git_hash")
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(ck["config"])
    model = build(cfg)
    model.load_state_dict(ck["state_dict"])
    return model, cfg, {**ck["extra"], "git_hash": ck["git_hash"]}
