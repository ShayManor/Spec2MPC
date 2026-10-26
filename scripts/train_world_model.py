"""S3: train the world model W on rollouts (no text, no rubric).

Usage: python scripts/train_world_model.py device=cuda data.roots=[data/rollouts/gautschi] out_dir=...
Writes {out_dir}/seed{seed}/model.pt and metrics.json (val per-feature R^2 in raw and robust z space for W,
a median predictor, ridge on raw phi and ridge on z). The checkpoint is the epoch with the lowest val loss.
"""
import json
import time
from pathlib import Path
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from spec2mpc.features.phi import PHI_NAMES
from spec2mpc.model.data import RolloutTensors, block_split, load_rollouts
from spec2mpc.model.io import save_checkpoint
from spec2mpc.model.world_model import WorldModel, build_world_model, wm_loss
from spec2mpc.sim.storage import code_digest, git_hash


def to_device(d: RolloutTensors, device: torch.device) -> RolloutTensors:
    return RolloutTensors(d.robot_tok.to(device), d.path_tok.to(device), d.theta_n.to(device), d.xi_n.to(device),
                          d.phi.to(device), d.task.to(device), d.block_id.to(device), d.episode_id.to(device),
                          d.shard.to(device), d.shard_dirs)


def r2(pred: Tensor, true: Tensor) -> Tensor:
    # pred, true [E, 20] -> [20], 1 - SSE / SST around the val mean; NaN where the val feature is constant
    sst = ((true - true.mean(0)) ** 2).sum(0)
    return torch.where(sst > 0, 1 - ((pred - true) ** 2).sum(0) / sst, torch.nan)


@torch.no_grad()
def predict(model: WorldModel, d: RolloutTensors, batch: int) -> Tensor:
    # -> phi_hat [E, 20] raw
    model.eval()
    out = [model(d.robot_tok[i:i + batch], d.path_tok[i:i + batch], d.theta_n[i:i + batch], d.xi_n[i:i + batch])
           for i in range(0, d.phi.shape[0], batch)]
    return torch.cat(out)


@torch.no_grad()
def val_loss(model: WorldModel, d: RolloutTensors, batch: int, huber_delta: float) -> float:
    # episode-weighted mean of wm_loss over d
    model.eval()
    E = d.phi.shape[0]
    idx = torch.arange(E, device=d.phi.device)
    return sum(wm_loss(model, d.index(idx[i:i + batch]), huber_delta).item() * min(batch, E - i)
               for i in range(0, E, batch)) / E


def flat_inputs(d: RolloutTensors) -> Tensor:
    # -> [E, 6*24 + 64*6 + 14 + 6]
    E = d.phi.shape[0]
    return torch.cat([d.robot_tok.reshape(E, -1), d.path_tok.reshape(E, -1), d.theta_n, d.xi_n], dim=-1).double()


def ridge_predict(tr: RolloutTensors, va: RolloutTensors, ytr: Tensor, lam: float) -> Tensor:
    # closed-form ridge on standardized flattened inputs, targets ytr [E_tr, 20] -> prediction on va [E_va, 20]
    xtr, xva = flat_inputs(tr), flat_inputs(va)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr, xva = (xtr - mu) / sd, (xva - mu) / sd
    ytr = ytr.double()
    ymu = ytr.mean(0)
    A = xtr.T @ xtr + lam * torch.eye(xtr.shape[1], dtype=xtr.dtype, device=xtr.device)
    w = torch.linalg.solve(A, xtr.T @ (ytr - ymu))
    return (xva @ w + ymu).float()


def summary(pred: Tensor, true: Tensor, model: WorldModel) -> Dict:
    # pred, true [E, 20] raw -> R^2 in raw space and in the robust z space, and z MSE
    zp, zt = model.phi_to_z(pred), model.phi_to_z(true)
    r, rz = r2(pred, true), r2(zp, zt)
    return {"r2": dict(zip(PHI_NAMES, r.tolist())), "r2_mean": r.nanmean().item(),
            "r2_z": dict(zip(PHI_NAMES, rz.tolist())), "r2_z_mean": rz.nanmean().item(), "z_mse": ((zp - zt) ** 2).mean().item()}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train/world_model")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    data = load_rollouts([Path(r) for r in cfg.data.roots], cfg.robot_tokens, cfg.mpc, cfg.sim)
    val = block_split(data.block_id, float(cfg.data.val_frac), int(cfg.data.split_seed))
    tr, va = to_device(data.index(~val), device), to_device(data.index(val), device)
    print(f"train {tr.phi.shape[0]} val {va.phi.shape[0]} episodes", flush=True)

    model = build_world_model(cfg).to(device)
    model.set_phi_stats(tr.phi)
    o = cfg.opt
    opt = torch.optim.AdamW(model.parameters(), lr=float(o.lr), weight_decay=float(o.weight_decay))
    E, bs = tr.phi.shape[0], int(o.batch)
    steps = int(o.epochs) * (E // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(int(cfg.seed))
    history, t0 = [], time.perf_counter()
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    for epoch in range(int(o.epochs)):
        model.train()
        perm = torch.randperm(E, generator=gen).to(device)
        tot = 0.0
        for k in range(E // bs):
            loss = wm_loss(model, tr.index(perm[k * bs:(k + 1) * bs]), float(cfg.huber_delta))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item()
        rec = {"epoch": epoch + 1, "train_loss": tot / (E // bs), "time_s": time.perf_counter() - t0}
        if (epoch + 1) % int(cfg.log_every) == 0 or epoch + 1 == int(o.epochs):
            rec["val_loss"] = val_loss(model, va, int(cfg.eval_batch), float(cfg.huber_delta))
            if rec["val_loss"] < best["val_loss"]:
                best = {"val_loss": rec["val_loss"], "epoch": epoch + 1,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        history.append(rec)
        print(json.dumps(rec), flush=True)

    model.load_state_dict(best["state"])
    pred = predict(model, va, int(cfg.eval_batch))
    median_pred = model.phi_center.expand_as(va.phi)
    lam = float(cfg.ridge_lambda)
    with torch.no_grad():
        ridge_z = model.z_to_phi(ridge_predict(tr, va, model.phi_to_z(tr.phi), lam))
    metrics = {
        "world_model": summary(pred, va.phi, model),
        "median_predictor": summary(median_pred, va.phi, model),
        "ridge": summary(ridge_predict(tr, va, tr.phi, lam), va.phi, model),
        "ridge_z": summary(ridge_z, va.phi, model),
        "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
        "n_train": int(tr.phi.shape[0]), "n_val": int(va.phi.shape[0]), "history": history,
        "git_hash": git_hash(), "code_digest": code_digest(), "config": OmegaConf.to_container(cfg, resolve=True),
    }
    out = Path(cfg.out_dir) / f"seed{int(cfg.seed)}"
    save_checkpoint(out / "model.pt", model.cpu(), cfg, {"epochs": int(o.epochs), "best_epoch": best["epoch"], "val_r2_mean": metrics["world_model"]["r2_mean"]})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: [metrics[k]["r2_mean"], metrics[k]["r2_z_mean"]] for k in ("world_model", "median_predictor", "ridge", "ridge_z")}
                     | {"best_epoch": best["epoch"]}), flush=True)


if __name__ == "__main__":
    main()
