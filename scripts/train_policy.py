"""S4: train the policy (distill theta*, then V(W) with W and V frozen) for one text_mode.

Usage: python scripts/train_policy.py device=cuda text_mode=full world_model=... critic=... oracle=... data.roots=[...]
Writes {out_dir}_{text_mode}/seed{seed}/model.pt and metrics.json (training history, row counts).
Only statement texts and card ids come from the cards; no rubric content reaches the model.
"""
import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from spec2mpc.critic.critic import build_critic
from spec2mpc.model.data import TASK_IDS, block_split, load_rollouts, load_theta_star
from spec2mpc.model.io import load_checkpoint, save_checkpoint
from spec2mpc.model.train_policy import check_split, distill_table, train_policy, vw_table
from spec2mpc.model.world_model import N_PATH, build_world_model
from spec2mpc.mpc.theta import normalize_theta
from spec2mpc.rubric.splits import statement_records
from spec2mpc.sim.problems import load_oracle_problems
from spec2mpc.sim.storage import code_digest, git_hash
from spec2mpc.text.encoder import make_text_encoder, text_features
from spec2mpc.tokenize.robot_tokens import robot_tokens


@hydra.main(version_base="1.3", config_path="../configs", config_name="train/policy")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    recs = [r for r in statement_records(cfg.splits) if r.role == "train"]
    encoder = make_text_encoder(cfg.text)
    emb, slots = text_features(encoder, [r.text for r in recs], cfg.text.slots)  # [S, d_text], [S, d_slot]
    W, w_cfg, w_extra = load_checkpoint(Path(cfg.world_model), build_world_model)
    V, v_cfg, v_extra = load_checkpoint(Path(cfg.critic), build_critic)
    check_split(cfg, w_cfg, v_cfg)
    assert v_extra["encoder"] == encoder.name and int(v_cfg.d_text) == emb.shape[1], \
        f"critic trained with {v_extra['encoder']} (d_text {v_cfg.d_text}), policy uses {encoder.name} ({emb.shape[1]})"

    # stage A rows: oracle problems rebuilt from the oracle run's own config.yaml (digest-checked against problems.pt
    # when present), one row per (record, train statement); theta* normalized with that run's mpc box
    problems, ocfg = load_oracle_problems(Path(cfg.oracle).parent)
    assert str(ocfg.split) == "train", f"policy distills the train oracle, got split {ocfg.split}"
    assert problems.path_tok.shape[1] == N_PATH, problems.path_tok.shape
    pid, cid, task, theta = load_theta_star(Path(cfg.oracle))
    p_task = dict(zip(problems.problem_id, problems.task))
    assert all(p_task.get(p) == t for p, t in zip(pid, task)), "oracle problems differ from the rebuilt train problems"
    distill = distill_table(problems.problem_id, robot_tokens(problems.robot, cfg.robot_tokens), problems.path_tok.float(),
                            emb, slots, [r.card_id for r in recs], pid, cid, normalize_theta(theta, ocfg.mpc))

    # stage B contexts: train blocks of the rollouts crossed with the train statements of the same task
    data = load_rollouts([Path(r) for r in cfg.data.roots], cfg.robot_tokens, cfg.mpc, cfg.sim)
    val = block_split(data.block_id, float(cfg.data.val_frac), int(cfg.data.split_seed))
    contexts = vw_table(data, ~val, emb, slots, torch.tensor([TASK_IDS[r.task] for r in recs]))
    print(f"distill rows {distill[0].batch_size} from {len(pid)} oracle records; V(W) contexts {contexts.batch_size} "
          f"from {contexts.robot_tok.shape[0]} train blocks x {len(recs)} train statements", flush=True)

    with open_dict(cfg):
        cfg.d_text, cfg.d_slot = int(emb.shape[1]), int(slots.shape[1])
    history = []
    policy = train_policy(cfg, distill, contexts, W, V, log=history)

    out = Path(f"{cfg.out_dir}_{cfg.text_mode}") / f"seed{int(cfg.seed)}"
    last = dict(history[-1]) if history else {}
    save_checkpoint(out / "model.pt", policy.cpu(), cfg, {"encoder": encoder.name, "text_mode": str(cfg.text_mode),
                                                         "world_model_git_hash": w_extra["git_hash"],
                                                         "critic_git_hash": v_extra["git_hash"], "last": last})
    metrics = {"n_distill_rows": distill[0].batch_size, "n_oracle_records": len(pid), "n_vw_contexts": contexts.batch_size,
               "statements": [r.text for r in recs], "history": history, "git_hash": git_hash(),
               "code_digest": code_digest(),
               "config": OmegaConf.to_container(cfg, resolve=True)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
