"""Evaluation: every method on the val oracle problems with true sim rollouts scored by the true rubric.

Usage: python scripts/run_eval.py device=cuda oracle_root=... models.critic=.../seed{seed}/model.pt out_dir=...
methods=[...] seeds=[...] select a subset. Existing {out_dir}/{method}_{K}_{seed}_{group}.parquet files are skipped
(resumable). At the end, when this group's default and oracle files exist, every result file present whose json records
this n_xi and xi_seed is summarized into {out_dir}/summary.parquet (each row normalized by the anchors of its own group;
rows of a group without anchors are dropped and counted) and the three figures are written to {plot_dir}/eval_*.png.
"""
import sys
from pathlib import Path
from typing import List

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig, OmegaConf

from spec2mpc.baselines.applv_lang import build_applv
from spec2mpc.critic.critic import build_critic
from spec2mpc.eval.metrics import attach_jhat, drop_unanchored, summarize
from spec2mpc.eval.plots import plot_critic_gap, plot_methods, plot_score_vs_budget
from spec2mpc.eval.runner import (SUMMARY_FILE, EvalContext, anchor_files, make_layout, result_files, run_settings,
                                  settings, val_units, write_table)
from spec2mpc.model.io import load_checkpoint
from spec2mpc.model.policy import build_policy
from spec2mpc.model.world_model import build_world_model
from spec2mpc.mpc.theta import theta_bounds
from spec2mpc.rubric.cards import load_cards
from spec2mpc.rubric.splits import statement_records
from spec2mpc.sim.problems import load_oracle_problems
from spec2mpc.sim.storage import code_digest, git_hash
from spec2mpc.text.encoder import make_text_encoder, text_features
from spec2mpc.tokenize.robot_tokens import robot_tokens

BUILDERS = {"world_model": build_world_model, "critic": build_critic, "policy_full": build_policy,
            "policy_zero": build_policy, "applv": build_applv}

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    problems, rcfg = load_oracle_problems(Path(cfg.oracle_root) / "val")
    assert str(rcfg.split) == "val", f"eval needs the val oracle run, got split {rcfg.split}"
    if not all(torch.equal(a, b) for a, b in zip(theta_bounds(cfg.mpc), theta_bounds(rcfg.mpc))):
        raise ValueError("eval mpc theta box differs from the val oracle run")
    root = Path(cfg.splits.cards_root)
    cards = {c.id: c for c in load_cards(root, "train") + load_cards(root, "val")}
    units = val_units(cfg.splits)
    layout = make_layout(units, problems, cards, int(cfg.n_xi), int(cfg.xi_seed), rcfg.sim)
    train = [r for r in statement_records(cfg.splits) if r.role == "train"]
    encoder = make_text_encoder(cfg.text)
    emb, slots = text_features(encoder, [u.statement for u in units] + [r.text for r in train], cfg.text.slots)
    U = len(units)
    loaders = {n: (lambda s, n=n: load_checkpoint(Path(str(cfg.models[n]).format(seed=s)), BUILDERS[n]))
               for n in BUILDERS}
    ctx = EvalContext(units=units, problems=problems, layout=layout, cards=cards, rcfg=rcfg, cfg=cfg, device=device,
                      unit_emb=emb[:U], unit_slots=slots[:U], train_emb=emb[U:], train_card=[r.card_id for r in train],
                      encoder_name=encoder.name, robot_tok=robot_tokens(problems.robot, cfg.robot_tokens),
                      loaders=loaders)
    n_pairs = layout.pairs()[0].numel()
    print(f"eval units={U} problems={problems.n} pairs={n_pairs} episodes/setting={layout.unit_idx.numel()} "
          f"encoder={encoder.name} device={device}", flush=True)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    group = str(cfg.group)
    tags = {"device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "code_digest": code_digest()}
    print(f"group={group} {tags}", flush=True)
    failed = run_settings(ctx, settings(cfg), out, float(cfg.min_gap),
                          {"git_hash": git_hash(), "config": OmegaConf.to_container(cfg, resolve=True)}, group, tags)

    anchors = anchor_files(group)
    if not all((out / a).exists() for a in anchors):
        print(f"summary skipped: {anchors} not all present in {out}", flush=True)
        _exit_on_failures(failed)
        return
    files, skipped = result_files(out, int(cfg.n_xi), int(cfg.xi_seed))
    if skipped:
        print(f"summary skips {len(skipped)} files without a json for n_xi={cfg.n_xi} xi_seed={cfg.xi_seed}: "
              f"{[p.name for p in skipped]}", flush=True)
    results, unanchored = drop_unanchored(pa.concat_tables([pq.read_table(p) for p in files]))
    for (m, k, kind, g), n in sorted(unanchored.items()):
        print(f"summary drops {n} rows of {m} K={k} {kind} group={g}: no default and oracle rows in that group",
              flush=True)
    results = attach_jhat(results, float(cfg.min_gap))
    summary = summarize(results, int(cfg.n_boot), int(cfg.xi_seed), unanchored)
    assert all(r == 0.0 for r in summary.column("floor_violation_rate").to_pylist()), "safety-floor violation in results"
    write_table(summary, out / SUMMARY_FILE, f"{group}_")
    for r in summary.to_pylist():
        print(f"{r['method']:>12} K={r['K']:<4} {r['kind']:<10} Jhat={r['Jhat_mean']:+.3f} "
              f"[{r['Jhat_lo']:+.3f}, {r['Jhat_hi']:+.3f}] n_cards={r['n_cards']} gap_ens={r['gap_ens_mean']:+.3f} "
              f"gap={r['gap_mean']:+.3f} excluded={r['n_excluded']} unanchored={r['n_unanchored']} "
              f"floor_viol={r['floor_violation_rate']:.3f} boundary_viol={r['boundary_violation_rate']:.3f}", flush=True)
    plots = Path(cfg.plot_dir)
    plot_score_vs_budget(summary, plots / "eval_score_vs_budget.png")
    plot_critic_gap(summary, plots / "eval_critic_gap.png")
    plot_methods(summary, plots / "eval_methods.png")
    print(f"wrote {out / SUMMARY_FILE} and {plots}/eval_*.png", flush=True)
    _exit_on_failures(failed)


def _exit_on_failures(failed: List[str]) -> None:
    if failed:
        print(f"FAILED settings ({len(failed)}): {failed}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
