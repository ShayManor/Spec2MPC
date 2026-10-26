"""S2: train the intent critic ensemble V_psi on train-card scores of train blocks.

Usage: python scripts/train_critic.py device=cuda text.encoder.backend=sentence_transformers out_dir=...
Writes {out_dir}/seed{seed}/model.pt, metrics.json (val-block Spearman per card for train cards with their
val_paraphrase statement and for val_intent cards, plus intent margin, J >= 0 Spearman and disagreement-pair
accuracy) and val_predictions.pt (per card and statement: true J, score). eval_only=<model.pt> skips training and evaluates that checkpoint.
"""
import json
import time
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor

from spec2mpc.critic.critic import CriticEnsemble, build_critic, critic_loss, flip_pair_weight
from spec2mpc.eval.stats import bootstrap_ci, disagreement_pairs, pair_accuracy, spearman
from spec2mpc.model.data import block_split, load_rollouts
from spec2mpc.model.io import load_checkpoint, save_checkpoint
from spec2mpc.rubric.cards import load_cards
from spec2mpc.rubric.labels import load_score_labels
from spec2mpc.rubric.splits import statement_records
from spec2mpc.sim.storage import code_digest, git_hash
from spec2mpc.text.encoder import make_text_encoder, text_features


@torch.no_grad()
def predict(v: CriticEnsemble, emb: Tensor, slots: Tensor, phi: Tensor, lam: float, batch: int) -> Tensor:
    # one statement emb [d_text], slots [d_slot]; phi [N, 20] raw -> [2, N] (score with lam, ensemble mean)
    v.eval()
    out = []
    for i in range(0, phi.shape[0], batch):
        p = phi[i:i + batch]
        m = v(emb.expand(p.shape[0], -1), slots.expand(p.shape[0], -1), p)  # [M, b]
        out.append(torch.stack([m.mean(0) - lam * m.std(0), m.mean(0)]))
    return torch.cat(out, dim=1) if out else torch.zeros(2, 0, device=phi.device)


def evaluate(v: CriticEnsemble, emb: Tensor, slots: Tensor, phi: Tensor, J: Tensor, ok: Tensor, val: Tensor,
             eval_stmts: Dict[int, List[int]], first_stmt: Dict[int, int], tasks: List[str], card_ids: List[str],
             cfg: DictConfig, keep: bool) -> Dict:
    # per card on val-block in-condition episodes, averaged over the card's eval statements:
    #   spearman: score vs true J; spearman_jge0: same on episodes with J >= 0;
    #   intent margin: own-statement Spearman minus mean Spearman when scored with other same-task cards' statements,
    #     label_spearman_other: mean Spearman(J_own, J_other) on joint episodes (margin ceiling = 1 - it);
    #   disagreement_acc: pairwise accuracy on pairs ordered oppositely by this card's J and another same-task card's J
    # each with mean and bootstrap CI over cards
    margin = float(cfg.rank_margin)
    need = sorted({st for c in eval_stmts for st in eval_stmts[c]} | set(first_stmt.values()))
    scores = {st: predict(v, emb[st], slots[st], phi, float(cfg.lam), int(cfg.eval_batch)) for st in need}  # [2, E]
    per_card, preds = {}, {}
    for c, stmts in eval_stmts.items():
        m_c = ok[:, c] & val
        e = m_c.nonzero().squeeze(1)
        others = [o for o in first_stmt if o != c and tasks[o] == tasks[c]]
        other_rho = [spearman(scores[first_stmt[o]][0, e], J[e, c]) for o in others]
        label_rho = []
        for o in others:
            joint = (m_c & ok[:, o]).nonzero().squeeze(1)
            label_rho.append(spearman(J[joint, c], J[joint, o]))
        rows = []
        for st in stmts:
            p = scores[st][:, e]
            pos = J[e, c] >= 0
            correct, n_pairs = 0.0, 0
            for o in others:
                joint = (m_c & ok[:, o]).nonzero().squeeze(1)
                acc, n = pair_accuracy(scores[st][0, joint], disagreement_pairs(J[joint, c], J[joint, o], margin))
                correct, n_pairs = correct + (acc * n if n else 0.0), n_pairs + n
            rho = spearman(p[0], J[e, c])
            rows.append({"statement": st, "spearman": rho, "spearman_mean_pred": spearman(p[1], J[e, c]),
                         "spearman_jge0": spearman(p[0, pos], J[e[pos], c]), "intent_margin": rho - sum(other_rho) / len(other_rho),
                         "disagreement_acc": correct / n_pairs if n_pairs else float("nan"), "disagreement_pairs": n_pairs})
            if keep:
                preds[(card_ids[c], st)] = {"J": J[e, c].cpu(), "score": p[0].cpu(), "mean": p[1].cpu()}
        avg = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
        per_card[card_ids[c]] = {"n_episodes": int(e.numel()), "per_statement": rows,
                                 **{k: avg(k) for k in ("spearman", "spearman_mean_pred", "spearman_jge0", "intent_margin", "disagreement_acc")},
                                 "other_statement_spearman": dict(zip([card_ids[o] for o in others], other_rho)),
                                 "label_spearman_other": torch.tensor(label_rho).nanmean().item()}
    out = {"per_card": per_card, "predictions": preds}
    for k in ("spearman", "spearman_jge0", "intent_margin", "disagreement_acc", "label_spearman_other"):
        vals = torch.tensor([r[k] for r in per_card.values()], dtype=torch.float64)
        vals = vals[~vals.isnan()]  # cards with constant labels or no pairs carry no value
        mean, lo, hi = bootstrap_ci(vals, torch.arange(vals.numel()), int(cfg.ci.n_boot), float(cfg.ci.alpha),
                                    torch.Generator().manual_seed(int(cfg.seed))) if vals.numel() else (float("nan"),) * 3
        out[f"mean_{k}"], out[f"ci_{k}"] = mean, [lo, hi]
    return out


def episode_rows(sel: Tensor, ep_start: Tensor, ep_count: Tensor) -> Tensor:
    # sel [b] episode indices -> all their row indices (rows of one episode are contiguous)
    counts = ep_count[sel]
    offs = torch.arange(int(counts.sum()), device=sel.device) - (torch.cumsum(counts, 0) - counts).repeat_interleave(counts)
    return ep_start[sel].repeat_interleave(counts) + offs


def train(v: CriticEnsemble, cfg: DictConfig, emb: Tensor, slots: Tensor, phi: Tensor, J: Tensor, ok: Tensor, val: Tensor,
          row_e: Tensor, row_c: Tensor, stmt_table: Tensor, stmt_count: Tensor, ep_start: Tensor, ep_count: Tensor,
          n_train_cards: int, eval_args: tuple) -> List[Dict]:
    # episode-major: each member bootstraps train episodes and every batch holds all in-condition train cards of its
    # episodes; otherwise rows are bootstrapped and batched independently
    device, N, n_ep = phi.device, row_e.numel(), ep_count.numel()
    M, o = int(cfg.members), cfg.opt
    opt = torch.optim.Adam(v.parameters(), lr=float(o.lr))
    gen = torch.Generator().manual_seed(int(cfg.seed))
    n_units = n_ep if cfg.episode_major else N
    boot = [torch.randint(0, n_units, (n_units,), generator=torch.Generator().manual_seed(int(cfg.seed) * 1000 + m)).to(device)
            for m in range(M)]  # member m's bootstrap resample of episodes (or rows)
    bs = max(1, round(int(o.batch) * n_ep / N)) if cfg.episode_major else int(o.batch)  # units per batch
    history, t0 = [], time.perf_counter()
    for epoch in range(int(o.epochs)):
        v.train()
        pick = (torch.rand(N, generator=gen).to(device) * stmt_count[row_c]).long()
        row_s = stmt_table[row_c, pick]  # [N] statement drawn uniformly from the card's train statements
        perms = [torch.randperm(n_units, generator=gen).to(device) for _ in range(M)]
        tot = 0.0
        for k in range(n_units // bs):
            loss = 0.0
            for m in range(M):
                u = boot[m][perms[m][k * bs:(k + 1) * bs]]
                r = episode_rows(u, ep_start, ep_count) if cfg.episode_major else u
                e, c, s = row_e[r], row_c[r], row_s[r]
                target = J[e, c]
                w = None
                if float(cfg.flip_upweight) != 1.0:
                    w = flip_pair_weight(J[e, :n_train_cards], ok[e, :n_train_cards], float(cfg.rank_margin), float(cfg.flip_upweight))
                pred = v.member(m, emb[s], slots[s], phi[e])[None]  # [1, B]
                loss = loss + critic_loss(pred, target, c, float(cfg.w_rank), float(cfg.rank_margin), pair_weight=w,
                                          mse_target=torch.asinh(target) if cfg.target_asinh else None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() / M
        rec = {"epoch": epoch + 1, "train_loss": tot / (n_units // bs), "time_s": time.perf_counter() - t0}
        if (epoch + 1) % int(cfg.log_every) == 0 or epoch + 1 == int(o.epochs):
            ev = evaluate(v, emb, slots, phi, J, ok, val, eval_args[0], *eval_args[1:], cfg, False)
            rec["val_spearman_paraphrase"] = ev["mean_spearman"]
            rec["val_disagreement_acc_paraphrase"] = ev["mean_disagreement_acc"]
        history.append(rec)
        print(json.dumps(rec), flush=True)
    return history


@hydra.main(version_base="1.3", config_path="../configs", config_name="train/critic")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    data = load_rollouts([Path(r) for r in cfg.data.roots], cfg.robot_tokens, cfg.mpc, cfg.sim)
    root = Path(cfg.splits.cards_root)
    train_cards, val_cards = load_cards(root, "train"), load_cards(root, "val")
    cards = train_cards + val_cards
    card_ids = [c.id for c in cards]
    J, ok = load_score_labels(data, cards)  # [E, C] normalized J, in_conditions
    val = block_split(data.block_id, float(cfg.val_frac), int(cfg.split_seed))

    recs = statement_records(cfg.splits)
    encoder = make_text_encoder(cfg.text)
    emb, slots = text_features(encoder, [r.text for r in recs], cfg.text.slots)  # [S, d_text], [S, d_slot]
    by_role = lambda role: {card_ids.index(cid): [i for i, r in enumerate(recs) if r.card_id == cid and r.role == role]  # noqa: E731
                            for cid in card_ids if any(r.card_id == cid and r.role == role for r in recs)}
    train_stmts, para_stmts, intent_stmts = by_role("train"), by_role("val_paraphrase"), by_role("val_intent")
    n_train_cards = len(train_cards)
    assert sorted(train_stmts) == list(range(n_train_cards)) and sorted(intent_stmts) == list(range(n_train_cards, len(cards)))
    width = max(len(s) for s in train_stmts.values())
    stmt_table = torch.tensor([train_stmts[c] + [train_stmts[c][0]] * (width - len(train_stmts[c])) for c in range(n_train_cards)])
    stmt_count = torch.tensor([len(train_stmts[c]) for c in range(n_train_cards)])

    # training rows: (episode, train card) in the card's conditions (False off-task), train blocks only
    row_e, row_c = (ok[:, :n_train_cards] & ~val[:, None]).nonzero(as_tuple=True)  # episode-major order
    N = row_e.numel()
    print(f"train rows {N} from {int((~val).sum())} train-block episodes; val-block episodes {int(val.sum())}", flush=True)
    eps, ep_count = torch.unique_consecutive(row_e, return_counts=True)
    ep_start = torch.cumsum(ep_count, 0) - ep_count
    tasks = [c.task for c in cards]
    first_stmt = {c: st[0] for c, st in {**para_stmts, **intent_stmts}.items()}

    emb, slots, phi, J, ok, val = emb.to(device), slots.to(device), data.phi.to(device), J.to(device), ok.to(device), val.to(device)
    row_e, row_c, stmt_table, stmt_count = row_e.to(device), row_c.to(device), stmt_table.to(device), stmt_count.to(device)
    ep_start, ep_count = ep_start.to(device), ep_count.to(device)
    with open_dict(cfg):
        cfg.d_text, cfg.d_slot = int(emb.shape[1]), int(slots.shape[1])
    if cfg.eval_only:
        v, _, _ = load_checkpoint(Path(cfg.eval_only), build_critic)
        v = v.to(device)
        history = []
    else:
        v = build_critic(cfg).to(device)
        v.set_phi_stats(phi[~val])
        history = train(v, cfg, emb, slots, phi, J, ok, val, row_e, row_c, stmt_table, stmt_count, ep_start, ep_count,
                        n_train_cards, (para_stmts, first_stmt, tasks, card_ids))

    ev_args = (first_stmt, tasks, card_ids, cfg, True)
    para = evaluate(v, emb, slots, phi, J, ok, val, para_stmts, *ev_args)
    intent = evaluate(v, emb, slots, phi, J, ok, val, intent_stmts, *ev_args)
    preds = {**para.pop("predictions"), **intent.pop("predictions")}
    metrics = {"val_paraphrase": para, "val_intent": intent, "statements": [r.text for r in recs],
               "n_train_rows": N, "history": history, "git_hash": git_hash(), "code_digest": code_digest(),
               "config": OmegaConf.to_container(cfg, resolve=True)}
    out = Path(cfg.out_dir) / f"seed{int(cfg.seed)}"
    if not cfg.eval_only:
        save_checkpoint(out / "model.pt", v.cpu(), cfg, {"encoder": encoder.name,
                                                        "mean_spearman_paraphrase": para["mean_spearman"],
                                                        "mean_spearman_intent": intent["mean_spearman"]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({f"{cid}|{st}": d for (cid, st), d in preds.items()}, out / "val_predictions.pt")
    print(f"M2 target 0.8: mean_spearman_paraphrase={para['mean_spearman']:.3f} "
          f"mean_spearman_intent={intent['mean_spearman']:.3f}", flush=True)
    for name, r in (("paraphrase", para), ("intent", intent)):
        print(f"{name}: disagreement_acc={r['mean_disagreement_acc']:.3f} "
              f"[{r['ci_disagreement_acc'][0]:.3f}, {r['ci_disagreement_acc'][1]:.3f}] "
              f"intent_margin={r['mean_intent_margin']:.3f} (ceiling {1 - r['mean_label_spearman_other']:.3f}) "
              f"spearman_jge0={r['mean_spearman_jge0']:.3f}", flush=True)


if __name__ == "__main__":
    main()
