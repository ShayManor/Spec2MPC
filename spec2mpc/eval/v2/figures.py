"""Round figures (matplotlib, minimal): pooled bars, per-family and per-goal dots, score vs test-time compute."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow as pa  # noqa: E402

from spec2mpc.eval.v2.metrics import load_round, pooled  # noqa: E402

PREVIEW_DIR = Path("data/previews")
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#7f7f7f")


def _base(ex: str) -> str:
    # executors run at several test-time budgets are named <name>@<k>
    return ex.split("@")[0]


def _clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> Path:
    if not fig.get_layout_engine():
        fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _headline(rows: list[dict]) -> list[dict]:
    # per base executor keep only its largest-compute variant
    best: dict[str, tuple[float, str]] = {}
    for r in rows:
        c = r["compute"] if r.get("compute") is not None else -1.0
        b = _base(r["executor"])
        if b not in best or c > best[b][0]:
            best[b] = (c, r["executor"])
    keep = {ex for _, ex in best.values()}
    return [r for r in rows if r["executor"] in keep]


def _colors(rows: list[dict]) -> dict[str, str]:
    bases = sorted({_base(r["executor"]) for r in rows})
    return {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(bases)}


def _pooled_bars(rows: list[dict], colors: dict, path: Path) -> Path:
    splits = list(dict.fromkeys(r["split"] for r in rows))
    exs = sorted({r["executor"] for r in rows})
    w = 0.8 / len(exs)
    fig, axes = plt.subplots(1, 2, figsize=(max(6.0, 1.6 * len(splits) + 3), 3.2), sharey=True)
    for ax, (key, ylabel) in zip(axes, (("success_rate", "success rate"), ("score", "mean score"))):
        for j, ex in enumerate(exs):
            xs, m, lo, hi = [], [], [], []
            for i, sp in enumerate(splits):
                sub = [r for r in rows if r["executor"] == ex and r["split"] == sp]
                if not sub:
                    continue
                p = pooled(sub)
                xs.append(i + (j - (len(exs) - 1) / 2) * w)
                m.append(p[key])
                lo.append(p[key] - p[f"{key}_lo"])
                hi.append(p[f"{key}_hi"] - p[key])
            ax.bar(xs, m, w * 0.9, yerr=[lo, hi], color=colors[_base(ex)], capsize=2,
                   error_kw={"lw": 0.8, "ecolor": "#333333"}, label=ex)
        ax.set_xticks(range(len(splits)), splits, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        _clean(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=len(exs))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, path)


def _dots(rows: list[dict], colors: dict, by: str, path: Path) -> Path:
    # mean score with 95% CI per category, one color per executor, all splits pooled
    cats = sorted({r[by] for r in rows})
    exs = sorted({r["executor"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.0, 0.32 * len(cats) * max(1, len(exs) / 2) + 1.0))
    off = 0.6 / max(1, len(exs))
    for j, ex in enumerate(exs):
        for i, c in enumerate(cats):
            sub = [r for r in rows if r["executor"] == ex and r[by] == c]
            if not sub:
                continue
            p = pooled(sub)
            y = i + (j - (len(exs) - 1) / 2) * off
            ax.plot([p["score_lo"], p["score_hi"]], [y, y], color=colors[_base(ex)], lw=1.2)
            ax.plot(p["score"], y, "o", color=colors[_base(ex)], ms=4, label=ex if i == 0 else None)
    ax.set_yticks(range(len(cats)), cats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean score")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(1.0, 0.0))
    _clean(ax)
    return _save(fig, path)


def _compute(rows: list[dict], colors: dict, path: Path) -> Path:
    # mean score vs test-time compute for executors run at several budgets; single-budget executors as flat lines
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    scaled = sorted({_base(r["executor"]) for r in rows if r.get("compute") is not None})
    for b in scaled:
        pts = sorted({r["compute"] for r in rows if _base(r["executor"]) == b and r.get("compute") is not None})
        ps = [pooled([r for r in rows if _base(r["executor"]) == b and r.get("compute") == c]) for c in pts]
        ax.fill_between(pts, [p["score_lo"] for p in ps], [p["score_hi"] for p in ps], color=colors[b], alpha=0.15, lw=0)
        ax.plot(pts, [p["score"] for p in ps], "o-", color=colors[b], ms=3.5, lw=1.5, label=b)
    for b in sorted({_base(r["executor"]) for r in rows} - set(scaled)):
        p = pooled(_headline([r for r in rows if _base(r["executor"]) == b]))
        ax.axhline(p["score"], color=colors[b], lw=1.0, ls="--", label=b)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("test-time compute")
    ax.set_ylabel("mean score")
    ax.legend(frameon=False, fontsize=8)
    _clean(ax)
    return _save(fig, path)


def figure_round(round_dir: str | Path, out_dir: str | Path = PREVIEW_DIR) -> list[Path]:
    # writes <out_dir>/v2_<round>_{pooled,family,goal[,compute]}.png from every result parquet of the round
    round_dir, out_dir = Path(round_dir), Path(out_dir)
    rows: list[dict] = load_round(round_dir).to_pylist()
    colors = _colors(rows)
    head = _headline(rows)
    pre = out_dir / f"v2_{round_dir.name}"
    paths = [_pooled_bars(head, colors, Path(f"{pre}_pooled.png")),
             _dots(head, colors, "family", Path(f"{pre}_family.png")),
             _dots(head, colors, "goal", Path(f"{pre}_goal.png"))]
    if any(r.get("compute") is not None for r in rows):
        paths.append(_compute(rows, colors, Path(f"{pre}_compute.png")))
    return paths
