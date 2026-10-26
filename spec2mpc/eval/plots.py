"""Evaluation figures (matplotlib, minimal)."""
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow as pa  # noqa: E402

KINDS = ("paraphrase", "intent")
MODE_COLORS = {"amortized": "#7f7f7f", "bon_pred": "#1f77b4", "refine": "#2ca02c", "bon_true": "#d62728",
               "oracle_bo": "#111111"}
METHOD_ORDER = ("default", "mean_theta", "nn_retrieval", "slot_only", "applv_lang", "amortized", "bon_pred", "refine",
                "bon_true", "oracle_bo", "oracle")
KIND_COLORS = {"paraphrase": "#1f77b4", "intent": "#ff7f0e"}


def _rows(summary: pa.Table, kind: str, method: str) -> List[Dict]:
    return sorted((r for r in summary.to_pylist() if r["kind"] == kind and r["method"] == method), key=lambda r: r["K"])


def _clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _curves(summary: pa.Table, out_png: Path, ys: tuple, modes: tuple, ylabel: str, zero_line: bool,
            dashed_label: str = "") -> None:
    # one panel per kind: each y in ys vs K on a log2 axis per mode, the first y solid with its CI band, later ones
    # dashed (legend entry dashed_label); amortized at K = 1
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, kind in zip(axes, KINDS):
        if zero_line:
            ax.axhline(0.0, color="#bbbbbb", lw=0.8)
        for mode in modes:
            for j, y in enumerate(ys):
                rs = [r for r in _rows(summary, kind, mode) if r[f"{y}_mean"] == r[f"{y}_mean"]]  # drop NaN
                if not rs:
                    continue
                c = MODE_COLORS[mode]
                label = None if j else ("oracle_bo@k" if mode == "oracle_bo" else mode)
                x = [1 if mode == "amortized" else r["K"] for r in rs]
                m, lo, hi = ([r[f"{y}_{s}"] for r in rs] for s in ("mean", "lo", "hi"))
                if mode == "amortized":
                    if j:
                        ax.plot(x, m, marker="_", ms=10, color=c)
                    else:
                        ax.errorbar(x, m, yerr=[[a - b for a, b in zip(m, lo)], [b - a for a, b in zip(m, hi)]],
                                    fmt="o", color=c, capsize=3, label=label)
                    continue
                ls = "--" if j or mode == "oracle_bo" else "-"
                ax.plot(x, m, marker="o" if not j else None, color=c, ms=3.5, lw=1.5 if not j else 1.0, label=label, ls=ls)
                if not j:
                    ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(f"budget K ({kind})")
        _clean(ax)
    axes[0].set_ylabel(ylabel)
    handles = {}  # one entry per mode over both panels
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    if dashed_label and handles:
        handles[dashed_label] = plt.Line2D([], [], color="#555555", ls="--", lw=1.0)
    if handles:
        axes[1].legend(list(handles.values()), list(handles), frameon=False, fontsize=8)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_score_vs_budget(summary: pa.Table, out_png: Path) -> None:
    # normalized score vs test-time budget per inference mode, oracle_bo@k on the same axes (x = k)
    _curves(summary, out_png, ("Jhat",), ("amortized", "bon_pred", "refine", "bon_true", "oracle_bo"), r"normalized score $\hat{J}$",
            False)


def plot_critic_gap(summary: pa.Table, out_png: Path) -> None:
    # critic gap (predicted minus true, critic space) vs budget per inference mode: ensemble mean solid with CI,
    # the penalized score (mean - lam * std) the modes select on dashed
    _curves(summary, out_png, ("gap_ens", "gap"), ("amortized", "bon_pred", "refine", "bon_true"), "critic gap", True,
            r"mean $-\ \lambda$ std")


def plot_methods(summary: pa.Table, out_png: Path) -> None:
    # normalized score with CI per method at its largest budget, paraphrase and intent side by side
    rows = summary.to_pylist()
    present = {r["method"] for r in rows}
    methods = [m for m in METHOD_ORDER if m in present] + sorted(present - set(METHOD_ORDER))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * len(methods) + 1.5), 3.4))
    ax.axhline(0.0, color="#bbbbbb", lw=0.8)
    ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=":")
    for j, kind in enumerate(KINDS):
        xs, m, lo, hi = [], [], [], []
        for i, meth in enumerate(methods):
            rs = _rows(summary, kind, meth)
            if not rs:
                continue
            r = rs[-1]
            xs.append(i + (j - 0.5) * 0.3)
            m.append(r["Jhat_mean"])
            lo.append(r["Jhat_mean"] - r["Jhat_lo"])
            hi.append(r["Jhat_hi"] - r["Jhat_mean"])
        ax.errorbar(xs, m, yerr=[lo, hi], fmt="o", color=KIND_COLORS[kind], capsize=3, ms=4, label=kind)
    labels = [m if m not in ("bon_pred", "refine", "bon_true", "oracle_bo")
              else f"{m}@{max(r['K'] for r in rows if r['method'] == m)}" for m in methods]
    ax.set_xticks(range(len(methods)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"normalized score $\hat{J}$")
    ax.legend(frameon=False, fontsize=8)
    _clean(ax)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
