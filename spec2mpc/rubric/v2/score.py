"""Score one episode trace against a hidden TaskTest (trace format: predicates.py docstring)."""
from __future__ import annotations

import numpy as np

from spec2mpc.core import TaskTest
from spec2mpc.rubric.v2.graded import GRADED, PENALTIES
from spec2mpc.rubric.v2.predicates import evaluate, normalize


def score(test: TaskTest, trace: dict[str, np.ndarray]) -> dict:
    """success: every predicate true at episode end. score: mean of (fraction of predicates true, each graded
    term in [0, 1]), so 0 <= score <= 1 before penalties; then sum(weight * magnitude) over penalties is subtracted.
    """
    truth = evaluate(test.predicates, trace)
    per_pred = {f"{i}:{normalize(d)[0]}": v for i, (d, v) in enumerate(zip(test.predicates, truth))}
    graded, pens = {}, {}
    for i, d in enumerate(test.graded):
        name, args, params = normalize(d)
        if name in PENALTIES:
            pens[f"{i}:{name}"] = float(params.get("weight", 1.0)) * PENALTIES[name](trace, args, params,
                                                                                    test.predicates)
        elif name in GRADED:
            graded[f"{i}:{name}"] = GRADED[name](trace, args, params, test.predicates)
        else:
            raise ValueError(f"unknown graded term {name!r}")
    parts = ([float(np.mean(truth))] if truth else []) + list(graded.values())
    s = float(np.mean(parts)) if parts else 0.0
    return {"success": all(truth), "score": s - sum(pens.values()), "per_predicate": per_pred,
            "graded": graded, "penalties": pens}
