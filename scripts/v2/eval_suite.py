"""Evaluate one executor on one split of the generated suite, then redraw the round's figures.

Usage: python scripts/v2/eval_suite.py --split val --executor spec2mpc --url http://host:8000
       --env-factory spec2mpc.some.module:make_env [--n-seeds 1 --workers 4 --round r0 --compute 16]
Resumable: rows already in data/v2/eval/<round>/<executor>_<split>.partial.jsonl are skipped.
"""
import argparse
import importlib

from spec2mpc.eval.v2 import figure_round, run_suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--executor", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--env-factory", required=True, help="module:function, TaskPublic -> SimEnv")
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--round", default="r0")
    ap.add_argument("--compute", type=float, default=None, help="test-time budget recorded with each row")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    mod, fn = a.env_factory.split(":")
    make_env = getattr(importlib.import_module(mod), fn)
    out = run_suite(a.split, a.executor, a.url, n_seeds=a.n_seeds, workers=a.workers, make_env=make_env,
                    round_name=a.round, compute=a.compute)
    print(out)
    if not a.no_figures:
        for p in figure_round(out.parent):
            print(p)


if __name__ == "__main__":
    main()
