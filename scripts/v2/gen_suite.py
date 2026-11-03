"""Generate the template task suite: goals x robots x scene seeds x styles -> data/v2/cards/<split>/<task_id>.json.

  uv run python scripts/v2/gen_suite.py --seeds 2 --draws 1                 # every family
  uv run python scripts/v2/gen_suite.py --families arm kart --paraphrases 2  # + gpt-6.1-sol paraphrases (needs key)
  uv run python scripts/v2/gen_suite.py --sources barn tracks               # only these sources (bare --sources: none)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from spec2mpc.robots.v2 import FAMILIES, list_robots, load_robot
from spec2mpc.tasks import (CARD_ROOT, SOURCE_MODULES, assign_splits, draft_commands, goals_for, make_card, save_cards,
                            source_cards, with_command)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--seeds", type=int, default=2, help="scene seeds per (goal, robot draw)")
    ap.add_argument("--draws", type=int, default=1, help="randomized parameter draws per robot member")
    ap.add_argument("--styles", nargs="*", default=None, help="default: every style of each goal")
    ap.add_argument("--paraphrases", type=int, default=0, help="LLM paraphrase variants per card (live calls)")
    ap.add_argument("--sources", nargs="*", default=None, choices=list(SOURCE_MODULES),
                    help="training-source adapters to add (default: every available one)")
    ap.add_argument("--heldout-type", default=None)
    ap.add_argument("--out", type=Path, default=CARD_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cards = []
    for fam in a.families:
        for name in list_robots(fam):
            for d in range(a.draws):
                robot = load_robot(fam, name, d)
                for goal in goals_for(fam):
                    for i in range(a.seeds):
                        for style in a.styles or goal.styles:
                            if style in goal.styles:
                                cards.append(make_card(goal, robot, d * 1000 + i, style, "train"))
    if a.paraphrases:
        extra = []
        for c in cards:
            extra += [with_command(c, cmd, k + 1) for k, cmd in enumerate(draft_commands(c, a.paraphrases))]
        cards += extra
    # source cards join after paraphrasing (draft_commands rebuilds generator scenes only)
    cards += [c for c in source_cards(a.sources) if c.public.family in a.families]
    cards = assign_splits(cards, a.heldout_type)
    counts = Counter(c.split for c in cards)
    srcs = Counter(c.source for c in cards)
    print(json.dumps({"n": len(cards), "splits": dict(counts),
                      "families": dict(Counter(c.public.family for c in cards)), "sources": dict(srcs)}, indent=1))
    if not a.dry_run:
        save_cards(cards, a.out)
        (a.out / "manifest.json").write_text(json.dumps(
            {"n": len(cards), "splits": dict(counts), "sources": dict(srcs),
             "args": {k: str(v) for k, v in vars(a).items()}}, indent=1))


if __name__ == "__main__":
    main()
