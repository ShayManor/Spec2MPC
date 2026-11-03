#!/usr/bin/env bash
# One-time setup of the HumanoidBench env (idempotent).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
commit=cb1189039151c8aadaaa987b442da54383c87fab
if [ ! -d "$here/humanoid-bench/.git" ]; then
  git clone https://github.com/carlosferrazza/humanoid-bench.git "$here/humanoid-bench"
fi
git -C "$here/humanoid-bench" checkout -q "$commit"
"${UV:-$HOME/.local/bin/uv}" sync --project "$here"
