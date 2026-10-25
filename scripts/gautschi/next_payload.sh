#!/usr/bin/env bash
# Gautschi slot payload: bench at B=ENVS (log only), then shards [FIRST_BLOCK, FIRST_BLOCK + N_BLOCKS) of S0, then scoring.
# Usage: bash next_payload.sh FIRST_BLOCK N_BLOCKS
# Uses the prebuilt venv directly (torch pinned to cu126); never uv sync / uv run, which would reinstall torch.
# Env overrides (ideas8 rehearsal only; Gautschi uses the defaults):
#   REPO (repo root), PY (venv python), ENVS (envs per batch), OUT (shard dir), GEN_EXTRA (extra Hydra overrides for gen),
#   GEN_TIMEOUT (gen wall limit, leaves time to score before the 12 h job limit)
set -euo pipefail

FIRST_BLOCK="${1:?usage: next_payload.sh FIRST_BLOCK N_BLOCKS}"
N_BLOCKS="${2:?usage: next_payload.sh FIRST_BLOCK N_BLOCKS}"
REPO="${REPO:-$HOME/spec2mpc}"
PY="${PY:-$HOME/spec2mpc/.venv/bin/python}"
ENVS="${ENVS:-1024}"
OUT="${OUT:-/scratch/gautschi/manors/spec2mpc/rollouts/kart}"
GEN_EXTRA="${GEN_EXTRA:-}"
GEN_TIMEOUT="${GEN_TIMEOUT:-11h}"

cd "$REPO"
# fail in seconds on a driver / wheel mismatch
"$PY" -c "import torch; assert torch.cuda.is_available(), torch.version.cuda; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
echo "payload first_block=$FIRST_BLOCK n_blocks=$N_BLOCKS envs=$ENVS out=$OUT host=$(hostname) start=$(date -Is)"

# the Triton perception fast path must match the reference copies bit for bit on this GPU; otherwise use the pre-speedup tree
REF_REPO="${REF_REPO:-/scratch/gautschi/manors/spec2mpc/code_ref}"
if ! "$PY" -m pytest -q -x -p no:cacheprovider tests/test_occupancy.py -m slow -k equivalence_to_reference_gpu; then
  echo "perception fast path not bit-identical on this GPU; generating with $REF_REPO"
  cd "$REF_REPO"
  export PYTHONPATH="$REF_REPO"
fi

# bench is informational; a bench failure must never cost the shard work
"$PY" scripts/bench_mpcc.py bench.B="$ENVS" bench.steps=50 device=cuda mpc.solver.compile=true || true

# unfinished shards are picked up by a later RESUME run; a timeout still scores the finished ones
# shellcheck disable=SC2086
RESUME=1 timeout "$GEN_TIMEOUT" "$PY" scripts/gen_rollouts.py n_blocks="$N_BLOCKS" first_block="$FIRST_BLOCK" envs_per_batch="$ENVS" \
  device=cuda mpc.solver.compile=true seed=0 out_dir="$OUT" $GEN_EXTRA || echo "gen exited with status $? at $(date -Is)"
# score only this job's shards: concurrent jobs share OUT, and scoring the same shard twice races on its temp file
for ((b = FIRST_BLOCK; b < FIRST_BLOCK + N_BLOCKS; b++)); do
  shard="$OUT/$(printf 'shard_%06d' "$b")"
  if [[ -d "$shard" ]]; then "$PY" scripts/score_rollouts.py "$shard"; fi
done
echo "payload done $(date -Is)"
