#!/usr/bin/env bash
# Gautschi slot payload: applv_lang RL stage (REINFORCE on train-card rewards from true rollouts) for each seed,
# starting from the bc checkpoints of train_payload.sh. All seeds share one wall deadline; a timed-out job is
# continued by queueing the same payload again (RESUME=1, finished seeds return after loading).
# Usage: bash applv_payload.sh
# Env overrides (ideas8 rehearsal only): MODELS (output root), APPLV_OUT (bc checkpoints and rl outputs), ORACLE (train oracle.parquet), SEEDS, DEVICE,
#   APPLV_TIMEOUT_S (wall limit in seconds), APPLV_EXTRA (extra Hydra overrides)
# shellcheck source=stage_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/stage_common.sh"

MODELS="${MODELS:-/scratch/gautschi/manors/spec2mpc/models}"
APPLV_OUT="${APPLV_OUT:-$MODELS/applv}"  # where train_payload.sh's bc line writes the bc checkpoints
ORACLE="${ORACLE:-/scratch/gautschi/manors/spec2mpc/oracle/train/oracle.parquet}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cuda}"
APPLV_EXTRA="${APPLV_EXTRA:-}"

deadline=$(( $(date +%s) + ${APPLV_TIMEOUT_S:-41400} ))  # one 11h30m wall limit across all seeds
for seed in $SEEDS; do
  left=$(( deadline - $(date +%s) ))
  if (( left <= 0 )); then echo "deadline reached before seed $seed"; break; fi
  # shellcheck disable=SC2086
  RESUME=1 timeout "${left}s" "$PY" scripts/train_applv.py stage=rl seed="$seed" device="$DEVICE" oracle="$ORACLE" \
    out_dir="$APPLV_OUT" mpc.solver.compile=true $APPLV_EXTRA || echo "applv seed $seed exited $? at $(date -Is)"
  echo "seed $seed done $(date -Is)"
done
echo "payload done $(date -Is)"
