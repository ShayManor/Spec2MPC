#!/usr/bin/env bash
# Gautschi slot payload: S1 oracle for one split. A timed-out job is continued by queueing the same payload again (RESUME).
# Usage: bash oracle_payload.sh SPLIT   (train | val)
# Env overrides (ideas8 rehearsal only): OUT (output root), ORACLE_TIMEOUT (wall limit), ORACLE_EXTRA (extra Hydra overrides)
SPLIT="${1:?usage: oracle_payload.sh SPLIT}"
# shellcheck source=stage_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/stage_common.sh"

OUT="${OUT:-/scratch/gautschi/manors/spec2mpc/oracle}"
ORACLE_EXTRA="${ORACLE_EXTRA:-}"
# shellcheck disable=SC2086
RESUME=1 timeout "${ORACLE_TIMEOUT:-690m}" "$PY" scripts/run_oracle.py split="$SPLIT" device=cuda mpc.solver.compile=true \
  out_dir="$OUT" $ORACLE_EXTRA || echo "oracle exited $? at $(date -Is)"
echo "payload done $(date -Is)"
