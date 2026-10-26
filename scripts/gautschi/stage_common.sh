#!/usr/bin/env bash
# Sourced by every E2E stage payload: strict mode, repo and venv paths, CUDA check, offline Hugging Face cache.
# Uses the prebuilt venv directly (torch pinned to cu126); never uv sync / uv run, which would reinstall torch.
# Env overrides (ideas8 rehearsal only; Gautschi uses the defaults): REPO (repo root), PY (venv python), HF_HOME,
#   REF_REPO (pre-speedup tree used when the perception fast path is not bit-identical on this GPU)
set -euo pipefail

REPO="${REPO:-/scratch/gautschi/manors/spec2mpc/code_e2e}"
PY="${PY:-$HOME/spec2mpc/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/scratch/gautschi/manors/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# the venv's editable install points at the S0 tree; import this tree instead
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
# fail in seconds on a driver / wheel mismatch
"$PY" -c "import torch; assert torch.cuda.is_available(), torch.version.cuda; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
# the Triton perception fast path must match the reference copies bit for bit on this GPU; otherwise use the pre-speedup tree
REF_REPO="${REF_REPO:-/scratch/gautschi/manors/spec2mpc/code_e2e_ref}"
if ! "$PY" -m pytest -q -x -p no:cacheprovider tests/test_occupancy.py -m slow -k equivalence_to_reference_gpu; then
  if [[ -d "$REF_REPO" ]]; then
    echo "perception fast path not bit-identical on this GPU; switching REPO $REPO -> $REF_REPO"
    export PYTHONPATH="$REF_REPO${PYTHONPATH#"$REPO"}"
    REPO="$REF_REPO"
    cd "$REPO"
  else
    echo "WARNING WARNING: perception fast path not bit-identical on this GPU and $REF_REPO is missing;"
    echo "WARNING WARNING: continuing on the fast path in $REPO (results may differ from the reference perception)"
  fi
fi
# run from a per-job copy, so re-syncing the shared tree while this job runs cannot mix code versions
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SNAP="/scratch/gautschi/manors/spec2mpc/snapshots/$SLURM_JOB_ID"
  mkdir -p "$SNAP"
  rsync -a "$REPO/" "$SNAP/"
  export PYTHONPATH="$SNAP${PYTHONPATH#"$REPO"}"
  REPO="$SNAP"
  cd "$REPO"
fi
"$PY" -c "import os, spec2mpc; p = os.path.dirname(spec2mpc.__file__); assert p.startswith('$REPO/'), p; print('spec2mpc from', p)"
echo "stage $(basename "$0") args=$* repo=$REPO host=$(hostname) start=$(date -Is)"
