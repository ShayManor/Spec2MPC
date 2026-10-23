#!/bin/bash
#SBATCH -p smallgpu
#SBATCH -A lilly-rob1
#SBATCH --gres=gpu:1
#SBATCH -c 64
#SBATCH -t 12:00:00
#SBATCH -J next
#SBATCH -o %x-%j.out
# Usage: sbatch deferred.sbatch <slot>
# Runs $HOME/next<slot>.sh when the job starts; exits at once if that slot is empty or a template.
set -uo pipefail

SLOT="${1:?usage: sbatch deferred.sbatch <slot>}"
export NEXT_FILE="$HOME/next${SLOT}.sh"
TEMPLATE="$HOME/next_template.sh"

if [[ ! -s "$NEXT_FILE" ]] || grep -q '^# SPEC2MPC_TEMPLATE' "$NEXT_FILE"; then
  echo "next${SLOT}.sh is empty or a template; exiting."
  exit 0
fi

# Snapshot the payload so edits during the run cannot change what executes.
RUN_DIR="$HOME/next_runs/${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"
cp "$NEXT_FILE" "$RUN_DIR/next${SLOT}.sh"
sha256sum "$RUN_DIR/next${SLOT}.sh" > "$RUN_DIR/sha256.txt"
echo "slot=$SLOT job=${SLURM_JOB_ID:-local} host=$(hostname) start=$(date -Is)" | tee "$RUN_DIR/info.txt"

# If the payload dies before its reset step, reset the slot anyway so it never runs twice.
trap 'grep -q "^# SPEC2MPC_TEMPLATE" "$NEXT_FILE" || cp "$TEMPLATE" "$NEXT_FILE"' EXIT

bash "$RUN_DIR/next${SLOT}.sh"
