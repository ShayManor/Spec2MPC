#!/usr/bin/env bash
# Gautschi slot payload: per seed W (S3), V (S2), policy full and zero (S4), and the APPLV BC baseline, all on the
# shard set frozen in $MODELS/manifest.txt (written by the first run: shards with meta.json and scores for every train
# and val card; the S0 jobs score their shards). A stage whose checkpoint exists is skipped, so a requeued job
# continues. Needs the train oracle output.
# Usage: bash train_payload.sh
# Env overrides (ideas8 rehearsal only): ROLL (rollout root), MODELS (output root), ORACLE (train oracle.parquet),
#   SEEDS, DEVICE, WM_EXTRA, CRITIC_EXTRA, POLICY_EXTRA, APPLV_EXTRA (extra Hydra overrides per stage)
# shellcheck source=stage_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/stage_common.sh"

ROLL="${ROLL:-/scratch/gautschi/manors/spec2mpc/rollouts/kart}"
MODELS="${MODELS:-/scratch/gautschi/manors/spec2mpc/models}"
ORACLE="${ORACLE:-/scratch/gautschi/manors/spec2mpc/oracle/train/oracle.parquet}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cuda}"
WM_EXTRA="${WM_EXTRA:-}" CRITIC_EXTRA="${CRITIC_EXTRA:-}" POLICY_EXTRA="${POLICY_EXTRA:-}" APPLV_EXTRA="${APPLV_EXTRA:-}"

MANIFEST="$MODELS/manifest.txt"
if [[ -f "$MANIFEST" ]]; then
  echo "manifest $MANIFEST exists ($(wc -l < "$MANIFEST") shards), reused"
else
  write='import sys; from spec2mpc.rubric.labels import write_manifest; print(len(write_manifest(*sys.argv[1:])), "shards")'
  "$PY" -c "$write" "$ROLL" data/cards "$MANIFEST"
fi
have() { [[ -f "$1" ]] && echo "skip $1 (exists)"; }
for seed in $SEEDS; do
  common=(seed="$seed" device="$DEVICE" "data.roots=[$MANIFEST]")
  # shellcheck disable=SC2086
  have "$MODELS/world_model/seed$seed/model.pt" || "$PY" scripts/train_world_model.py "${common[@]}" \
    out_dir="$MODELS/world_model" $WM_EXTRA
  # shellcheck disable=SC2086
  have "$MODELS/critic/seed$seed/model.pt" || "$PY" scripts/train_critic.py "${common[@]}" \
    out_dir="$MODELS/critic" $CRITIC_EXTRA
  for mode in full zero; do
    # shellcheck disable=SC2086
    have "$MODELS/policy_$mode/seed$seed/model.pt" || "$PY" scripts/train_policy.py "${common[@]}" text_mode="$mode" \
      out_dir="$MODELS/policy" oracle="$ORACLE" world_model="$MODELS/world_model/seed$seed/model.pt" \
      critic="$MODELS/critic/seed$seed/model.pt" $POLICY_EXTRA
  done
  # shellcheck disable=SC2086
  have "$MODELS/applv/seed$seed/bc.pt" || "$PY" scripts/train_applv.py stage=bc "${common[@]}" oracle="$ORACLE" \
    out_dir="$MODELS/applv" $APPLV_EXTRA
  echo "seed $seed done $(date -Is)"
done
echo "payload done $(date -Is)"
