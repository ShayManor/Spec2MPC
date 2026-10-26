#!/usr/bin/env bash
# Gautschi slot payload: evaluation (true rollouts, true rubric) on the val oracle problems, in two groups of ~7 h.
#   a: default, oracle, oracle_bo, mean_theta, nn_retrieval, and slot_only, applv_lang, amortized, bon_pred, refine
#      for seeds 0 and 1
#   b: default, oracle, then slot_only, applv_lang, amortized, bon_pred, refine for seed 2, and bon_true for seeds 0 to 2
#   s: single-seed run in one job: group a's methods for seed 0, then bon_true for seed 0
# Each group rolls out its own default and oracle anchors in the same job (files tagged _a / _b), and Jhat pairs every
# row with the anchors of its own group only, so the groups may run on different GPUs. Finished settings are skipped,
# so a timed-out job is continued by queueing the same group again. The summary and figures are rebuilt at the end of
# either group once that group's anchors exist.
# Usage: bash eval_payload.sh GROUP   (a | b | s)
# Env overrides (ideas8 rehearsal only): ORACLE_ROOT (dir with train/ and val/), MODELS (train_payload.sh output root),
#   APPLV (applv_payload.sh output dir), OUT (results dir), DEVICE, EVAL_TIMEOUT_S (wall limit in seconds), EVAL_EXTRA (Hydra overrides)
GROUP="${1:?usage: eval_payload.sh GROUP (a | b | s)}"
# shellcheck source=stage_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/stage_common.sh"

ORACLE_ROOT="${ORACLE_ROOT:-/scratch/gautschi/manors/spec2mpc/oracle}"
MODELS="${MODELS:-/scratch/gautschi/manors/spec2mpc/models}"
APPLV="${APPLV:-$MODELS/applv}"  # where train_payload.sh (bc) and applv_payload.sh (rl) write
OUT="${OUT:-/scratch/gautschi/manors/spec2mpc/eval}"
DEVICE="${DEVICE:-cuda}"
EVAL_EXTRA="${EVAL_EXTRA:-}"

# quoted values: Hydra parses a bare {seed} as a dict
common=(group="$GROUP" device="$DEVICE" oracle_root="$ORACLE_ROOT" out_dir="$OUT" plot_dir="$OUT/plots"
  "models.world_model='$MODELS/world_model/seed{seed}/model.pt'" "models.critic='$MODELS/critic/seed{seed}/model.pt'"
  "models.policy_full='$MODELS/policy_full/seed{seed}/model.pt'" "models.policy_zero='$MODELS/policy_zero/seed{seed}/model.pt'"
  "models.applv='$APPLV/seed{seed}/model.pt'")
learned="[slot_only,applv_lang,amortized,bon_pred,refine]"
deadline=$(( $(date +%s) + ${EVAL_TIMEOUT_S:-41400} ))  # one 11h30m wall limit across the runs of a group

run() {
  local left=$(( deadline - $(date +%s) ))
  if (( left <= 0 )); then echo "deadline reached before $*"; return 0; fi
  # shellcheck disable=SC2086
  timeout "${left}s" "$PY" scripts/run_eval.py "${common[@]}" "$@" $EVAL_EXTRA || echo "run_eval $* exited $? at $(date -Is)"
}

case "$GROUP" in
  a) run "methods=[default,oracle,oracle_bo,mean_theta,nn_retrieval]"
     run "methods=$learned" "seeds=[0,1]" ;;
  b) run "methods=[default,oracle]"
     run "methods=$learned" "seeds=[2]"
     run "methods=[bon_true]" "seeds=[0,1,2]" ;;
  s) run "methods=[default,oracle,oracle_bo,mean_theta,nn_retrieval]"
     run "methods=$learned" "seeds=[0]"
     run "methods=[bon_true]" "seeds=[0]" ;;
  *) echo "unknown GROUP $GROUP (a | b | s)"; exit 2 ;;
esac
echo "payload done $(date -Is)"
