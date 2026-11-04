#!/usr/bin/env bash
# Localization (DESIGN steps 1-2 + statistics + directions) for one (model, revision, tier):
#   extract (train + val of every needed dataset; HelpSteer2 values share the extract-once set in mode pythia)
#   -> probe (head-only primary + [head ; layer mean] comparison) -> probe-stats (bootstrap CIs, permutation null,
#   significant set, confirmation split) -> directions (mass-mean + sigma). Every step is idempotent.
#
#   HF_HOME=<cache> VH_ROOT=<run dir> bash scripts/run_localize.sh [model] [revision] [tier] [values...]
#   bash scripts/run_localize.sh pythia-160m step143000 t4 honesty harmless helpful helpfulness complexity verbosity coherence
#
# Environment: VH_ROOT (default ./vh_runs), VH_FEATURES (feature dir, e.g. /content/features on Colab), VH_MODE
# (pythia | paper, default pythia), VH_DEVICE (auto | cpu | cuda), VH_BOOTSTRAP / VH_PERMS (default 1000 / 100;
# tier smoke: 200 / 20), VH_STATS_DEVICE (cpu | cuda, default = cuda when VH_DEVICE is cuda), VH_MAX_SAMPLES_TRAIN /
# VH_MAX_SAMPLES_VAL (override the tier table, 0 = all).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-pythia-160m}
REV=${2:-step143000}
TIER=${3:-smoke}
shift $(( $# > 3 ? 3 : $# )) || true
VALUES=${*:-honesty}
export VH_ROOT=${VH_ROOT:-./vh_runs}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
MODE=${VH_MODE:-pythia}
DEVICE=${VH_DEVICE:-auto}
if [ "$TIER" = smoke ]; then B=${VH_BOOTSTRAP:-200}; P=${VH_PERMS:-20}; else B=${VH_BOOTSTRAP:-1000}; P=${VH_PERMS:-100}; fi
STATS_DEVICE=${VH_STATS_DEVICE:-$([ "$DEVICE" = cuda ] && echo cuda || echo cpu)}
CLI="python3 -m valueheads_pythia.cli"
COMMON=(--model "$MODEL" --revision "$REV" --mode "$MODE")
T0=$(date +%s)
step() { echo; echo "### [$(( $(date +%s) - T0 ))s] $*"; }

# datasets to extract: a value's own set, or the shared helpsteer2 set for HelpSteer2 values in mode pythia
DATASETS=$(python3 - "$MODE" $VALUES <<'PY'
import sys
from valueheads_pythia import config
mode, values = sys.argv[1], sys.argv[2:]
seen = []
for v in values:
    d = config.probing_dataset_of(v, mode)
    if d not in seen:
        seen.append(d)
print(" ".join(seen))
PY
)
step "localize $MODEL @ $REV: tier $TIER, mode $MODE, values [$VALUES], datasets [$DATASETS], B=$B P=$P, root $VH_ROOT"

for DS in $DATASETS; do
  for SPLIT in train val; do
    ARGS=(extract "${COMMON[@]}" --dataset "$DS" --split "$SPLIT" --tier "$TIER" --device "$DEVICE" --no_progress)
    if [ "$SPLIT" = train ] && [ -n "${VH_MAX_SAMPLES_TRAIN:-}" ]; then ARGS+=(--max_samples "$VH_MAX_SAMPLES_TRAIN"); fi
    if [ "$SPLIT" = val ] && [ -n "${VH_MAX_SAMPLES_VAL:-}" ]; then ARGS+=(--max_samples "$VH_MAX_SAMPLES_VAL"); fi
    step "extract $DS/$SPLIT"
    $CLI "${ARGS[@]}"
  done
done

for VALUE in $VALUES; do
  step "probe $VALUE (head-only + head+mean)"
  $CLI probe "${COMMON[@]}" --value "$VALUE" --features both --quiet
  step "probe-stats $VALUE (B=$B, P=$P on $STATS_DEVICE)"
  $CLI probe-stats "${COMMON[@]}" --value "$VALUE" --features both --bootstrap "$B" --perms "$P" --device "$STATS_DEVICE" --quiet
  step "directions $VALUE"
  $CLI directions "${COMMON[@]}" --value "$VALUE"
done

# Extra TruthfulQA question splits for honesty (ranking stability, DESIGN 5.6) are produced by the sweep command:
#   VH_SWEEP_ARGS="--split_seeds 1 2 3 4" bash scripts/run_sweep.sh <model> <tier> honesty

step "done; head tables:"
ls -1 "$VH_ROOT/results/heads/" | grep "^${MODEL}_${REV}_" || true
