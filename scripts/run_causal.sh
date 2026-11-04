#!/usr/bin/env bash
# Causal tests (DESIGN 5.9): likelihood metrics under Base / TopK / layer-matched Random / Steer for the end checkpoint
# plus two early ones. Needs results/heads/{model}_{rev}_{value}_{mode}.json and results/directions/{model}_{rev}_{value}.npz
# (scripts/run_localize.sh or run_smoke.sh) under VH_ROOT.
#
#   HF_HOME=<cache> VH_ROOT=<run dir> bash scripts/run_causal.sh [model] [tier] [value] [revisions...]
#
# Tiers (sizes per DESIGN 6.2 / 6.3): smoke = CPU, < 10 min for one checkpoint (30 TQA questions, 60 HH pairs, 20 HelpSteer2
# pairs, 12 MC questions, 5k WikiText tokens, R = 3); t4 = reduced suite (full pair sets, 50k WikiText tokens, R = 20);
# a100 = full suite with R = 50. Every command is idempotent (stored conditions are reused).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-pythia-160m}
TIER=${2:-smoke}
VALUE=${3:-honesty}
shift 3 2>/dev/null || shift $#
REVS=("$@")
if [ ${#REVS[@]} -eq 0 ]; then
  if [ "$TIER" = smoke ]; then REVS=(step143000); else REVS=(step143000 step16000 step2000); fi
fi
export VH_ROOT=${VH_ROOT:-./vh_runs}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CLI="python3 -m valueheads_pythia.cli eval-likelihood --model $MODEL --value $VALUE --no_progress"
case "$TIER" in
  smoke) TQA="--max_items 30"; MC="--max_items 12"; HH="--max_items 60"; HS="--max_items 20"; WT="--max_tokens 5000"; R=3; RMC=2; ALPHA="--alpha 0.5";;
  t4)    TQA=""; MC=""; HH=""; HS=""; WT="--max_tokens 50000"; R=20; RMC=20; ALPHA="--alpha 0.3 0.5 1.0";;
  *)     TQA=""; MC=""; HH=""; HS=""; WT=""; R=50; RMC=50; ALPHA="--alpha 0.3 0.5 1.0";;
esac
# HelpSteer2 evaluation value: the head value itself when it is a HelpSteer2 value, else verbosity (cross-value control)
case "$VALUE" in helpfulness|complexity|verbosity|coherence|coherence_strict) HSV=$VALUE;; *) HSV=verbosity;; esac
case "$VALUE" in harmless|helpful) HHV=$VALUE;; *) HHV=harmless;; esac
T0=$(date +%s)
step() { echo; echo "### [$(( $(date +%s) - T0 ))s] $*"; }

for REV in "${REVS[@]}"; do
  step "$REV tqa_pairwise mask"
  $CLI --revision "$REV" --metric tqa_pairwise $TQA --intervention mask --select topk --random_sets $R
  step "$REV wikitext_ppl mask"
  $CLI --revision "$REV" --metric wikitext_ppl $WT --intervention mask --select topk --random_sets $R
  step "$REV hh_pairwise ($HHV) mask"
  $CLI --revision "$REV" --metric hh_pairwise --eval_value $HHV $HH --intervention mask --select topk --random_sets $R
  step "$REV helpsteer2_pairwise ($HSV) mask"
  $CLI --revision "$REV" --metric helpsteer2_pairwise --eval_value $HSV $HS --intervention mask --select topk --random_sets $R
  step "$REV tqa_mc1_mc2 mask"
  $CLI --revision "$REV" --metric tqa_mc1_mc2 $MC --intervention mask --select topk --random_sets $RMC
  step "$REV tqa_pairwise steer"
  $CLI --revision "$REV" --metric tqa_pairwise $TQA --intervention steer --select topk --random_sets $RMC $ALPHA
done
step "done; summary: $VH_ROOT/results/eval/summary.csv"
python3 - <<'PY'
import os, pandas as pd
df = pd.read_csv(os.path.join(os.environ["VH_ROOT"], "results", "eval", "summary.csv"))
cols = [c for c in ["model", "revision", "metric", "eval_value", "intervention", "alpha", "primary", "R", "random_mean", "random_std",
                    "p_emp_primary", "delta_mean", "delta_ci_lo", "delta_ci_hi", "p_emp_delta"] if c in df.columns]
print(df[cols].to_string(index=False))
PY
