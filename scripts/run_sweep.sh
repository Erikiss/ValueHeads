#!/usr/bin/env bash
# Checkpoint sweep (DESIGN 3.4 / 7): extract -> probe -> probe-stats -> directions per revision, skipping finished
# artifacts, then sweep curves + manifest; finally the analysis (heatmaps, curves, correlations, angles, report).
#
#   HF_HOME=<cache> VH_ROOT=<run dir> bash scripts/run_sweep.sh [model] [tier] [values...]
#   VH_SMOKE=1 bash scripts/run_sweep.sh                      # step0 / step2000 / step143000, tier smoke, B=200, P=20
#   VH_REVISIONS="step0 step2000 step143000" ...              # explicit revisions (default: config.CHECKPOINTS_DEFAULT)
#   VH_CLEANUP_CACHE=1 ...                                    # delete each HF revision after its probing (keeps main/step143000)
#   VH_EVAL_REVISIONS="step2000 step16000 step143000" ...     # also run eval-likelihood on these revisions (if installed)
#   VH_EVAL_ARGS="--metric tqa_pairwise --intervention mask --select topk --random_sets 20" ...   # eval-likelihood arguments
#   VH_SWEEP_ARGS="--perms 100 --bootstrap 1000 --split_seeds 1 2" ...   # anything else for the sweep command (shell-quoted
#                                                             #   values are honoured, e.g. VH_SWEEP_ARGS='--eval_args "--metric wikitext_ppl --intervention mask"')
#   VH_CLEANUP_FEATURES=1 ...                                 # delete non-cacheable train features after each revision
#   VH_DRY_RUN=1 ...                                          # print the plan only
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-pythia-160m}
TIER=${2:-smoke}
shift $(( $# > 2 ? 2 : $# )) || true
VALUES=${*:-honesty harmless helpful helpfulness complexity verbosity coherence}
export VH_ROOT=${VH_ROOT:-./vh_runs}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CLI="python3 -m valueheads_pythia.cli"
ARGS=(--model "$MODEL" --tier "$TIER" --values $VALUES --no_progress)
[ "${VH_SMOKE:-0}" = "1" ] && ARGS+=(--smoke)
[ -n "${VH_REVISIONS:-}" ] && ARGS+=(--revisions $VH_REVISIONS)
[ "${VH_CLEANUP_CACHE:-0}" = "1" ] && ARGS+=(--cleanup_cache)
[ -n "${VH_EVAL_REVISIONS:-}" ] && ARGS+=(--eval_revisions $VH_EVAL_REVISIONS)
[ -n "${VH_EVAL_ARGS:-}" ] && ARGS+=(--eval_args "$VH_EVAL_ARGS")
[ "${VH_CLEANUP_FEATURES:-0}" = "1" ] && ARGS+=(--cleanup_features)
[ "${VH_DRY_RUN:-0}" = "1" ] && ARGS+=(--dry_run)
# VH_SWEEP_ARGS is parsed like a shell command line (quotes kept), so --eval_args "..." survives (review finding F7)
EXTRA=()
if [ -n "${VH_SWEEP_ARGS:-}" ]; then eval "EXTRA=(${VH_SWEEP_ARGS})"; fi
T0=$(date +%s)
echo "### sweep: $CLI sweep ${ARGS[*]} ${EXTRA[*]:-}"
$CLI sweep "${ARGS[@]}" "${EXTRA[@]}"
if [ "${VH_DRY_RUN:-0}" != "1" ]; then
  echo "### [$(( $(date +%s) - T0 ))s] analyze"
  $CLI analyze --model "$MODEL"
fi
echo "### done in $(( $(date +%s) - T0 ))s; results under $VH_ROOT/results, figures under $VH_ROOT/figures"
