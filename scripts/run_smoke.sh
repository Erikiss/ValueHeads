#!/usr/bin/env bash
# Smoke tier end-to-end on pythia-160m (CPU: float32 / eager; T4: same numerics policy), DESIGN 6.3 "Smoke-Ziel".
# Runs the complete chain exactly as a Colab user would through the CLI, with tiny subsets; every single command
# stays well below 10 minutes on 4 CPU cores (the whole chain takes ~25-35 min on a shared CPU, ~5 min on a T4):
#
#   tests      pytest (hook layout, mask positions, data counts, probing vs. reference repo, stats, eval, judge, ...)
#   selftest   hook layout + mask positions + quick data counts
#   extract    honesty train/val (smoke tier 1000/300), HelpSteer2 extract-once (160/80 rows, labels per value),
#              harmless-base (300/100 texts)
#   localize   probe (head-only primary + [head ; layer mean]) -> probe-stats (B=200 cluster bootstrap, P=20 permutations,
#              significant set, confirmation split) -> directions (+ sigma) for honesty, one HelpSteer2 value, harmless;
#              plus the grouped 5-fold CV table (probe --cv 5) of the HelpSteer2 value (DESIGN 5.3)
#   eval       eval-likelihood: TruthfulQA pairwise (30 questions) and WikiText-2 PPL (5k tokens, min context 512) under base /
#              TopK mask / 3 layer-matched random sets, steering (alpha 0.5) on TruthfulQA pairwise, and the lm-eval
#              6-shot Q:/A: context column of TruthfulQA pairwise (base only); then the tiny CLI coverage runs (R = 1):
#              hh_pairwise (harmless, 6 pairs), helpsteer2_pairwise (verbosity, 6 pairs), tqa_mc1_mc2 (4 questions),
#              multi-value steering honesty + harmless (union mask / composite direction), --select signif, and the
#              disjoint protocol --disjoint (heads ranked on half A of the confirmation split, eval on half B)
#   generate   eval-generate honesty / fewshot_qa (8 questions): base vs. TopK mask (paper hook semantics)
#   judge      refusal heuristic on the generations (the CPU tier has no judge model; `--judge auto --tier cpu` falls back)
#   analyze    heatmaps, layer profiles, summary, report (results/report_{model}.md)
#   sweep      sweep over two checkpoints (step0 + step143000) for honesty with the same tiny sizes plus one extra
#              TruthfulQA question split (--split_seeds 1: stability CSV + mean-over-splits ranking), then analyze again
#              (sweep curve, Spearman / Jaccard between the checkpoints, split stability)
#
#   HF_HOME=<cache> VH_ROOT=<run dir> bash scripts/run_smoke.sh [model] [revision] [early revision]
#
# Environment: HF_HOME (model / dataset cache), VH_ROOT (default ./vh_runs), VH_FEATURES (feature dir), VH_DEVICE
# (auto | cpu | cuda), VH_SKIP_TESTS=1 (skip the pytest stage), VH_SMOKE_STAGES="selftest extract ..." (subset of the
# stages above, default all), VH_SMOKE_HS_VALUE (HelpSteer2 value, default helpfulness), VH_BOOTSTRAP / VH_PERMS /
# VH_RANDOM_SETS (200 / 20 / 3), VH_TQA_ITEMS / VH_WIKITEXT_TOKENS / VH_GEN_ITEMS (30 / 5000 / 8), VH_SPLIT_SEEDS (extra
# question-split seeds of the sweep stage, default 1), VH_CV (folds of the HelpSteer2 CV table, default 5), VH_TEST_DATA=full
# (extend the data-count test), VH_REPO (reference repo for the paper-mode comparison test; skipped if absent).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-pythia-160m}
REV=${2:-step143000}
EARLY_REV=${3:-step0}
export VH_ROOT=${VH_ROOT:-./vh_runs}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
DEVICE=${VH_DEVICE:-auto}
HS_VALUE=${VH_SMOKE_HS_VALUE:-helpfulness}
B=${VH_BOOTSTRAP:-200}; P=${VH_PERMS:-20}; R=${VH_RANDOM_SETS:-3}      # P = 20 = DESIGN 5.4 smoke value
SPLIT_SEEDS=${VH_SPLIT_SEEDS:-1}; CV=${VH_CV:-5}
TQA_N=${VH_TQA_ITEMS:-30}; WT_TOKENS=${VH_WIKITEXT_TOKENS:-5000}; GEN_N=${VH_GEN_ITEMS:-8}
STAGES=${VH_SMOKE_STAGES:-"selftest extract localize eval generate judge analyze sweep"}
CLI="python3 -m valueheads_pythia.cli"
M=(--model "$MODEL" --revision "$REV" --device "$DEVICE")
T0=$(date +%s)
TS=$T0
step() { local now; now=$(date +%s); echo; echo "### [$((now - T0))s total, +$((now - TS))s] $*"; TS=$now; }
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
EVAL_DIR="$VH_ROOT/results/eval/${MODEL}_${REV}"

if [ "${VH_SKIP_TESTS:-0}" != "1" ]; then
  step "tests"
  VH_TEST_MODEL=$MODEL VH_TEST_REVISION=$REV python3 -m pytest tests -q -p no:cacheprovider
fi

if want selftest; then
  step "selftest (hook layout, mask positions, quick data counts)"
  $CLI selftest --model "$MODEL" --revision "$REV" --device "$DEVICE" --data quick
fi

if want extract; then
  for SPLIT in train val; do
    step "extract honesty/$SPLIT (tier smoke: 1000 / 300 texts)"
    $CLI extract "${M[@]}" --dataset honesty --split $SPLIT --tier smoke --no_progress
  done
  for SPLIT in train val; do
    N=$([ "$SPLIT" = train ] && echo 160 || echo 80)
    step "extract helpsteer2/$SPLIT once ($N rows, labels per value)"
    $CLI extract "${M[@]}" --dataset helpsteer2 --split $SPLIT --tier smoke --max_samples "$N" --no_progress
  done
  for SPLIT in train val; do
    N=$([ "$SPLIT" = train ] && echo 300 || echo 100)
    step "extract harmless/$SPLIT (harmless-base, $N texts)"
    $CLI extract "${M[@]}" --dataset harmless --split $SPLIT --tier smoke --max_samples "$N" --no_progress
  done
fi

if want localize; then
  for V in honesty "$HS_VALUE" harmless; do
    step "probe $V (head-only + head+mean, AUC, class_weight balanced)"
    $CLI probe --model "$MODEL" --revision "$REV" --value "$V" --features both --quiet
    step "probe-stats $V (B=$B, P=$P; permutation scheme per DESIGN 5.4)"
    $CLI probe-stats --model "$MODEL" --revision "$REV" --value "$V" --features both --bootstrap "$B" --perms "$P" --device cpu
    step "directions $V (mass-mean + sigma)"
    $CLI directions --model "$MODEL" --revision "$REV" --value "$V"
  done
  step "probe $HS_VALUE --cv $CV (grouped stratified $CV-fold over train + val, DESIGN 5.3)"
  $CLI probe --model "$MODEL" --revision "$REV" --value "$HS_VALUE" --cv "$CV" --quiet
  step "probe honesty --pca 16 (capacity equalization, DESIGN 3.5 -> *_pca16.json)"
  $CLI probe --model "$MODEL" --revision "$REV" --value honesty --pca 16 --quiet
fi

if want eval; then
  E=("${M[@]}" --value honesty --positions all --no_progress)
  step "eval-likelihood tqa_pairwise ($TQA_N questions): base / TopK mask / $R layer-matched random sets"
  $CLI eval-likelihood "${E[@]}" --metric tqa_pairwise --max_items "$TQA_N" --intervention mask --select topk --random_sets "$R"
  step "eval-likelihood wikitext_ppl ($WT_TOKENS tokens): base / TopK mask / $R random sets"
  $CLI eval-likelihood "${E[@]}" --metric wikitext_ppl --max_tokens "$WT_TOKENS" --intervention mask --select topk --random_sets "$R"
  step "eval-likelihood tqa_pairwise steer alpha 0.5: TopK / $R random sets"
  $CLI eval-likelihood "${E[@]}" --metric tqa_pairwise --max_items "$TQA_N" --intervention steer --select topk --random_sets "$R" --alpha 0.5
  step "eval-likelihood tqa_pairwise --context lmeval ($TQA_N questions, 6-shot Q:/A: primer): base only (second column, DESIGN 3.2)"
  $CLI eval-likelihood "${M[@]}" --metric tqa_pairwise --context lmeval --max_items "$TQA_N" --intervention none --no_progress
  step "eval-likelihood hh_pairwise harmless (6 pairs, R = 1), helpsteer2_pairwise verbosity (6 pairs), tqa_mc1_mc2 (4 questions): CLI coverage"
  $CLI eval-likelihood "${E[@]}" --metric hh_pairwise --eval_value harmless --max_items 6 --intervention mask --select topk --random_sets 1
  $CLI eval-likelihood "${E[@]}" --metric helpsteer2_pairwise --eval_value verbosity --max_items 6 --intervention mask --select topk --random_sets 1
  $CLI eval-likelihood "${E[@]}" --metric tqa_mc1_mc2 --max_items 4 --intervention mask --select topk --random_sets 1
  step "eval-likelihood tqa_pairwise: multi-value steering honesty + harmless (6 questions, alpha 0.5, R = 1) and --select signif"
  $CLI eval-likelihood "${M[@]}" --values honesty harmless --weights 0.5 0.5 --positions all --no_progress --metric tqa_pairwise --max_items 6 \
      --intervention steer --select topk --random_sets 1 --alpha 0.5
  $CLI eval-likelihood "${E[@]}" --metric tqa_pairwise --max_items 6 --intervention mask --select signif --random_sets 1
  step "eval-likelihood tqa_pairwise --disjoint (heads ranked on half A, questions of half B only; DESIGN 4.1)"
  $CLI eval-likelihood "${E[@]}" --metric tqa_pairwise --disjoint --max_items "$TQA_N" --intervention mask --select topk --random_sets 1
fi

if want generate; then
  step "eval-generate honesty / fewshot_qa ($GEN_N questions): base vs. TopK mask"
  $CLI eval-generate "${M[@]}" --eval_set honesty --prompt_style fewshot_qa --num_eval "$GEN_N" --value honesty \
      --intervention mask --select topk --examples 2 --no_progress
fi

if want judge; then
  step "judge heuristic (refusal / no-comment / degeneration flags; base rate + paired delta)"
  $CLI judge --judge heuristic --tier cpu --inputs "$EVAL_DIR"/gen-honesty-fewshot_qa_*.jsonl --value harmless --no_progress
  step "judge auto on tier cpu (falls back to the heuristic)"
  $CLI judge --judge auto --tier cpu --inputs "$EVAL_DIR"/gen-honesty-fewshot_qa_none.jsonl --no_progress
fi

if want analyze; then
  step "analyze (heatmaps, layer profiles, summary, report)"
  $CLI analyze --model "$MODEL" --revision "$REV"
fi

if want sweep; then
  step "sweep honesty over $EARLY_REV + $REV (tier smoke, B=$B, P=$P, split seeds $SPLIT_SEEDS; finished artifacts are skipped)"
  # shellcheck disable=SC2086
  $CLI sweep --model "$MODEL" --revisions "$EARLY_REV" "$REV" --values honesty --tier smoke --features both \
      --bootstrap "$B" --perms "$P" --device "$DEVICE" --split_seeds $SPLIT_SEEDS --no_progress
  step "analyze (all checkpoints: sweep curve, Spearman / Jaccard to $REV)"
  $CLI analyze --model "$MODEL"
fi

step "done; results under $VH_ROOT"
find "$VH_ROOT/results" "$VH_ROOT/figures" -type f 2>/dev/null | sed "s#^$VH_ROOT/##" | sort
echo "report: $VH_ROOT/results/report_${MODEL}.md"
