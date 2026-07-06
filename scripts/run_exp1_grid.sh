#!/usr/bin/env bash
# =============================================================================
# run_exp1_grid.sh -- Experiment 1 backbone x readout grid.
#
# Runs the full 7-model x 2-readout comparison your TA asked for:
#   - 7 LLM backbones (Qwen 0.5/3/7/14B, Gemma-2 2/9B, Llama-2-7b-chat)
#   - 2 readouts: Type-1 "head" (trained classifier) and Type-2 "generative"
#     (the LLM's own LM head over emotion-word tokens, no added head)
#   - 5 speaker-independent folds each
# = 14 cells x 5 folds = 70 training runs, then a per-cell age-gap analysis.
#
# Everything runs on the SAME cached features (data/crema_d/features) and the
# SAME folds, so cells are directly comparable. Outputs go to:
#   <OUT>/<modelshort>_<readout>/foldN/{predictions.csv,metrics.json,best.pt}
#   <OUT>/<modelshort>_<readout>/analysis/{regression.txt,per_actor.csv,age_gap.png}
# Aggregate everything into one table with scripts/aggregate_exp1_grid.py.
#
# Idempotent/resumable: a fold whose metrics.json already exists is skipped, so
# you can stop and re-run freely (it picks up where it left off).
#
# USAGE (on Atlas, from ~/elder-mer):
#   bash scripts/run_exp1_grid.sh                     # all 14 cells on GPU 0
#   GPU=1 bash scripts/run_exp1_grid.sh               # use GPU 1
#   EPOCHS=15 OUT=data/crema_d/exp1_grid bash scripts/run_exp1_grid.sh
#   MODELS_ONLY="qwen7b gemma9b" bash scripts/run_exp1_grid.sh   # subset by shortname
#
# ! RESERVE A GPU on the Atlas sheet first -- this is a multi-hour job.
#   Run it in the background (e.g. tmux, or `nohup bash scripts/run_exp1_grid.sh &`).
# =============================================================================
set -uo pipefail

PY="${PY:-$HOME/miniconda3/envs/elder-mer/bin/python}"
GPU="${GPU:-0}"
OUT="${OUT:-data/crema_d/exp1_grid}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-16}"
NFOLDS="${NFOLDS:-5}"
DTYPE="${DTYPE:-bfloat16}"
MODELS_ONLY="${MODELS_ONLY:-}"   # optional space-separated shortnames to restrict to

# model_id : shortname   (shortname is used for the output dir + table label)
MODELS=(
  "Qwen/Qwen2.5-0.5B:qwen0.5b"
  "Qwen/Qwen2.5-3B:qwen3b"
  "Qwen/Qwen2.5-7B:qwen7b"
  "Qwen/Qwen2.5-14B:qwen14b"
  "unsloth/gemma-2-2b:gemma2b"
  "unsloth/gemma-2-9b:gemma9b"
  "NousResearch/Llama-2-7b-chat-hf:llama7b"
)
READOUTS=(head generative)

NOISE='newly initialized|TRAIN this model|^Some weights|UserWarning|warnings.warn|Loading checkpoint|it/s\]|resume_download|pkg_resources'

echo "============================================================"
echo "Experiment 1 grid  |  GPU $GPU  |  out=$OUT  |  epochs=$EPOCHS  dtype=$DTYPE"
echo "models: ${MODELS_ONLY:-all 7}   readouts: ${READOUTS[*]}   folds: $NFOLDS"
echo "! Ensure GPU $GPU is reserved on the Atlas sheet (multi-hour job)."
echo "============================================================"

done_cells=0; skipped_folds=0; ran_folds=0
for pair in "${MODELS[@]}"; do
  mid="${pair%%:*}"; short="${pair##*:}"
  if [[ -n "$MODELS_ONLY" ]] && [[ " $MODELS_ONLY " != *" $short "* ]]; then continue; fi
  for ro in "${READOUTS[@]}"; do
    cell="$OUT/${short}_${ro}"
    echo; echo "===== CELL  $short / $ro  ->  $cell ====="
    for f in $(seq 0 $((NFOLDS-1))); do
      if [[ -s "$cell/fold$f/metrics.json" ]]; then
        echo "  fold $f: already done, skipping"; skipped_folds=$((skipped_folds+1)); continue
      fi
      echo "  fold $f: training ..."
      CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/train_backbone.py \
        --fold "$f" --n-folds "$NFOLDS" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --llm "$mid" --dtype "$DTYPE" --device cuda --readout "$ro" \
        --out "$cell" 2>&1 | grep -vE "$NOISE"
      ran_folds=$((ran_folds+1))
    done
    # per-cell age-gap analysis once folds exist
    if ls "$cell"/fold*/predictions.csv >/dev/null 2>&1; then
      "$PY" scripts/analyze_exp1_age_gap.py \
        --pred-glob "$cell/fold*/predictions.csv" \
        --out "$cell/analysis" --model "${short}/${ro}" 2>&1 \
        | grep -E "overall:|accuracy   slope|VERDICT" | head -3
    fi
    done_cells=$((done_cells+1))
  done
done

echo; echo "============================================================"
echo "GRID PASS COMPLETE  |  cells touched: $done_cells  folds run: $ran_folds  skipped: $skipped_folds"
echo "Aggregate the comparison table with:"
echo "  $PY scripts/aggregate_exp1_grid.py --grid $OUT"
echo "============================================================"
