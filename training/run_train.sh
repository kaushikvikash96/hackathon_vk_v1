#!/usr/bin/env bash
# Fine-tune Nemotron for grounded answer synthesis, on the fine-tuning/model node.
#
#   bash training/run_train.sh --quick     # ~30-45 min, 25 steps, 600 examples
#   bash training/run_train.sh             # ~2-3 hours, 100 steps, full set
#   bash training/run_train.sh --dry-run   # print the resolved settings only
#
# Options:
#   --quick            short profile (training/configs/lora_quick.yaml)
#   --full             full profile  (training/configs/lora_r32.yaml), default
#   --steps N          override the step count for the chosen profile
#   --skip-smoke       skip the 30-second pipeline smoke test
#   --dry-run          resolve and print settings without training
#
# Wraps the organizer-supplied NeMo scripts rather than replacing them, so the
# run stays reproducible against the environment the judges have. Always run
# inside tmux: earlyoom kills long training runs on a disconnected session.
#
#   tmux new-session -s train "bash training/run_train.sh --quick"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINETUNE_DIR="${FINETUNE_DIR:-$HOME/Cognitivo_Training/finagent-finetune}"

PROFILE="${PROFILE:-full}"
STEPS_OVERRIDE=""
SKIP_SMOKE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)      PROFILE=quick ;;
    --full)       PROFILE=full ;;
    --steps)      STEPS_OVERRIDE="$2"; shift ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --dry-run)    DRY_RUN=1 ;;
    -h|--help)    sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# --- profile ----------------------------------------------------------------
# Warmup is scaled with the run, not inherited. A 25-step run with the
# baseline's 50-step warmup would end before the learning rate peaked.
if [[ "$PROFILE" == "quick" ]]; then
  CONFIG="training/configs/lora_quick.yaml"
  MAX_STEPS=25
  WARMUP_STEPS=10
  CHECKPOINT_EVERY=5
  TRAIN_FILE="$REPO_ROOT/training/data/train_quick_nemo.jsonl"
  RUN_NAME="${RUN_NAME:-nemotron8b-synth-lora-r32-quick}"
  EXPECTED="30-45 minutes"
else
  CONFIG="training/configs/lora_r32.yaml"
  MAX_STEPS=100
  WARMUP_STEPS=50
  CHECKPOINT_EVERY=20
  TRAIN_FILE="$REPO_ROOT/training/data/train_nemo.jsonl"
  RUN_NAME="${RUN_NAME:-nemotron8b-synth-lora-r32}"
  EXPECTED="2-3 hours"
fi

if [[ -n "$STEPS_OVERRIDE" ]]; then
  MAX_STEPS="$STEPS_OVERRIDE"
  # Keep warmup and checkpointing sane for whatever length was requested.
  WARMUP_STEPS=$(( MAX_STEPS * 2 / 5 ))
  (( WARMUP_STEPS < 1 )) && WARMUP_STEPS=1
  CHECKPOINT_EVERY=$(( MAX_STEPS / 5 ))
  (( CHECKPOINT_EVERY < 1 )) && CHECKPOINT_EVERY=1
  RUN_NAME="${RUN_NAME}-s${MAX_STEPS}"
  EXPECTED="scaled from the ${PROFILE} profile"
fi

LOG_FILE="${LOG_FILE:-$REPO_ROOT/training/logs/${RUN_NAME}.log}"

# shellcheck disable=SC1090
[[ -f "$HOME/team.env" ]] && source "$HOME/team.env"

mkdir -p "$REPO_ROOT/training/logs" "$REPO_ROOT/training/metrics"

cat <<SETTINGS
profile          $PROFILE
config           $CONFIG
train file       $(basename "$TRAIN_FILE")
max steps        $MAX_STEPS
warmup steps     $WARMUP_STEPS
checkpoint every $CHECKPOINT_EVERY
lora rank        32       learning rate 5e-5      seq len 512
run name         $RUN_NAME
expected time    $EXPECTED
log              $LOG_FILE

SETTINGS

[[ "$DRY_RUN" == "1" ]] && exit 0

echo "== 1/4 verifying the training data =="
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "missing $TRAIN_FILE - run: python training/make_dataset.py" >&2
  exit 1
fi
echo "  $(wc -l < "$TRAIN_FILE") training examples"
for split in val test; do
  file="$REPO_ROOT/training/data/${split}_nemo.jsonl"
  [[ -f "$file" ]] || { echo "missing $file - run: python training/make_dataset.py" >&2; exit 1; }
  echo "  $(wc -l < "$file") examples in ${split}_nemo.jsonl"
done

if [[ "$SKIP_SMOKE" == "0" ]]; then
  echo "== 2/4 smoke test (~30s) - confirms container, GPU, and checkpoint saving =="
  ( cd "$FINETUNE_DIR" && bash scripts/02_smoke_test.sh )
else
  echo "== 2/4 smoke test skipped =="
fi

echo "== 3/4 training =="
(
  cd "$FINETUNE_DIR"
  TRAIN_DATA="$TRAIN_FILE" \
  VAL_DATA="$REPO_ROOT/training/data/val_nemo.jsonl" \
  LORA_RANK=32 \
  LR=5e-5 \
  MAX_STEPS="$MAX_STEPS" \
  WARMUP_STEPS="$WARMUP_STEPS" \
  MAX_SEQ_LEN=512 \
  BATCH_SIZE=2 \
  GRAD_ACCUM=4 \
  CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  RUN_NAME="$RUN_NAME" \
  bash scripts/03_train_1node.sh
) 2>&1 | tee "$LOG_FILE"

echo "== 4/4 adapters produced =="
find "${MODELS_DIR:-$HOME/local-llm-setup/models}/checkpoints" -type d -name hf_adapter | sed 's/^/  /'

cat <<NEXT

Next:
  1. Serve a checkpoint:
       ADAPTER_CHECKPOINT=<path-to>/hf_adapter bash scripts/04_export_and_serve.sh
  2. Compare it against the base model on identical evidence:
       python training/eval_base_vs_ft.py --ft-model domain-ft --base-model nemotron-base
  3. Point the agent at it and re-score the calibration questions:
       export DOMAIN_PREDICT_MODE=llm
       python scripts/eval_public.py
NEXT

if [[ "$PROFILE" == "quick" ]]; then
  cat <<'QUICKNOTE'
This was the quick profile. It exists to get a real fine-tuned adapter serving
early so the whole pipeline can be measured. If time allows, start the full run
(bash training/run_train.sh) and report whichever checkpoint measures better.
QUICKNOTE
fi
