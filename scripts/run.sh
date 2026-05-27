#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

OUT_DIR="${OUT_DIR:-checkpoints/token_humanline_grpo}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --pool-stale-batches
  --balanced-train
  --train-steps "${TRAIN_STEPS:-128}"
  --sft-steps "${SFT_STEPS:-1000}"
  --train-size "${TRAIN_SIZE:-512}"
  --eval-size "${EVAL_SIZE:-160}"
  --batch-size "${BATCH_SIZE:-16}"
  --group-size "${GROUP_SIZE:-6}"
  --hidden-size "${HIDDEN_SIZE:-128}"
  --max-new-tokens "${MAX_NEW_TOKENS:-24}"
  --temperature "${TEMPERATURE:-0.8}"
  --sft-lr "${SFT_LR:-2e-3}"
  --rl-lr "${RL_LR:-5e-4}"
  --beta-kl "${BETA_KL:-0.01}"
  --log-epsilon-p "${LOG_EPSILON_P:--1.5}"
  --log-epsilon-r "${LOG_EPSILON_R:-1.5}"
  --min-value "${MIN_VALUE:-0}"
  --max-value "${MAX_VALUE:-9}"
  --log-every "${LOG_EVERY:-32}"
  --output-dir "${OUT_DIR}"
)

python -m method.humanline_grpo \
  --name stale_grpo_64x_offpolicy \
  --refresh-interval 64 \
  --sync-k "${SYNC_K:-12}" \
  "${COMMON_ARGS[@]}"

python -m method.humanline_grpo \
  --name offline_humanline_grpo \
  --humanline \
  --refresh-interval 64 \
  --sync-k "${SYNC_K:-12}" \
  "${COMMON_ARGS[@]}"

bash scripts/evaluate.sh --runs-dir "${OUT_DIR}"
