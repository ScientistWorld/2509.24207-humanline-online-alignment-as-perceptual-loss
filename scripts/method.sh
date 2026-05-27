#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python -m method.humanline_grpo \
  --name "${METHOD_NAME:-offline_humanline_grpo}" \
  --humanline \
  --pool-stale-batches \
  --balanced-train \
  --refresh-interval "${REFRESH_INTERVAL:-64}" \
  --sync-k "${SYNC_K:-12}" \
  --train-steps "${TRAIN_STEPS:-128}" \
  --sft-steps "${SFT_STEPS:-1000}" \
  --train-size "${TRAIN_SIZE:-512}" \
  --eval-size "${EVAL_SIZE:-160}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --group-size "${GROUP_SIZE:-6}" \
  --hidden-size "${HIDDEN_SIZE:-128}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --rl-lr "${RL_LR:-5e-4}" \
  --beta-kl "${BETA_KL:-0.01}" \
  --log-epsilon-p "${LOG_EPSILON_P:--1.5}" \
  --log-epsilon-r "${LOG_EPSILON_R:-1.5}" \
  --min-value "${MIN_VALUE:-0}" \
  --max-value "${MAX_VALUE:-9}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-24}" \
  --log-every "${LOG_EVERY:-32}" \
  --output-dir checkpoints/humanline_grpo

bash scripts/evaluate.sh --runs-dir checkpoints/humanline_grpo
