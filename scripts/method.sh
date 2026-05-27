#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python -m method.answer_grpo \
  --name "${METHOD_NAME:-offline_humanline_grpo}" \
  --humanline \
  --refresh-interval "${REFRESH_INTERVAL:-64}" \
  --sync-k "${SYNC_K:-12}" \
  --train-steps "${TRAIN_STEPS:-256}" \
  --sft-steps "${SFT_STEPS:-500}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --group-size "${GROUP_SIZE:-4}" \
  --hidden-size "${HIDDEN_SIZE:-128}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --rl-lr "${RL_LR:-5e-4}" \
  --beta-kl "${BETA_KL:-0.01}" \
  --log-epsilon-p "${LOG_EPSILON_P:--1.5}" \
  --log-epsilon-r "${LOG_EPSILON_R:-1.5}" \
  --min-value "${MIN_VALUE:-0}" \
  --max-value "${MAX_VALUE:-19}" \
  --output-dir checkpoints/humanline_grpo

bash scripts/evaluate.sh --runs-dir checkpoints/humanline_grpo
