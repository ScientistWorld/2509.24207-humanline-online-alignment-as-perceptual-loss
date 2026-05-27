#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

python -m baseline.stale_grpo \
  --name "${BASELINE_NAME:-stale_grpo_64x_offpolicy}" \
  --pool-stale-batches \
  --balanced-train \
  --refresh-interval "${REFRESH_INTERVAL:-64}" \
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
