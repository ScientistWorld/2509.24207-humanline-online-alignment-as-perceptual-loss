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

export TRAIN_STEPS="${TRAIN_STEPS:-1024}"
export SFT_STEPS="${SFT_STEPS:-20}"
export REFRESH_INTERVAL="${REFRESH_INTERVAL:-64}"
export SYNC_K="${SYNC_K:-12}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export GROUP_SIZE="${GROUP_SIZE:-16}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export RL_LR="${RL_LR:-3e-4}"
export BETA_KL="${BETA_KL:-0.05}"
export LOG_EPSILON_P="${LOG_EPSILON_P:--5.0}"
export LOG_EPSILON_R="${LOG_EPSILON_R:-5.0}"

for sync_k in 8 16 20 24; do
  METHOD_NAME="offline_humanline_grpo_64x_sync${sync_k}" \
  REFRESH_INTERVAL=64 \
  SYNC_K="${sync_k}" \
  bash scripts/method.sh
done

bash scripts/evaluate.sh --runs-dir checkpoints/humanline_grpo
