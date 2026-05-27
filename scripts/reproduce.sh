#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

rm -rf checkpoints/humanline_grpo
mkdir -p checkpoints/humanline_grpo

export BASELINE_NAME="${BASELINE_NAME:-stale_grpo_64x_offpolicy}"
export METHOD_NAME="${METHOD_NAME:-offline_humanline_grpo}"
bash scripts/baseline.sh
bash scripts/method.sh

if [ "${RUN_EXTRA_REFRESH:-1}" = "1" ]; then
  BASELINE_NAME="stale_grpo_128x_offpolicy" \
  METHOD_NAME="offline_humanline_grpo_128x" \
  REFRESH_INTERVAL=128 \
  bash scripts/baseline.sh

  BASELINE_NAME="stale_grpo_128x_offpolicy" \
  METHOD_NAME="offline_humanline_grpo_128x" \
  REFRESH_INTERVAL=128 \
  bash scripts/method.sh
fi

if [ "${RUN_EXTRA_32_REFRESH:-1}" = "1" ]; then
  BASELINE_NAME="stale_grpo_32x_offpolicy" \
  METHOD_NAME="offline_humanline_grpo_32x" \
  REFRESH_INTERVAL=32 \
  bash scripts/baseline.sh

  BASELINE_NAME="stale_grpo_32x_offpolicy" \
  METHOD_NAME="offline_humanline_grpo_32x" \
  REFRESH_INTERVAL=32 \
  bash scripts/method.sh
fi

if [ "${RUN_ABLATIONS:-1}" = "1" ]; then
  bash scripts/ablation.sh
fi

if [ "${RUN_SYNC_SWEEP:-1}" = "1" ]; then
  for sync_k in 8 16 20 24; do
    METHOD_NAME="offline_humanline_grpo_64x_sync${sync_k}" \
    REFRESH_INTERVAL=64 \
    SYNC_K="${sync_k}" \
    bash scripts/method.sh
  done
fi

bash scripts/evaluate.sh --runs-dir checkpoints/humanline_grpo
