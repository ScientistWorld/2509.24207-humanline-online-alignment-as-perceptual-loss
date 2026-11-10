#!/bin/bash
# Run the paper's method (humanline GRPO) at both fixed operating points, for
# every required training seed (the evaluators score the mean over seeds 7-14).
#
#   stale64: completions are sampled from a frozen policy snapshot every 64
#            optimizer updates (8 sampling rounds, 131,072 completions).
#   offline: one sampling round from the starting policy, 512 updates.
#
# Writes checkpoints/<experiment>/<name>/seed<seed>/{policy.pt,run.json,train_log.jsonl}.
# Hyperparameters per experiment were selected on the visible validation slice
# (per-method learning rate, as in the paper's Table 3, and sync interval k):
#   stale64: lr 1e-4, k 48      offline: lr 1e-4, k 192
# Usage: bash scripts/method.sh        (env vars override hyperparameters)

set -e
cd /home/user

NAME="${METHOD_NAME:-humanline_grpo}"
SEEDS="${SEEDS:-7 8 9 10 11 12 13 14}"
PARALLEL="${PARALLEL:-4}"
LR_STALE="${LR_STALE:-1e-4}"
SYNC_K_STALE="${SYNC_K_STALE:-48}"
LR_OFFLINE="${LR_OFFLINE:-1e-4}"
SYNC_K_OFFLINE="${SYNC_K_OFFLINE:-192}"
OUT="${OUTPUT_ROOT:-checkpoints}"
source scripts/launch.sh

for SEED in ${SEEDS}; do
  launch "${OUT}/stale64/${NAME}/seed${SEED}" method.humanline_grpo --name "${NAME}" --experiment stale64 \
    --humanline --sync-k "${SYNC_K_STALE}" --refresh-interval 64 --train-steps 512 \
    --seed "${SEED}" --lr "${LR_STALE}" ${EXTRA_ARGS:-}
  launch "${OUT}/offline/${NAME}/seed${SEED}" method.humanline_grpo --name "${NAME}" --experiment offline \
    --humanline --sync-k "${SYNC_K_OFFLINE}" --refresh-interval 512 --train-steps 512 \
    --seed "${SEED}" --lr "${LR_OFFLINE}" ${EXTRA_ARGS:-}
done
finish
