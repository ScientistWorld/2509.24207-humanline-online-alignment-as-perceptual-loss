#!/bin/bash
# Run the paper's method (humanline GRPO) at both fixed operating points.
#
#   stale64: completions are sampled from a frozen policy snapshot every 64
#            optimizer updates (8 sampling rounds, 131,072 completions).
#   offline: one sampling round from the starting policy, 512 updates.
#
# Writes checkpoints/<experiment>/<name>/{policy.pt,run.json,train_log.jsonl}.
# Hyperparameters per experiment were selected on the visible validation slice
# (per-method learning rate, as in the paper's Table 3, and sync interval k):
#   stale64: lr 1e-4, k 48      offline: lr 1e-4, k 192
# Usage: bash scripts/method.sh        (env vars override hyperparameters)

set -e
cd /home/user

NAME="${METHOD_NAME:-humanline_grpo}"
SEED="${SEED:-7}"
LR_STALE="${LR_STALE:-1e-4}"
SYNC_K_STALE="${SYNC_K_STALE:-48}"
LR_OFFLINE="${LR_OFFLINE:-1e-4}"
SYNC_K_OFFLINE="${SYNC_K_OFFLINE:-192}"
OUT="${OUTPUT_ROOT:-checkpoints}"

python -m method.humanline_grpo --name "${NAME}" --experiment stale64 --output-root "${OUT}" \
  --humanline --sync-k "${SYNC_K_STALE}" --refresh-interval 64 --train-steps 512 \
  --seed "${SEED}" --lr "${LR_STALE}" ${EXTRA_ARGS:-}

python -m method.humanline_grpo --name "${NAME}" --experiment offline --output-root "${OUT}" \
  --humanline --sync-k "${SYNC_K_OFFLINE}" --refresh-interval 512 --train-steps 512 \
  --seed "${SEED}" --lr "${LR_OFFLINE}" ${EXTRA_ARGS:-}
