#!/bin/bash
# Run the paper's method (humanline GRPO) at both fixed operating points.
#
#   stale64: completions are sampled from a frozen policy snapshot every 64
#            optimizer updates (8 sampling rounds, 131,072 completions).
#   offline: one sampling round from the starting policy, 512 updates.
#
# Writes checkpoints/<experiment>/<name>/{policy.pt,run.json,train_log.jsonl}.
# Usage: bash scripts/method.sh        (env vars override hyperparameters)

set -e
cd /home/user

NAME="${METHOD_NAME:-humanline_grpo}"
SEED="${SEED:-7}"
LR="${LR:-3e-4}"
SYNC_K="${SYNC_K:-12}"
OUT="${OUTPUT_ROOT:-checkpoints}"

python -m method.humanline_grpo --name "${NAME}" --experiment stale64 --output-root "${OUT}" \
  --humanline --sync-k "${SYNC_K}" --refresh-interval 64 --train-steps 512 \
  --seed "${SEED}" --lr "${LR}" ${EXTRA_ARGS:-}

python -m method.humanline_grpo --name "${NAME}" --experiment offline --output-root "${OUT}" \
  --humanline --sync-k "${SYNC_K}" --refresh-interval 512 --train-steps 512 \
  --seed "${SEED}" --lr "${LR}" ${EXTRA_ARGS:-}
