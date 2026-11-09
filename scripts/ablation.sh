#!/bin/bash
# Humanline component ablations at both operating points:
#   humanline_sync_only: reference syncing every k steps, no upstream clipping.
#   humanline_clip_only: upstream log-ratio clipping, reference fixed per round.
# Usage: bash scripts/ablation.sh

set -e
cd /home/user

SEED="${SEED:-7}"
# Same hyperparameters as the full method in each experiment.
LR_STALE="${LR_STALE:-1e-4}"
SYNC_K_STALE="${SYNC_K_STALE:-48}"
LR_OFFLINE="${LR_OFFLINE:-1e-4}"
SYNC_K_OFFLINE="${SYNC_K_OFFLINE:-192}"
OUT="${OUTPUT_ROOT:-checkpoints}"

for EXP in stale64 offline; do
  if [ "${EXP}" = "stale64" ]; then
    REFRESH=64; LR="${LR_STALE}"; SYNC_K="${SYNC_K_STALE}"
  else
    REFRESH=512; LR="${LR_OFFLINE}"; SYNC_K="${SYNC_K_OFFLINE}"
  fi
  python -m method.humanline_grpo --name humanline_sync_only --experiment "${EXP}" --output-root "${OUT}" \
    --humanline --disable-humanline-clipping --sync-k "${SYNC_K}" --refresh-interval "${REFRESH}" \
    --train-steps 512 --seed "${SEED}" --lr "${LR}"
  python -m method.humanline_grpo --name humanline_clip_only --experiment "${EXP}" --output-root "${OUT}" \
    --humanline --disable-humanline-sync --sync-k "${SYNC_K}" --refresh-interval "${REFRESH}" \
    --train-steps 512 --seed "${SEED}" --lr "${LR}"
done
