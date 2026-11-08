#!/bin/bash
# Humanline component ablations at both operating points:
#   humanline_sync_only: reference syncing every k steps, no upstream clipping.
#   humanline_clip_only: upstream log-ratio clipping, reference fixed per round.
# Usage: bash scripts/ablation.sh

set -e
cd /home/user

SEED="${SEED:-7}"
LR="${LR:-3e-4}"
SYNC_K="${SYNC_K:-12}"
OUT="${OUTPUT_ROOT:-checkpoints}"

for EXP in stale64 offline; do
  if [ "${EXP}" = "stale64" ]; then REFRESH=64; else REFRESH=512; fi
  python -m method.humanline_grpo --name humanline_sync_only --experiment "${EXP}" --output-root "${OUT}" \
    --humanline --disable-humanline-clipping --sync-k "${SYNC_K}" --refresh-interval "${REFRESH}" \
    --train-steps 512 --seed "${SEED}" --lr "${LR}"
  python -m method.humanline_grpo --name humanline_clip_only --experiment "${EXP}" --output-root "${OUT}" \
    --humanline --disable-humanline-sync --sync-k "${SYNC_K}" --refresh-interval "${REFRESH}" \
    --train-steps 512 --seed "${SEED}" --lr "${LR}"
done
