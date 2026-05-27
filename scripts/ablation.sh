#!/bin/bash
# Run Humanline GRPO ablations for the scaled stale-data setting.
#
# These ablations isolate the two pieces of the paper's Humanline design:
# reference syncing and upstream policy/reference log-ratio clipping.

set -e

cd /home/user

python -m method.answer_grpo \
  --name "${SYNC_ONLY_NAME:-humanline_sync_only_grpo}" \
  --humanline \
  --disable-humanline-clipping \
  --refresh-interval "${REFRESH_INTERVAL:-64}" \
  --sync-k "${SYNC_K:-12}" \
  --train-steps "${TRAIN_STEPS:-1024}" \
  --sft-steps "${SFT_STEPS:-20}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --group-size "${GROUP_SIZE:-16}" \
  --hidden-size "${HIDDEN_SIZE:-64}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --rl-lr "${RL_LR:-3e-4}" \
  --beta-kl "${BETA_KL:-0.05}" \
  --log-epsilon-p "${LOG_EPSILON_P:--5.0}" \
  --log-epsilon-r "${LOG_EPSILON_R:-5.0}" \
  --min-value "${MIN_VALUE:-0}" \
  --max-value "${MAX_VALUE:-19}" \
  --output-dir checkpoints/humanline_grpo

python -m method.answer_grpo \
  --name "${CLIP_ONLY_NAME:-humanline_clip_only_grpo}" \
  --humanline \
  --disable-humanline-sync \
  --refresh-interval "${REFRESH_INTERVAL:-64}" \
  --sync-k "${SYNC_K:-12}" \
  --train-steps "${TRAIN_STEPS:-1024}" \
  --sft-steps "${SFT_STEPS:-20}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --group-size "${GROUP_SIZE:-16}" \
  --hidden-size "${HIDDEN_SIZE:-64}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --rl-lr "${RL_LR:-3e-4}" \
  --beta-kl "${BETA_KL:-0.05}" \
  --log-epsilon-p "${LOG_EPSILON_P:--5.0}" \
  --log-epsilon-r "${LOG_EPSILON_R:-5.0}" \
  --min-value "${MIN_VALUE:-0}" \
  --max-value "${MAX_VALUE:-19}" \
  --output-dir checkpoints/humanline_grpo

bash scripts/evaluate.sh --runs-dir checkpoints/humanline_grpo
