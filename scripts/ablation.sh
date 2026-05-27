#!/bin/bash
# Run Humanline GRPO ablations for the scaled stale-data setting.
#
# These ablations isolate the two pieces of the paper's Humanline design:
# reference syncing and upstream policy/reference log-ratio clipping.

set -e

cd /home/user

python -m method.humanline_grpo \
  --name "${SYNC_ONLY_NAME:-humanline_sync_only_grpo}" \
  --humanline \
  --pool-stale-batches \
  --disable-humanline-clipping \
  --refresh-interval "${REFRESH_INTERVAL:-64}" \
  --sync-k "${SYNC_K:-12}" \
  --train-steps "${TRAIN_STEPS:-256}" \
  --sft-steps "${SFT_STEPS:-1200}" \
  --train-size "${TRAIN_SIZE:-512}" \
  --eval-size "${EVAL_SIZE:-160}" \
  --batch-size "${BATCH_SIZE:-24}" \
  --group-size "${GROUP_SIZE:-6}" \
  --hidden-size "${HIDDEN_SIZE:-128}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --rl-lr "${RL_LR:-5e-4}" \
  --beta-kl "${BETA_KL:-0.01}" \
  --log-epsilon-p "${LOG_EPSILON_P:--1.5}" \
  --log-epsilon-r "${LOG_EPSILON_R:-1.5}" \
  --min-value "${MIN_VALUE:-0}" \
  --max-value "${MAX_VALUE:-9}" \
  --output-dir checkpoints/humanline_grpo

python -m method.humanline_grpo \
  --name "${CLIP_ONLY_NAME:-humanline_clip_only_grpo}" \
  --humanline \
  --pool-stale-batches \
  --disable-humanline-sync \
  --refresh-interval "${REFRESH_INTERVAL:-64}" \
  --sync-k "${SYNC_K:-12}" \
  --train-steps "${TRAIN_STEPS:-256}" \
  --sft-steps "${SFT_STEPS:-1200}" \
  --train-size "${TRAIN_SIZE:-512}" \
  --eval-size "${EVAL_SIZE:-160}" \
  --batch-size "${BATCH_SIZE:-24}" \
  --group-size "${GROUP_SIZE:-6}" \
  --hidden-size "${HIDDEN_SIZE:-128}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --rl-lr "${RL_LR:-5e-4}" \
  --beta-kl "${BETA_KL:-0.01}" \
  --log-epsilon-p "${LOG_EPSILON_P:--1.5}" \
  --log-epsilon-r "${LOG_EPSILON_R:-1.5}" \
  --min-value "${MIN_VALUE:-0}" \
  --max-value "${MAX_VALUE:-9}" \
  --output-dir checkpoints/humanline_grpo

bash scripts/evaluate.sh --runs-dir checkpoints/humanline_grpo
