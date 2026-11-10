#!/bin/bash
# Run the comparison baselines with the same GRPO code path, without humanline,
# for every required training seed (7-14).
#
#   grpo        (stale64): sample every 64 updates; the reference is the policy
#               that generated the current round (the paper's 64x-stale GRPO).
#   grpo        (offline): one pool from the starting policy; the reference never
#               moves (the paper's offline GRPO, epsilon 0.5 as in its Table 3).
#   online_grpo (both):    fresh samples every update.  It deliberately violates
#               both operating points, so the evaluators score it as failed and
#               report its accuracy only as `unconstrained_pass_at_1`.
#
# Learning rates were selected per method on the visible validation slice:
#   grpo 1e-4 (both experiments), online_grpo 3e-4.
# Writes checkpoints/<experiment>/<name>/seed<seed>/.
# Usage: bash scripts/baseline.sh

set -e
cd /home/user

SEEDS="${SEEDS:-7 8 9 10 11 12 13 14}"
PARALLEL="${PARALLEL:-4}"
LR_STALE="${LR_STALE:-1e-4}"
LR_OFFLINE="${LR_OFFLINE:-1e-4}"
LR_ONLINE="${LR_ONLINE:-3e-4}"
OUT="${OUTPUT_ROOT:-checkpoints}"
source scripts/launch.sh

for SEED in ${SEEDS}; do
  launch "${OUT}/stale64/grpo/seed${SEED}" baseline.stale_grpo --name grpo --experiment stale64 \
    --refresh-interval 64 --train-steps 512 --seed "${SEED}" --lr "${LR_STALE}"
  launch "${OUT}/offline/grpo/seed${SEED}" baseline.stale_grpo --name grpo --experiment offline \
    --refresh-interval 512 --train-steps 512 --grpo-epsilon 0.5 --seed "${SEED}" --lr "${LR_OFFLINE}"
  for EXP in stale64 offline; do
    launch "${OUT}/${EXP}/online_grpo/seed${SEED}" baseline.stale_grpo --name online_grpo --experiment "${EXP}" \
      --refresh-interval 1 --train-steps 512 --seed "${SEED}" --lr "${LR_ONLINE}"
  done
done
finish
