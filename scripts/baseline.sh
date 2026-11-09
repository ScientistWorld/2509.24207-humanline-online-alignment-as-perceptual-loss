#!/bin/bash
# Run the comparison baselines with the same GRPO code path, without humanline.
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
# Usage: bash scripts/baseline.sh

set -e
cd /home/user

SEED="${SEED:-7}"
LR_STALE="${LR_STALE:-1e-4}"
LR_OFFLINE="${LR_OFFLINE:-1e-4}"
LR_ONLINE="${LR_ONLINE:-3e-4}"
OUT="${OUTPUT_ROOT:-checkpoints}"

python -m baseline.stale_grpo --name grpo --experiment stale64 --output-root "${OUT}" \
  --refresh-interval 64 --train-steps 512 --seed "${SEED}" --lr "${LR_STALE}"

python -m baseline.stale_grpo --name grpo --experiment offline --output-root "${OUT}" \
  --refresh-interval 512 --train-steps 512 --grpo-epsilon 0.5 --seed "${SEED}" --lr "${LR_OFFLINE}"

for EXP in stale64 offline; do
  python -m baseline.stale_grpo --name online_grpo --experiment "${EXP}" --output-root "${OUT}" \
    --refresh-interval 1 --train-steps 512 --seed "${SEED}" --lr "${LR_ONLINE}"
done
