#!/bin/bash
# Run the comparison baselines with the same GRPO code path, without humanline.
#
#   stale_grpo   (stale64): sample every 64 updates; the reference is the
#                policy that generated the current round (paper's 64x-stale GRPO).
#   offline_grpo (offline): one pool from the starting policy; the reference
#                never moves (paper's offline GRPO, epsilon 0.5 as in Table 3).
#   online_grpo  (both):    fresh samples every update.  It deliberately violates
#                both operating points, so the evaluators score it as failed and
#                report its accuracy only as `unconstrained_pass_at_1`.
#
# Usage: bash scripts/baseline.sh

set -e
cd /home/user

SEED="${SEED:-7}"
LR="${LR:-3e-4}"
OUT="${OUTPUT_ROOT:-checkpoints}"

python -m baseline.stale_grpo --name stale_grpo --experiment stale64 --output-root "${OUT}" \
  --refresh-interval 64 --train-steps 512 --seed "${SEED}" --lr "${LR}"

python -m baseline.stale_grpo --name offline_grpo --experiment offline --output-root "${OUT}" \
  --refresh-interval 512 --train-steps 512 --grpo-epsilon 0.5 --seed "${SEED}" --lr "${LR}"

for EXP in stale64 offline; do
  python -m baseline.stale_grpo --name online_grpo --experiment "${EXP}" --output-root "${OUT}" \
    --refresh-interval 1 --train-steps 512 --seed "${SEED}" --lr "${LR}"
done
