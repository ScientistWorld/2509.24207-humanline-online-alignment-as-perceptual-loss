#!/bin/bash
# Reproduce the reference runs end to end from the committed starting policy
# (data/base_policy.pt), then score them on the visible slice.
#
# Assumes: container built, working directory /home/user.  No downloads are
# needed.  Runs on GPU when available, otherwise CPU.

set -e
cd /home/user

rm -rf checkpoints/stale64 checkpoints/offline
mkdir -p checkpoints

bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/ablation.sh

bash scripts/evaluate.sh
