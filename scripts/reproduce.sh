#!/bin/bash
# Reproduce the reference runs end to end from the committed starting policy
# (data/base_policy.pt): baselines, the method and its ablations, each for the
# eight required training seeds (7-14), then score them on the visible slice.
#
# Assumes: container built, working directory /home/user.  No downloads are
# needed.  Runs on GPU when available (about 15 minutes for the 80 runs with
# 4 in parallel), otherwise CPU.

set -e
cd /home/user

rm -rf checkpoints/stale64 checkpoints/offline
mkdir -p checkpoints

bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/ablation.sh

bash scripts/evaluate.sh
