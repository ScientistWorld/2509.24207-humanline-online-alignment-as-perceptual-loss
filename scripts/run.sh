#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

export TRAIN_STEPS="${TRAIN_STEPS:-256}"
export SFT_STEPS="${SFT_STEPS:-500}"
export REFRESH_INTERVAL="${REFRESH_INTERVAL:-64}"
export SYNC_K="${SYNC_K:-12}"

bash scripts/reproduce.sh
python validate.py --compare
