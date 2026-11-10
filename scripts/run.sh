#!/bin/bash
# Reproduce the reference end to end: train the 80 reference runs (baselines,
# humanline GRPO and its two ablations; 2 experiments x seeds 7-14) from the
# committed starting policy, then score them on the visible slice
# (scoring/scores.json, scoring/scores_train.json) and the hidden test
# (scoring/scores_test.json).  About 20 minutes on one GPU.
#
# The final hardening job (variants, extra seeds, shortcut cases, analysis) is
#   python -m redteam.hardening.final
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user

bash scripts/reproduce.sh
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh > /dev/null
