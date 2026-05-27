#!/bin/bash
# Regenerate the scaled reproduction artifacts, then score them on the visible
# train slice and the held-out test slice.

set -euo pipefail

cd /home/user

RUNS_DIR="${RUNS_DIR:-checkpoints/humanline_grpo}"

bash scripts/reproduce.sh

bash scripts/evaluate_train.sh --runs-dir "${RUNS_DIR}" --output scoring/scores_train.json
bash scripts/evaluate_test.sh --runs-dir "${RUNS_DIR}" --train-scores scoring/scores_train.json --output scoring/scores_test.json
