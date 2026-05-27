#!/bin/bash
# Score the frozen final method artifacts on the visible train slice and the
# held-out test slice. This job intentionally does not train or modify methods.

set -euo pipefail

cd /home/user

RUNS_DIR="${RUNS_DIR:-checkpoints/token_humanline_grpo}"

bash scripts/evaluate_train.sh --runs-dir "${RUNS_DIR}" --output scoring/scores_train.json
bash scripts/evaluate_test.sh --runs-dir "${RUNS_DIR}" --train-scores scoring/scores_train.json --output scoring/scores_test.json
