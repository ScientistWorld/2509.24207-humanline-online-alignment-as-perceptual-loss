#!/bin/bash
# Visible train-slice evaluation: scores every method found under checkpoints/
# on the 2,000 visible validation prompts (same rules as the hidden test: mean
# over training seeds 7-14) and writes scoring/scores_train.json.
# Optional args: --runs-root DIR --output FILE --device auto|cpu|cuda.

set -e
cd /home/user
mkdir -p scoring

PYTHONPATH="/home/user:${PYTHONPATH:-}" python eval/train/evaluate.py "$@"
