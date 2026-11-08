#!/bin/bash
# Visible train-slice evaluation: scores every run under checkpoints/ on the
# visible validation prompts and writes scoring/scores_train.json.
# Optional args: --runs-root DIR --output FILE.

set -e
cd /home/user
mkdir -p scoring

PYTHONPATH="/home/user:${PYTHONPATH:-}" python eval/train/evaluate.py "$@"
