#!/bin/bash
# Held-out test evaluation: scores every method found under checkpoints/
# (any run.json, at any depth) on the 16,000 prompts held only in eval/test/
# and writes scoring/scores_test.json.  Each (experiment, method) row is the
# mean over the required training seeds 7-14.  Runs with no arguments.
# Optional args: --runs-root DIR --output FILE --device auto|cpu|cuda.

set -e
cd /home/user
mkdir -p scoring

PYTHONPATH="/home/user:${PYTHONPATH:-}" python eval/test/evaluate.py "$@"
