#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: reads method-produced run JSON and prediction
# JSONL artifacts, then writes a scores file under scoring/. The evaluator
# defaults to scoring the same method keys already present in scores_train.json.

set -e

cd /home/user

mkdir -p scoring

PYTHONPATH="/home/user:${PYTHONPATH:-}" python eval/test/evaluate.py "$@"
