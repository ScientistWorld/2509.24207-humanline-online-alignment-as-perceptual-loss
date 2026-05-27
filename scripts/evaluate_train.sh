#!/bin/bash
# Visible train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: reads method-produced run JSON and prediction
# JSONL artifacts, then writes a scores file under scoring/.

set -e

cd /home/user

mkdir -p scoring

PYTHONPATH="/home/user:${PYTHONPATH:-}" python eval/train/evaluate.py "$@"
