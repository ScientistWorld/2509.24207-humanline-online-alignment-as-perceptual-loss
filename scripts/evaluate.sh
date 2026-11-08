#!/bin/bash
# Standard (visible) evaluation: scores every run under checkpoints/ on the
# visible validation prompts and writes scoring/scores.json.
#
# The evaluator loads each run's policy.pt into its own copy of the fixed
# architecture and decodes greedily itself; it does not import method/ or
# data/.  Optional args: --runs-root DIR --output FILE.

set -e
cd /home/user
mkdir -p scoring

PYTHONPATH="/home/user:${PYTHONPATH:-}" python eval/train/evaluate.py --output scoring/scores.json "$@"
