#!/bin/bash
# Held-out test evaluation: scores every run under checkpoints/ on prompts
# held only in eval/test/ and writes scoring/scores_test.json.  Runs with no
# arguments.  Optional args: --runs-root DIR --output FILE.

set -e
cd /home/user
mkdir -p scoring

PYTHONPATH="/home/user:${PYTHONPATH:-}" python eval/test/evaluate.py "$@"
