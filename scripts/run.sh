#!/bin/bash
# Final hardening job (last iteration).  See redteam/hardening/final.py:
#   smoke      fail-fast checks of the multi-seed evaluators
#   reference  scripts/reproduce.sh (80 runs, seeds 7-14) + visible/hidden scoring,
#              CPU-vs-GPU decoding and retrain-determinism checks
#   variants   better / weaker / simple-baseline variants (8 seeds each)
#   pool       extra seeds 15-30 of humanline_grpo and grpo (noise estimate)
#   shortcuts  every shortcut case through the hidden test (+ the pre-fix evaluator)
#   analyze    redteam/hardening/results_final/summary.json
#   reference_json  scoring/reference.json from scoring/scores_test.json
#
# For the plain reference reproduction alone, run:
#   bash scripts/reproduce.sh && bash scripts/evaluate_train.sh && bash scripts/evaluate_test.sh
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user

python -m redteam.hardening.final
python validate.py || true
