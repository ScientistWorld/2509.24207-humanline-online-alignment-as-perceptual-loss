#!/bin/bash
# Hardening job (iteration 11): smoke test, base-policy build, pre-registered
# LR sweep, reference runs into checkpoints/, extra seeds, variants, shortcut
# submissions, and visible + hidden evaluation.  Resumable.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
rm -rf redteam/hardening/shortcuts_runs/reward_artifact
python -m redteam.hardening.pipeline
