#!/bin/bash
# Final hardening job.
#   0. sanity: the new evaluators run on the job-4 checkpoints still on disk
#   1. regenerate the reference set end to end from the final scripts
#      (scripts/reproduce.sh: baselines, method, ablations, visible scores)
#   2. score it on the visible slice and the enlarged hidden test
#   3. check training reproduced job 4 exactly (visible slice is unchanged)
#   4. rescore the saved variants / seeds / candidates through the final
#      evaluators, rebuild every shortcut case, write the analysis summary
#
# For the plain reference reproduction alone, run: bash scripts/reproduce.sh
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user

H=redteam/hardening
mkdir -p tmp/hardening "$H/results"

# 0. evaluator sanity check on the existing checkpoints (fails fast on a bug)
python eval/train/evaluate.py --output tmp/hardening/precheck_train.json > /dev/null
python eval/test/evaluate.py --output tmp/hardening/precheck_test.json > /dev/null
echo "precheck ok"

# clear outputs that the final evaluators regenerate (job-4 copies are in $H/results_job4/)
rm -rf tmp/hardening/correct tmp/hardening/correct_train "$H/shortcuts_runs"
find "$H/results" -name '*.json' ! -name selection.json ! -name 'sweep_*_train.json' -delete

# 1-2. reference set from the final code
bash scripts/reproduce.sh > tmp/reproduce_stdout.log 2>&1
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh > /dev/null

# 3. reproduction check against job 4 on the (unchanged) visible slice
python - <<'PY'
import json
new = json.load(open("scoring/scores_train.json"))["experiments"]
old = json.load(open("redteam/hardening/results_job4/reference_scores_train.json"))["experiments"]
alias = {("stale64", "grpo"): "stale_grpo", ("offline", "grpo"): "offline_grpo"}
rows = {}
for e in new:
    for m, r in new[e]["results"].items():
        o = old[e]["results"].get(alias.get((e, m), m), {})
        rows[f"{e}/{m}"] = {"job4": o.get("unconstrained_pass_at_1"), "final": r["unconstrained_pass_at_1"]}
identical = all(v["job4"] == v["final"] for v in rows.values())
json.dump({"identical": identical, "rows": rows}, open("redteam/hardening/results/repro_check.json", "w"), indent=2)
print("final reference set identical to job 4 on the visible slice:", identical)
PY

# 4. rescoring, shortcuts, analysis
python -m redteam.hardening.pipeline evaluate,shortcuts
