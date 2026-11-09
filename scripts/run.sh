#!/bin/bash
# Hardening job 4 (final): regenerate the reference set from the final scripts,
# score it (visible + hidden), check it matches the selected sweep runs, then
# run the remaining headroom candidates, shortcuts and the analysis.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user

bash scripts/reproduce.sh > tmp/reproduce_stdout.log 2>&1
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh > /dev/null

python - <<'PY'
import json
new = json.load(open("scoring/scores_test.json"))["experiments"]
old = json.load(open("redteam/hardening/results_job3/reference_scores_test.json"))["experiments"]
diff = {f"{e}/{m}": {"job3_sweep_copy": old[e]["results"].get(m, {}).get("unconstrained_pass_at_1"),
                     "reproduce_sh": r["unconstrained_pass_at_1"]}
        for e in new for m, r in new[e]["results"].items()}
identical = all(v["job3_sweep_copy"] == v["reproduce_sh"] for v in diff.values())
json.dump({"identical": identical, "rows": diff}, open("redteam/hardening/results/repro_check.json", "w"), indent=2)
print("reproduce.sh identical to selected sweep runs:", identical)
PY

rm -rf redteam/hardening/shortcuts_runs/reward_artifact
python -m redteam.hardening.pipeline candidates2,evaluate,shortcuts
