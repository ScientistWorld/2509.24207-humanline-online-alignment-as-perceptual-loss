# Scoring audit

Run `bash redteam/run_all.sh` as a CPU job in the gym container. No packages or
network are needed beyond the original environment. Each directory has a
`run.sh` that installs its attempt, runs it, runs the visible evaluation, then
executes exactly `bash scripts/evaluate_test.sh` with no arguments in a new
process. Each case retains `scores_train.json`, `scores_test.json`, `rank.json`,
`result.json`, and the submitted JSON/JSONL artifacts. Rejected cases retain
`test_error.txt` instead of claiming a score. Large model weights and logs are
ignored; training can regenerate weights. Delete a case's `result.json` to
rerun it through the bundle, or call its `run.sh` directly to rerun immediately.

First run `reference_seed7/run.sh`. Shortcut producers reuse that genuinely
trained run and only use future-agent-visible task definitions and artifacts;
they never import test code or read test scores. The harness preserves the
other original comparator rows for a like-for-like rank calculation. The
original score snapshot and rank coefficients are used only by the post-run
audit summarizer, not by shortcut methods.

All training variants use the original algorithm and model with the changes
in their `config.json`. Training seeds 7, 8, and 9 share evaluation prompts
from the default seed 10007. The reference uses the original procedure exactly.
Each evaluator invocation has its own Python process; no monkey patches,
background processes, import-path changes, or environment modifications carry
into hidden evaluation. Only files written by each attempt reach evaluation.
Evaluator source hashes are recorded and checked in each `result.json`.
The 22 cases were bundled in two CPU job containers. Each evaluation has a fresh
child interpreter; there was not a separate container launch for each case.

`_workspace_backup/` is ignored harness state. The harness restores run
artifacts, the task definition, and original scores between cases and after
the bundle. No changes to evaluator sources are made. See `../REDTEAM.md` for
the completed audit findings, including eight paying shortcuts and the inverted
quality ordering caused by the raw generated-example coefficient.

`python3 redteam/verify_evidence.py` performs only lightweight syntax, recorded
rank, hash, and restoration checks. Its final output is saved in
`verification.json`. It does not run training or either evaluator.
