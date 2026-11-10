# Evaluation Targets

Every target is scored by `scripts/evaluate_test.sh` into `scoring/scores_test.json` (visible slice: `scripts/evaluate_train.sh` → `scoring/scores_train.json`). Rows are means over training seeds 7-14. Reproduce the reference with `bash scripts/reproduce.sh`.

## Primary: stale64 — `pass_at_1` (higher is better, coefficient 1.0, weight 0.5)

Held-out exact-match accuracy when completions are sampled at most every 64 updates (512 updates, ≤ 8 rounds, ≤ 131,072 completions per run).
- Reference `humanline_grpo`: 0.662. Baseline `grpo` (64x-stale GRPO): 0.646. Starting policy: 0.360.
- Comparator: `online_grpo` (sampling every update, same completion volume) reaches 0.760 but violates the operating point, so it is scored 0.
- Paper claim tested: the stale humanline variant improves on stale GRPO toward online GRPO.

## Primary: offline — `pass_at_1` (higher is better, coefficient 1.0, weight 0.5)

Held-out accuracy with a single sampling round from the starting policy (512 updates, 1 round, ≤ 131,072 completions per run).
- Reference `humanline_grpo`: 0.558. Baseline `grpo` (offline GRPO, fixed reference, eps 0.5): 0.546.
- Paper claim tested: the offline humanline variant improves on offline GRPO.

## Secondary: both experiments — `hard_pass_at_1` (higher is better, coefficient 0.5)

Accuracy on the hidden prompts whose addition needs at least two carries (the harder half), so gains concentrated on easy prompts count less. Reference: 0.620 (stale64), 0.557 (offline).

## Constraint: both experiments — `operating_point_ok` (gate, coefficient 0)

Every run's declared schedule must fit the experiment's budget (refresh interval, sampling rounds, generated completions, optimizer updates). A violation or an impossible value in any run zeroes the method's scored metrics in that experiment; `unconstrained_pass_at_1` (coefficient 0) shows the accuracy it would have had.

## Diagnostics (in `scoring/reference.json`, not targets)

`humanline_sync_only` and `humanline_clip_only` ablate the two components of the reference method (offline: 0.557 and 0.473; stale64: 0.655 and 0.655).
