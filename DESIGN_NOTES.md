# Train/Test Evaluation Design Notes

## Checked

- Compared `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`; the train wrapper calls `eval/train/evaluate.py` and writes `scoring/scores_train.json`, while the test wrapper calls `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Grepped for cross-imports between `eval/train/` and `eval/test/`; none were present after the edits.
- Checked for paper-agreement metric names and formulas in `scoring/reference.json`; none are used.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
- Ran `bash -n` on both train/test wrapper scripts.

## Changed

- Replaced the first-pass even/odd row split with a stable prompt-hash split. This makes the split independent of prediction row order and closes the easy shortcut where a method can reorder or duplicate rows to favor the visible slice.
- Added minimum slice-size and unique-prompt checks. This forces methods to produce enough method-generated prediction artifacts for both slices instead of stuffing a small set of easy prompts.
- Added train/test metrics that triangulate the scientific quantity beyond headline exact accuracy:
  - `numeric_mae` penalizes near-miss and constant-answer shortcuts.
  - `carry_pass_at_1` checks harder carry prompts separately.
  - `unique_prompt_rate` discourages duplicate prompt stuffing.
  - `evaluated_rows` records slice size as a zero-weight diagnostic.
- Updated `scoring/reference.json` so every metric has `higher_is_better`, `coefficient`, and `description`, and adjusted coefficients to put most weight on answer quality while retaining resource and starting-model constraints.

## Runtime Concerns For Next Runner

- The prompt-hash split is stricter than the previous row-parity split. If a future method emits very few predictions or a narrow prompt distribution, the evaluator now fails instead of returning a misleading score; this is intentional, but the runner should verify normal reproduction artifacts satisfy the `MIN_SLICE_ROWS=32` and `MIN_UNIQUE_PROMPTS=16` checks.
- `carry_pass_at_1` returns `0.0` when a slice contains no carry prompts. With the current arithmetic generation this should not happen, but it is worth checking if the task range is changed.
- The train/test evaluators still rescore existing method-produced JSONL artifacts and do not run inference. If a future method changes output naming, it must still produce either `<run_stem>_predictions.jsonl` or `<method>_predictions.jsonl` next to each run JSON.
