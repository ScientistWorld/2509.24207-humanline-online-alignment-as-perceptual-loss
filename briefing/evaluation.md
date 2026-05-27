# Evaluation

## Metrics

- `pass_at_1`: fraction of greedy completions whose parsed answer matches the ground-truth arithmetic answer. Higher is better.
- `mean_reward`: average verifiable reward combining required answer-tag formatting and correctness. Higher is better.
- `format_rate`: fraction of completions containing the expected answer tags. Higher is better.
- `numeric_mae`: mean absolute numeric error after parsing an answer. Lower is better.
- `carry_pass_at_1`: exact-answer accuracy on prompts whose ones digits require carrying. Higher is better.
- `generated_examples`: number of sampled completions used during training. Lower is better when task quality is held fixed.

## Baselines and Targets

The original paper evaluates stale-feedback post-training on mathematical reasoning. This gym uses a small deterministic arithmetic benchmark to test the same research question at compute-feasible scale. Because the scored task is not the paper's full mathematical-reasoning benchmark, `scoring/reference.json` records calibrated reference values for this reduced benchmark and includes notes documenting that scale change.

## Evaluation Protocol

Run `scripts/reproduce.sh` to regenerate the reference run artifacts, then run `scripts/evaluate.sh`. Evaluation reads run metadata and prediction JSONL files produced under `checkpoints/humanline_grpo/` and writes `scoring/scores.json`. The evaluator is method-agnostic and does not import from `method/`.

Any competing method should write one JSON metadata file and one matching `*_predictions.jsonl` file per method under the evaluated run directory. Each prediction row must include `prompt`, `answer`, and `completion`.
