# Evaluation

## Metrics

- `pass_at_1`: fraction of greedy completions whose parsed answer matches the ground-truth arithmetic answer. Higher is better.
- `mean_reward`: average verifiable reward combining required answer-tag formatting and correctness. Higher is better.
- `format_rate`: fraction of completions containing the expected answer tags. Higher is better.
- `generated_examples`: number of sampled completions used during training. Lower is better when task quality is held fixed.

## Baselines and Targets

The paper reports that online GRPO and its stale-data humanline variant both reach MATH500 Pass@1 of 0.593 +/- 0.019 after 1600 steps, while the stale non-humanline run is significantly worse. This reproduction uses a small synthetic arithmetic benchmark to validate the same stale/off-policy verifiable-reward claim at compute-feasible scale. Because the scored task is not MATH500, `scoring/reference.json` records calibrated reference values for this reduced benchmark rather than the paper-scale MATH500 Pass@1.

## Evaluation Protocol

Run `scripts/reproduce.sh` to train the baseline and proposed variants, then run `scripts/evaluate.sh`. Evaluation reads JSON artifacts produced under `checkpoints/humanline_grpo/` and writes `scoring/scores.json`. The evaluator is method-agnostic and does not import from `method/`.

Set `RUN_ABLATIONS=1` when running `scripts/reproduce.sh` to include syncing-only and clipping-only Humanline GRPO ablations for the component claim.
