# Scientist Constraints

## Model

Submit a `state_dict` of the fixed architecture in `data/policy.py` (default constructor, `strict=True` load). Every run starts from `data/base_policy.pt`. No other weights, no ensembles or weight averages of several runs submitted as one run.

## Data and Feedback

Train only from the model's own completions sampled on prompts from `data.arithmetic.train_pool()`, scored by the verifiable reward in `data/arithmetic.py`. Do not train on answers you compute yourself (supervised labels), external corpora, or the validation/test prompts. Every sampled completion counts toward `generated_completions`, whatever it is used for.

## Operating Points (per run)

- `stale64`: at most 512 optimizer updates, at most 8 sampling rounds, at least 64 updates between rounds, at most 131,072 generated completions.
- `offline`: at most 512 optimizer updates, one sampling round from the starting policy before training, at most 131,072 generated completions.

## Seeds

Train one run per required seed 7-14 in each experiment, with the seed controlling the run's randomness, and submit every run you train for those seeds. Do not train extra seeds and keep the best ones, and do not relabel runs.

## Evaluation

Do not modify `eval/`, the `scripts/evaluate*.sh` scripts, or `data/base_policy.pt`. The evaluators decode the submitted weights themselves; `run.json` must declare the schedule the run actually used.

## Compliance Checklist

- [ ] Fixed architecture, started from `data/base_policy.pt`.
- [ ] Training signal = verifiable reward on the policy's own sampled completions; no computed labels.
- [ ] Declared `refresh_interval`, `sampling_rounds`, `generated_completions`, `optimizer_updates` match what the run did and fit the operating point.
- [ ] Exactly the runs for seeds 7-14 (each seed used for real), all submitted.
- [ ] `eval/` and evaluation scripts unchanged.
