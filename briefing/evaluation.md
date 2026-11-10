# Evaluation

## What a Submission Is

One training run per (experiment, training seed), each in its own directory anywhere under `checkpoints/` (the reference uses `checkpoints/<experiment>/<method>/seed<seed>/`):

- `policy.pt`: a plain `state_dict` of the fixed architecture in `data/policy.py` (loaded with `strict=True`; finite floating-point tensors only).
- `run.json`: `method` (name, `[A-Za-z0-9_.-]`), `experiment` (`stale64` or `offline`), `seed` (one of 7-14), and the declared schedule: `refresh_interval`, `sampling_rounds`, `generated_completions`, `optimizer_updates` (integers).

Nothing else the method writes is read: no predictions, scores, flags or claimed metrics.

## How It Is Scored

`eval/test/evaluate.py` (hidden test, 16,000 prompts that exist only in `eval/test/`) and `eval/train/evaluate.py` (visible slice, 2,000 prompts in `eval/train/val_prompts.json`) apply the same rules:

1. The evaluator loads each `policy.pt` into its own copy of the architecture and decodes greedily itself (four answer tokens), then scores exact-match accuracy.
2. Each run's declared schedule is checked against the experiment's operating point (`stale64`: at most 512 updates, 8 sampling rounds, 131,072 completions, refresh interval at least 64; `offline`: at most 512 updates, 1 round, 131,072 completions). A run outside it, or with an impossible or missing value, fails the method's whole row in that experiment (scored metrics 0).
3. A row is the mean over the eight required training seeds 7-14. A seed with no run, an unloadable checkpoint, or a policy whose greedy outputs repeat another seed's (identical outputs on at least 90% of the prompts where either run errs) contributes 0. Several runs declaring the same seed score the worst of them. Runs declaring any other seed are ignored.
4. Every method found is scored in every experiment; a method with no run in an experiment gets a failed row there.
5. `base_policy` is an evaluator-owned row: the pinned starting policy, evaluated once.

## Metrics

| Metric | Direction | Coefficient | Meaning |
|---|---|---|---|
| `pass_at_1` | higher | 1.0 | seed-mean exact-match accuracy on the slice's prompts |
| `hard_pass_at_1` | higher | 0.5 | the same on prompts needing at least two carries (about half) |
| `unconstrained_pass_at_1` | higher | 0 | diagnostic: accuracy ignoring the operating-point gate |
| `operating_point_ok` | higher | 0 | diagnostic: 1 if every run fits the operating point |

The two experiments have weight 0.5 each. Ranking uses only `scoring/scores_test.json`.

## Reference Numbers (hidden test, mean over seeds 7-14)

| Method | stale64 `pass_at_1` | stale64 `hard` | offline `pass_at_1` | offline `hard` |
|---|---:|---:|---:|---:|
| `base_policy` (no post-training) | 0.360 | 0.356 | 0.360 | 0.356 |
| `grpo` (stale / offline GRPO baseline) | 0.646 | 0.621 | 0.546 | 0.545 |
| `humanline_grpo` (reference) | **0.662** | **0.620** | **0.558** | **0.557** |
| `online_grpo` (samples every update; fails both operating points) | 0 (0.760 unconstrained) | 0 | 0 (0.760 unconstrained) | 0 |

`scoring/reference.json` also lists the two reference ablation rows. On the visible slice the trained methods score within about 0.005 of these (the starting policy: 0.335).

## Noise

Training-seed noise dominates prompt-sampling noise. One run's accuracy has a standard deviation of about 0.015 across seeds for the stale and offline methods (0.035 for the online comparator); an 8-seed mean, about 0.005. Two independent 8-seed sets of the same method differ in the ranked score (weight × coefficient × change) by about 0.0054 in stale64 and 0.0063 in offline, or 0.008 summed over both experiments (one standard deviation). Re-running the reference method with seeds 15-22 or 23-30 changed it by -0.002 and +0.001. Differences that are small relative to this are not evidence of an improvement.

## Protocol

```bash
bash scripts/reproduce.sh       # trains the reference runs (seeds 7-14), writes scoring/scores.json
bash scripts/evaluate_train.sh  # visible slice  -> scoring/scores_train.json
bash scripts/evaluate_test.sh   # hidden test    -> scoring/scores_test.json (no arguments)
```

The evaluators are self-contained (no imports from `data/`, `method/`, or each other) and use the GPU when available; CPU and GPU decoding gave identical scores for the reference runs.
