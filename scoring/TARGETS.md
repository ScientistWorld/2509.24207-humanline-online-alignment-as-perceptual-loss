# Evaluation Targets

## Primary: synthetic_math_stale_grpo

Tests whether a model trained from stale generated groups can still improve verifiable arithmetic answer accuracy.
- **Metric**: `pass_at_1` (higher is better)
- **Run**: `bash scripts/reproduce.sh`, then inspect `scoring/scores.json`
- **Expansion**: `scripts/reproduce.sh` also runs an optional 128-step stale refresh setting by default; disable with `RUN_EXTRA_REFRESH=0`.

## Constraint: synthetic_math_stale_grpo

Tests whether quality is achieved without simply increasing the amount of generated training data.
- **Metric**: `generated_examples` (lower is better when answer quality is comparable)
- **Run**: `bash scripts/evaluate.sh --runs-dir checkpoints/humanline_grpo`

## Ablation: synthetic_math_stale_grpo

Tests whether Humanline's gain is primarily attributable to reference syncing rather than clipping alone.
- **Metric**: `pass_at_1` and `mean_reward`
- **Run**: `RUN_ABLATIONS=1 bash scripts/reproduce.sh`
