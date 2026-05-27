# Evaluation Targets

## Primary: synthetic_math_stale_grpo

Tests whether a model trained from stale generated groups can still improve verifiable arithmetic answer accuracy.
- **Metric**: `pass_at_1` (higher is better)
- **Run**: `bash scripts/reproduce.sh`, then inspect `scoring/scores.json`
- **Comparator**: the fresh-data baseline refreshes generated groups every update; the stale-data methods should remain close to that comparator while reusing generated groups for 64 updates.
- **Expansion**: `scripts/reproduce.sh` also runs an optional 128-step stale refresh setting by default; disable with `RUN_EXTRA_REFRESH=0`.
- **Schedule sweep**: `scripts/reproduce.sh` runs 64-step schedule variants by default; at least one variant should improve over the default stale-data baseline. Disable with `RUN_SYNC_SWEEP=0`.

## Constraint: synthetic_math_stale_grpo

Tests whether quality is achieved without simply increasing the amount of generated training data.
- **Metric**: `generated_examples` (lower is better when answer quality is comparable)
- **Run**: `bash scripts/evaluate.sh --runs-dir checkpoints/humanline_grpo`

## Ablation: synthetic_math_stale_grpo

Tests whether the reproduced method's component choices matter beyond the generic stale-data setup.
- **Metric**: `pass_at_1` and `mean_reward`
- **Run**: `RUN_ABLATIONS=1 bash scripts/reproduce.sh`
- **Status**: included for scientific completeness; in the active reduced benchmark this ablation does not support the paper's component-attribution claim.
