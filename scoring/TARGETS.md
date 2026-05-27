# Evaluation Targets

## Primary: synthetic_math_stale_grpo

Tests whether a model trained from stale generated groups can still improve verifiable arithmetic answer accuracy.
- **Metric**: `pass_at_1` (higher is better)
- **Run**: `bash scripts/reproduce.sh`, then inspect `scoring/scores.json`

## Constraint: synthetic_math_stale_grpo

Tests whether quality is achieved without simply increasing the amount of generated training data.
- **Metric**: `generated_examples` (lower is better when answer quality is comparable)
- **Run**: `bash scripts/evaluate.sh --runs-dir checkpoints/humanline_grpo`
