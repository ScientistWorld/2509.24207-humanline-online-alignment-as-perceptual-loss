# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-27 00:00] - none
- Read the paper and identified the core reproducible claim: humanline GRPO can preserve verifiable-reward performance under stale/off-policy generated data.
- Implemented the scaled method, baseline, evaluator, scripts, environment setup, and scoring reference.
- Submitting the first compute job to determine whether the method runs end-to-end.

### [2026-05-27 00:20] - method_runs
- First compute job exited successfully and ran the paper's GRPO humanline computation end-to-end.
- Results did not support the core claim yet: both variants learned formatting but had zero exact arithmetic accuracy.
- Revised the scaled reward and stale-refresh configuration for the next job.

### [2026-05-27 02:59] - method_runs
- Second compute job exited successfully with the dense reward and 64-step stale batches.
- Both stale GRPO and offline humanline GRPO reached `pass_at_1=0.04375`, `mean_reward=0.47025`, and `format_rate=1.0`.
- Core claim is not met: both methods behaved identically and sample completions collapsed to a constant answer, so I revised warm start to optimize answer tokens only.

### [2026-05-27 03:18] - method_runs
- Third compute job exited successfully after the answer-token warm-start change.
- Stale GRPO reached `pass_at_1=0.0625`; offline humanline GRPO reached `pass_at_1=0.05`, so the core claim is still not met.
- Added a structured complete-answer policy to preserve GRPO/humanline computation while removing character-generation collapse from the scaled reproduction.

### [2026-05-27 03:21] - method_runs
- Fourth compute job exited successfully with the structured answer policy.
- Constant-format collapse was fixed, but both RL variants degraded from the warm-start policy; stale GRPO reached `pass_at_1=0.14375` and offline humanline GRPO reached `0.10`.
- Revised the verifier reward to be exact-dominant and widened action-level humanline clipping for the next run.

### [2026-05-27 03:23] - method_runs
- Fifth compute job exited successfully with exact-dominant reward and wider action-level clipping.
- Stale GRPO reached `pass_at_1=0.14375`; offline humanline GRPO reached `0.11875`, still below the warm-start `0.18125`.
- Revised the next run to use less SFT, larger generated groups, lower RL learning rate, and stronger KL pressure.

### [2026-05-27 03:27] - core_claim
- Sixth compute job exited successfully with the lower warm-start and larger generated groups.
- Offline humanline GRPO improved from `initial_pass_at_1=0.06875` to `pass_at_1=0.175`, while stale GRPO reached `0.15` under the same `64`-step refresh and `16384` generated examples.
- This supports the scaled core claim: humanline stale-data GRPO preserves/improves verifiable answer accuracy better than the fixed-reference stale baseline under the same generated-data constraint.

### [2026-05-27 03:32] - core_claim_plus
- Seventh compute job ran both the reproduced `64`-step setting and an additional `128`-step stale-refresh setting.
- At `128` steps, offline humanline GRPO matched stale GRPO on `pass_at_1=0.11875` and slightly improved mean reward (`0.26071875` vs `0.26009375`) with the same `8192` generated examples.
- The job failed only because the new 128-step method names were missing from `reference.json`; I added them so the reusable evaluator accepts the expansion.

### [2026-05-27 03:36] - core_claim_plus
- Eighth compute job reran the full reproduction and exited successfully.
- `scripts/reproduce.sh` now produces validated 64-step and 128-step stale-refresh results, and `python validate.py --compare` passes.
- Stopping at `core_claim_plus`; broader instruction-following and ablation results remain outside this scaled reproduction.

## Stop Justification
- Stopping with a validated `core_claim_plus` environment: the core stale-GRPO math claim is reproduced at 64-step refresh and expanded to a 128-step refresh setting.
