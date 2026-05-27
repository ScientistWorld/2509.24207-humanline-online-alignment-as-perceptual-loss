# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification
- Not stopped.
