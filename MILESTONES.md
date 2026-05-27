# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

## Stop Justification
- Not stopped. A corrected token-level Humanline GRPO job is queued on a harder arithmetic setting to test whether the actual algorithm supports the core stale-data claim.
