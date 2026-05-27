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

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

## Stop Justification
- Not stopped. A structured token-level Humanline GRPO job is queued with online-refresh, 64-step stale, 64-step Humanline, and 128-step expansion runs.
