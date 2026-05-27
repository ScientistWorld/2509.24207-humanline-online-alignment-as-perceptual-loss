# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

## Stop Justification
- Not stopped. A token-level Humanline GRPO job is queued to test whether the actual algorithm supports the core stale-data claim.
