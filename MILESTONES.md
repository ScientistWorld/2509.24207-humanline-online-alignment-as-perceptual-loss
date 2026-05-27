# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-27 03:41] - core_claim_plus
- Validated the packaged scaled reproduction with `python validate.py --compare`.
- The 64-step stale/off-policy setting supports the core claim: offline humanline GRPO improves Pass@1 and mean reward over stale GRPO with the same generated-example budget.
- The 128-step stale-refresh expansion is included as an additional setting, with evaluation artifacts and reference entries packaged.

### [2026-05-27 03:43] - core_claim
- Continuing audit reset the active milestone to `core_claim`.
- Method and evaluator audit found the core Humanline GRPO computation and evaluator separation intact.
- Added runnable ablations for syncing-only and clipping-only Humanline GRPO to test the paper's secondary component claim in the scaled setting.

## Stop Justification
- Active job will measure secondary ablations before restoring or advancing the milestone.
