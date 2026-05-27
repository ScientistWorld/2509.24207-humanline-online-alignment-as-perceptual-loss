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

### [2026-05-27 03:50] - core_claim
- Job `1444d95f-1d6` completed successfully.
- The 64-step stale/off-policy setting still supports the core claim: offline humanline GRPO reached `pass_at_1=0.16875` versus `0.15` for stale GRPO at the same generated-data budget.
- Component ablations support the paper's secondary syncing claim: syncing-only matched full humanline, while clipping-only matched the stale baseline.
- The 128-step expansion failed to support the core claim in this rerun, so a 32-step expansion is queued before restoring a higher milestone.

## Stop Justification
- Active job will measure a 32-step stale-refresh expansion before restoring or advancing the milestone.
