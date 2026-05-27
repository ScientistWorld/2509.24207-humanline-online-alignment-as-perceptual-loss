# Reproduction Milestones

**Current: secondary_claims**

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

### [2026-05-27 04:00] - core_claim
- Job `b4150f04-d9c` completed successfully and evaluated the 32-step stale-refresh expansion.
- The 32-step setting improved absolute accuracy but did not support a Humanline advantage: stale GRPO reached `pass_at_1=0.2375`, while offline humanline GRPO reached `0.21875`.
- Added the 32-step results to `reference.json` and queued a 64-step Humanline sync-frequency sweep before considering any milestone above `core_claim`.

### [2026-05-27 04:03] - secondary_claims
- Job `639ed704-46f` completed successfully and evaluated 64-step Humanline sync-frequency variants.
- Sync cadence variants `sync8`, `sync16`, and `sync24` all beat the stale 64-step baseline at the same generated-data budget; `sync20` tied the baseline.
- Together with the completed syncing-only and clipping-only ablations, the reduced reproduction now supports the core stale-data claim plus the paper's secondary component claim that reference syncing is the important Humanline component in this setting.

## Stop Justification
- Reached `secondary_claims` on the scaled arithmetic reproduction. Remaining paper sections, especially instruction-following judge experiments, would require a different large-model evaluation stack beyond the current reduced gym.
