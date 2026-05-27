# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-27 03:41] - core_claim_plus
- Validated the packaged scaled reproduction with `python validate.py --compare`.
- The 64-step stale/off-policy setting supports the core claim: offline humanline GRPO improves Pass@1 and mean reward over stale GRPO with the same generated-example budget.
- The 128-step stale-refresh expansion is included as an additional setting, with evaluation artifacts and reference entries packaged.

## Stop Justification
- Completed at milestone `core_claim_plus`.
