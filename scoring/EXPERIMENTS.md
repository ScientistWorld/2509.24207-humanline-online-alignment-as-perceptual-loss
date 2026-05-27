# Experiment Overview

## synthetic_math_stale_grpo

Tests whether a post-training approach remains effective when generated feedback data is reused for multiple updates instead of being refreshed at every step. The experiment measures answer correctness, formatting reliability, and generated-data usage on a small verifiable arithmetic task.

The default reproduction includes a fresh-data comparator, a 64-update stale-data setting, a harder 128-update stale-data diagnostic, component ablations from the reproduced method, and several schedule variants. All scored methods use the same generated-example budget in the main setting, so improvements should come from better use of stale feedback rather than from more sampling. The 128-update setting is retained as a diagnostic because it is harder in this reduced benchmark; one schedule variant provides the additional positive setting counted toward `core_claim_plus`.
