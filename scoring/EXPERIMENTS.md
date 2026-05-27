# Experiment Overview

## synthetic_math_stale_grpo

Tests whether a post-training approach remains effective when generated feedback data is reused for multiple updates instead of being refreshed at every step. The experiment measures answer correctness, formatting reliability, and generated-data usage on a small verifiable arithmetic task.

The default reproduction includes an online-refresh comparator, the paper-aligned 64-step stale refresh setting, a harder 128-step refresh expansion, a more frequently refreshed 32-step expansion, component ablations that isolate reference syncing from upstream ratio clipping, and a 64-step sync-cadence sweep. The 64-step setting compares the same generated-data budget across online refresh, stale GRPO, full humanline GRPO, syncing-only humanline GRPO, clipping-only humanline GRPO, and sync-cadence variants. The 32-step and 128-step expansions are retained as diagnostic settings: they measure whether the effect persists as the generated-feedback refresh cadence changes, even when a particular expansion does not favor the proposed method.
