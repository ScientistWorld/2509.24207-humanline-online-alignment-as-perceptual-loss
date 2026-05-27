# Experiment Overview

## synthetic_math_stale_grpo

Tests whether a post-training approach remains effective when generated feedback data is reused for multiple updates instead of being refreshed at every step. The experiment measures answer correctness, formatting reliability, and generated-data usage on a small verifiable arithmetic task.

The default reproduction includes the paper-aligned 64-step stale refresh setting and a harder 128-step refresh expansion. Both settings compare the same generated-data budget within each refresh interval.
