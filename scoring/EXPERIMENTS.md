# Experiment Overview

## synthetic_math_stale_grpo

Tests whether a post-training approach remains effective when generated feedback data is reused for multiple updates instead of being refreshed at every step. The experiment measures answer correctness, formatting reliability, and generated-data usage on a small verifiable arithmetic task.

The default reproduction includes a fresh-data comparator, a 64-update stale-data setting, a harder 128-update stale-data diagnostic, diagnostic variants, and several refresh schedules. All scored methods use the same generated-example budget in the main setting, so improvements should come from better use of stale feedback rather than from more sampling. The diagnostic entries are retained to show whether a result is robust beyond one refresh configuration; the primary interpretation is still whether answer quality can be maintained while generated feedback is reused for many updates.
