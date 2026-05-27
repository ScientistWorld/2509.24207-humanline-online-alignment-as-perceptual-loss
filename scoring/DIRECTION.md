# Research Direction

## Research Problem

The benchmark asks how to keep post-training effective when model-generated feedback data is stale or off-policy. Improvements should address alignment under reduced sampling frequency.

## Core Contribution

Work must remain focused on post-training generated-text models from verifiable feedback when the feedback data may be stale or off-policy.

## Approach Scope

Changes may modify the learning objective, feedback weighting, reward shaping, data-refresh schedule, or training schedule, as long as training still uses generated completions with verifiable feedback and stale/off-policy reuse remains central.

## Out of Bounds

- Replacing the task with supervised answer memorization.
- Using a deterministic calculator or parser in generation.
- Solving the benchmark by refreshing data every step while claiming stale-data performance.
- Switching to a larger pretrained model to bypass the post-training question.
