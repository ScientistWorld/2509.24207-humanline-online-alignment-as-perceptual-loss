# Research Direction

## Research Problem

Keep post-training from verifiable feedback effective when the training completions are stale (sampled every 64 updates) or offline (sampled once, before training), at a fixed number of updates and completions.

## Core Contribution

Work must improve how a model learns from completions sampled by an earlier version of itself: the objective, how stale or off-policy samples are weighted, selected or reused, the reference model, regularization, the optimizer schedule, or how the sampling budget is spread across prompts and rounds within the operating point.

## Approach Scope

Anything inside one training run that respects the operating point: e.g. changes to the loss, ratio corrections, clipping, reference-model handling, group construction, advantage estimation, which prompts each round samples, temperature, learning-rate schedule, hyperparameters.

## Out of Bounds

- Supervised training on answers the method computes itself (the task is addition, so labels are trivially available; using them bypasses the research question).
- Sampling more often, or generating more completions, than the operating point allows, whatever `run.json` declares.
- Selecting among extra training seeds, relabelling seeds, or submitting one run under several seeds.
- Changing the architecture, the starting policy, or the evaluators.
