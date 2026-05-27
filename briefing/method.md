# Paper's Method

## Key Contribution

The paper introduces humanline variants of reference-model-based alignment objectives. The goal is to make offline or stale feedback data behave more like the human-perceived distribution of model outcomes.

## Approach

For each token in an aligned completion, compute the likelihood ratio between the policy and the reference model. Humanline clipping clamps this token-wise log-ratio to an asymmetric range before the normal objective sees it. Humanline syncing periodically updates the reference model from the previous policy state, after loss computation and before the optimizer step. For GRPO, these changes are applied upstream of the standard GRPO clipped surrogate, so the ratio is clipped once by the humanline rule and again by GRPO.

The active scaled implementation preserves this computation with a small token-level policy that generates formatted arithmetic completions. The default reduced-benchmark policy is constrained to valid answer-tag strings and exposes autoregressive log-probabilities for answer characters; the earlier unconstrained character GRU remains available with `--policy-type gru` but was too weak to learn prompt-conditioned arithmetic in the CPU budget. Training reuses stale generated completion pools, computes group-relative advantages from verifiable rewards, forms the GRPO token ratio from policy/reference log-probabilities as in the paper appendix, applies optional Humanline log-ratio clipping upstream of the GRPO surrogate, and syncs a lagged reference model from the policy.

## Main Claims

Primary: offline or stale-data humanline variants can match online alignment performance on instruction following and mathematical reasoning. Secondary: most of the improvement comes from syncing, clipping gives additional benefit, and stale humanline GRPO can sample mathematical reasoning data up to 64x less often without Pass@1 degradation.
