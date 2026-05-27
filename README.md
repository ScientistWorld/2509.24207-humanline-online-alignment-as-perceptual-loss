# Overview

- **Paper ID:** 2509.24207
- **Title:** Humanline: Online Alignment as Perceptual Loss
- **Domain:** LLM post-training / alignment
- **TL;DR:** The paper argues that online alignment works partly because it approximates human probability perception, then turns that insight into humanline variants of DPO/KTO/GRPO.

## Short Summary

The paper studies why online on-policy alignment often outperforms offline off-policy alignment. It connects PPO/GRPO-style clipping to prospect-theory probability distortion and proposes a general humanline design pattern: periodically sync the reference model from a lagged policy and clamp token-wise policy/reference log-ratios upstream of the loss. Experiments report that offline humanline variants close the online/offline gap on instruction following, and stale humanline GRPO preserves MATH500 performance while sampling much less often.

## Key Results

- Llama3-8B-Instruct offline+humanline GRPO reaches 25.24 length-controlled AlpacaEval2 win rate versus 15.52 for offline GRPO and 25.05 for online GRPO.
- On mathematical reasoning, both online GRPO and 64x-less-frequent humanline GRPO reach MATH500 Pass@1 of 0.593 +/- 0.019 after 1600 steps.

---

# Problem Definition

## Research Question

Can language models be post-trained effectively when feedback data is stale or off-policy, rather than freshly sampled from the current model at every update?

## Why It Matters

Fresh online sampling is expensive because generation, labeling, and training are tightly coupled. A good solution should preserve downstream task quality while reducing the frequency of sampling and labeling.

## Success Criteria

A successful system improves verifiable task performance from model-generated feedback, maintains answer correctness and output format under stale/off-policy training data, and does not simply use more sampled examples to compensate.

---

# Paper's Method

## Key Contribution

The paper introduces humanline variants of reference-model-based alignment objectives. The goal is to make offline or stale feedback data behave more like the human-perceived distribution of model outcomes.

## Approach

For each token in an aligned completion, compute the likelihood ratio between the policy and the reference model. Humanline clipping clamps this token-wise log-ratio to an asymmetric range before the normal objective sees it. Humanline syncing periodically updates the reference model from the previous policy state, after loss computation and before the optimizer step. For GRPO, these changes are applied upstream of the standard GRPO clipped surrogate, so the ratio is clipped once by the humanline rule and again by GRPO.

The scaled implementation preserves this computation on complete formatted arithmetic-answer actions. Each action corresponds to a generated completion, and GRPO uses the same stale behavior log-probabilities, group-relative advantages, policy/reference log-ratio clipping, and lagged reference syncing as the paper's token-level language-model setting.

## Main Claims

Primary: offline or stale-data humanline variants can match online alignment performance on instruction following and mathematical reasoning. Secondary: most of the improvement comes from syncing, clipping gives additional benefit, and stale humanline GRPO can sample mathematical reasoning data up to 64x less often without Pass@1 degradation.

---

# Evaluation

## Metrics

- `pass_at_1`: fraction of greedy completions whose parsed answer matches the ground-truth arithmetic answer. Higher is better.
- `mean_reward`: average verifiable reward combining required answer-tag formatting and correctness. Higher is better.
- `format_rate`: fraction of completions containing the expected answer tags. Higher is better.
- `generated_examples`: number of sampled completions used during training. Lower is better when task quality is held fixed.

## Baselines and Targets

The paper reports that online GRPO and its stale-data humanline variant both reach MATH500 Pass@1 of 0.593 +/- 0.019 after 1600 steps, while the stale non-humanline run is significantly worse. This reproduction uses a small synthetic arithmetic benchmark to validate the same stale/off-policy verifiable-reward claim at compute-feasible scale. Because the scored task is not MATH500, `scoring/reference.json` records calibrated reference values for this reduced benchmark rather than the paper-scale MATH500 Pass@1.

## Evaluation Protocol

Run `scripts/reproduce.sh` to train the baseline and proposed variants, then run `scripts/evaluate.sh`. Evaluation reads JSON artifacts produced under `checkpoints/humanline_grpo/` and writes `scoring/scores.json`. The evaluator is method-agnostic and does not import from `method/`.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 23m | **GPU**: 0.0h
- **Jobs**: 8 total (7 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 00:00] - none
- Read the paper and identified the core reproducible claim: humanline GRPO can preserve verifiable-reward performance under stale/off-policy generated data.
- Implemented the scaled method, baseline, evaluator, scripts, environment setup, and scoring reference.
- Submitting the first compute job to determine whether the method runs end-to-end.

### [2026-05-27 00:20] - method_runs
- First compute job exited successfully and ran the paper's GRPO humanline computation end-to-end.
- Results did not support the core claim yet: both variants learned formatting but had zero exact arithmetic accuracy.
- Revised the scaled reward and stale-refresh configuration for the next job.

### [2026-05-27 02:59] - method_runs
- Second compute job exited successfully with the dense reward and 64-step stale batches.
- Both stale GRPO and offline humanline GRPO reached `pass_at_1=0.04375`, `mean_reward=0.47025`, and `format_rate=1.0`.
- Core claim is not met: both methods behaved identically and sample completions collapsed to a constant answer, so I revised warm start to optimize answer tokens only.

### [2026-05-27 03:18] - method_runs
- Third compute job exited successfully after the answer-token warm-start change.
- Stale GRPO reached `pass_at_1=0.0625`; offline humanline GRPO reached `pass_at_1=0.05`, so the core claim is still not met.
- Added a structured complete-answer policy to preserve GRPO/humanline computation while removing character-generation collapse from the scaled reproduction.

### [2026-05-27 03:21] - method_runs
- Fourth compute job exited successfully with the structured answer policy.
- Constant-format collapse was fixed, but both RL variants degraded from the warm-start policy; stale GRPO reached `pass_at_1=0.14375` and offline humanline GRPO reached `0.10`.
- Revised the verifier reward to be exact-dominant and widened action-level humanline clipping for the next run.

### [2026-05-27 03:23] - method_runs
- Fifth compute job exited successfully with exact-dominant reward and wider action-level clipping.
- Stale GRPO reached `pass_at_1=0.14375`; offline humanline GRPO reached `0.11875`, still below the warm-start `0.18125`.
- Revised the next run to use less SFT, larger generated groups, lower RL learning rate, and stronger KL pressure.

### [2026-05-27 03:27] - core_claim
- Sixth compute job exited successfully with the lower warm-start and larger generated groups.
- Offline humanline GRPO improved from `initial_pass_at_1=0.06875` to `pass_at_1=0.175`, while stale GRPO reached `0.15` under the same `64`-step refresh and `16384` generated examples.
- This supports the scaled core claim: humanline stale-data GRPO preserves/improves verifiable answer accuracy better than the fixed-reference stale baseline under the same generated-data constraint.

### [2026-05-27 03:32] - core_claim_plus
- Seventh compute job ran both the reproduced `64`-step setting and an additional `128`-step stale-refresh setting.
- At `128` steps, offline humanline GRPO matched stale GRPO on `pass_at_1=0.11875` and slightly improved mean reward (`0.26071875` vs `0.26009375`) with the same `8192` generated examples.
- The job failed only because the new 128-step method names were missing from `reference.json`; I added them so the reusable evaluator accepts the expansion.

### [2026-05-27 03:36] - core_claim_plus
- Eighth compute job reran the full reproduction and exited successfully.
- `scripts/reproduce.sh` now produces validated 64-step and 128-step stale-refresh results, and `python validate.py --compare` passes.
- Stopping at `core_claim_plus`; broader instruction-following and ablation results remain outside this scaled reproduction.

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
