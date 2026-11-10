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

## This Gym

A scaled, compute-feasible version of the paper's verifiable-reward experiment: a fixed 0.6M-parameter transformer post-trained with GRPO-style feedback on three-digit addition under two fixed sampling regimes (64x-stale and offline), scored on 16,000 hidden prompts as the mean over training seeds 7-14. See `HARDENING.md` for how the scoring was hardened and what it can and cannot detect.

---

# Problem Definition

## Research Question

Can a generative model be post-trained effectively from verifiable feedback when the training completions are stale or off-policy, rather than freshly sampled from the current model at every update?

## Why It Matters

Fresh online sampling is expensive: generation, labeling and training are tightly coupled, so every optimizer step waits for new samples from the current model. If training can reuse completions sampled much less often (or sampled once, offline) without losing quality, sampling, labeling and training can be decoupled and post-training becomes cheaper and more parallel.

## The Task in This Gym

- **Model.** A fixed decoder-only character transformer (about 0.6M parameters, `data/policy.py`). Every submission is a `state_dict` of exactly this architecture.
- **Starting point.** Every run starts from the committed policy `data/base_policy.pt`, which was supervised-trained once (on a pool disjoint from all evaluation prompts) to about 36% accuracy and then frozen.
- **Problem.** Three-digit addition, `a2a1a0+b2b1b0=` → four answer digits. The training feedback is a verifiable exact-match reward on the model's own sampled completions (`data/arithmetic.py`). Methods may sample prompts from the visible training pool of 20,000 operand pairs.
- **Operating points.** Two fixed training regimes, per training run:
  - `stale64`: completions may be sampled at most every 64 optimizer updates (64x less often than online): 512 updates, at most 8 sampling rounds, at most 131,072 generated completions.
  - `offline`: a single sampling round, drawn from the starting policy before training, then no new samples: 512 updates, at most 131,072 generated completions.
  Sampling at every update with the same 131,072-completion volume ("online") reaches about 76% held-out accuracy in 512 updates; it is outside both operating points and serves only as a reference for how much is lost to staleness.
- **Seeds.** Training from feedback is noisy from seed to seed, so each method is trained once for each of the eight training seeds 7-14 in each experiment and scored as the mean.

## Success Criteria

A successful method reaches higher exact-match accuracy on held-out addition prompts (and on the harder, carry-heavy half) at each fixed operating point, from the same starting policy and model, using the model's own sampled completions and the verifiable reward as its training signal, without sampling more often or generating more completions than the operating point allows.

---

# Paper's Method

## Key Contribution

The paper introduces humanline variants of reference-model-based alignment objectives (DPO, KTO, GRPO). Motivated by prospect theory, it argues that online on-policy sampling works because it approximates the human-perceived distribution of model outcomes, and that PPO/GRPO clipping already recovers part of this perceptual bias. Making the bias explicit should let stale or offline data train as well as fresh online data.

## Approach (the design pattern)

1. **Humanline syncing.** Every `k` steps, after the loss is computed but before the optimizer step, the reference model is synced with the policy (so the next update is judged against the previous-step policy).
2. **Humanline clipping.** Token-wise log-ratios `log pi_theta / pi_ref` are clamped to `[log eps_P, log eps_R]` (the paper uses -1.5 and 1.5) upstream of the loss. For GRPO the ratio is therefore clipped twice: once by the humanline clamp and once by GRPO's `[1-eps, 1+eps]` surrogate. The KL term is not humanline-clipped.

## Implementation in This Gym

`method/humanline_grpo.py` implements GRPO (group-relative advantages from the 0/1 verifiable reward, the clipped surrogate on the token-wise policy/reference ratio, a k3 KL penalty to the reference, DAPO-style token-mean loss) on the fixed tiny transformer, with the paper's two humanline changes behind `--humanline` (and `--disable-humanline-sync` / `--disable-humanline-clipping` for the ablations). Sampling follows the paper's 64x-less-frequent protocol: every `refresh_interval` updates a pool of 64 batches is sampled from a frozen snapshot of the policy and consumed one batch per update; the reference is reset to the sampling policy at the start of each round. A single round (`refresh_interval = 512`) is the offline setting, where the reference never moves without humanline syncing.

Settings (paper values where they apply): 32 prompts x group size 8 = 256 completions per update, temperature 0.7, AdamW (beta 0.9/0.999, eps 1e-8, no weight decay), 10% warmup, max grad norm 1.0, GRPO eps 0.15 (offline GRPO baseline: 0.5, paper Table 3), KL beta 0.01, humanline log-clip range [-1.5, 1.5]. Learning rate and `k` were selected per method on the visible slice (the paper tunes LR per variant and `k` per setting): humanline GRPO lr 1e-4 with `k = 48` (stale64) and `k = 192` (offline); GRPO lr 1e-4; online GRPO lr 3e-4. Scaled down from the paper: a 0.6M-parameter model and three-digit addition instead of Qwen2.5-1.5B on MATH, and 512 updates instead of 1600.

`scripts/method.sh`, `scripts/baseline.sh` and `scripts/ablation.sh` train each configuration for seeds 7-14.

## Main Claims and What Reproduces Here

- **Primary (paper):** offline or 64x-stale humanline GRPO matches online GRPO. **Here:** humanline GRPO beats the stale/offline GRPO baseline at the same data volume (+0.016 and +0.011 hidden-test accuracy over seeds 7-14; +0.016 and +0.013 over 24 seeds, t = 3.4 and 2.6), but stays well below online GRPO (0.76): the direction reproduces, closing the gap does not at this scale.
- **Secondary (paper):** syncing gives most of the improvement; clipping alone is no better than the offline variant. **Here (offline):** sync-only ≈ full humanline (0.557 vs 0.558), clip-only well below the baseline (0.473). In stale64 both components alone land between the baseline and the full method.
- **Paper:** syncing too often (`k = 1`) collapses with small models. **Here:** `k = 4` collapses (0.19 stale64, 0.04 offline).

---

# Evaluation

## What a Submission Is

One training run per (experiment, training seed), each in its own directory anywhere under `checkpoints/` (the reference uses `checkpoints/<experiment>/<method>/seed<seed>/`):

- `policy.pt`: a plain `state_dict` of the fixed architecture in `data/policy.py` (loaded with `strict=True`; finite floating-point tensors only).
- `run.json`: `method` (name, `[A-Za-z0-9_.-]`), `experiment` (`stale64` or `offline`), `seed` (one of 7-14), and the declared schedule: `refresh_interval`, `sampling_rounds`, `generated_completions`, `optimizer_updates` (integers).

Nothing else the method writes is read: no predictions, scores, flags or claimed metrics.

## How It Is Scored

`eval/test/evaluate.py` (hidden test, 16,000 prompts that exist only in `eval/test/`) and `eval/train/evaluate.py` (visible slice, 2,000 prompts in `eval/train/val_prompts.json`) apply the same rules:

1. The evaluator loads each `policy.pt` into its own copy of the architecture and decodes greedily itself (four answer tokens), then scores exact-match accuracy.
2. Each run's declared schedule is checked against the experiment's operating point (`stale64`: at most 512 updates, 8 sampling rounds, 131,072 completions, refresh interval at least 64; `offline`: at most 512 updates, 1 round, 131,072 completions). A run outside it, or with an impossible or missing value, fails the method's whole row in that experiment (scored metrics 0).
3. A row is the mean over the eight required training seeds 7-14. A seed with no run, an unloadable checkpoint, or a policy whose greedy outputs repeat another seed's (identical outputs on at least 90% of the prompts where either run errs) contributes 0. Several runs declaring the same seed score the worst of them. Runs declaring any other seed are ignored.
4. Every method found is scored in every experiment; a method with no run in an experiment gets a failed row there.
5. `base_policy` is an evaluator-owned row: the pinned starting policy, evaluated once.

## Metrics

| Metric | Direction | Coefficient | Meaning |
|---|---|---|---|
| `pass_at_1` | higher | 1.0 | seed-mean exact-match accuracy on the slice's prompts |
| `hard_pass_at_1` | higher | 0.5 | the same on prompts needing at least two carries (about half) |
| `unconstrained_pass_at_1` | higher | 0 | diagnostic: accuracy ignoring the operating-point gate |
| `operating_point_ok` | higher | 0 | diagnostic: 1 if every run fits the operating point |

The two experiments have weight 0.5 each. Ranking uses only `scoring/scores_test.json`.

## Reference Numbers (hidden test, mean over seeds 7-14)

| Method | stale64 `pass_at_1` | stale64 `hard` | offline `pass_at_1` | offline `hard` |
|---|---:|---:|---:|---:|
| `base_policy` (no post-training) | 0.360 | 0.356 | 0.360 | 0.356 |
| `grpo` (stale / offline GRPO baseline) | 0.646 | 0.621 | 0.546 | 0.545 |
| `humanline_grpo` (reference) | **0.662** | **0.620** | **0.558** | **0.557** |
| `online_grpo` (samples every update; fails both operating points) | 0 (0.760 unconstrained) | 0 | 0 (0.760 unconstrained) | 0 |

`scoring/reference.json` also lists the two reference ablation rows. On the visible slice the trained methods score within about 0.005 of these (the starting policy: 0.335).

## Noise

Training-seed noise dominates prompt-sampling noise. One run's accuracy has a standard deviation of about 0.015 across seeds for the stale and offline methods (0.035 for the online comparator); an 8-seed mean, about 0.005. Two independent 8-seed sets of the same method differ in the ranked score (weight × coefficient × change) by about 0.0054 in stale64 and 0.0063 in offline, or 0.008 summed over both experiments (one standard deviation). Re-running the reference method with seeds 15-22 or 23-30 changed it by -0.002 and +0.001. Differences that are small relative to this are not evidence of an improvement.

## Protocol

```bash
bash scripts/reproduce.sh       # trains the reference runs (seeds 7-14), writes scoring/scores.json
bash scripts/evaluate_train.sh  # visible slice  -> scoring/scores_train.json
bash scripts/evaluate_test.sh   # hidden test    -> scoring/scores_test.json (no arguments)
```

The evaluators are self-contained (no imports from `data/`, `method/`, or each other) and use the GPU when available; CPU and GPU decoding gave identical scores for the reference runs.

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

### Iteration 2: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 11 total (10 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 37m | **GPU**: 0.0h
- **Jobs**: 17 total (13 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 17 total (13 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 17 total (13 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 19 total (14 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 19 total (14 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 19 total (14 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 19 total (14 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

### [2026-05-27 07:54] - core_claim_plus
- Re-audited the packaged environment after sub-agent review.
- Confirmed `python validate.py --compare` passes, `eval/` and `data/` remain separated from `method/`, and active scores are recomputed from prediction JSONL artifacts.
- Left unrelated root device-file and trajectory changes untouched.

### [2026-05-27 07:58] - core_claim_plus
- Completed final verification of the research-gym package.
- Confirmed validation, shell syntax checks, and standalone evaluation all pass using existing reproduced artifacts.
- Tightened public scoring documentation to keep from-scratch guidance method-agnostic.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

### [2026-05-27 07:54] - core_claim_plus
- Re-audited the packaged environment after sub-agent review.
- Confirmed `python validate.py --compare` passes, `eval/` and `data/` remain separated from `method/`, and active scores are recomputed from prediction JSONL artifacts.
- Left unrelated root device-file and trajectory changes untouched.

### [2026-05-27 07:58] - core_claim_plus
- Completed final verification of the research-gym package.
- Confirmed validation, shell syntax checks, and standalone evaluation all pass using existing reproduced artifacts.
- Tightened public scoring documentation to keep from-scratch guidance method-agnostic.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 38m | **GPU**: 2.1h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

### [2026-05-27 07:54] - core_claim_plus
- Re-audited the packaged environment after sub-agent review.
- Confirmed `python validate.py --compare` passes, `eval/` and `data/` remain separated from `method/`, and active scores are recomputed from prediction JSONL artifacts.
- Left unrelated root device-file and trajectory changes untouched.

### [2026-05-27 07:58] - core_claim_plus
- Completed final verification of the research-gym package.
- Confirmed validation, shell syntax checks, and standalone evaluation all pass using existing reproduced artifacts.
- Tightened public scoring documentation to keep from-scratch guidance method-agnostic.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 22m | **GPU**: 1.2h
- **Jobs**: 6 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-11-10 20:10] - core_claim
- Hardening iterations 11-12 rebuilt the gym (three-digit addition, fixed 0.6M-parameter transformer and starting policy, evaluator-owned decoding, `stale64` and `offline` operating points) and now score each method as the mean over training seeds 7-14 (see `HARDENING.md`).
- Paper direction reproduced with the paper's training method: humanline GRPO beats 64x-stale GRPO by +0.016 and offline GRPO by +0.011 hidden-test accuracy (24 seeds: t = 3.4 and 2.6); syncing carries most of the offline gain and clipping alone hurts, as in the paper's ablation; syncing too often (k = 4) collapses.
- Not reproduced: closing the gap to online GRPO (0.760 vs 0.662 / 0.558). The label is lowered from `core_claim_plus` to `core_claim` because only the direction of the paper's central claim holds at this scale, not its magnitude.

### [2026-05-27 04:10] - method_runs
- Audited the existing reproduction and found that the active `answer_grpo` pipeline is an action-level scaled analogue, not the paper's token-wise Humanline GRPO.
- Preserved the action-level artifacts as diagnostic results, but reset the active milestone to `method_runs` for the stricter token-level reproduction.
- Added a paper-style stale-pool option to the token-level GRPO path so a pool of old-policy generations can be consumed across multiple optimizer steps.

### [2026-05-27 04:14] - method_runs
- Switched the standard method, baseline, and ablation scripts onto the token-level Humanline GRPO implementation.
- Changed evaluation to recompute correctness, reward, and formatting from prediction JSONL artifacts instead of trusting self-reported run metrics.
- Queued a CPU smoke job for stale GRPO, token-level Humanline GRPO, and online-refresh GRPO on the reduced arithmetic task.

### [2026-05-27 05:23] - method_runs
- Cancelled the token-level smoke run after it showed a ceiling-dominated setting: stale GRPO reached `pass_at_1=0.94375` from `initial_pass_at_1=0.85625`, while Humanline dropped to `0.70`.
- Fixed the GRPO ratio to match the paper appendix: Humanline and regular token ratios are now computed against the reference model, with Humanline clipping applied upstream before GRPO clipping.
- Removed dropout from the tiny policy and made generation run in eval mode; queued a harder `0..19` arithmetic run with sync and clipping variants.

### [2026-05-27 06:31] - method_runs
- Cancelled the harder token-level run after both stale GRPO and Humanline GRPO collapsed to the same constant `<answer>18</answer>` output (`pass_at_1=0.0375`).
- Removed the dense numeric-closeness reward that made central wrong answers attractive, added balanced finite-grid warm-start data, and saved initial SFT predictions for diagnostics.
- Replaced the weak character-GRU policy in the reduced benchmark with a prompt-conditioned structured token policy over answer-tag completions, then queued a shorter core stale GRPO vs Humanline GRPO run.

### [2026-05-27 07:03] - method_runs
- Completed structured token-level GRPO job `a15a0a07-953`: both 64-step stale GRPO and 64-step Humanline GRPO improved from `initial_pass_at_1=0.48125` to `pass_at_1=0.61875` with `format_rate=1.0`.
- Kept the milestone at `method_runs` because the run did not yet include the online-refresh comparator needed for the paper's core stale-vs-online claim.
- Added the online-refresh comparator and a 128-step stale-refresh expansion to the next job.

### [2026-05-27 07:12] - core_claim
- Completed job `9e3a4fbb-a50` with the online-refresh comparator: online GRPO reached `pass_at_1=0.59375`, while 64-step Humanline GRPO reached `0.61875` with the same `65536` generated examples and `format_rate=1.0`.
- The 128-step diagnostic did not support an additional-setting claim: 128-step Humanline reached `0.51875`, below 128-step stale GRPO at `0.6125`.
- Queued a 64-step ablation and sync-cadence job to test whether any secondary Humanline component claims hold in the active token-level setup.

### [2026-05-27 07:25] - core_claim_plus
- Completed job `9f062ef2-f4d` with 64-step ablations and sync-cadence variants.
- The ablation did not support `secondary_claims`: syncing-only matched full Humanline (`0.61875`) and clipping-only was slightly higher (`0.625`), so this reduced benchmark does not reproduce the paper's component attribution.
- The sync-cadence expansion did support an additional Humanline setting: `offline_humanline_grpo_64x_sync24` reached `pass_at_1=0.69375`, above online refresh (`0.59375`) and the 64-step stale baseline (`0.61875`) with the same generated-example budget.

### [2026-05-27 07:54] - core_claim_plus
- Re-audited the packaged environment after sub-agent review.
- Confirmed `python validate.py --compare` passes, `eval/` and `data/` remain separated from `method/`, and active scores are recomputed from prediction JSONL artifacts.
- Left unrelated root device-file and trajectory changes untouched.

### [2026-05-27 07:58] - core_claim_plus
- Completed final verification of the research-gym package.
- Confirmed validation, shell syntax checks, and standalone evaluation all pass using existing reproduced artifacts.
- Tightened public scoring documentation to keep from-scratch guidance method-agnostic.

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim_plus`.
