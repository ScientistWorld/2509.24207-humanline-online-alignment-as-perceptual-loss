# Hardening Report

This gym reproduces the verifiable-reward experiment of *Humanline: Online Alignment as Perceptual Loss* at small scale. A fixed 0.6M-parameter transformer is post-trained with GRPO-style feedback on three-digit addition, from a fixed starting policy, under two sampling regimes: `stale64` (sample every 64 updates) and `offline` (sample once). It is scored on 16,000 hidden prompts.

Two hardening iterations followed the audit in `REDTEAM.md`:

- **Iteration 11** rebuilt the task and evaluator: evaluator-owned prompts and decoding, a fixed starting policy and architecture, operating-point gates instead of a cost reward, and paper-faithful GRPO baselines.
- **Iteration 12 (this one)** made scoring robust to training-seed noise and closed the seed-aggregation shortcuts that this opens. It also reran the reference set, variants and shortcuts from the final code (job `8584707b-feb`), and rewrote the briefing and scoring docs to match the final gym.

Ranks below use the raw form of the gym's rank: Σ over experiments and scored metrics of weight × |coefficient| × (value − reference). That is 0.5 per experiment, coefficient 1.0 on `pass_at_1` and 0.5 on `hard_pass_at_1`. The real ranker also divides by each metric's spread across attempts, which does not change signs. All numbers are hidden-test numbers unless stated. Evidence is in `redteam/hardening/results_final/summary.json`, the per-case `redteam/hardening/shortcuts_final/*/scores_*.json`, and the job log (`jobs/8584707b-feb/stdout.log`). The harness is `redteam/hardening/final.py`, with analysis in `analyze_final.py`.

## What changed in this iteration

1. **Scoring over fixed seeds.** Before, each method was scored on one training run. One run's accuracy has a standard deviation of about 0.015 across seeds (0.035 for online GRPO). The paper's effect here (humanline minus GRPO) is about +0.013, so the old ranking mostly measured seed luck. Both evaluators now score each (experiment, method) as the mean over the required seeds 7–14:
   - A missing or unloadable seed counts 0.
   - Several runs declaring the same seed score the worst one.
   - Runs declaring other seeds are ignored.
   - A run whose greedy outputs repeat an earlier seed's counts 0. "Repeat" means identical outputs on ≥ 90% of the prompts where either run errs. Honest seeds are both wrong on at most 52% (stale64) and 68% (offline) of those prompts, an upper bound on identical outputs. Copies with weight noise share 99%.
   - A schedule violation or impossible value in any run fails the whole row.
2. **Cached greedy decoding, on GPU when available.** The key/value-cached decode matched uncached decoding on every checked prompt (0 mismatches in 6,000). CPU and GPU gave identical per-seed scores for the 16 reference runs checked. The hidden test of the 81 reference checkpoints takes seconds on a GPU and about 15 minutes on one CPU.
3. **Scripts.** `method.sh`, `baseline.sh` and `ablation.sh` train seeds 7–14, four at a time (`scripts/launch.sh`). `reproduce.sh` regenerates all 80 reference runs in about 17 minutes on one GPU. Retraining reproduces bit-identical weights, and the seed-7 runs match iteration 11's visible scores exactly.
4. **Docs.** `briefing/`, `scoring/*.md` and `DETAILS.md` described the original gym (prediction files, cost metric, a different task). They now describe this one. `scoring/reference.json` holds the 8-seed reference rows and keeps the paper's numbers under `_paper_reported`.

## Do better results score higher?

**Yes, now by more than the noise.** Two independent 8-seed sets of the same method differ in rank by 0.0083 over both experiments (one standard deviation, estimated from 24 seeds of the reference method). Per experiment the figure is 0.0054 (stale64) and 0.0063 (offline). Rerunning the reference method with seeds 15–22 and 23–30 moved it by −0.0017 and +0.0014. Under the single-run regime, two runs differ by 0.023.

"Before" is the single-run regime this iteration replaced: each method's seed-7 run against the seed-7 reference run. "After" is the final 8-seed scoring. The paired column is the per-seed accuracy difference against the reference on the same seeds and prompts (mean, t over 8 seeds).

| Variant (8 seeds each) | Expected | Rank before | Rank after | Accuracy vs reference (t) |
|---|---|---:|---:|---:|
| stale64 `humanline_informative` (each update drawn from groups with mixed rewards) | better | +0.0234 | **+0.0213** | +0.030 (4.9) |
| offline `humanline_informative` | better | +0.0179 | **+0.0257** | +0.036 (5.2) |
| offline `humanline_eps05` (GRPO ε 0.5) | better | **−0.0041** | **+0.0196** | +0.031 (3.7) |
| stale64 `humanline_eps03` (GRPO ε 0.3) | ≈ | −0.0034 | +0.0047 | +0.005 (0.7) |
| stale64 `grpo_eps03` (plain GRPO, ε 0.3) | simple baseline | +0.0050 | +0.0075 | +0.011 (0.9) |
| stale64 `humanline_k24` (paper's k range) | weaker here | −0.0109 | −0.0093 | −0.015 (−2.2) |
| stale64 `humanline_one_round` (64 updates) | weaker | −0.1977 | −0.1635 | −0.228 |
| stale64 `humanline_k4` / offline `humanline_k4` (syncing too often collapses) | weaker | −0.378 / −0.411 | −0.352 / −0.395 | −0.47 / −0.52 |
| stale64 `humanline_double_budget` (1,024 updates) | over budget | −0.494 | −0.486 (failed row) | +0.077 unconstrained |
| reference-set `grpo` (stale64 / offline) | weaker (paper) | −0.0078 / −0.0378 | −0.0079 / −0.0087 | −0.016 (−1.9) / −0.011 (−1.5) |
| `humanline_sync_only` (stale64 / offline) | ablation | −0.0128 / −0.0016 | −0.0027 / −0.0025 | |
| `humanline_clip_only` (stale64 / offline) | ablation | −0.0026 / −0.0919 | −0.0019 / −0.0659 | |

- Under the single-run regime, the genuinely better offline `humanline_eps05` (t = 3.7 over 8 seeds) ranked below the reference. Most other before-differences were well inside the 0.023 single-run noise.
- After the change, every variant expected to be better is above the reference by 3.1–4.1 standard deviations of the comparison noise in its experiment. Applied in both experiments, informative groups give +0.047, 5.7 standard deviations.
- Every weaker variant is below the reference, and the over-budget variant fails.
- In the original gym (`REDTEAM.md`) the ordering was inverted by an unnormalised cost term. More RL ranked −45,874 and a one-update run ranked +6,528. Iteration 11 removed that term, and the ordering above is the final state.

## Reference score before and after

| | stale64 `pass_at_1` / `hard` | offline `pass_at_1` / `hard` |
|---|---|---|
| Original gym (REDTEAM.md; different task, 81 hidden rows) | 0.654 test accuracy, rank dominated by −0.1 per generated example | — |
| Iteration 11, seed-7 run, pre-fix evaluator, 16k test | 0.676 / 0.625 | 0.585 / 0.583 |
| **Final, mean of seeds 7–14** | **0.662 / 0.620** | **0.558 / 0.557** |

The drop is not a regression. Seed 7 was one of the luckier seeds: the best of eight offline, and +0.015 above the stale64 mean. The training code and its seed-7 outputs are unchanged.

## The paper's effect against seed noise

Humanline GRPO minus the GRPO baseline, per training seed, raw test accuracy:

| | 8 scored seeds | 24 seeds (7–30) | Online GRPO minus humanline (8 seeds) |
|---|---|---|---|
| stale64 | +0.016 (se 0.009, t 1.85) | +0.016 (se 0.005, t 3.4) | +0.098 |
| offline | +0.011 (se 0.008, t 1.5) | +0.013 (se 0.005, t 2.6) | +0.202 |

- The paper's direction reproduces in both experiments: humanline beats 64x-stale and offline GRPO at the same data volume, and is clearly established over 24 seeds.
- In ranked terms the effect is about twice the comparison noise: 0.0166 against a standard deviation of 0.0083. Per experiment it is 1.5–1.9 standard errors.
- The paper's claimed magnitude does not reproduce. Humanline recovers about 14% (stale64) and 5% (offline) of the gap to online GRPO, not all of it.
- The secondary claim reproduces in the offline experiment:
  - Syncing provides most of the gain: sync-only 0.557 against full 0.558.
  - Clipping alone is worse than the offline baseline: 0.473 against 0.546.
  - In stale64 each component alone recovers about half the gain.
- The paper's observation that syncing too often collapses small models also reproduces: k = 4 collapses.
- I did not enlarge to 16 or 24 required seeds. Every future attempt would pay proportionally, and the better variants are already separated at 8.

## Shortcuts

Rank change summed over both experiments. "Before" for the REDTEAM cases is the original gym, in its own units. "Before" for the seed-aggregation cases is the pre-fix single-run evaluator (`redteam/hardening/prev_eval_test.py`) against its own score of the seed-7 reference. All "after" numbers are the final hidden test.

| Shortcut | Fix | Before | After |
|---|---|---:|---:|
| Label copy (write correct answers / claimed scores) | Iteration 11: the evaluator decodes the weights itself and reads no predictions or claims. Now also: base weights copied across seeds count once. | +0.745 | −0.836 |
| Zero / forged cost metadata | Iteration 11: cost is not scored, only gated, so the same weights get the same score | +6,553.6 | 0.000 |
| Negative cost, null cost | Impossible or missing values fail the row | +106,651 / undefined | −0.904 / −0.904 |
| Zero-training constant answer | Evaluator-owned decoding; identical seeds count once | +6,551.0 | −0.904 |
| Constant-task / cherry-picked population | Iteration 11: evaluator-owned hidden prompts | +0.745 / +0.745 | 0.000 |
| Reward artifact (edit `data/arithmetic.py`) | Iteration 11: evaluators import nothing from `data/`; test rows identical | +6.67 | 0.000 |
| Wrong architecture / NaN weights | Strict load; an unevaluable seed counts 0 | (iteration 11) fail | −0.904 / −0.904 |
| Missing experiment (submit only the better one) | A missing experiment is a failed row | −0.437 | −0.397 |
| Copy the best seed into all seed slots | Output-repeat detection | +0.0020 | −0.787 |
| Same, with weight noise per copy | Output-repeat detection (99% of errors shared) | +0.0015 | −0.787 |
| Submit only the best seed | Missing seeds count 0 | +0.0020 | −0.787 |
| Best of 24 seeds relabelled as seed 7 | Missing seeds count 0 | +0.0033 | −0.787 |
| Two runs per seed hoping for best-of | Worst run per seed | −0.086 | −0.013 |
| Runs with seeds outside 7–14 | Ignored; the method gets failed rows | −0.082 | −0.904 |

The small "before" gains of the seed cases reflect that seed 7 happened to be a good seed. The expected gain of best-of-8 under the single-run regime is about +0.02.

### Shortcuts left open

All three work by misreporting or bypassing the training procedure. The hidden test only sees the final weights and must not retrain, so they are left to the grader. `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md` state them explicitly.

- **Online training declared as stale or offline (+0.219).** Iteration 11's single-run test scored it +0.073. The weights of an online run are indistinguishable from a stale one's. Verifying the sampling schedule would mean replaying training. The grader can compare the training code and log against `run.json`.
- **Supervised training on computed labels (+0.589, accuracy 0.996).** The answers to addition are trivially computable, and an honest supervised run can even declare zero sampling rounds and pass the budget gate. The task design cannot prevent it, and the weights do not reveal it. This is the largest open hole, and it is out of bounds in `DIRECTION.md`.
- **Seed shopping (+0.021).** Train 24 seeds and submit the best 8 relabelled as 7–14 (selected on the visible slice). That is about 2.5 standard deviations of the comparison noise. It requires falsifying the `seed` field. Multi-seed scoring roughly halves the expected gain compared with best-of-24 under the single-run regime (about +0.03), but cannot remove it. The grader can rerun one declared seed and compare.

## Headroom

- No scored metric of the reference sits at its bound: stale64 0.662 / 0.620 and offline 0.558 / 0.557, against 1.0.
- The online comparator reaches 0.760 with the same updates and completions, and supervised training reaches 0.996, so neither the model nor the update budget is the limit.
- Honest alternatives inside the operating points beat the reference by more than the noise in both experiments (informative groups; GRPO ε 0.5 offline).
- No simple variant reaches the bound, and the brute force the paper avoids (sampling every update) is gated out.
- The simple baseline is well below the reference: the starting policy at 0.360. The plain GRPO baseline is below it by about 1.5 standard deviations per experiment (2 combined).
- Plain GRPO with a wider trust region (ε 0.3) ties the reference in stale64 (+0.0075, about 1.4 standard deviations; t 0.9 over seeds). Combined with the plain offline baseline (−0.0087) it gives about −0.001 overall. A future agent copying a simple baseline therefore does not outrank the reference; it is an honest alternative the test cannot separate from the reference.
- I added no experiment: the existing two have room above the reference.

## Weights

Unchanged: 0.5 per experiment, coefficient 1.0 on `pass_at_1` and 0.5 on `hard_pass_at_1`. The two experiments are the paper's two headline settings: 64x-less-frequent sampling for verifiable math, and offline data for the offline-vs-online claim. Neither matters more in the paper.

## What I could not fix

- **Reproduction magnitude.** The paper's central claim that humanline variants match online alignment does not reproduce at this scale; only its direction does. The selected k = 48 differs from the paper's 12–24 range, which performs worse here (k = 24: −0.015). I did not change the training method or tune it toward the paper's numbers.
- **Procedure-level shortcuts.** The three cases above remain, as stated.
- **Effect size against noise.** With the 8 required seeds the paper's effect is about 2 standard deviations of the comparison noise, not more. A future improvement of the paper's own effect size will be only marginally distinguishable.
- **Device dependence.** CPU and GPU decoding agreed exactly on the checked runs. Floating-point near-ties could in principle flip a few of 16,000 answers between devices.
