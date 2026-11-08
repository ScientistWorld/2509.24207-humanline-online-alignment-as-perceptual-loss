# Gym scoring audit

**Better results do not reliably score higher.** The unchanged hidden test ranks a much stronger run below the reference and a much weaker run above it. The main cause is the unnormalized coefficient of **−0.1 per generated training example**, added directly to quality metrics whose useful changes are fractions of one point. Eight paying shortcuts were reproduced. The task also permits perfect scores without stale-feedback post-training.

All rank changes below use the signed coefficients and experiment weight from `scoring/reference.json`, anchored to the **original `scoring/scores_test.json`**, not the full-evaluation values embedded in `reference.json`. Reference rank is zero. The formula is `sum(weight × coefficient × (attempt − original_test_reference))` over scored metrics and result rows. Unchanged comparator rows cancel. [Machine-readable results](redteam/summary.json), [rank definition](redteam/rank_definition.json), and [original test snapshot](redteam/reference_test_snapshot.json) are retained.

**Quality ordering checks.** Every row below is an actual training run of the supplied model and algorithm, with only the stated change. The reference uses 160 supervised warm-start updates, 256 RL updates, batch size 32, group size 8, and stale refresh interval 64. Evaluation uses the same 81 hidden rows in every row of this table.

| Run and change | Test accuracy | Numeric MAE | Generated examples | Rank change |
|---|---:|---:|---:|---:|
| [Reference, seed 7](redteam/reference_seed7/run.sh) | 65.43% | 0.4691 | 65,536 | 0.000000 |
| [More RL: 2,048 updates](redteam/better_more_rl_2048/run.sh) | **97.53%** | **0.0247** | 524,288 | **−45,874.495916** |
| [More RL: 512 updates](redteam/better_more_rl/run.sh) | 66.67% | 0.4568 | 131,072 | −6,553.561681 |
| [More warm-start: 480 updates](redteam/better_warmstart/run.sh) | **100.00%** | **0.0000** | 65,536 | **+0.692949** |
| [Less RL: one update](redteam/weaker_less_rl/run.sh) | **48.15%** | 0.5432 | 256 | **+6,527.685660** |
| [Less warm-start: 16 updates](redteam/weaker_warmstart/run.sh) | **17.28%** | 2.1605 | 65,536 | **−1.259349** |
| [Humanline removed](redteam/weaker_no_humanline/run.sh) | 61.73% | 0.4198 | 65,536 | −0.012583 |

The 2,048-update run raises accuracy by **32.10 percentage points**, reduces numeric error, and reaches 100% carry accuracy. Its quality contribution improves by +0.704084, but the training-count term contributes −45,875.2. Conversely, the one-update run loses 17.28 accuracy points yet gains thousands of rank points. Both accuracy differences exceed the measured seed spread. The weaker warm-start run is correctly penalized; the Humanline ablation and the 512-update accuracy gain are within seed noise.

At fixed reported generated count, more supervised training produces a real, stronger learned model and a rank gain larger than seed noise. However, its initial accuracy is already 100%: it establishes that the reference can be beaten, not a benefit from stale-feedback post-training. The supervised-only shortcut below confirms this independently.

**Bounds, task failure, and scale.** None of the six metrics with nonzero coefficients is at its optimization bound in any of the ten original result rows. The bounds are accuracy 1, reward 1.05, numeric MAE 0, carry accuracy 1, generated count 0, and initial accuracy 0 under its negative coefficient. `format_rate` is already 1 for all ten rows, but has coefficient zero. Coverage, uniqueness, and stale interval also have zero coefficients. [Full bound check](redteam/bounds.json).

There is room above the reference in task quality, but the scale does not enforce “lower cost when quality is held fixed.” Even a constant answer with **truthfully reported zero training**, 1.23% test accuracy, zero carry accuracy, and MAE 6.6049 outranks the reference by **+6,551.005249**. Thus a method failing the task can win without falsifying cost. At the reference count and initial accuracy, perfect quality offers only +0.744824; saving eight reported completions earns +0.8. The weak constant baseline is clearly worse on quality but overwhelmingly better on rank.

**Three-seed spread.** Training seeds 7, 8, and 9 were rerun with the supplied procedure. Evaluation generation was held at the reference seed 10007 inside the producer process so seed changes do not change the test population; all share 81 test rows and 69 distinct prompts. The evaluators themselves were unchanged.

| Training seed | Test accuracy | Rank change |
|---|---:|---:|
| [7](redteam/reference_seed7/run.sh) | 65.43% | 0.000000 |
| [8](redteam/reference_seed8/run.sh) | 62.96% | −0.061469 |
| [9](redteam/reference_seed9/run.sh) | 58.02% | −0.112908 |

The observed test-rank range is **0.112908**, and the accuracy range is **7.41 percentage points**. Gains smaller than these spreads are noise for this audit. Three seeds give an empirical range, not a significance guarantee. Seed 7 reproduces the stored reference to floating-point rounding. [Exact spread](redteam/seed_spread.json).

**Reproduction check.** Against the supplied test scores, the default method beats the online and stale-64 baselines in accuracy, **65.43% versus 61.73%**, matching the direction described in the visible briefing. The independently rerun Humanline ablation also gives 61.73%. But the 3.70-point accuracy advantage and +0.012583 composite advantage are both below seed spread; baseline numeric MAE and carry accuracy are better. The online and stale-64 baseline test metrics are identical. All report 65,536 generated examples, so these metrics do not demonstrate fewer generated examples.

At stale interval 128, the supplied method scores **48.15% versus the baseline’s 60.49%**, reversing the intended direction. The supplied sync-only row ties the default on every scored metric: **an honest alternative the test cannot tell apart**, not an independently rerun shortcut. Clip-only ties default accuracy and improves MAE/carry; these rows do not establish a benefit from combining the components. These comparisons use the supplied, reevaluated comparator artifacts, except for the explicitly rerun ablation. The paper was not used, as required by the audit. [All original comparison metrics](redteam/baseline_comparison.json).

**Shortcut runs.** Every linked directory contains a `run.sh`, producer, retained submitted JSON/JSONL artifacts, `result.json`, and test scores or a rejection error. Each script installs its case, runs the visible evaluation, and invokes exactly `bash scripts/evaluate_test.sh` with no arguments. Rank changes in this table include all emitted result rows. Except for the reward-artifact shortcut, comparator rows are unchanged and total rank change equals the changed main row.

| Shortcut | What it does | Test accuracy | Rank change vs reference | Result |
|---|---|---:|---:|---|
| [Label-copy shortcut](redteam/label_copy_shortcut/run.sh) | Regenerates the default evaluation examples using the visible generator and seed, writes `Example.answer` into completions; retains reference cost/start metadata to isolate answer exposure. | 100.00% | **+0.744824** | **Works** |
| [Zero-metadata shortcut](redteam/metadata_zero_shortcut/run.sh) | Reuses identical trained predictions; writes generated count and initial accuracy as zero. | 65.43% | **+6,553.648125** | **Works** |
| [Negative-cost shortcut](redteam/constant_negative_cost_shortcut/run.sh) | Always answers 18; writes generated count −1,000,000 and initial accuracy −1000. | 1.23% | **+106,651.008999** | **Works** |
| [Zero-training constant shortcut](redteam/constant_zero_training_shortcut/run.sh) | Constructs a constant-18 policy with no training; reports actual zero generated count and actual full-set initial accuracy 3.75%. | 1.23% | **+6,551.005249** | **Works** |
| [Warm-start-only shortcut](redteam/warmstart_only_shortcut/run.sh) | Actually trains the supplied model for 480 supervised updates on the visible balanced grid; performs no RL or stale-feedback updates. Initial and final predictions are identical. | 100.00% | **+6,554.292949** | **Works** |
| [Constant-task shortcut](redteam/constant_task_shortcut/run.sh) | Chooses only valid arithmetic prompts summing to 18; uses leading spaces to create distinct raw prompt strings; always answers 18. | 100.00% | **+0.744824** | **Works** |
| [Cherry-pick shortcut](redteam/cherry_pick_shortcut/run.sh) | Retains only the existing model’s correct rows using visible labels, duplicates them, and submits the reduced population. | 100.00% | **+0.744824** | **Works** |
| [Reward-artifact shortcut](redteam/reward_artifact_shortcut/run.sh) | Replaces the correctness rule in visible `data/synthetic_math.py` with unconditional correctness. Predictions and evaluator sources are unchanged. | 100.00% reported | **+6.667616 total**; +0.627540 main row | **Works** |
| [Forged-scores shortcut](redteam/forged_scores_shortcut/run.sh) | Always answers 18; writes fake quality scores, success flags, negative timing, null status, and a fabricated `scores_test.json`. | 1.23% | −2.639126 | Fails: test recomputes quality and overwrites scores |
| [Empty-output shortcut](redteam/empty_output_shortcut/run.sh) | Writes empty completions with unchanged cost metadata. | 0.00% | −6.278262 | Fails |
| [Copied-input shortcut](redteam/copied_input_shortcut/run.sh) | Returns the input prompt as its completion with unchanged cost metadata. | 0.00% | −6.278262 | Fails |
| [Wrong-label shortcut](redteam/wrong_labels_shortcut/run.sh) | Sets both submitted answers and completions to 18 on the original prompts. | Rejected | No score | Fails: prompt-derived answer validation rejects it |
| [Null-cost shortcut](redteam/null_cost_shortcut/run.sh) | Writes null generated count and initial accuracy. | 65.43% | Undefined | Not paying: test exits 0 and emits nulls, but the specified arithmetic rank cannot be computed |

The answer-copy shortcut needs no hidden split information: the default method exposes the generator and seed. The finite domain has just 400 possible operand pairs, all covered twice by the balanced 800-example training set. The warm-start-only shortcut stays within the supplied model size and solves this domain before generated-feedback training. It is a shortcut for the stated stale-feedback research question, despite being a real supervised solution to the arithmetic task.

The comparison-population shortcuts also need no hidden hash or test answers. The test selects a hash slice of **submitted rows**, rather than requiring a fixed prompt manifest. Cherry-picking passes with 106 hidden rows and uniqueness 0.434. The constant-task shortcut passes with 370 hidden rows and reported uniqueness 1.0, although the whitespace variants add no new arithmetic problems. Minimum row and raw-string uniqueness checks do not enforce a representative test distribution.

The reward-artifact shortcut changes an agent-visible task artifact that the restored evaluator imports as fixed ground truth. Its numerical MAE remains 0.4691, exposing unchanged wrong answers, while the reward-derived accuracy and carry metrics become perfect. The same replacement inflates all ten comparator rows, explaining its larger aggregate gain. This uses a persistent task-file edit; it does not modify evaluator source, its process, or the Python environment. No model or evaluator artifact replacement was needed beyond this task rule.

**Execution and retained evidence.** Two CPU jobs completed successfully: `0a7ba4bf-3f8` and `7bbf1684-1af`. Together they ran 22 cases: nine training variants/seeds and thirteen shortcuts. There were 21 successful hidden-test exits, including the unrankable null case, and one deliberate rejection. [First job log](redteam/first_job_stdout.txt), [second job log](redteam/second_job_stdout.txt), and [training trace](redteam/training_trace.json) are retained. No further heavy computation ran on the login node.

Runs were bundled within these two job containers. Each method producer exits before evaluation starts in a fresh child interpreter; cases are not claimed to have each received a separate container. All shortcut effects are persistent files and require no surviving producer process. The harness checks evaluator-source hashes for every case and restores task data, checkpoints, and scoring files between cases. Method producers do not read hidden test modules, the reference score snapshot, or rank coefficients. Those are used only for audit verification and post-run rank calculation. [Harness and rerun instructions](redteam/README.md).

The priority fixes are to enforce a quality floor before any cost reward, normalize or constrain cost, validate metadata against measured work, keep task reward code evaluator-owned, and have the evaluator own a fixed evaluation population and model inference. The current arithmetic scale also needs a task that cannot be solved by the permitted warm-start stage alone. [Final verification](redteam/verification.json) independently checks the retained ranks, evaluator hashes, syntax, and restoration. Final deliverables are confined to `redteam/` and this report, with the required job-control `action.yaml`; original method, data, scoring, evaluator, and run-script files are restored.
