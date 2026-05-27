# Verification Report

## Initial Assessment

The workspace started at claimed milestone `core_claim_plus`. The reproduced method artifacts and progress log support that milestone: the standard run includes the core stale-feedback comparison plus additional schedule and ablation settings. The environment was mostly packaged, but several gym-facing documents leaked method details and the scored metric set was not fully emitted by the default evaluator.

## Issues Found

- Critical: `reference.json` assigned nonzero coefficients to metrics that `scripts/evaluate.sh` did not write to `scores.json`, so future methods would not be scored on the full intended rubric.
- Minor: method-agnostic and method-safe documents included implementation-specific names and details that would compromise from-scratch mode.
- Minor: the local default checkpoint directory contained stale ignored artifacts without prediction JSONL files, so standalone evaluation with the default path was not usable until the reproduced artifacts were restored there.

## Fixes Applied

- Updated `eval/evaluate_math.py` so the standard evaluator reads method output files and computes all weighted metrics from predictions, including numeric error, carry-case accuracy, prompt coverage, evaluated row count, and initial accuracy when initial predictions are available.
- Regenerated `scoring/scores.json`, `scoring/scores_train.json`, and `scoring/scores_test.json` from the reproduced artifacts.
- Updated `scoring/reference.json` to contain the full calibrated reduced-benchmark metric set and an explicit note that these values are for the synthetic scaled benchmark rather than the paper's full mathematical-reasoning benchmark.
- Cleaned method leakage from `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, `scoring/CONSTRAINTS.md`, `scoring/DIRECTION.md`, and comments in `data/synthetic_math.py`.
- Restored the local default ignored checkpoint directory from the reproduced token-level artifacts so `scripts/evaluate.sh` works standalone with its default arguments.

Validation run:

```text
python validate.py --compare
bash -n scripts/*.sh
```

Both checks passed.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can run the packaged method and compare against the current scored artifacts; from-scratch agents can use the method-agnostic problem, data, and evaluation interface without needing method internals. The main limitation is that the gym uses a reduced synthetic arithmetic benchmark rather than the paper's full mathematical-reasoning benchmark, but that limitation is documented and the evaluation remains nontrivial.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: reduced synthetic arithmetic benchmark instead of the full paper-scale math benchmark
