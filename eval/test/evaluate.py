from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from data.synthetic_math import reward_completion


EXPERIMENT_NAME = "synthetic_math_stale_grpo"
SLICE_NAME = "test"
SLICE_REMAINDER = 1
PROMPT_RE = re.compile(r"^\s*(-?\d+)\+(-?\d+)=\?\s*$")


def load_run(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def answer_from_prompt(prompt: str) -> int:
    match = PROMPT_RE.match(prompt)
    if not match:
        raise ValueError(f"Cannot parse arithmetic prompt: {prompt!r}")
    return int(match.group(1)) + int(match.group(2))


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"prompt", "answer", "completion"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing required fields: {sorted(missing)}")
            prompt_answer = answer_from_prompt(str(row["prompt"]))
            recorded_answer = int(row["answer"])
            if recorded_answer != prompt_answer:
                raise ValueError(
                    f"{path}:{line_no} answer {recorded_answer} does not match prompt-derived answer {prompt_answer}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no prediction rows")
    return rows


def select_slice(rows: list[dict]) -> list[dict]:
    selected = [row for idx, row in enumerate(rows) if idx % 2 == SLICE_REMAINDER]
    if not selected:
        raise ValueError(f"{SLICE_NAME} slice has no rows after deterministic 50/50 split")
    return selected


def score_predictions(rows: list[dict]) -> dict[str, float]:
    rewards = [
        reward_completion(str(row["completion"]), answer_from_prompt(str(row["prompt"])))
        for row in rows
    ]
    correct = sum(1 for reward in rewards if reward >= 1.0)
    formatted = sum(
        1
        for row in rows
        if "<answer>" in str(row["completion"]) and "</answer>" in str(row["completion"])
    )
    return {
        "pass_at_1": correct / len(rows),
        "mean_reward": sum(rewards) / len(rewards),
        "format_rate": formatted / len(rows),
    }


def prediction_path_for_run(run_path: Path, run: dict) -> Path:
    candidates = [
        run_path.with_name(f"{run_path.stem}_predictions.jsonl"),
        run_path.with_name(f"{run['method']}_predictions.jsonl"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No predictions JSONL found for {run_path}; searched {searched}")


def methods_from_train_scores(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    with path.open() as f:
        scores = json.load(f)
    experiments = scores.get("experiments", scores)
    exp = experiments.get(EXPERIMENT_NAME)
    if not isinstance(exp, dict):
        raise ValueError(f"{path} does not contain experiment {EXPERIMENT_NAME!r}")
    results = exp.get("results", exp)
    if not isinstance(results, dict):
        raise ValueError(f"{path} experiment {EXPERIMENT_NAME!r} has no results mapping")
    methods = {name for name in results if not str(name).startswith("_")}
    if not methods:
        raise ValueError(f"{path} contains no train-slice method keys")
    return methods


def score_run(run_path: Path, run: dict) -> tuple[str, dict[str, float]]:
    method = str(run["method"])
    pred_path = prediction_path_for_run(run_path, run)
    rows = select_slice(load_predictions(pred_path))
    scored = score_predictions(rows)
    return method, {
        "pass_at_1": scored["pass_at_1"],
        "mean_reward": scored["mean_reward"],
        "format_rate": scored["format_rate"],
        "stale_refresh_interval": run["stale_refresh_interval"],
        "generated_examples": run["generated_examples"],
        "initial_pass_at_1": run.get("initial_pass_at_1", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="checkpoints/humanline_grpo")
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    requested_methods = methods_from_train_scores(Path(args.train_scores))
    results: dict[str, dict[str, float]] = {}
    for path in sorted(runs_dir.glob("*.json")):
        run = load_run(path)
        method = str(run["method"])
        if requested_methods is not None and method not in requested_methods:
            continue
        method, metrics = score_run(path, run)
        if method in results:
            raise ValueError(f"Duplicate method key {method!r} from {path}")
        results[method] = metrics

    if requested_methods is not None:
        missing = sorted(requested_methods - set(results))
        if missing:
            raise SystemExit(f"Missing run artifacts for train-scored methods: {missing}")
    if not results:
        raise SystemExit(f"No run JSON files found in {runs_dir}")

    scores = {
        "experiments": {
            EXPERIMENT_NAME: {
                "slice": SLICE_NAME,
                "split": "deterministic_odd_index_rows",
                "results": results,
            }
        }
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
