from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from data.synthetic_math import extract_answer, reward_completion


PROMPT_RE = re.compile(r"^\s*(-?\d+)\+(-?\d+)=\?\s*$")


def answer_from_prompt(prompt: str) -> int:
    match = PROMPT_RE.match(prompt)
    if not match:
        raise ValueError(f"Cannot parse arithmetic prompt: {prompt!r}")
    return int(match.group(1)) + int(match.group(2))


def operands_from_prompt(prompt: str) -> tuple[int, int]:
    match = PROMPT_RE.match(prompt)
    if not match:
        raise ValueError(f"Cannot parse arithmetic prompt: {prompt!r}")
    return int(match.group(1)), int(match.group(2))


def load_run(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


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


def score_predictions(rows: list[dict]) -> dict[str, float]:
    rewards = [
        reward_completion(str(row["completion"]), answer_from_prompt(str(row["prompt"])))
        for row in rows
    ]
    correct = sum(1 for reward in rewards if reward >= 1.0)
    parsed_answers = [extract_answer(str(row["completion"])) for row in rows]
    absolute_errors = [
        (
            abs(parsed - answer_from_prompt(str(row["prompt"])))
            if parsed is not None
            else abs(answer_from_prompt(str(row["prompt"]))) + 1
        )
        for row, parsed in zip(rows, parsed_answers)
    ]
    carry_rows = [row for row in rows if is_carry_prompt(str(row["prompt"]))]
    formatted = sum(1 for row in rows if "<answer>" in str(row["completion"]) and "</answer>" in str(row["completion"]))
    return {
        "pass_at_1": correct / len(rows),
        "mean_reward": sum(rewards) / len(rewards),
        "format_rate": formatted / len(rows),
        "numeric_mae": sum(absolute_errors) / len(absolute_errors),
        "carry_pass_at_1": rate([correctness(row) for row in carry_rows]),
        "unique_prompt_rate": len({str(row["prompt"]) for row in rows}) / len(rows),
        "evaluated_rows": float(len(rows)),
    }


def is_carry_prompt(prompt: str) -> bool:
    a, b = operands_from_prompt(prompt)
    return abs(a) % 10 + abs(b) % 10 >= 10


def correctness(row: dict) -> bool:
    return reward_completion(str(row["completion"]), answer_from_prompt(str(row["prompt"]))) >= 1.0


def rate(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


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


def initial_prediction_path_for_run(run_path: Path, run: dict) -> Path | None:
    candidates = [
        run_path.with_name(f"{run_path.stem}_initial_predictions.jsonl"),
        run_path.with_name(f"{run['method']}_initial_predictions.jsonl"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="checkpoints/humanline_grpo")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    results = {}
    for path in sorted(runs_dir.glob("*.json")):
        run = load_run(path)
        pred_path = prediction_path_for_run(path, run)
        scored = score_predictions(load_predictions(pred_path))
        initial_pred_path = initial_prediction_path_for_run(path, run)
        if initial_pred_path is not None:
            initial_pass_at_1 = score_predictions(load_predictions(initial_pred_path))["pass_at_1"]
        else:
            initial_pass_at_1 = run.get("initial_pass_at_1", 0.0)
        results[run["method"]] = {
            "pass_at_1": scored["pass_at_1"],
            "mean_reward": scored["mean_reward"],
            "format_rate": scored["format_rate"],
            "numeric_mae": scored["numeric_mae"],
            "carry_pass_at_1": scored["carry_pass_at_1"],
            "unique_prompt_rate": scored["unique_prompt_rate"],
            "evaluated_rows": scored["evaluated_rows"],
            "stale_refresh_interval": run["stale_refresh_interval"],
            "generated_examples": run["generated_examples"],
            "initial_pass_at_1": initial_pass_at_1,
        }

    if not results:
        raise SystemExit(f"No run JSON files found in {runs_dir}")

    scores = {"experiments": {"synthetic_math_stale_grpo": {"results": results}}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
