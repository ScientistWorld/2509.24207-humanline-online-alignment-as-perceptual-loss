from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.synthetic_math import reward_completion


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
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no prediction rows")
    return rows


def score_predictions(rows: list[dict]) -> dict[str, float]:
    rewards = [reward_completion(str(row["completion"]), int(row["answer"])) for row in rows]
    correct = sum(1 for reward in rewards if reward >= 1.0)
    formatted = sum(1 for row in rows if "<answer>" in str(row["completion"]) and "</answer>" in str(row["completion"]))
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
        results[run["method"]] = {
            "pass_at_1": scored["pass_at_1"],
            "mean_reward": scored["mean_reward"],
            "format_rate": scored["format_rate"],
            "stale_refresh_interval": run["stale_refresh_interval"],
            "generated_examples": run["generated_examples"],
            "initial_pass_at_1": run.get("initial_pass_at_1", 0.0),
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
