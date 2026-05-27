from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="checkpoints/humanline_grpo")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    results = {}
    for path in sorted(runs_dir.glob("*.json")):
        run = load_run(path)
        results[run["method"]] = {
            "pass_at_1": run["pass_at_1"],
            "mean_reward": run["mean_reward"],
            "format_rate": run["format_rate"],
            "stale_refresh_interval": run["stale_refresh_interval"],
            "generated_examples": run["generated_examples"],
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
