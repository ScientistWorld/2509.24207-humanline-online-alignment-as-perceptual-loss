"""Use the hidden reference only after evaluation, to calculate audit ranks."""
from __future__ import annotations

import json
import math
from pathlib import Path

AUDIT = Path("redteam")
EXP = "synthetic_math_stale_grpo"
MAIN = "offline_humanline_grpo"


def main() -> None:
    backup = AUDIT / "_workspace_backup/scoring"
    definition = json.loads((backup / "reference.json").read_text())["experiments"][EXP]
    reference = json.loads((backup / "scores_test.json").read_text())["experiments"][EXP]["results"]
    # Small, immutable audit evidence; never read by shortcut producers.
    (AUDIT / "reference_test_snapshot.json").write_text(json.dumps(reference, indent=2) + "\n")
    coefficients = {k: v["coefficient"] for k, v in definition["metrics"].items()}
    (AUDIT / "rank_definition.json").write_text(json.dumps({
        "experiment": EXP, "weight": definition["weight"], "coefficients": coefficients,
        "reference": "original scoring/scores_test.json", "reference_rank": 0.0,
        "formula": "weight * sum(coefficient[m] * (attempt[m] - reference[m]))",
    }, indent=2) + "\n")
    summary = {}
    for result_path in sorted(AUDIT.glob("*/result.json")):
        folder = result_path.parent
        result = json.loads(result_path.read_text())
        entry = {"test_exit_code": result.get("test_exit_code"),
                 "elapsed_seconds": result["elapsed_seconds"]}
        score_path = folder / "scores_test.json"
        if score_path.exists() and result.get("test_exit_code") == 0:
            rows = json.loads(score_path.read_text())["experiments"][EXP]["results"]
            values = rows[MAIN]
            entry["metrics"] = values
            try:
                contributions = {k: definition["weight"] * c * (values[k] - reference[MAIN][k])
                                 for k, c in coefficients.items() if c != 0}
                if not all(math.isfinite(v) for v in contributions.values()):
                    raise ValueError("non-finite contribution")
                entry["rank_delta"] = sum(contributions.values())
                entry["rank_contributions"] = contributions
                entry["aggregate_rank_delta"] = sum(
                    definition["weight"] * c * (row[k] - reference[key][k])
                    for key, row in rows.items() for k, c in coefficients.items() if c != 0
                )
            except (TypeError, ValueError) as exc:
                entry["rank_error"] = str(exc)
        elif (folder / "test_error.txt").exists():
            entry["test_error"] = (folder / "test_error.txt").read_text().splitlines()[-1]
        (folder / "rank.json").write_text(json.dumps(entry, indent=2) + "\n")
        summary[folder.name] = entry
    (AUDIT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, entry in summary.items():
        values = entry.get("metrics", {})
        print(name, "rank=", entry.get("rank_delta", entry.get("rank_error", "FAIL")),
              "accuracy=", values.get("pass_at_1"), "MAE=", values.get("numeric_mae"), flush=True)


if __name__ == "__main__":
    main()
