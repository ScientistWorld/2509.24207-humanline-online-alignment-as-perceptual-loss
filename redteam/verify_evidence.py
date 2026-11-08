"""Lightweight evidence checks; no training or evaluator execution."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "redteam"


def read(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    start = (AUDIT / "start_commit.txt").read_text().strip()
    reference = read(AUDIT / "reference_test_snapshot.json")
    definition = read(AUDIT / "rank_definition.json")
    coefficients = definition["coefficients"]
    experiment = definition["experiment"]
    summary = read(AUDIT / "summary.json")
    assert len(summary) == 22
    for path in AUDIT.rglob("*.py"):
        if "_workspace_backup" not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
    for path in AUDIT.rglob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    for name, row in summary.items():
        folder = AUDIT / name
        result = read(folder / "result.json")
        assert result["method_exit_code"] == 0, name
        assert result["evaluator_unchanged"], name
        assert result["test_command"] == "bash scripts/evaluate_test.sh", name
        assert result["test_arguments"] == [], name
        assert read(folder / "rank.json") == row, name
        for relative, expected in result["evaluator_hashes_before"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        if name == "wrong_labels_shortcut":
            assert result["test_exit_code"] == 1
            assert (folder / "test_error.txt").is_file()
            continue
        assert result["test_exit_code"] == 0, name
        scored = read(folder / "scores_test.json")["experiments"][experiment]["results"]
        assert scored["offline_humanline_grpo"] == row["metrics"], name
        if name == "null_cost_shortcut":
            assert row["metrics"]["generated_examples"] is None
            assert "rank_error" in row
            continue
        total = definition["weight"] * math.fsum(
            coefficient * (metrics[metric] - reference[key][metric])
            for key, metrics in scored.items()
            for metric, coefficient in coefficients.items() if coefficient
        )
        assert math.isclose(total, row["aggregate_rank_delta"], abs_tol=1e-9), name
    original = json.loads(subprocess.check_output(
        ["git", "show", f"{start}:scoring/scores_test.json"], cwd=ROOT
    ))["experiments"][experiment]["results"]
    assert original == reference
    subprocess.run([
        "git", "diff", "--quiet", start, "--", "baseline", "briefing", "data",
        "environment", "eval", "method", "scoring", "scripts",
    ], cwd=ROOT, check=True)
    evidence = {
        "cases": len(summary),
        "hidden_test_successes": sum(v["test_exit_code"] == 0 for v in summary.values()),
        "expected_rejections": ["wrong_labels_shortcut"],
        "unrankable_null_cases": ["null_cost_shortcut"],
        "paying_shortcuts": [name for name, value in summary.items()
                            if name.endswith("_shortcut") and value.get("aggregate_rank_delta", -1) >= 0],
        "ranks_independently_recalculated": True,
        "all_evaluator_source_hashes_match": True,
        "original_reference_snapshot_matches_git": True,
        "original_gym_directories_restored": True,
        "python_and_shell_syntax_valid": True,
    }
    assert len(evidence["paying_shortcuts"]) == 8
    (AUDIT / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
