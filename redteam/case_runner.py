"""Install one file-only attempt and invoke both stock evaluators, no args.

Backups and the official reference scores are harness-only. Shortcut producers
cannot inspect them; they consume only the newly trained seed-7 output.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/user")
AUDIT = ROOT / "redteam"
BACKUP = AUDIT / "_workspace_backup"
RUNS = ROOT / "checkpoints/humanline_grpo"
MAIN = "offline_humanline_grpo"
SAVED = ["data/synthetic_math.py", "scoring/scores_train.json",
         "scoring/scores_test.json", "scoring/scores.json"]


def snapshot() -> None:
    if (BACKUP / "ready").exists():
        return
    BACKUP.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUNS, BACKUP / "runs", dirs_exist_ok=True)
    for relative in SAVED + ["scoring/reference.json"]:
        target = BACKUP / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    (BACKUP / "ready").write_text("Original workspace snapshot\n")


def restore() -> None:
    shutil.rmtree(RUNS, ignore_errors=True)
    shutil.copytree(BACKUP / "runs", RUNS)
    for relative in SAVED:
        shutil.copy2(BACKUP / relative, ROOT / relative)


def evaluator_hashes() -> dict[str, str]:
    paths = [ROOT / "scripts/evaluate_test.sh", *sorted((ROOT / "eval").rglob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def command(args: list[str], log: Path) -> int:
    with log.open("w") as handle:
        return subprocess.run(args, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def install_reference_cache() -> None:
    cache = AUDIT / "reference_seed7/artifacts"
    if not (cache / f"{MAIN}.json").exists():
        raise RuntimeError("Run reference_seed7/run.sh first (cache must be genuinely trained)")
    for source in cache.glob(f"{MAIN}*"):
        shutil.copy2(source, RUNS / source.name)


def main() -> None:
    os.chdir(ROOT)
    name = sys.argv[1]
    folder = AUDIT / name
    spec = json.loads((folder / "config.json").read_text())
    snapshot()
    restore()
    before = evaluator_hashes()
    started = time.time()
    result = {"case": name, "config": spec, "test_command": "bash scripts/evaluate_test.sh",
              "test_arguments": [], "persistent_files_only": True,
              "evaluator_hashes_before": before}
    try:
        if spec["kind"] == "shortcut":
            install_reference_cache()
        code = command([sys.executable, "-m", f"redteam.{name}.method"], folder / "method.log")
        result["method_exit_code"] = code
        if code:
            raise RuntimeError((folder / "method.log").read_text()[-4000:])
        artifacts = folder / "artifacts"
        artifacts.mkdir(exist_ok=True)
        for source in RUNS.glob(f"{MAIN}*"):
            shutil.copy2(source, artifacts / source.name)
        train_code = command(["bash", "scripts/evaluate_train.sh"], folder / "train.log")
        result["train_exit_code"] = train_code
        if train_code == 0:
            shutil.copy2(ROOT / "scoring/scores_train.json", folder / "scores_train.json")
        # Fresh child interpreter, no arguments, unchanged test sources. Neither
        # training nor producer process survives into this evaluation process.
        test_code = command(["bash", "scripts/evaluate_test.sh"], folder / "test.log")
        result["test_exit_code"] = test_code
        if test_code == 0:
            shutil.copy2(ROOT / "scoring/scores_test.json", folder / "scores_test.json")
        else:
            (folder / "test_error.txt").write_text((folder / "test.log").read_text()[-6000:])
        result["evaluator_unchanged"] = before == evaluator_hashes()
        if not result["evaluator_unchanged"]:
            raise RuntimeError("Evaluator sources changed")
    except Exception as exc:
        result["harness_error"] = str(exc)
    finally:
        result["elapsed_seconds"] = round(time.time() - started, 3)
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        restore()
    print(json.dumps({k: v for k, v in result.items() if k != "evaluator_hashes_before"}), flush=True)
    if "harness_error" in result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
