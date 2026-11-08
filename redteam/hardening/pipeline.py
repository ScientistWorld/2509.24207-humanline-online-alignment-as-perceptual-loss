"""Hardening job: build the gym's reference runs, variants, seeds and shortcuts.

Phases (each resumable; finished runs are skipped):
  0 smoke     tiny end-to-end run of every code path; fails fast on bugs
  1 base      build data/base_policy.pt and pin its hash in eval/{train,test}
  2 sweep     pre-registered learning-rate choice: online/stale/humanline GRPO,
              seed 7, stale64, LR grid; LR* = argmax of the *online* baseline's
              visible-validation accuracy (ties -> smaller LR)
  3 reference reference set at LR* into checkpoints/ (scored by default)
  4 seeds     extra training seeds at LR*
  5 variants  better / weaker candidates at LR*, plus an over-budget run
  6 shortcuts shortcut submissions through the hidden-test evaluator
  7 evaluate  visible + hidden evaluation of every root, then analysis
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/user")
H = ROOT / "redteam" / "hardening"
RUNS = H / "runs"
SHORT = H / "shortcuts_runs"
RESULTS = H / "results"
TMP = ROOT / "tmp" / "hardening"
CKPT = ROOT / "checkpoints"
PY = sys.executable

LRS = ["1e-4", "3e-4", "1e-3"]
EXTRA_SEEDS = [8, 9, 10, 11]
ENV = {**os.environ, "PYTHONPATH": str(ROOT), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def cuda_available() -> bool:
    out = subprocess.run([PY, "-c", "import torch;print(int(torch.cuda.is_available()))"], capture_output=True, text=True, env=ENV)
    return out.stdout.strip() == "1"


CUDA = cuda_available()
CPUS = os.cpu_count() or 1
WORKERS = 4 if CUDA else max(1, min(4, CPUS // 2))
THREADS = max(1, CPUS // WORKERS)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------- specs ----
def spec_args(spec: str, experiment: str) -> tuple[str, list[str]]:
    refresh = "64" if experiment == "stale64" else "512"
    method = "method.humanline_grpo"
    baseline = "baseline.stale_grpo"
    table = {
        "online_grpo": (baseline, ["--refresh-interval", "1"]),
        "stale_grpo": (baseline, ["--refresh-interval", "64"]),
        "offline_grpo": (baseline, ["--refresh-interval", "512", "--grpo-epsilon", "0.5"]),
        "humanline_grpo": (method, ["--humanline", "--refresh-interval", refresh]),
        "humanline_sync_only": (method, ["--humanline", "--disable-humanline-clipping", "--refresh-interval", refresh]),
        "humanline_clip_only": (method, ["--humanline", "--disable-humanline-sync", "--refresh-interval", refresh]),
    }
    return table[spec]


def train(run_dir: Path, name: str, experiment: str, module: str, args: list[str], allow_fail: bool = False) -> Path:
    if (run_dir / "run.json").exists() and (run_dir / "policy.pt").exists():
        return run_dir
    if allow_fail and (run_dir / "FAILED").exists():
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", module, "--name", name, "--experiment", experiment, "--run-dir", str(run_dir),
           "--threads", str(THREADS), *args]
    t0 = time.time()
    with (run_dir / "stdout.log").open("w") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        tail = (run_dir / "stdout.log").read_text()[-2000:]
        if not allow_fail:
            raise RuntimeError(f"training failed: {run_dir}\n{tail}")
        (run_dir / "FAILED").write_text(tail)
        log(f"FAILED (allowed) {run_dir.relative_to(ROOT)}")
        return run_dir
    log(f"trained {run_dir.relative_to(ROOT)} in {time.time() - t0:.0f}s")
    return run_dir


def train_many(jobs: list[tuple], allow_fail: bool = False) -> None:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for fut in [pool.submit(train, *job, allow_fail=allow_fail) for job in jobs]:
            fut.result()


def evaluate(slice_name: str, runs_root: Path, output: Path, dump: Path | None = None) -> dict:
    script = ROOT / "eval" / slice_name / "evaluate.py"
    cmd = [PY, str(script), "--runs-root", str(runs_root), "--output", str(output)]
    if dump is not None:
        cmd += ["--dump-correct", str(dump)]
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=ENV, cwd=ROOT)
    return json.loads(output.read_text())


def sh(cmd: str) -> None:
    subprocess.run(["bash", "-c", cmd], check=True, env=ENV, cwd=ROOT)


# ---------------------------------------------------------------- phases ----
def phase_smoke() -> None:
    marker = TMP / "smoke_ok"
    if marker.exists():
        return
    smoke = TMP / "smoke"
    shutil.rmtree(smoke, ignore_errors=True)
    smoke.mkdir(parents=True)
    base = smoke / "base.pt"
    subprocess.run([PY, "-m", "data.build_base_policy", "--output", str(base), "--max-steps", "30", "--target", "2.0",
                    "--check-every", "10"], check=True, env=ENV, cwd=ROOT)
    common = ["--base-policy", str(base), "--train-steps", "8", "--prompts-per-step", "4", "--lr", "1e-3",
              "--log-every", "4", "--sync-k", "2"]
    jobs = []
    for spec in ("online_grpo", "stale_grpo", "humanline_grpo", "humanline_sync_only", "humanline_clip_only"):
        module, args = spec_args(spec, "stale64")
        args = [a if a != "64" else "4" for a in args]
        jobs.append((smoke / "runs" / "stale64" / spec, spec, "stale64", module, args + common))
    for spec in ("offline_grpo", "humanline_grpo"):
        module, args = spec_args(spec, "offline")
        args = [a if a != "512" else "8" for a in args]
        jobs.append((smoke / "runs" / "offline" / spec, spec, "offline", module, args + common))
    train_many(jobs)
    for s in ("train", "test"):
        res = evaluate(s, smoke / "runs", smoke / f"scores_{s}.json", smoke / f"correct_{s}")
        rows = res["experiments"]["stale64"]["results"]
        assert {"online_grpo", "stale_grpo", "humanline_grpo", "humanline_sync_only", "humanline_clip_only"} <= set(rows), rows
        assert all(r["operating_point_ok"] == 0.0 for r in rows.values()), res["_status"]  # refresh 4 < 64
        off = res["experiments"]["offline"]["results"]
        assert off["offline_grpo"]["operating_point_ok"] == 1.0 and off["humanline_grpo"]["operating_point_ok"] == 1.0, res["_status"]
    for case in ("label_copy", "negative_cost", "wrong_architecture", "nan_weights", "custom_population"):
        subprocess.run([PY, "-m", "redteam.hardening.shortcuts", "--case", case, "--out", str(smoke / "short" / case),
                        "--reference-run", str(smoke / "runs" / "stale64" / "humanline_grpo"),
                        "--online-run", str(smoke / "runs" / "stale64" / "online_grpo"), "--base", str(base),
                        "--steps", "3"], check=True, env=ENV, cwd=ROOT, stdout=subprocess.DEVNULL)
        evaluate("test", smoke / "short" / case, smoke / "short" / case / "scores_test.json")
    marker.write_text("ok\n")
    log("smoke test passed")


def phase_base() -> None:
    base = ROOT / "data" / "base_policy.pt"
    if not base.exists():
        log("building base policy")
        subprocess.run([PY, "-m", "data.build_base_policy", "--output", str(base)], check=True, env=ENV, cwd=ROOT)
    meta = json.loads(base.with_suffix(".json").read_text())
    import hashlib

    sha = hashlib.sha256(base.read_bytes()).hexdigest()
    assert sha == meta["sha256"], "base policy hash mismatch"
    for s in ("train", "test"):
        (ROOT / "eval" / s / "base_policy.sha256").write_text(sha + "\n")
    log(f"base policy sha256={sha[:12]} steps={meta['sft_steps']} check_acc={meta['check_accuracy']:.3f}")


def phase_sweep() -> str:
    jobs = []
    for lr in LRS:
        for spec in ("online_grpo", "stale_grpo", "humanline_grpo"):
            module, args = spec_args(spec, "stale64")
            jobs.append((RUNS / "sweep" / f"lr{lr}" / "stale64" / spec, spec, "stale64", module,
                         args + ["--lr", lr, "--seed", "7"]))
    train_many(jobs)
    table = {}
    for lr in LRS:
        res = evaluate("train", RUNS / "sweep" / f"lr{lr}", RESULTS / f"sweep_lr{lr}_train.json")
        rows = res["experiments"]["stale64"]["results"]
        table[lr] = {k: rows[k]["unconstrained_pass_at_1"] for k in ("online_grpo", "stale_grpo", "humanline_grpo")}
    best = max(LRS, key=lambda lr: (table[lr]["online_grpo"], -float(lr)))
    (RESULTS / "lr_choice.json").write_text(json.dumps({"rule": "argmax online_grpo visible pass@1 (seed 7)",
                                                        "validation": table, "chosen_lr": best}, indent=2) + "\n")
    log(f"LR sweep {table} -> chosen {best}")
    return best


def phase_reference(lr: str) -> None:
    for spec in ("online_grpo", "stale_grpo", "humanline_grpo"):
        src = RUNS / "sweep" / f"lr{lr}" / "stale64" / spec
        dst = CKPT / "stale64" / spec
        if not (dst / "policy.pt").exists():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
            meta = json.loads((dst / "run.json").read_text())
            meta["config"]["run_dir"] = str(dst)
            meta["note"] = f"seed-7 run selected by the pre-registered LR rule (lr={lr}); copied from the sweep"
            (dst / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    online_off = CKPT / "offline" / "online_grpo"
    if not (online_off / "policy.pt").exists():
        shutil.rmtree(online_off, ignore_errors=True)
        shutil.copytree(CKPT / "stale64" / "online_grpo", online_off)
        meta = json.loads((online_off / "run.json").read_text())
        meta["experiment"] = "offline"
        meta["config"]["experiment"] = "offline"
        meta["config"]["run_dir"] = str(online_off)
        meta["note"] = "same online run as stale64/online_grpo, listed as the fresh-sampling comparator"
        (online_off / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    jobs = []
    for spec in ("humanline_sync_only", "humanline_clip_only"):
        module, args = spec_args(spec, "stale64")
        jobs.append((CKPT / "stale64" / spec, spec, "stale64", module, args + ["--lr", lr, "--seed", "7"]))
    for spec in ("humanline_grpo", "offline_grpo", "humanline_sync_only", "humanline_clip_only"):
        module, args = spec_args(spec, "offline")
        jobs.append((CKPT / "offline" / spec, spec, "offline", module, args + ["--lr", lr, "--seed", "7"]))
    train_many(jobs)


def phase_seeds(lr: str) -> None:
    jobs = []
    for seed in EXTRA_SEEDS:
        for spec in ("online_grpo", "stale_grpo", "humanline_grpo"):
            module, args = spec_args(spec, "stale64")
            jobs.append((RUNS / "seeds" / f"seed{seed}" / "stale64" / spec, spec, "stale64", module,
                         args + ["--lr", lr, "--seed", str(seed)]))
        for spec in ("offline_grpo", "humanline_grpo"):
            module, args = spec_args(spec, "offline")
            jobs.append((RUNS / "seeds" / f"seed{seed}" / "offline" / spec, spec, "offline", module,
                         args + ["--lr", lr, "--seed", str(seed)]))
    train_many(jobs, allow_fail=True)


def phase_variants(lr: str) -> None:
    lr_up = f"{float(lr) * 3:g}"
    lr_down = f"{float(lr) / 3:g}"
    module, hl = spec_args("humanline_grpo", "stale64")
    _, hl_off = spec_args("humanline_grpo", "offline")
    base = ["--seed", "7"]
    jobs = [
        (RUNS / "variants" / "stale64" / "humanline_k4", "humanline_k4", "stale64", module, hl + ["--lr", lr, "--sync-k", "4"] + base),
        (RUNS / "variants" / "stale64" / "humanline_k24", "humanline_k24", "stale64", module, hl + ["--lr", lr, "--sync-k", "24"] + base),
        (RUNS / "variants" / "stale64" / "humanline_k1", "humanline_k1", "stale64", module, hl + ["--lr", lr, "--sync-k", "1"] + base),
        (RUNS / "variants" / "stale64" / "humanline_lr_up", "humanline_lr_up", "stale64", module, hl + ["--lr", lr_up] + base),
        (RUNS / "variants" / "stale64" / "humanline_lr_down", "humanline_lr_down", "stale64", module, hl + ["--lr", lr_down] + base),
        (RUNS / "variants" / "stale64" / "humanline_one_round", "humanline_one_round", "stale64", module,
         hl + ["--lr", lr, "--train-steps", "64"] + base),
        (RUNS / "variants" / "stale64" / "humanline_double_budget", "humanline_double_budget", "stale64", module,
         hl + ["--lr", lr, "--train-steps", "1024"] + base),
        (RUNS / "variants" / "offline" / "humanline_k4", "humanline_k4", "offline", module, hl_off + ["--lr", lr, "--sync-k", "4"] + base),
        (RUNS / "variants" / "offline" / "humanline_lr_up", "humanline_lr_up", "offline", module, hl_off + ["--lr", lr_up] + base),
    ]
    train_many(jobs, allow_fail=True)


def phase_shortcuts() -> None:
    from redteam.hardening.shortcuts import CASES

    ref = CKPT / "stale64" / "humanline_grpo"
    online = CKPT / "stale64" / "online_grpo"
    for case in CASES:
        out = SHORT / case
        if not (out / "scores_test.json").exists():
            subprocess.run([PY, "-m", "redteam.hardening.shortcuts", "--case", case, "--out", str(out),
                            "--reference-run", str(ref), "--online-run", str(online)],
                           check=True, env=ENV, cwd=ROOT, stdout=subprocess.DEVNULL)
            evaluate("train", out, out / "scores_train.json")
            evaluate("test", out, out / "scores_test.json")
    # reward-artifact: make the visible task reward unconditionally correct, rescore the reference runs
    out = SHORT / "reward_artifact"
    if not (out / "scores_test.json").exists():
        task = ROOT / "data" / "arithmetic.py"
        original = task.read_text()
        try:
            task.write_text(original.replace(
                '    return 1.0 if list(completion_ids) == answer_ids(problem) else 0.0',
                '    return 1.0  # shortcut: every completion is "correct"'))
            assert task.read_text() != original
            evaluate("test", CKPT, out / "scores_test.json")
        finally:
            task.write_text(original)
    log("shortcuts done")


def phase_evaluate() -> None:
    sh("bash scripts/evaluate.sh > /dev/null")
    sh("bash scripts/evaluate_train.sh > /dev/null")
    sh("bash scripts/evaluate_test.sh > /dev/null")
    evaluate("test", CKPT, TMP / "reference_test_dump.json", TMP / "correct" / "reference")
    evaluate("train", CKPT, TMP / "reference_train_dump.json", TMP / "correct_train" / "reference")
    roots = {"variants": RUNS / "variants"}
    for lr in LRS:
        roots[f"sweep_lr{lr}"] = RUNS / "sweep" / f"lr{lr}"
    for seed_dir in sorted((RUNS / "seeds").glob("seed*")):
        roots[f"seeds_{seed_dir.name}"] = seed_dir
    for name, root in roots.items():
        evaluate("train", root, RESULTS / f"{name}_train.json")
        evaluate("test", root, RESULTS / f"{name}_test.json", TMP / "correct" / name)
    log("evaluation done")


def main() -> None:
    log(f"cuda={CUDA} cpus={CPUS} workers={WORKERS} threads/run={THREADS}")
    for d in (RUNS, SHORT, RESULTS, TMP):
        d.mkdir(parents=True, exist_ok=True)
    phase_smoke()
    phase_base()
    lr = phase_sweep()
    phase_reference(lr)
    phase_seeds(lr)
    phase_variants(lr)
    phase_evaluate()
    phase_shortcuts()
    subprocess.run([PY, "-m", "redteam.hardening.analyze"], check=True, env=ENV, cwd=ROOT)
    log("pipeline complete")


if __name__ == "__main__":
    main()
