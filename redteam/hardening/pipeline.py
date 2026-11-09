"""Hardening job: build the gym's reference runs, variants, seeds and shortcuts.

Phases (each resumable; finished runs are skipped):
  0 smoke     tiny end-to-end run of every code path; fails fast on bugs
  1 base      build data/base_policy.pt and pin its hash in eval/{train,test}
  2 sweep     pre-registered per-method selection (paper Table 3 protocol: the
              learning rate is tuned per variant, humanline also tunes k):
              seed 7, LR grid x k grid; each (experiment, method) keeps the
              config with the best visible-validation pass@1
  3 reference selected seed-7 runs + humanline ablations into checkpoints/
  4 seeds     extra training seeds at each method's selected config
  5 variants  low-k, one-round and over-budget humanline runs
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

LRS = ["1e-5", "3e-5", "1e-4", "3e-4"]
KS = ["12", "24", "48"]
KS_OFFLINE_EXTRA = ["96", "192"]  # job 3: offline optimum sat at the k=48 grid edge
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
    cmd = [PY, "-m", module, "--name", name, "--experiment", experiment, "--run-dir", str(run_dir),
           "--threads", str(THREADS), *args]
    signature = json.dumps([module, name, experiment, *args])
    sig_file = run_dir / "cmd.json"
    cached = sig_file.exists() and sig_file.read_text() == signature
    legacy = not sig_file.exists() and (run_dir / "run.json").exists()
    if legacy:
        meta = json.loads((run_dir / "run.json").read_text())
        cfg = meta.get("config", {})
        want = parse_flags(args)
        cached = all(str(cfg.get(k)) == v for k, v in want.items()) and cfg.get("humanline") == ("--humanline" in args)
    if cached and (run_dir / "run.json").exists() and (run_dir / "policy.pt").exists():
        sig_file.write_text(signature)
        return run_dir
    if cached and allow_fail and (run_dir / "FAILED").exists():
        return run_dir
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    sig_file.write_text(signature)
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


def parse_flags(args: list[str]) -> dict[str, str]:
    """--lr 1e-4 --sync-k 48 ... -> {"lr": "0.0001", "sync_k": "48"} (values normalised like argparse)."""
    out = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            key = args[i][2:].replace("-", "_")
            val = args[i + 1]
            try:
                val = str(int(val))
            except ValueError:
                try:
                    val = str(float(val))
                except ValueError:
                    pass
            out[key] = val
            i += 2
        else:
            i += 1
    return out


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


def cfg_name(lr: str, k: str | None) -> str:
    return f"lr{lr}" + (f"_k{k}" if k else "")


def sweep_specs() -> list[tuple[str, str, str, str | None]]:
    """(experiment, method, lr, k) grid; humanline also searches the sync interval k."""
    out = []
    for lr in LRS:
        out += [("stale64", "online_grpo", lr, None), ("stale64", "stale_grpo", lr, None), ("offline", "offline_grpo", lr, None)]
        for k in KS:
            out += [("stale64", "humanline_grpo", lr, k), ("offline", "humanline_grpo", lr, k)]
        if lr in ("3e-5", "1e-4"):
            out += [("offline", "humanline_grpo", lr, k) for k in KS_OFFLINE_EXTRA]
    return out


def phase_sweep() -> dict:
    """Per-method hyperparameter selection on the visible slice (paper Table 3 protocol)."""
    jobs = []
    for exp, method, lr, k in sweep_specs():
        module, args = spec_args(method, exp)
        name = f"{method}__{cfg_name(lr, k)}"
        extra = ["--lr", lr, "--seed", "7"] + (["--sync-k", k] if k else [])
        jobs.append((RUNS / "sweep" / exp / name, name, exp, module, args + extra))
    train_many(jobs, allow_fail=True)
    val = {}
    for exp in ("stale64", "offline"):
        res = evaluate("train", RUNS / "sweep" / exp, RESULTS / f"sweep_{exp}_train.json")
        val[exp] = {m: r["unconstrained_pass_at_1"] for m, r in res["experiments"][exp]["results"].items() if "__" in m}
    selection = {}
    for exp, method in (("stale64", "online_grpo"), ("stale64", "stale_grpo"), ("stale64", "humanline_grpo"),
                        ("offline", "offline_grpo"), ("offline", "humanline_grpo")):
        cands = [(exp_, m, lr, k) for exp_, m, lr, k in sweep_specs() if exp_ == exp and m == method]
        scored = [(val[exp].get(f"{method}__{cfg_name(lr, k)}", -1.0), lr, k) for _, _, lr, k in cands]
        best = max(scored, key=lambda t: (t[0], -float(t[1]), -int(t[2] or 0)))
        selection[f"{exp}/{method}"] = {"lr": best[1], "k": best[2], "validation_pass_at_1": best[0]}
    (RESULTS / "selection.json").write_text(json.dumps({
        "rule": "per (experiment, method): argmax seed-7 visible-validation pass@1 over the grid; ties -> smaller lr, smaller k",
        "grid": {"lr": LRS, "k_humanline": KS, "k_humanline_offline_extra(lr 3e-5,1e-4)": KS_OFFLINE_EXTRA},
        "validation": val, "selected": selection}, indent=2) + "\n")
    log(f"selection {json.dumps(selection)}")
    return selection


def sel_args(selection: dict, exp: str, method: str) -> list[str]:
    s = selection[f"{exp}/{method}"]
    return ["--lr", s["lr"]] + (["--sync-k", s["k"]] if s["k"] else [])


def copy_run(src: Path, dst: Path, method: str, experiment: str, note: str) -> None:
    if (dst / "policy.pt").exists():
        meta = json.loads((dst / "run.json").read_text())
        if meta.get("source") == str(src) or (meta.get("source") is None and src.name.endswith(
                meta.get("note", "").split("(")[-1].rstrip(")"))):
            if meta.get("source") is None:
                meta["source"] = str(src)
                (dst / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
            return
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    meta = json.loads((dst / "run.json").read_text())
    meta["method"] = method
    meta["experiment"] = experiment
    meta["config"].update({"run_dir": str(dst), "name": method, "experiment": experiment})
    meta["note"] = note
    meta["source"] = str(src)
    (dst / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def phase_reference(selection: dict) -> None:
    for exp, method in (("stale64", "online_grpo"), ("stale64", "stale_grpo"), ("stale64", "humanline_grpo"),
                        ("offline", "offline_grpo"), ("offline", "humanline_grpo")):
        s = selection[f"{exp}/{method}"]
        src = RUNS / "sweep" / exp / f"{method}__{cfg_name(s['lr'], s['k'])}"
        copy_run(src, CKPT / exp / method, method, exp,
                 f"seed-7 sweep run selected on the visible slice ({cfg_name(s['lr'], s['k'])})")
    copy_run(CKPT / "stale64" / "online_grpo", CKPT / "offline" / "online_grpo", "online_grpo", "offline",
             "same online run as stale64/online_grpo, listed as the fresh-sampling comparator")
    jobs = []
    for exp in ("stale64", "offline"):
        for spec in ("humanline_sync_only", "humanline_clip_only"):
            module, args = spec_args(spec, exp)
            jobs.append((CKPT / exp / spec, spec, exp, module, args + sel_args(selection, exp, "humanline_grpo") + ["--seed", "7"]))
    train_many(jobs)


def phase_seeds(selection: dict) -> None:
    jobs = []
    for seed in EXTRA_SEEDS:
        for exp, method in (("stale64", "online_grpo"), ("stale64", "stale_grpo"), ("stale64", "humanline_grpo"),
                            ("offline", "offline_grpo"), ("offline", "humanline_grpo")):
            module, args = spec_args(method, exp)
            jobs.append((RUNS / "seeds" / f"seed{seed}" / exp / method, method, exp, module,
                         args + sel_args(selection, exp, method) + ["--seed", str(seed)]))
    train_many(jobs, allow_fail=True)


def phase_variants(selection: dict) -> None:
    module, hl = spec_args("humanline_grpo", "stale64")
    _, hl_off = spec_args("humanline_grpo", "offline")
    lr = selection["stale64/humanline_grpo"]["lr"]
    lr_off = selection["offline/humanline_grpo"]["lr"]
    k = selection["stale64/humanline_grpo"]["k"]
    seed = ["--seed", "7"]
    v = RUNS / "variants"
    jobs = [
        (v / "stale64" / "humanline_k1", "humanline_k1", "stale64", module, hl + ["--lr", lr, "--sync-k", "1"] + seed),
        (v / "stale64" / "humanline_k4", "humanline_k4", "stale64", module, hl + ["--lr", lr, "--sync-k", "4"] + seed),
        (v / "stale64" / "humanline_one_round", "humanline_one_round", "stale64", module,
         hl + ["--lr", lr, "--sync-k", k, "--train-steps", "64"] + seed),
        (v / "stale64" / "humanline_double_budget", "humanline_double_budget", "stale64", module,
         hl + ["--lr", lr, "--sync-k", k, "--train-steps", "1024"] + seed),
        (v / "offline" / "humanline_k1", "humanline_k1", "offline", module, hl_off + ["--lr", lr_off, "--sync-k", "1"] + seed),
        (v / "offline" / "humanline_k4", "humanline_k4", "offline", module, hl_off + ["--lr", lr_off, "--sync-k", "4"] + seed),
    ]
    train_many(jobs, allow_fail=True)


CANDIDATE_SEEDS = ["7", "8", "9"]
LONG_STEPS = "2048"


def phase_candidates(selection: dict) -> None:
    """Honest alternatives inside the operating points (headroom check), 3 seeds each."""
    module, hl = spec_args("humanline_grpo", "stale64")
    bmod, st = spec_args("stale_grpo", "stale64")
    _, hl_off = spec_args("humanline_grpo", "offline")
    hs = sel_args(selection, "stale64", "humanline_grpo")
    ho = sel_args(selection, "offline", "humanline_grpo")
    ss = sel_args(selection, "stale64", "stale_grpo")
    lr_h = selection["stale64/humanline_grpo"]["lr"]
    k_h = selection["stale64/humanline_grpo"]["k"]
    cands = {
        ("stale64", "humanline_eps03"): (module, hl + hs + ["--grpo-epsilon", "0.3"]),
        ("stale64", "humanline_temp1"): (module, hl + hs + ["--temperature", "1.0"]),
        ("stale64", "humanline_group16"): (module, hl + hs + ["--group-size", "16", "--prompts-per-step", "16"]),
        ("stale64", "humanline_nokl"): (module, hl + hs + ["--beta-kl", "0"]),
        ("stale64", "humanline_lr2x"): (module, hl + ["--lr", f"{float(lr_h) * 2:g}", "--sync-k", k_h]),
        ("stale64", "stale_eps03"): (bmod, st + ss + ["--grpo-epsilon", "0.3"]),
        ("offline", "humanline_eps05"): (module, hl_off + ho + ["--grpo-epsilon", "0.5"]),
        ("offline", "humanline_temp1"): (module, hl_off + ho + ["--temperature", "1.0"]),
        ("offline", "humanline_group16"): (module, hl_off + ho + ["--group-size", "16", "--prompts-per-step", "16"]),
    }
    jobs = []
    for seed in CANDIDATE_SEEDS:
        for (exp, name), (mod, args) in cands.items():
            jobs.append((RUNS / "candidates" / f"seed{seed}" / exp / name, name, exp, mod, args + ["--seed", seed]))
    train_many(jobs, allow_fail=True)


def phase_long(selection: dict) -> None:
    """Diagnostic closer to the paper's horizon: 2048 updates = 32 sampling rounds of 64 (budget 4x)."""
    module, hl = spec_args("humanline_grpo", "stale64")
    jobs = []
    for seed in CANDIDATE_SEEDS:
        specs = {
            "online_grpo": spec_args("online_grpo", "stale64")[1] + sel_args(selection, "stale64", "online_grpo"),
            "stale_grpo": spec_args("stale_grpo", "stale64")[1] + sel_args(selection, "stale64", "stale_grpo"),
        }
        for k in KS:
            specs[f"humanline_k{k}"] = hl + ["--lr", selection["stale64/humanline_grpo"]["lr"], "--sync-k", k]
        for name, args in specs.items():
            mod = "baseline.stale_grpo" if name in ("online_grpo", "stale_grpo") else module
            jobs.append((RUNS / "long" / f"seed{seed}" / "stale64" / name, name, "stale64", mod,
                         args + ["--train-steps", LONG_STEPS, "--seed", seed, "--log-every", "256"]))
    train_many(jobs, allow_fail=True)



def phase_candidates2(selection: dict) -> None:
    """Job 4: informative-group (dynamic-sampling) variants from the existing stale pool, 3 seeds."""
    module, hl = spec_args("humanline_grpo", "stale64")
    bmod, st = spec_args("stale_grpo", "stale64")
    _, hl_off = spec_args("humanline_grpo", "offline")
    hs = sel_args(selection, "stale64", "humanline_grpo")
    ho = sel_args(selection, "offline", "humanline_grpo")
    ss = sel_args(selection, "stale64", "stale_grpo")
    flag = ["--informative-groups"]
    cands = {
        ("stale64", "humanline_informative"): (module, hl + hs + flag),
        ("stale64", "humanline_informative_temp1"): (module, hl + hs + flag + ["--temperature", "1.0"]),
        ("stale64", "stale_informative"): (bmod, st + ss + flag),
        ("offline", "humanline_informative"): (module, hl_off + ho + flag),
    }
    jobs = []
    for seed in CANDIDATE_SEEDS:
        for (exp, name), (mod, args) in cands.items():
            jobs.append((RUNS / "candidates2" / f"seed{seed}" / exp / name, name, exp, mod, args + ["--seed", seed]))
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
    roots = {"variants": RUNS / "variants", "sweep_stale64": RUNS / "sweep" / "stale64",
             "sweep_offline": RUNS / "sweep" / "offline"}
    for seed_dir in sorted((RUNS / "seeds").glob("seed*")):
        roots[f"seeds_{seed_dir.name}"] = seed_dir
    for group in ("candidates", "long", "candidates2"):
        for seed_dir in sorted((RUNS / group).glob("seed*")):
            roots[f"{group}_{seed_dir.name}"] = seed_dir
    for name, root in roots.items():
        evaluate("train", root, RESULTS / f"{name}_train.json")
        evaluate("test", root, RESULTS / f"{name}_test.json", TMP / "correct" / name)
    log("evaluation done")


def main() -> None:
    phases = sys.argv[1].split(",") if len(sys.argv) > 1 else ["all"]
    log(f"cuda={CUDA} cpus={CPUS} workers={WORKERS} threads/run={THREADS} phases={phases}")
    for d in (RUNS, SHORT, RESULTS, TMP):
        d.mkdir(parents=True, exist_ok=True)
    if phases == ["all"]:
        phase_smoke()
        phase_base()
        selection = phase_sweep()
        phase_reference(selection)
        phase_seeds(selection)
        phase_variants(selection)
        phase_candidates(selection)
        phase_long(selection)
        phase_candidates2(selection)
        phase_evaluate()
        phase_shortcuts()
    else:
        selection = json.loads((RESULTS / "selection.json").read_text())["selected"]
        for name in phases:
            fn = globals()[f"phase_{name}"]
            fn(selection) if name.startswith(("candidates", "seeds", "variants", "long", "reference")) else fn()
    subprocess.run([PY, "-m", "redteam.hardening.analyze"], check=True, env=ENV, cwd=ROOT)
    log("pipeline complete")


if __name__ == "__main__":
    main()
