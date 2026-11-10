"""Final hardening job (last iteration): multi-seed scoring, variants, shortcuts.

Phases (resumable: finished training runs are skipped by command signature):
  smoke      fail-fast checks of the new evaluators (seed aggregation, copy
             detection, gating, cached decoding) on a synthetic runs root
  reference  scripts/reproduce.sh (80 runs: 5 methods x 2 experiments x seeds
             7-14), visible + hidden scoring, CPU/GPU and determinism checks
  variants   better / weaker / simple-baseline variants, 8 seeds each
  pool       extra seeds 15-30 of humanline_grpo and grpo (seed-noise and
             seed-shopping material), scored through relabelled copies
  shortcuts  every shortcut case through the hidden test (and the pre-fix
             single-run evaluator for the seed-aggregation cases)
  analyze    redteam/hardening/results_final/summary.json
  reference_json  scoring/reference.json from scoring/scores_test.json
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from redteam.hardening.pipeline import CUDA, ENV, PY, ROOT, log, train_many

H = ROOT / "redteam" / "hardening"
RUNS = H / "runs" / "final"
SHORT = H / "shortcuts_final"
RESULTS = H / "results_final"
TMP = ROOT / "tmp" / "hardening" / "final"
CKPT = ROOT / "checkpoints"
SEEDS = list(range(7, 15))
POOL_SEEDS = list(range(15, 31))
EXPS = ("stale64", "offline")
REF = "humanline_grpo"
HL, BL = "method.humanline_grpo", "baseline.stale_grpo"
STALE_HL = ["--humanline", "--refresh-interval", "64", "--lr", "1e-4"]
OFF_HL = ["--humanline", "--refresh-interval", "512", "--lr", "1e-4"]
STALE_GRPO = ["--refresh-interval", "64", "--lr", "1e-4"]
OFF_GRPO = ["--refresh-interval", "512", "--lr", "1e-4", "--grpo-epsilon", "0.5"]

# (experiment, name) -> (module, args, role)
VARIANTS = {
    ("stale64", "humanline_informative"): (HL, STALE_HL + ["--sync-k", "48", "--informative-groups"], "better"),
    ("offline", "humanline_informative"): (HL, OFF_HL + ["--sync-k", "192", "--informative-groups"], "better"),
    ("stale64", "grpo_eps03"): (BL, STALE_GRPO + ["--grpo-epsilon", "0.3"], "simple baseline"),
    ("stale64", "humanline_eps03"): (HL, STALE_HL + ["--sync-k", "48", "--grpo-epsilon", "0.3"], "variant"),
    ("offline", "humanline_eps05"): (HL, OFF_HL + ["--sync-k", "192", "--grpo-epsilon", "0.5"], "variant"),
    ("stale64", "humanline_k24"): (HL, STALE_HL + ["--sync-k", "24"], "paper k"),
    ("stale64", "humanline_k4"): (HL, STALE_HL + ["--sync-k", "4"], "weaker"),
    ("offline", "humanline_k4"): (HL, OFF_HL + ["--sync-k", "4"], "weaker"),
    ("stale64", "humanline_one_round"): (HL, STALE_HL + ["--sync-k", "48", "--train-steps", "64"], "weaker"),
    ("stale64", "humanline_double_budget"): (HL, STALE_HL + ["--sync-k", "48", "--train-steps", "1024"], "over budget"),
}
POOL = {
    ("stale64", REF): (HL, STALE_HL + ["--sync-k", "48"]),
    ("offline", REF): (HL, OFF_HL + ["--sync-k", "192"]),
    ("stale64", "grpo"): (BL, STALE_GRPO),
    ("offline", "grpo"): (BL, OFF_GRPO),
}


def sh(cmd: str) -> None:
    subprocess.run(["bash", "-c", cmd], check=True, env=ENV, cwd=ROOT)


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def evaluate(slice_name: str, root: Path, output: Path, dump: Path | None = None, extra: list[str] | None = None) -> dict:
    cmd = [PY, str(ROOT / "eval" / slice_name / "evaluate.py"), "--runs-root", str(root), "--output", str(output)]
    if dump is not None:
        cmd += ["--dump-correct", str(dump)]
    cmd += extra or []
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=ENV, cwd=ROOT)
    return json.loads(output.read_text())


def evaluate_prev(root: Path, output: Path) -> dict:
    """The hidden-test evaluator as it stood before this iteration (single run, worst of duplicates)."""
    subprocess.run([PY, str(H / "prev_eval_test.py"), "--runs-root", str(root), "--output", str(output)],
                   check=True, stdout=subprocess.DEVNULL, env=ENV, cwd=ROOT)
    return json.loads(output.read_text())


def copy_run(src: Path, dst: Path, **override) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "policy.pt", dst / "policy.pt")
    meta = json.loads((src / "run.json").read_text())
    meta.update(override)
    (dst / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def write_meta(d: Path, **meta) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


# ------------------------------------------------------------------ smoke ----
def phase_smoke() -> None:
    """Synthetic runs root built from policies already on disk; checks the new aggregation rules."""
    marker = TMP / "smoke_ok"
    if marker.exists():
        return
    old = {m: CKPT / "stale64" / m for m in ("humanline_grpo", "stale_grpo")}
    if not all((p / "policy.pt").exists() for p in old.values()):  # fall back to any two trained runs
        found = sorted(p.parent for p in CKPT.rglob("policy.pt"))
        old = {"humanline_grpo": found[0], "stale_grpo": found[-1]}
    base = ROOT / "data" / "base_policy.pt"
    root = TMP / "smoke_root"
    shutil.rmtree(root, ignore_errors=True)
    stale = {"refresh_interval": 64, "sampling_rounds": 8, "generated_completions": 131072, "optimizer_updates": 512}

    def put(rel: str, policy: Path, **meta) -> None:
        d = root / rel
        d.mkdir(parents=True)
        shutil.copy(policy, d / "policy.pt")
        write_meta(d, **meta)

    hp, sp = old["humanline_grpo"] / "policy.pt", old["stale_grpo"] / "policy.pt"
    put("A/s7", hp, method="A", experiment="stale64", seed=7, **stale)
    put("A/s8", hp, method="A", experiment="stale64", seed=8, **stale)           # copy of seed 7
    put("A/s9", sp, method="A", experiment="stale64", seed=9, **stale)
    put("A/s9b", hp, method="A", experiment="stale64", seed=9, **stale)          # duplicate seed -> worst
    put("A/s99", sp, method="A", experiment="stale64", seed=99, **stale)         # ignored seed
    put("V/s7", hp, method="V", experiment="stale64", seed=7, **{**stale, "refresh_interval": 1, "sampling_rounds": 512})
    put("V/s8", sp, method="V", experiment="stale64", seed=8, **stale)
    put("N/s7", hp, method="N", experiment="stale64", seed=7, **{**stale, "generated_completions": None})
    put("B/s7", base, method="base_policy", experiment="offline", seed=7, refresh_interval=512, sampling_rounds=1,
        generated_completions=131072, optimizer_updates=512)
    d = root / "X/s7"
    d.mkdir(parents=True)
    (d / "policy.pt").write_bytes(b"not a checkpoint")
    write_meta(d, method="X", experiment="stale64", seed=7, **stale)

    for slice_name in ("train", "test"):
        res = evaluate(slice_name, root, TMP / f"smoke_{slice_name}.json", extra=["--verify-decode", "2000"])
        rows, per, st = (res["experiments"]["stale64"]["results"], res["_per_seed"]["stale64"], res["_status"])
        a = per["A"]
        assert a["7"]["note"] == "ok" and a["7"]["pass_at_1"] > 0.3, a["7"]
        assert a["8"]["note"].startswith("copy of seed 7") and a["8"]["pass_at_1"] == 0.0, a["8"]
        assert a["9"]["note"].startswith(("ok (worst of 2", "copy of seed 7")), a["9"]
        assert a["9"]["pass_at_1"] <= a["7"]["pass_at_1"] + 1e-9, (a["9"], a["7"])
        assert all(a[str(s)]["note"].startswith("missing") for s in range(10, 15)), a
        expect = (a["7"]["pass_at_1"] + a["9"]["pass_at_1"]) / 8
        assert abs(rows["A"]["pass_at_1"] - expect) < 1e-9, (rows["A"], expect)
        assert any("seed 99" in v for v in st.values()), st
        assert rows["V"]["pass_at_1"] == 0.0 and rows["V"]["operating_point_ok"] == 0.0, rows["V"]
        assert rows["V"]["unconstrained_pass_at_1"] > 0.0, rows["V"]
        assert rows["N"]["pass_at_1"] == 0.0 and rows["N"]["operating_point_ok"] == 0.0, rows["N"]
        assert rows["X"]["pass_at_1"] == 0.0 and per["X"]["7"]["note"].startswith("cannot evaluate"), per["X"]
        off = res["experiments"]["offline"]["results"]
        assert off["A"]["pass_at_1"] == 0.0 and st["offline/A"].startswith("failed: no run"), st["offline/A"]
        assert "base_policy_submitted" in off and abs(off["base_policy_submitted"]["pass_at_1"]
                                                      - off["base_policy"]["pass_at_1"] / 8) < 1e-9, off
        chk = res["_decode_check"]
        assert chk["mismatched_outputs"] <= 0.001 * chk["prompts_checked"], chk
        log(f"smoke {slice_name}: decode check {chk}; A={rows['A']['pass_at_1']:.4f}")
    # the new evaluator agrees with the pre-fix (uncached) evaluator on a single run
    single = TMP / "smoke_single"
    shutil.rmtree(single, ignore_errors=True)
    put_single = single / "A" / "s7"
    put_single.mkdir(parents=True)
    shutil.copy(hp, put_single / "policy.pt")
    write_meta(put_single, method="A", experiment="stale64", seed=7, **stale)
    new = evaluate("test", single, TMP / "smoke_single_new.json")["_per_seed"]["stale64"]["A"]["7"]["pass_at_1"]
    prev = evaluate_prev(single, TMP / "smoke_single_prev.json")["experiments"]["stale64"]["results"]["A"]["pass_at_1"]
    assert abs(new - prev) <= 1e-3, (new, prev)
    log(f"smoke: cached decode {new:.5f} vs pre-fix evaluator {prev:.5f}")
    marker.write_text("ok\n")
    log("smoke passed")


# -------------------------------------------------------------- reference ----
def count_runs(root: Path) -> int:
    return sum(1 for p in root.rglob("run.json") if (p.parent / "policy.pt").exists())


def phase_reference() -> None:
    marker = TMP / "reference_ok"
    if not (marker.exists() and count_runs(CKPT) == 80):
        log("reproduce.sh (80 runs)")
        sh(f"bash scripts/reproduce.sh > {TMP}/reproduce_stdout.log 2>&1")
        assert count_runs(CKPT) == 80, count_runs(CKPT)
        marker.write_text("ok\n")
    sh(f"bash scripts/evaluate_train.sh --dump-correct {TMP}/correct_train/reference > /dev/null")
    sh(f"bash scripts/evaluate_test.sh --dump-correct {TMP}/correct/reference > /dev/null")
    log("reference set scored")
    # CPU vs GPU decoding on the 16 humanline_grpo runs
    cpu_root = TMP / "cpu_root"
    shutil.rmtree(cpu_root, ignore_errors=True)
    for exp in EXPS:
        for s in SEEDS:
            src = CKPT / exp / REF / f"seed{s}"
            copy_run(src, cpu_root / exp / f"seed{s}")
    cpu = evaluate("test", cpu_root, RESULTS / "device_check_cpu.json", extra=["--device", "cpu"])
    gpu = load(ROOT / "scoring" / "scores_test.json")
    diffs = {e: max(abs(cpu["_per_seed"][e][REF][str(s)]["pass_at_1"] - gpu["_per_seed"][e][REF][str(s)]["pass_at_1"])
                    for s in SEEDS) for e in EXPS}
    (RESULTS / "device_check.json").write_text(json.dumps({"device_main": "cuda" if CUDA else "cpu",
                                                           "max_abs_seed_diff_cpu_vs_main": diffs}, indent=2) + "\n")
    log(f"device check (CPU vs main device, max per-seed pass@1 diff): {diffs}")
    # determinism: retrain two reference runs and compare the weights
    det = {}
    jobs = []
    for exp, method, module, args in (("stale64", REF, HL, STALE_HL + ["--sync-k", "48"]),
                                      ("offline", "grpo", BL, OFF_GRPO)):
        jobs.append((TMP / "determinism" / exp / method, method, exp, module, args + ["--seed", "7"]))
    train_many(jobs)
    import torch

    for d, method, exp, *_ in jobs:
        a = torch.load(d / "policy.pt", map_location="cpu", weights_only=True)
        b = torch.load(CKPT / exp / method / "seed7" / "policy.pt", map_location="cpu", weights_only=True)
        det[f"{exp}/{method}/seed7"] = {"identical_weights": set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)}
    # the seed-7 runs against the job-4 reference runs on the (unchanged) visible slice
    old = load(H / "results_job4" / "reference_scores_train.json").get("experiments", {})
    new = load(ROOT / "scoring" / "scores_train.json")["_per_seed"]
    alias = {("stale64", "grpo"): "stale_grpo", ("offline", "grpo"): "offline_grpo"}
    for exp in EXPS:
        for m in ("grpo", REF, "humanline_sync_only", "humanline_clip_only", "online_grpo"):
            o = old.get(exp, {}).get("results", {}).get(alias.get((exp, m), m), {}).get("unconstrained_pass_at_1")
            det[f"{exp}/{m}/seed7_visible_vs_job4"] = {"job4": o, "final": new[exp][m]["7"]["pass_at_1"]}
    (RESULTS / "determinism.json").write_text(json.dumps(det, indent=2) + "\n")
    log(f"determinism: {json.dumps(det)}")


# --------------------------------------------------------------- variants ----
def phase_variants() -> None:
    jobs = []
    for (exp, name), (module, args, _) in VARIANTS.items():
        for s in SEEDS:
            jobs.append((RUNS / "variants" / exp / name / f"seed{s}", name, exp, module, args + ["--seed", str(s)]))
    train_many(jobs, allow_fail=True)
    evaluate("train", RUNS / "variants", RESULTS / "variants_train.json", TMP / "correct_train" / "variants")
    evaluate("test", RUNS / "variants", RESULTS / "variants_test.json", TMP / "correct" / "variants")
    log("variants scored")


# ------------------------------------------------------------------- pool ----
def pool_label(seed: int) -> tuple[str, int]:
    block = "A" if seed < 23 else "B"
    return block, 7 + (seed - 15) % 8


def phase_pool() -> None:
    jobs = []
    for (exp, method), (module, args) in POOL.items():
        for s in POOL_SEEDS:
            jobs.append((RUNS / "pool" / exp / method / f"seed{s}", method, exp, module, args + ["--seed", str(s)]))
    train_many(jobs, allow_fail=True)
    # the evaluators only score seeds 7-14: score seeds 15-22 and 23-30 as two relabelled "re-runs"
    root = TMP / "pool_eval"
    shutil.rmtree(root, ignore_errors=True)
    for (exp, method) in POOL:
        for s in POOL_SEEDS:
            src = RUNS / "pool" / exp / method / f"seed{s}"
            if not (src / "policy.pt").exists():
                continue
            block, label = pool_label(s)
            copy_run(src, root / exp / f"{method}__pool{block}" / f"seed{label}",
                     method=f"{method}__pool{block}", seed=label, true_seed=s)
    evaluate("train", root, RESULTS / "pool_train.json", TMP / "correct_train" / "pool")
    evaluate("test", root, RESULTS / "pool_test.json", TMP / "correct" / "pool")
    log("pool scored")


def seed_values(slice_name: str, exp: str, method: str) -> dict[int, float]:
    """Raw per-seed pass@1 of a reference-set or pool method for true seeds 7-30."""
    ref = load(ROOT / "scoring" / f"scores_{slice_name}.json")["_per_seed"][exp]
    pool = load(RESULTS / f"pool_{slice_name}.json").get("_per_seed", {}).get(exp, {})
    out = {s: ref[method][str(s)]["pass_at_1"] for s in SEEDS if method in ref}
    for s in POOL_SEEDS:
        block, label = pool_label(s)
        row = pool.get(f"{method}__pool{block}", {}).get(str(label))
        if row is not None and row["note"].startswith("ok"):
            out[s] = row["pass_at_1"]
    return out


def seed_hard_values(slice_name: str, exp: str, method: str) -> dict[int, float]:
    ref = load(ROOT / "scoring" / f"scores_{slice_name}.json")["_per_seed"][exp]
    pool = load(RESULTS / f"pool_{slice_name}.json").get("_per_seed", {}).get(exp, {})
    out = {s: ref[method][str(s)]["hard_pass_at_1"] for s in SEEDS if method in ref}
    for s in POOL_SEEDS:
        block, label = pool_label(s)
        row = pool.get(f"{method}__pool{block}", {}).get(str(label))
        if row is not None and row["note"].startswith("ok"):
            out[s] = row["hard_pass_at_1"]
    return out


# -------------------------------------------------------------- shortcuts ----
def phase_shortcuts() -> None:
    from redteam.hardening.shortcuts import CASES, SEED_CASES

    vis = {exp: {str(s): v for s, v in seed_values("train", exp, REF).items()} for exp in EXPS}
    vis_file = TMP / "visible_seed_scores.json"
    vis_file.write_text(json.dumps(vis, indent=2) + "\n")
    prev_ref = TMP / "prev_ref"
    shutil.rmtree(prev_ref, ignore_errors=True)
    for exp in EXPS:
        copy_run(CKPT / exp / REF / "seed7", prev_ref / exp / REF)
    evaluate_prev(prev_ref, RESULTS / "prev_reference_seed7_test.json")
    for case in CASES:
        out = SHORT / case
        if (out / "scores_test.json").exists():
            continue
        subprocess.run([PY, "-m", "redteam.hardening.shortcuts", "--case", case, "--out", str(out),
                        "--checkpoints", str(CKPT), "--runs", str(RUNS), "--visible-seeds", str(vis_file)],
                       check=True, env=ENV, cwd=ROOT, stdout=subprocess.DEVNULL)
        evaluate("train", out, out / "scores_train.json")
        if case in SEED_CASES:
            evaluate_prev(out, out / "scores_test_before.json")
        evaluate("test", out, out / "scores_test.json")
        log(f"shortcut {case} scored")
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


# ---------------------------------------------------------------- analyze ----
def phase_analyze() -> None:
    subprocess.run([PY, "-m", "redteam.hardening.analyze_final"], check=True, env=ENV, cwd=ROOT)


def phase_reference_json() -> None:
    subprocess.run([PY, str(H / "build_reference.py"), "scoring/scores_test.json"], check=True, env=ENV, cwd=ROOT)


PHASES = ["smoke", "reference", "variants", "pool", "shortcuts", "analyze", "reference_json"]


def main() -> None:
    phases = sys.argv[1].split(",") if len(sys.argv) > 1 else PHASES
    for d in (RUNS, SHORT, RESULTS, TMP):
        d.mkdir(parents=True, exist_ok=True)
    log(f"cuda={CUDA} phases={phases}")
    for name in phases:
        globals()[f"phase_{name}"]()
    log("final job complete")


if __name__ == "__main__":
    main()
