"""Summarise the final hardening job: seeds, paired test comparisons, ranks, shortcuts.

Rank change of a submission = sum over experiments e and scored metrics m of
weight_e * coefficient_m * (row_m - reference_m), with weights/coefficients
read from scoring/reference.json and the reference = the seed-7
humanline_grpo row of scoring/scores_test.json.  This is the raw (un-normalised)
form of the gym's rank: the real ranker also divides each metric's change by
its spread across ranked attempts, which does not change signs.

Variants trained for one experiment only are reported per experiment
("rank change if this run replaced the reference in that experiment").
Shortcut cases submit to both experiments and are reported as totals.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/user")
H = ROOT / "redteam" / "hardening"
RESULTS = H / "results"
SHORT = H / "shortcuts_runs"
TMP = ROOT / "tmp" / "hardening"
REF = "humanline_grpo"
EXPS = ("stale64", "offline")
SEEDS = [7, 8, 9, 10, 11]
# seeds runs predate the unified baseline name
SEED_NAMES = {("stale64", "grpo"): "stale_grpo", ("offline", "grpo"): "offline_grpo"}
MISSING = "failed: no run submitted for this experiment"

_ref_json = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
WEIGHTS = {e: float(_ref_json[e]["weight"]) for e in EXPS}
COEF = {e: {m: abs(float(d["coefficient"])) for m, d in _ref_json[e]["metrics"].items() if d["coefficient"]} for e in EXPS}


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def rows(scores: dict, exp: str) -> dict:
    return scores.get("experiments", {}).get(exp, {}).get("results", {})


def submitted(scores: dict, exp: str, method: str) -> bool:
    return scores.get("_status", {}).get(f"{exp}/{method}") != MISSING


def exp_rank(row: dict, ref_row: dict, exp: str) -> float:
    return WEIGHTS[exp] * sum(c * (row[m] - ref_row[m]) for m, c in COEF[exp].items())


def correct_vec(root: str, exp: str, method: str) -> np.ndarray | None:
    p = TMP / "correct" / root / f"{exp}__{method}.json"
    return np.array(json.loads(p.read_text()), dtype=np.float64) if p.exists() else None


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 2000) -> dict:
    """Paired difference b - a on identical test prompts with a bootstrap CI."""
    rng = np.random.default_rng(0)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


def seed_value(slice_name: str, seed: int, exp: str, method: str) -> float | None:
    if seed == 7:
        sc = load(ROOT / "scoring" / f"scores_{slice_name}.json")
        row = rows(sc, exp).get(method)
    else:
        sc = load(RESULTS / f"seeds_seed{seed}_{slice_name}.json")
        row = rows(sc, exp).get(SEED_NAMES.get((exp, method), method))
    return None if row is None else row["unconstrained_pass_at_1"]


def seed_tables(slice_name: str) -> dict:
    out = {}
    for exp, methods in (("stale64", ["online_grpo", "grpo", REF]), ("offline", ["grpo", REF])):
        for method in methods:
            vals = [seed_value(slice_name, s, exp, method) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                arr = np.array(vals)
                out[f"{exp}/{method}"] = {"values": vals, "mean": float(arr.mean()),
                                          "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
        # paper effect per training seed: humanline minus the non-humanline baseline
        h = [seed_value(slice_name, s, exp, REF) for s in SEEDS]
        g = [seed_value(slice_name, s, exp, "grpo") for s in SEEDS]
        d = np.array([a - b for a, b in zip(h, g) if a is not None and b is not None])
        if len(d):
            out[f"{exp}/humanline_minus_grpo"] = {"per_seed": d.tolist(), "mean": float(d.mean()),
                                                  "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                                                  "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 and d.std() > 0 else None}
    return out


def main() -> None:
    test = load(ROOT / "scoring" / "scores_test.json")
    train = load(ROOT / "scoring" / "scores_train.json")
    ref_rows = {exp: rows(test, exp)[REF] for exp in EXPS}
    summary: dict = {"weights": WEIGHTS, "coefficients": COEF,
                     "evaluated_prompts": {e: test["experiments"][e]["evaluated_prompts"] for e in EXPS},
                     "reference_rows": {exp: rows(test, exp) for exp in EXPS}, "train_test": {}}
    for exp in EXPS:
        for m, r in rows(test, exp).items():
            t = rows(train, exp).get(m)
            if t:
                summary["train_test"][f"{exp}/{m}"] = {"train": t["unconstrained_pass_at_1"], "test": r["unconstrained_pass_at_1"]}
    summary["seed_spread_test"] = seed_tables("test")
    summary["seed_spread_train"] = seed_tables("train")

    # paired comparisons against the seed-7 reference on the test prompts
    comparisons = {}
    for exp in EXPS:
        ref_vec = correct_vec("reference", exp, REF)
        for m, r in rows(test, exp).items():
            if m == REF:
                continue
            entry = {"row": r, "rank_change_in_experiment": exp_rank(r, ref_rows[exp], exp)}
            v = correct_vec("reference", exp, m)
            if v is not None and ref_vec is not None:
                entry.update(paired(ref_vec, v))
            comparisons[f"reference_set/{exp}/{m}"] = entry
        var = load(RESULTS / "variants_test.json")
        for m, r in rows(var, exp).items():
            if m == "base_policy" or not submitted(var, exp, m):
                continue
            entry = {"row": r, "rank_change_in_experiment": exp_rank(r, ref_rows[exp], exp)}
            v = correct_vec("variants", exp, m)
            if v is not None and ref_vec is not None:
                entry.update(paired(ref_vec, v))
            comparisons[f"variants/{exp}/{m}"] = entry
    summary["paired_vs_reference"] = comparisons

    # candidates (3 seeds) and the long-horizon diagnostic
    for group in ("candidates", "long", "candidates2"):
        table = {}
        for seed in (7, 8, 9):
            for slice_name in ("test", "train"):
                sc = load(RESULTS / f"{group}_seed{seed}_{slice_name}.json")
                for exp in EXPS:
                    for m, r in rows(sc, exp).items():
                        if m == "base_policy" or not submitted(sc, exp, m):
                            continue
                        t = table.setdefault(f"{exp}/{m}", {"test": [], "train": [], "test_scored": []})
                        t[slice_name].append(r["unconstrained_pass_at_1"])
                        if slice_name == "test":
                            t["test_scored"].append(r["pass_at_1"])
                        if slice_name == "test" and seed == 7:
                            t["rank_change_in_experiment_seed7"] = exp_rank(r, ref_rows[exp], exp)
                            v = correct_vec(f"{group}_seed7", exp, m)
                            ref_vec = correct_vec("reference", exp, REF)
                            if v is not None and ref_vec is not None:
                                t["paired_seed7"] = paired(ref_vec, v)
        for key, t in table.items():
            exp = key.split("/")[0]
            for sl in ("test", "train"):
                if t[sl]:
                    t[f"{sl}_mean"] = float(np.mean(t[sl]))
            ref_seeds = [seed_value("test", s, exp, REF) for s in (7, 8, 9)]
            if len(t["test"]) == 3 and all(v is not None for v in ref_seeds):
                d = np.array(t["test"]) - np.array(ref_seeds)
                t["minus_reference_same_seed"] = {"per_seed": d.tolist(), "mean": float(d.mean())}
        summary[group] = table

    # shortcuts: total rank change over both experiments
    shortcuts = {}
    for case_dir in sorted(p for p in SHORT.iterdir() if p.is_dir()) if SHORT.exists() else []:
        sc = load(case_dir / "scores_test.json")
        if not sc:
            continue
        case = case_dir.name
        if case == "reward_artifact":
            identical = all(rows(sc, e) == rows(test, e) for e in EXPS)
            shortcuts[case] = {"test_rows_identical_to_unmodified": identical, "rank_change": 0.0 if identical else None}
            continue
        entry = {"rows": {}, "status": {}, "rank_change": 0.0}
        for exp in EXPS:
            r = rows(sc, exp).get(case)
            if r is None:
                continue
            entry["rows"][exp] = r
            entry["status"][exp] = sc.get("_status", {}).get(f"{exp}/{case}")
            entry["rank_change"] += exp_rank(r, ref_rows[exp], exp)
        before = load(case_dir / "scores_test_before.json")
        if before:
            # pre-fix evaluator on the 4000-prompt test set, against the job-4 reference rows
            old_ref = load(H / "results_job4" / "reference_scores_test.json")
            b = {"rows": {}, "rank_change_by_row": {}}
            for exp in EXPS:
                for m, r in rows(before, exp).items():
                    if not m.startswith(case):
                        continue
                    b["rows"][f"{exp}/{m}"] = r
                    b["rank_change_by_row"][f"{exp}/{m}"] = exp_rank(r, rows(old_ref, exp)[REF], exp)
            if case == "missing_experiment":
                b["rank_change"] = sum(b["rank_change_by_row"].values())  # the empty experiment is skipped
            else:
                per_exp = {e: [v for k, v in b["rank_change_by_row"].items() if k.startswith(e + "/")] for e in EXPS}
                b["rank_change_best_row_per_experiment"] = sum(max(v) for v in per_exp.values() if v)
                b["rank_change_worst_row_per_experiment"] = sum(min(v) for v in per_exp.values() if v)
            entry["before"] = b
        shortcuts[case] = entry
    summary["shortcuts"] = shortcuts

    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("weights", "evaluated_prompts", "seed_spread_test")}, indent=2))
    for k, v in comparisons.items():
        print(k, {kk: v[kk] for kk in ("diff", "ci95", "rank_change_in_experiment") if kk in v})
    for group in ("candidates", "candidates2", "long"):
        for k, v in summary[group].items():
            print(group, k, {kk: v.get(kk) for kk in ("test_mean", "rank_change_in_experiment_seed7", "minus_reference_same_seed")})
    for k, v in shortcuts.items():
        print("shortcut", k, v.get("rank_change"), v.get("status"), v.get("before", {}).get("rank_change"),
              v.get("before", {}).get("rank_change_best_row_per_experiment"))


if __name__ == "__main__":
    main()
