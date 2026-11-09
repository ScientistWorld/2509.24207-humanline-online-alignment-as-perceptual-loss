"""Summarise the hardening job: seeds, paired test comparisons, ranks, shortcuts.

Rank change of a row = sum over experiments e and scored metrics m of
weight_e * coefficient_m * (row_m - reference_m), where the reference is the
seed-7 humanline_grpo row of the same experiment in scoring/scores_test.json.
Rows submitted to one experiment only contribute to that experiment.
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
WEIGHTS = {"stale64": 0.6, "offline": 0.4}
COEF = {"pass_at_1": 1.0, "hard_pass_at_1": 0.5}
REF = "humanline_grpo"
SEEDS = [7, 8, 9, 10, 11]


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def rows(scores: dict, exp: str) -> dict:
    return scores.get("experiments", {}).get(exp, {}).get("results", {})


def rank_change(row: dict, ref_row: dict, exp: str) -> float:
    return WEIGHTS[exp] * sum(c * (row[m] - ref_row[m]) for m, c in COEF.items())


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
        "b_only_correct": int(((b == 1) & (a == 0)).sum()),
        "a_only_correct": int(((a == 1) & (b == 0)).sum()),
    }


def seed_table(slice_name: str) -> dict:
    out = {}
    ref_scores = load(ROOT / "scoring" / f"scores_{slice_name}.json")
    for exp, methods in (("stale64", ["online_grpo", "stale_grpo", "humanline_grpo"]), ("offline", ["offline_grpo", "humanline_grpo"])):
        for method in methods:
            vals = []
            for seed in SEEDS:
                sc = ref_scores if seed == 7 else load(RESULTS / f"seeds_seed{seed}_{slice_name}.json")
                row = rows(sc, exp).get(method)
                if row is not None:
                    vals.append(row["unconstrained_pass_at_1"])
            if vals:
                arr = np.array(vals)
                out[f"{exp}/{method}"] = {"values": vals, "mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                                          "range": float(arr.max() - arr.min())}
    return out


def main() -> None:
    test = load(ROOT / "scoring" / "scores_test.json")
    train = load(ROOT / "scoring" / "scores_train.json")
    summary: dict = {"selection": load(RESULTS / "selection.json"), "reference_rows": {}, "train_test": {}}
    ref_rows = {exp: rows(test, exp)[REF] for exp in WEIGHTS}
    summary["reference_rows"] = {exp: rows(test, exp) for exp in WEIGHTS}
    for exp in WEIGHTS:
        for m, r in rows(test, exp).items():
            t = rows(train, exp).get(m)
            if t:
                summary["train_test"][f"{exp}/{m}"] = {"train": t["unconstrained_pass_at_1"], "test": r["unconstrained_pass_at_1"]}

    summary["seed_spread_test"] = seed_table("test")
    summary["seed_spread_train"] = seed_table("train")

    # paired comparisons against the seed-7 reference on the test prompts
    comparisons = {}
    for exp in WEIGHTS:
        ref_vec = correct_vec("reference", exp, REF)
        if ref_vec is None:
            continue
        for m in rows(test, exp):
            if m == REF:
                continue
            v = correct_vec("reference", exp, m)
            if v is not None:
                comparisons[f"reference_set/{exp}/{m}"] = paired(ref_vec, v)
        for root in ("variants", f"sweep_{exp}"):
            var = load(RESULTS / f"{root}_test.json")
            for m, r in rows(var, exp).items():
                if m == "base_policy":
                    continue
                v = correct_vec(root, exp, m)
                entry = {"row": r, "rank_change": rank_change(r, ref_rows[exp], exp)}
                if v is not None:
                    entry.update(paired(ref_vec, v))
                comparisons[f"{root}/{exp}/{m}"] = entry
        for seed in SEEDS[1:]:
            sc = load(RESULTS / f"seeds_seed{seed}_test.json")
            for m, r in rows(sc, exp).items():
                if m == "base_policy":
                    continue
                v = correct_vec(f"seeds_seed{seed}", exp, m)
                entry = {"row": r, "rank_change": rank_change(r, ref_rows[exp], exp)}
                if v is not None:
                    entry.update(paired(ref_vec, v))
                comparisons[f"seed{seed}/{exp}/{m}"] = entry
    summary["paired_vs_reference"] = comparisons

    # sweep rows (diagnostic only): visible-validation and test accuracy per config
    sweep = {}
    for exp in WEIGHTS:
        tr = rows(load(RESULTS / f"sweep_{exp}_train.json"), exp)
        te = rows(load(RESULTS / f"sweep_{exp}_test.json"), exp)
        sweep[exp] = {m: {"train": tr[m]["unconstrained_pass_at_1"], "test": r["unconstrained_pass_at_1"]}
                      for m, r in te.items() if m in tr}
    summary["sweep"] = sweep

    # candidates (3 seeds) and the long-horizon diagnostic: mean test accuracy per config
    for group in ("candidates", "long", "candidates2"):
        table = {}
        for seed in (7, 8, 9):
            for slice_name in ("test", "train"):
                sc = load(RESULTS / f"{group}_seed{seed}_{slice_name}.json")
                for exp in WEIGHTS:
                    for m, r in rows(sc, exp).items():
                        if m == "base_policy":
                            continue
                        t = table.setdefault(f"{exp}/{m}", {"test": [], "train": [], "rank_change_seed7": None})
                        t[slice_name].append(r["unconstrained_pass_at_1"])
                        if slice_name == "test" and seed == 7:
                            t["rank_change_seed7"] = rank_change(r, ref_rows[exp], exp)
                            v = correct_vec(f"{group}_seed7", exp, m)
                            ref_vec = correct_vec("reference", exp, REF)
                            if v is not None and ref_vec is not None:
                                t["paired_seed7"] = paired(ref_vec, v)
        for t in table.values():
            for sl in ("test", "train"):
                if t[sl]:
                    t[f"{sl}_mean"] = float(np.mean(t[sl]))
        summary[group] = table

    # shortcuts
    shortcuts = {}
    for case_dir in sorted(p for p in SHORT.iterdir() if p.is_dir()) if SHORT.exists() else []:
        sc = load(case_dir / "scores_test.json")
        if not sc:
            continue
        if case_dir.name == "reward_artifact":
            changed = {exp: {m: (r, rows(test, exp).get(m)) for m, r in rows(sc, exp).items()} for exp in WEIGHTS}
            identical = all(a == b for exp in changed.values() for a, b in exp.values())
            shortcuts["reward_artifact"] = {"test_rows_identical_to_unmodified": identical,
                                            "rank_change": 0.0 if identical else None}
            continue
        r = rows(sc, "stale64").get(case_dir.name)
        if r is None:
            shortcuts[case_dir.name] = {"row": None, "status": sc.get("_status")}
            continue
        shortcuts[case_dir.name] = {"row": r, "rank_change": rank_change(r, ref_rows["stale64"], "stale64"),
                                    "status": sc.get("_status", {}).get(f"stale64/{case_dir.name}")}
    summary["shortcuts"] = shortcuts

    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("selection", "seed_spread_test", "shortcuts")}, indent=2))
    for k, v in comparisons.items():
        print(k, {kk: v[kk] for kk in ("diff", "ci95", "rank_change") if kk in v})


if __name__ == "__main__":
    main()
