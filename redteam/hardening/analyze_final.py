"""Summarise the final hardening job: paper effect vs seed noise, variants, shortcuts.

Rank change of a submission = sum over experiments e and scored metrics m of
weight_e * |coefficient_m| * (row_m - reference_m), with weights/coefficients
read from scoring/reference.json and the reference = the humanline_grpo row of
scoring/scores_test.json (mean over seeds 7-14).  This is the raw form of the
gym's rank: the real ranker also divides each metric's change by its spread
across ranked attempts, which does not change signs.

"before" numbers use the single-run regime this iteration replaced: the
seed-7 run of each method against the seed-7 reference run (variants), or the
pre-fix evaluator redteam/hardening/prev_eval_test.py (seed-aggregation
shortcut cases), against its own score of the seed-7 reference run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/user")
H = ROOT / "redteam" / "hardening"
RESULTS = H / "results_final"
SHORT = H / "shortcuts_final"
TMP = ROOT / "tmp" / "hardening" / "final"
REF = "humanline_grpo"
EXPS = ("stale64", "offline")
SEEDS = list(range(7, 15))
POOL_SEEDS = list(range(15, 31))
SCORED = ("pass_at_1", "hard_pass_at_1")

_ref_json = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
WEIGHTS = {e: float(_ref_json[e]["weight"]) for e in EXPS}
COEF = {e: {m: abs(float(d["coefficient"])) for m, d in _ref_json[e]["metrics"].items() if d["coefficient"]} for e in EXPS}


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


TEST = load(ROOT / "scoring" / "scores_test.json")
TRAIN = load(ROOT / "scoring" / "scores_train.json")
REF_ROWS = {e: TEST["experiments"][e]["results"][REF] for e in EXPS}


def rows(scores: dict, exp: str) -> dict:
    return scores.get("experiments", {}).get(exp, {}).get("results", {})


def exp_rank(row: dict, ref_row: dict, exp: str) -> float:
    return WEIGHTS[exp] * sum(c * (row[m] - ref_row[m]) for m, c in COEF[exp].items())


def stats(d) -> dict:
    d = np.asarray(d, dtype=np.float64)
    out = {"n": int(len(d)), "mean": float(d.mean()) if len(d) else None}
    if len(d) > 1:
        sd = float(d.std(ddof=1))
        out.update({"sd": sd, "se": sd / np.sqrt(len(d)), "t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else None})
    return out


def per_seed(scores: dict, exp: str, method: str, metric: str = "pass_at_1", gated: bool = True) -> list[float]:
    """Per-seed values for seeds 7-14 (0 for missing seeds; 0 if the row is gated out and gated=True)."""
    seeds = scores.get("_per_seed", {}).get(exp, {}).get(method, {})
    row = rows(scores, exp).get(method, {})
    if gated and row.get("operating_point_ok", 1.0) == 0.0:
        return [0.0] * len(SEEDS)
    return [float(seeds.get(str(s), {}).get(metric, 0.0)) for s in SEEDS]


def seed_rank(scores: dict, exp: str, method: str, seed_index: int, ref_scores: dict = TEST, ref_method: str = REF) -> float:
    """Single-run rank change in one experiment: seed ``SEEDS[seed_index]`` against the same reference seed."""
    vals = {m: per_seed(scores, exp, method, m)[seed_index] for m in SCORED}
    ref = {m: per_seed(ref_scores, exp, ref_method, m)[seed_index] for m in SCORED}
    return exp_rank(vals, ref, exp)


def pool_seed_values(slice_scores: dict, pool_scores: dict, exp: str, method: str, metric: str = "pass_at_1") -> dict:
    out = {s: v for s, v in zip(SEEDS, per_seed(slice_scores, exp, method, metric, gated=False))}
    for s in POOL_SEEDS:
        block, label = ("A" if s < 23 else "B"), 7 + (s - 15) % 8
        row = pool_scores.get("_per_seed", {}).get(exp, {}).get(f"{method}__pool{block}", {}).get(str(label))
        if row is not None and row["note"].startswith("ok"):
            out[s] = row[metric]
    return out


def mean_vec(group: str, exp: str, method: str):
    p = TMP / "correct" / group / f"{exp}__{method}.json"
    return np.array(json.loads(p.read_text())["mean"], dtype=np.float64) if p.exists() else None


def prompt_paired(a: np.ndarray, b: np.ndarray, n_boot: int = 2000) -> dict:
    """Seed-mean accuracy difference b - a on identical prompts; bootstrap over prompts (test-sampling noise)."""
    rng = np.random.default_rng(0)
    d = b - a
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "se_prompts": float(d.std(ddof=1) / np.sqrt(len(d))),
            "ci95_prompts": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def compare(scores: dict, group: str, exp: str, method: str, train_scores: dict | None = None) -> dict:
    row = rows(scores, exp)[method]
    entry = {
        "row_test": row,
        "status": scores.get("_status", {}).get(f"{exp}/{method}"),
        "rank_change_after": exp_rank(row, REF_ROWS[exp], exp),
        "rank_change_before_seed7": seed_rank(scores, exp, method, 0),
        "per_seed_test": per_seed(scores, exp, method, gated=False),
    }
    if train_scores is not None and method in rows(train_scores, exp):
        entry["row_train"] = rows(train_scores, exp)[method]
    raw = np.array(per_seed(scores, exp, method, gated=False))
    ref = np.array(per_seed(TEST, exp, REF, gated=False))
    entry["paired_by_seed_raw_accuracy"] = stats(raw - ref)
    a, b = mean_vec("reference", exp, REF), mean_vec(group, exp, method)
    if a is not None and b is not None and row.get("operating_point_ok", 1.0) == 1.0:
        entry["paired_by_prompt_scored"] = prompt_paired(a, b)
    return entry


def main() -> None:
    summary: dict = {"weights": WEIGHTS, "coefficients": COEF,
                     "evaluated_prompts": {e: TEST["experiments"][e]["evaluated_prompts"] for e in EXPS},
                     "required_seeds": SEEDS}
    pool_test, pool_train = load(RESULTS / "pool_test.json"), load(RESULTS / "pool_train.json")

    # ---- reference set
    ref_set = {}
    for exp in EXPS:
        for m in rows(TEST, exp):
            entry = compare(TEST, "reference", exp, m, TRAIN) if m != "base_policy" else {
                "row_test": rows(TEST, exp)[m], "row_train": rows(TRAIN, exp).get(m),
                "rank_change_after": exp_rank(rows(TEST, exp)[m], REF_ROWS[exp], exp)}
            ref_set[f"{exp}/{m}"] = entry
    summary["reference_set"] = ref_set
    summary["reference_total_rank_terms"] = {e: {m: REF_ROWS[e][m] for m in SCORED} for e in EXPS}
    summary["bounds"] = {e: {m: {"reference": REF_ROWS[e][m], "bound": 1.0, "at_bound": REF_ROWS[e][m] >= 0.999}
                             for m in SCORED} for e in EXPS}

    # ---- paper effect: humanline - grpo per training seed (24 seeds where the pool ran), raw test accuracy
    effect = {}
    for exp in EXPS:
        h = pool_seed_values(TEST, pool_test, exp, REF)
        g = pool_seed_values(TEST, pool_test, exp, "grpo")
        common = sorted(set(h) & set(g))
        effect[exp] = {
            "humanline_minus_grpo_scored_seeds": stats([h[s] - g[s] for s in SEEDS]),
            "humanline_minus_grpo_all_seeds": stats([h[s] - g[s] for s in common]),
            "humanline_all_seeds": stats([h[s] for s in common]),
            "grpo_all_seeds": stats([g[s] for s in common]),
            "online_minus_humanline_scored_seeds": stats(
                np.array(per_seed(TEST, exp, "online_grpo", gated=False)) - np.array(per_seed(TEST, exp, REF, gated=False))),
            "sync_only_minus_grpo_scored_seeds": stats(
                np.array(per_seed(TEST, exp, "humanline_sync_only", gated=False)) - np.array(per_seed(TEST, exp, "grpo", gated=False))),
            "clip_only_minus_grpo_scored_seeds": stats(
                np.array(per_seed(TEST, exp, "humanline_clip_only", gated=False)) - np.array(per_seed(TEST, exp, "grpo", gated=False))),
            "per_seed_humanline": h, "per_seed_grpo": g,
        }
        a, b = mean_vec("reference", exp, "grpo"), mean_vec("reference", exp, REF)
        if a is not None and b is not None:
            effect[exp]["humanline_minus_grpo_by_prompt"] = prompt_paired(a, b)
    summary["paper_effect"] = effect

    # ---- seed noise of the scored rank (reference method, all 24 seeds)
    contrib = []
    hp = {e: pool_seed_values(TEST, pool_test, e, REF) for e in EXPS}
    hh = {e: pool_seed_values(TEST, pool_test, e, REF, "hard_pass_at_1") for e in EXPS}
    for s in sorted(set(hp["stale64"]) & set(hp["offline"])):
        contrib.append(sum(WEIGHTS[e] * (COEF[e]["pass_at_1"] * hp[e][s] + COEF[e]["hard_pass_at_1"] * hh[e][s]) for e in EXPS))
    sd_r = float(np.std(contrib, ddof=1)) if len(contrib) > 1 else None
    reruns = {}
    for block in ("A", "B"):
        name = f"{REF}__pool{block}"
        if all(name in rows(pool_test, e) for e in EXPS):
            reruns[f"seeds_{'15-22' if block == 'A' else '23-30'}"] = sum(
                exp_rank(rows(pool_test, e)[name], REF_ROWS[e], e) for e in EXPS)
    summary["noise"] = {
        "single_seed_rank_contribution_sd": sd_r,
        "n_seeds": len(contrib),
        "rank_sd_of_8_seed_mean": sd_r / np.sqrt(8) if sd_r else None,
        "rank_sd_of_difference_between_independent_8_seed_sets": sd_r * np.sqrt(2 / 8) if sd_r else None,
        "rank_sd_of_difference_between_single_runs": sd_r * np.sqrt(2) if sd_r else None,
        "reference_method_with_fresh_seeds_rank_change": reruns,
    }

    # ---- variants
    var_test, var_train = load(RESULTS / "variants_test.json"), load(RESULTS / "variants_train.json")
    variants = {}
    for exp in EXPS:
        for m in rows(var_test, exp):
            if m == "base_policy" or var_test["_status"].get(f"{exp}/{m}", "").startswith("failed: no run"):
                continue
            variants[f"{exp}/{m}"] = compare(var_test, "variants", exp, m, var_train)
    summary["variants"] = variants

    # ---- shortcuts
    prev_ref = load(RESULTS / "prev_reference_seed7_test.json")
    shortcuts = {}
    for case_dir in sorted(p for p in SHORT.iterdir() if p.is_dir()) if SHORT.exists() else []:
        sc = load(case_dir / "scores_test.json")
        if not sc:
            continue
        case = case_dir.name
        if case == "reward_artifact":
            identical = all(rows(sc, e) == rows(TEST, e) for e in EXPS)
            shortcuts[case] = {"test_rows_identical_to_unmodified": identical, "rank_change": 0.0 if identical else None}
            continue
        entry = {"rows": {}, "status": {}, "rank_change": 0.0}
        for exp in EXPS:
            r = rows(sc, exp).get(case)
            if r is None:
                continue
            entry["rows"][exp] = r
            entry["status"][exp] = sc.get("_status", {}).get(f"{exp}/{case}")
            entry["rank_change"] += exp_rank(r, REF_ROWS[exp], exp)
        before = load(case_dir / "scores_test_before.json")
        if before and prev_ref:
            b = {"rows": {}, "rank_change": 0.0}
            for exp in EXPS:
                r = rows(before, exp).get(case)
                if r is None:
                    continue
                b["rows"][exp] = r
                b["rank_change"] += exp_rank(r, rows(prev_ref, exp)[REF], exp)
            entry["before_prev_evaluator"] = b
        shortcuts[case] = entry
    summary["shortcuts"] = shortcuts
    summary["prev_evaluator_reference_seed7"] = {e: rows(prev_ref, e).get(REF) for e in EXPS}

    # ---- train vs test, device and determinism checks
    summary["train_vs_test"] = {f"{e}/{m}": {"train": rows(TRAIN, e)[m]["unconstrained_pass_at_1"],
                                             "test": rows(TEST, e)[m]["unconstrained_pass_at_1"]}
                                for e in EXPS for m in rows(TEST, e) if m in rows(TRAIN, e)}
    summary["device_check"] = load(RESULTS / "device_check.json")
    summary["determinism"] = load(RESULTS / "determinism.json")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    # ---- human-readable digest
    print("== reference set (test, 8-seed means) ==")
    for k, v in ref_set.items():
        r = v["row_test"]
        print(f"{k:34s} pass {r['pass_at_1']:.4f} hard {r['hard_pass_at_1']:.4f} unc {r['unconstrained_pass_at_1']:.4f} "
              f"rank {v['rank_change_after']:+.4f} train {v.get('row_train', {}) and v['row_train']['unconstrained_pass_at_1']}")
    print("== paper effect ==")
    for exp, v in effect.items():
        print(exp, {k: v[k] for k in v if k.endswith(("seeds", "prompt"))})
    print("== noise ==", json.dumps(summary["noise"]))
    print("== variants ==")
    for k, v in variants.items():
        print(f"{k:34s} pass {v['row_test']['pass_at_1']:.4f} after {v['rank_change_after']:+.4f} "
              f"before(seed7) {v['rank_change_before_seed7']:+.4f} by-seed {v['paired_by_seed_raw_accuracy']} "
              f"by-prompt {v.get('paired_by_prompt_scored')}")
    print("== shortcuts ==")
    for k, v in shortcuts.items():
        print(f"{k:28s} after {v.get('rank_change')} before {v.get('before_prev_evaluator', {}).get('rank_change')} "
              f"status {v.get('status')}")
    print("== device/determinism ==", json.dumps(summary["device_check"]), json.dumps(summary["determinism"])[:2000])


if __name__ == "__main__":
    main()
