"""Write scoring/reference.json from a scores_test.json (reference runs only)."""
import json, sys
from pathlib import Path

src = json.loads(Path(sys.argv[1]).read_text())["experiments"]
alias = {"stale_grpo": "grpo", "offline_grpo": "grpo"}
TYPES = {"humanline_grpo": "proposed", "humanline_sync_only": "proposed", "humanline_clip_only": "proposed",
         "grpo": "baseline", "online_grpo": "baseline", "base_policy": "baseline"}
METRICS = {
    "pass_at_1": {"higher_is_better": True, "coefficient": 1.0,
                  "description": "Exact-match accuracy of the evaluator's own greedy decoding of the submitted policy on the hidden test prompts. Set to 0 when the run violates the experiment's fixed operating point or cannot be evaluated."},
    "hard_pass_at_1": {"higher_is_better": True, "coefficient": 0.5,
                       "description": "Same accuracy restricted to hidden prompts whose addition needs at least two carries (about half of the prompts, the harder half). Set to 0 on a failed run."},
    "unconstrained_pass_at_1": {"higher_is_better": True, "coefficient": 0.0,
                                "description": "Diagnostic only: accuracy ignoring the operating-point gate (e.g. the online comparator's accuracy). Never ranked, so exceeding the sampling budget cannot pay."},
    "operating_point_ok": {"higher_is_better": True, "coefficient": 0.0,
                           "description": "Constraint gate: 1 if the declared training schedule fits the experiment's fixed budget (sampling rounds, refresh interval, generated completions, optimizer updates), else 0. A 0 zeroes the scored metrics."},
}
PAPER = {
    "stale64": {
        "source": "Section 5.2.2 and Figure 6: Qwen2.5-1.5B-Instruct aligned on the MATH500 training set, evaluated on MATH500 test after 1600 steps",
        "online_grpo_pass_at_1": 0.593,
        "humanline_grpo_64x_less_frequent_sampling_pass_at_1": 0.593,
        "pass_at_1_plus_minus": 0.019,
        "grpo_64x_less_frequent_sampling": "significantly worse than online GRPO (p < 0.05); no number reported in the text",
        "humanline_settings": "log epsilon_P = -1.5, log epsilon_R = 1.5; sync interval k in [12, 24]; k = 1 collapses",
    },
    "offline": {
        "source": "Table 3 (instruction following, Llama3-8B-Instruct, AlpacaEval2 length-controlled win rate); the paper runs offline GRPO only in this setting",
        "offline_grpo_lc_winrate": 15.52,
        "offline_humanline_grpo_lc_winrate": 25.24,
        "online_grpo_lc_winrate": 25.05,
        "online_humanline_grpo_lc_winrate": 26.10,
        "offline_grpo_epsilon": 0.50,
        "humanline_grpo_epsilon": 0.15,
    },
}
DESC = {
    "stale64": "Post-training from verifiable feedback when completions may only be sampled every 64 optimizer updates (64x less often than online): 512 updates, at most 8 sampling rounds, at most 131,072 generated completions, starting from the fixed base policy.",
    "offline": "Post-training from verifiable feedback with a single sampling round drawn from the starting policy and no further sampling: 512 updates, at most 131,072 generated completions, starting from the fixed base policy.",
}
NOTES = {
    "stale64": "Scaled setup: three-digit addition with a 0.6M-parameter transformer instead of MATH500 with Qwen2.5-1.5B, 512 updates instead of 1600. The paper's numbers are on a different task and scale and are kept under _paper_reported only. Results rows are the seed-7 reference runs scored by eval/test/evaluate.py; online_grpo is the fresh-sampling comparator, scored as failed because it samples every update (its accuracy is in unconstrained_pass_at_1).",
    "offline": "Scaled setup as in stale64. The paper reports offline GRPO only for instruction following (AlpacaEval2 win rates, kept under _paper_reported); this experiment applies the paper's offline GRPO protocol (one pool, fixed reference, epsilon 0.5) to the gym's verifiable task. online_grpo is the same fresh-sampling comparator, scored as failed here.",
}
out = {"experiments": {}}
for exp in ("stale64", "offline"):
    results = {}
    for m, r in src[exp]["results"].items():
        name = alias.get(m, m)
        results[name] = {"type": TYPES[name], **{k: r[k] for k in METRICS}}
    out["experiments"][exp] = {
        "description": DESC[exp], "weight": 0.5, "primary_metric": "pass_at_1",
        "metrics": METRICS, "results": results, "notes": NOTES[exp], "_paper_reported": PAPER[exp],
    }
Path("scoring/reference.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({e: {m: r["pass_at_1"] for m, r in v["results"].items()} for e, v in out["experiments"].items()}, indent=1))
