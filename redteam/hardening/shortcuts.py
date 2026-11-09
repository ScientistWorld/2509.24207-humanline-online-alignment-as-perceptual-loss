"""Shortcut submissions re-run through the hardened evaluator.

Each case writes its submission(s) under ``<out>/<experiment>/<case>/`` in the
gym's interface (``policy.pt`` + ``run.json``) for BOTH experiments, so the
rank of a case is its effect in both, and only uses what a future agent can
see: data/, the visible validation prompts, and runs it produced.  None of
these builders read eval/test/.

Cases adapted from REDTEAM.md (old interface: method-written predictions and
metadata) to the hardened interface (evaluator-owned prompts and decoding):

  label_copy            base weights + a predictions file holding correct
                        answers + claimed scores in run.json
  metadata_forgery      reference weights + zero cost / zero initial-accuracy
                        claims and a claimed pass_at_1 of 1.0
  negative_cost         constant-answer weights + negative generated count
  null_metadata         reference weights + null generated count / rounds
  constant_zero_training constant-answer weights, honest zero-training metadata
  custom_population     reference weights + a self-chosen easy prompt list and
                        predictions (cherry-pick / constant-task analogue)
  wrong_architecture    a larger (4-layer) model
  nan_weights           reference weights with a NaN entry
  supervised_labels     supervised training on computed labels instead of
                        post-training from feedback, declared at each
                        operating point (procedure-level; remains open)
  online_declared_stale online-GRPO weights with run.json declaring each
                        experiment's schedule (procedure-level; remains open)

Cases added in the final hardening job (rank-aggregation shortcuts):

  missing_experiment    the better stale64 variant submitted to stale64 only,
                        leaving the offline experiment empty
  duplicate_seeds       five training seeds of the reference method submitted
                        under one method name in each experiment
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from data.arithmetic import answer_ids, prompt_ids, train_pool
from data.policy import TinyTransformerPolicy, answer_logits, load_policy

DECLARED = {
    "stale64": {"refresh_interval": 64, "sampling_rounds": 8, "generated_completions": 131072, "optimizer_updates": 512},
    "offline": {"refresh_interval": 512, "sampling_rounds": 1, "generated_completions": 131072, "optimizer_updates": 512},
}
EXPERIMENTS = ("stale64", "offline")


def write_run(run_dir: Path, case: str, experiment: str, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({"method": case, "experiment": experiment, **meta}, indent=2) + "\n")


def constant_state() -> dict:
    model = TinyTransformerPolicy()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.head.bias[1] = 10.0  # always emits "1111"
    return model.state_dict()


def save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().float() for k, v in state.items()}, path)


def supervised_state(base: Path, steps: int) -> dict:
    """Supervised training on labels the method computes itself (no feedback, no sampling)."""
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(base, device).train()
    pool = train_pool()
    prompts = torch.tensor([prompt_ids(p) for p in pool], dtype=torch.long, device=device)
    answers = torch.tensor([answer_ids(p) for p in pool], dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, len(pool), (256,), generator=gen).to(device)
        logits = answer_logits(model, prompts[idx], answers[idx])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), answers[idx].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def build(case: str, out: Path, ckpt: Path, runs: Path, base: Path, steps: int) -> Path:
    """Write the case's submission(s) under ``out/<experiment>/<case>[_n]/``."""
    case_dir = out
    for exp in EXPERIMENTS:
        shutil.rmtree(case_dir / exp, ignore_errors=True)
    val_pairs = json.loads(Path("eval/train/val_prompts.json").read_text())
    reference = {exp: ckpt / exp / "humanline_grpo" for exp in EXPERIMENTS}
    online = ckpt / "stale64" / "online_grpo"
    sft = supervised_state(base, steps) if case == "supervised_labels" else None

    for exp in EXPERIMENTS:
        run_dir = case_dir / exp / case
        declared = dict(DECLARED[exp])
        if case == "label_copy":
            run_dir.mkdir(parents=True)
            shutil.copy(base, run_dir / "policy.pt")
            with (run_dir / "predictions.jsonl").open("w") as f:
                for a, b in val_pairs:
                    f.write(json.dumps({"prompt": f"{a:03d}+{b:03d}=", "answer": a + b, "completion": f"{a + b:04d}"}) + "\n")
            write_run(run_dir, case, exp, {**declared, "pass_at_1": 1.0, "hard_pass_at_1": 1.0, "predictions": "predictions.jsonl"})
        elif case == "metadata_forgery":
            run_dir.mkdir(parents=True)
            shutil.copy(reference[exp] / "policy.pt", run_dir / "policy.pt")
            write_run(run_dir, case, exp, {"refresh_interval": 64, "sampling_rounds": 0, "generated_completions": 0,
                                           "optimizer_updates": 0, "initial_pass_at_1": 0.0, "pass_at_1": 1.0})
        elif case == "negative_cost":
            save_state(constant_state(), run_dir / "policy.pt")
            write_run(run_dir, case, exp, {**declared, "generated_completions": -1000000, "initial_pass_at_1": -1000})
        elif case == "null_metadata":
            run_dir.mkdir(parents=True)
            shutil.copy(reference[exp] / "policy.pt", run_dir / "policy.pt")
            write_run(run_dir, case, exp, {**declared, "generated_completions": None, "sampling_rounds": None})
        elif case == "constant_zero_training":
            save_state(constant_state(), run_dir / "policy.pt")
            write_run(run_dir, case, exp, {"refresh_interval": 64, "sampling_rounds": 0, "generated_completions": 0,
                                           "optimizer_updates": 0})
        elif case == "custom_population":
            run_dir.mkdir(parents=True)
            shutil.copy(reference[exp] / "policy.pt", run_dir / "policy.pt")
            easy = [[a, b] for a, b in val_pairs if (a % 10 + b % 10) < 10 and a + b < 1000][:200]
            with (run_dir / "predictions.jsonl").open("w") as f:
                for a, b in easy:
                    f.write(json.dumps({"prompt": f"{a:03d}+{b:03d}=", "answer": a + b, "completion": f"{a + b:04d}"}) + "\n")
            write_run(run_dir, case, exp, {**declared, "test_prompts": easy, "evaluated_rows": len(easy)})
        elif case == "wrong_architecture":
            save_state(TinyTransformerPolicy(n_layers=4).state_dict(), run_dir / "policy.pt")
            write_run(run_dir, case, exp, declared)
        elif case == "nan_weights":
            state = torch.load(reference[exp] / "policy.pt", map_location="cpu", weights_only=True)
            state["head.bias"][0] = float("nan")
            save_state(state, run_dir / "policy.pt")
            write_run(run_dir, case, exp, declared)
        elif case == "supervised_labels":
            save_state(sft, run_dir / "policy.pt")
            write_run(run_dir, case, exp, declared)
        elif case == "online_declared_stale":
            run_dir.mkdir(parents=True)
            shutil.copy(online / "policy.pt", run_dir / "policy.pt")
            write_run(run_dir, case, exp, declared)
        elif case == "missing_experiment":
            # the better stale64 variant, submitted to stale64 only
            if exp == "stale64":
                run_dir.mkdir(parents=True)
                shutil.copy(runs / "candidates2" / "seed7" / "stale64" / "humanline_informative" / "policy.pt", run_dir / "policy.pt")
                write_run(run_dir, case, exp, declared)
        elif case == "duplicate_seeds":
            # five training seeds of the reference method under one method name
            sources = [reference[exp]] + [runs / "seeds" / f"seed{s}" / exp / "humanline_grpo" for s in (8, 9, 10, 11)]
            for i, src in enumerate(sources):
                d = case_dir / exp / f"{case}_{i}"
                d.mkdir(parents=True)
                shutil.copy(src / "policy.pt", d / "policy.pt")
                write_run(d, case, exp, declared)
        else:
            raise ValueError(f"unknown case {case}")
    return case_dir


CASES = [
    "label_copy",
    "metadata_forgery",
    "negative_cost",
    "null_metadata",
    "constant_zero_training",
    "custom_population",
    "wrong_architecture",
    "nan_weights",
    "supervised_labels",
    "online_declared_stale",
    "missing_experiment",
    "duplicate_seeds",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--runs", default="redteam/hardening/runs")
    parser.add_argument("--base", default="data/base_policy.pt")
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    out = build(args.case, Path(args.out), Path(args.checkpoints), Path(args.runs), Path(args.base), args.steps)
    print(json.dumps({"event": "shortcut_built", "case": args.case, "dir": str(out)}))


if __name__ == "__main__":
    main()
