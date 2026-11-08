"""Shortcut submissions re-run through the hardened evaluator.

Each case writes one submission directory ``<out>/stale64/<case>/`` in the
gym's interface (``policy.pt`` + ``run.json``) and only uses what a future
agent can see: data/, the visible validation prompts, and runs it produced.
None of these builders read eval/test/.

Cases adapted from REDTEAM.md (old interface: method-written predictions and
metadata) to the hardened interface (evaluator-owned prompts and decoding):

  label_copy            base weights + a predictions file holding correct
                        answers + claimed scores in run.json
  metadata_forgery      reference weights + zero cost / zero initial-accuracy
                        claims and a claimed pass_at_1 of 1.0
  negative_cost         constant-answer weights + negative generated count
  constant_zero_training constant-answer weights, honest zero-training metadata
  custom_population     reference weights + a self-chosen easy prompt list and
                        predictions (cherry-pick / constant-task analogue)
  wrong_architecture    a larger (4-layer) model
  nan_weights           reference weights with a NaN entry
  supervised_labels     supervised training on computed labels instead of
                        post-training from feedback, declared as stale64
                        (procedure-level; expected to remain open)
  online_declared_stale online-GRPO weights with run.json claiming the stale64
                        schedule (procedure-level; expected to remain open)
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
from data.policy import TinyTransformerPolicy, answer_logits, load_policy, save_policy

STALE_META = {"refresh_interval": 64, "sampling_rounds": 8, "generated_completions": 131072, "optimizer_updates": 512}


def write_run(run_dir: Path, case: str, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({"method": case, "experiment": "stale64", **meta}, indent=2) + "\n")


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


def build(case: str, out: Path, reference_run: Path, online_run: Path, base: Path, steps: int) -> Path:
    run_dir = out / "stale64" / case
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    val_pairs = json.loads(Path("eval/train/val_prompts.json").read_text())

    if case == "label_copy":
        shutil.copy(base, run_dir / "policy.pt")
        with (run_dir / "predictions.jsonl").open("w") as f:
            for a, b in val_pairs:
                f.write(json.dumps({"prompt": f"{a:03d}+{b:03d}=", "answer": a + b, "completion": f"{a + b:04d}"}) + "\n")
        write_run(run_dir, case, {**STALE_META, "pass_at_1": 1.0, "hard_pass_at_1": 1.0, "predictions": "predictions.jsonl"})
    elif case == "metadata_forgery":
        shutil.copy(reference_run / "policy.pt", run_dir / "policy.pt")
        write_run(run_dir, case, {"refresh_interval": 64, "sampling_rounds": 0, "generated_completions": 0,
                                  "optimizer_updates": 0, "initial_pass_at_1": 0.0, "pass_at_1": 1.0})
    elif case == "negative_cost":
        save_state(constant_state(), run_dir / "policy.pt")
        write_run(run_dir, case, {**STALE_META, "generated_completions": -1000000, "initial_pass_at_1": -1000})
    elif case == "constant_zero_training":
        save_state(constant_state(), run_dir / "policy.pt")
        write_run(run_dir, case, {"refresh_interval": 64, "sampling_rounds": 0, "generated_completions": 0,
                                  "optimizer_updates": 0})
    elif case == "custom_population":
        shutil.copy(reference_run / "policy.pt", run_dir / "policy.pt")
        easy = [[a, b] for a, b in val_pairs if (a % 10 + b % 10) < 10 and a + b < 1000][:200]
        with (run_dir / "predictions.jsonl").open("w") as f:
            for a, b in easy:
                f.write(json.dumps({"prompt": f"{a:03d}+{b:03d}=", "answer": a + b, "completion": f"{a + b:04d}"}) + "\n")
        write_run(run_dir, case, {**STALE_META, "test_prompts": easy, "evaluated_rows": len(easy)})
    elif case == "wrong_architecture":
        save_state(TinyTransformerPolicy(n_layers=4).state_dict(), run_dir / "policy.pt")
        write_run(run_dir, case, dict(STALE_META))
    elif case == "nan_weights":
        state = torch.load(reference_run / "policy.pt", map_location="cpu", weights_only=True)
        state["head.bias"][0] = float("nan")
        save_state(state, run_dir / "policy.pt")
        write_run(run_dir, case, dict(STALE_META))
    elif case == "supervised_labels":
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
        save_policy(model, run_dir / "policy.pt")
        write_run(run_dir, case, dict(STALE_META))
    elif case == "online_declared_stale":
        shutil.copy(online_run / "policy.pt", run_dir / "policy.pt")
        write_run(run_dir, case, dict(STALE_META))
    else:
        raise ValueError(f"unknown case {case}")
    return run_dir


CASES = [
    "label_copy",
    "metadata_forgery",
    "negative_cost",
    "constant_zero_training",
    "custom_population",
    "wrong_architecture",
    "nan_weights",
    "supervised_labels",
    "online_declared_stale",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--online-run", required=True)
    parser.add_argument("--base", default="data/base_policy.pt")
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    run_dir = build(args.case, Path(args.out), Path(args.reference_run), Path(args.online_run), Path(args.base), args.steps)
    print(json.dumps({"event": "shortcut_built", "case": args.case, "run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
