"""FROZEN COPY of eval/test/evaluate.py as it stood before the final hardening
job (commit 7f52758), kept only to score the "before" side of the
rank-aggregation shortcut cases.  It reads the first 4000 hidden test prompts
(the pre-enlargement test set).  Not part of the gym.

Original docstring: held-out test evaluator for the stale-feedback post-training gym.

For every ``run.json`` found under ``--runs-root`` (default ``checkpoints/``)
this evaluator loads the run's ``policy.pt`` into its own copy of the fixed
policy architecture, runs greedy decoding itself on held-out prompts that only
this directory holds, and scores exact-match accuracy.  Nothing the method
wrote is trusted as a score: run.json is read only for the experiment name,
the method name, and the declared operating point, which can only fail a run.

Self-contained on purpose: no imports from data/, method/ or eval/train/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

SLICE = "test"
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
PROMPTS_FILE = WORKSPACE / "eval" / "test" / "test_prompts.json"
BASE_HASH_FILE = WORKSPACE / "eval" / "test" / "base_policy.sha256"
BASE_POLICY = WORKSPACE / "data" / "base_policy.pt"
DEFAULT_OUTPUT = HERE / "unused_scores_test.json"
BASE_ROW = "base_policy"

# Fixed operating points.  Budgets are per run.
EXPERIMENTS = {
    # sampling at most every 64 optimizer updates (64x less often than online)
    "stale64": {"max_generated": 131072, "max_updates": 512, "max_rounds": 8, "min_refresh": 64},
    # one sampling round drawn from the starting policy, then no new samples
    "offline": {"max_generated": 131072, "max_updates": 512, "max_rounds": 1, "min_refresh": 1},
}
HARD_MIN_CARRIES = 2
METHOD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}$")

# ---------------------------------------------------------------- task ----
VOCAB_SIZE = 13
PLUS, EQUALS, BOS = 10, 11, 12
PROMPT_LEN = 9
ANSWER_DIGITS = 4


def encode_prompt(a: int, b: int) -> list[int]:
    return [BOS] + [int(c) for c in f"{a:03d}"] + [PLUS] + [int(c) for c in f"{b:03d}"] + [EQUALS]


def encode_answer(a: int, b: int) -> list[int]:
    return [int(c) for c in f"{a + b:04d}"]


def carries(a: int, b: int) -> int:
    carry = count = 0
    for _ in range(3):
        carry = 1 if a % 10 + b % 10 + carry >= 10 else 0
        count += carry
        a //= 10
        b //= 10
    return count


# ---------------------------------------------------------- architecture ----
class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(dim, dim=-1)
        dh = dim // self.n_heads
        q = q.view(bsz, seq, self.n_heads, dh).transpose(1, 2)
        k = k.view(bsz, seq, self.n_heads, dh).transpose(1, 2)
        v = v.view(bsz, seq, self.n_heads, dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        att = att.masked_fill(~mask[:seq, :seq], float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(bsz, seq, dim)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class _Policy(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, max_len=16, d_model=128, n_layers=3, n_heads=4, d_ff=512):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1]))[None]
        for block in self.blocks:
            x = block(x, self.causal_mask)
        return self.head(self.ln_f(x))


def load_checkpoint(path: Path) -> _Policy:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("policy.pt is not a plain state_dict of tensors")
    model = _Policy()
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"state_dict keys differ from the fixed architecture ({len(state)} vs {len(expected)})")
    for key, value in state.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(f"{key}: shape {tuple(value.shape)} != {tuple(expected[key].shape)}")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key}: non-floating or non-finite values")
    model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=True)
    return model.eval()


@torch.no_grad()
def greedy_correct(model: _Policy, prompts: torch.Tensor, answers: torch.Tensor, chunk: int = 1000) -> torch.Tensor:
    out = []
    for i in range(0, len(prompts), chunk):
        ids = prompts[i : i + chunk]
        for _ in range(ANSWER_DIGITS):
            nxt = model(ids)[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        out.append((ids[:, PROMPT_LEN:] == answers[i : i + chunk]).all(dim=1))
    return torch.cat(out)


# ------------------------------------------------------------- gating ----
def _int_field(meta: dict, key: str, minimum: int) -> int:
    value = meta.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"run.json field {key!r} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"run.json field {key!r}={value} is impossible (minimum {minimum})")
    return value


def operating_point_violation(meta: dict, experiment: str) -> str | None:
    spec = EXPERIMENTS[experiment]
    generated = _int_field(meta, "generated_completions", 0)
    updates = _int_field(meta, "optimizer_updates", 0)
    rounds = _int_field(meta, "sampling_rounds", 0)
    refresh = _int_field(meta, "refresh_interval", 1)
    if generated > spec["max_generated"]:
        return f"generated_completions {generated} > budget {spec['max_generated']}"
    if updates > spec["max_updates"]:
        return f"optimizer_updates {updates} > budget {spec['max_updates']}"
    if rounds > spec["max_rounds"]:
        return f"sampling_rounds {rounds} > {spec['max_rounds']}"
    if spec["max_rounds"] > 1:
        if refresh < spec["min_refresh"]:
            return f"refresh_interval {refresh} < {spec['min_refresh']}"
        if updates > 0 and rounds > math.ceil(updates / refresh):
            return f"sampling_rounds {rounds} inconsistent with {updates} updates at refresh {refresh}"
    if generated > 0 and rounds == 0:
        return "generated completions without any sampling round"
    return None


# ---------------------------------------------------------------- main ----
def load_prompts():
    pairs = json.loads(PROMPTS_FILE.read_text())[:4000]
    prompts = torch.tensor([encode_prompt(a, b) for a, b in pairs], dtype=torch.long)
    answers = torch.tensor([encode_answer(a, b) for a, b in pairs], dtype=torch.long)
    hard = torch.tensor([carries(a, b) >= HARD_MIN_CARRIES for a, b in pairs], dtype=torch.bool)
    return prompts, answers, hard


def metrics_from(correct: torch.Tensor, hard: torch.Tensor, ok: bool) -> dict[str, float]:
    acc = float(correct.float().mean())
    hard_acc = float(correct[hard].float().mean())
    return {
        "pass_at_1": acc if ok else 0.0,
        "hard_pass_at_1": hard_acc if ok else 0.0,
        "unconstrained_pass_at_1": acc,
        "operating_point_ok": 1.0 if ok else 0.0,
    }


FAILED = {"pass_at_1": 0.0, "hard_pass_at_1": 0.0, "unconstrained_pass_at_1": 0.0, "operating_point_ok": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default=str(WORKSPACE / "checkpoints"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dump-correct", default="", help="optional directory for per-prompt correctness vectors")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    prompts, answers, hard = load_prompts()
    results: dict[str, dict[str, dict[str, float]]] = {name: {} for name in EXPERIMENTS}
    status: dict[str, str] = {}
    dump_dir = Path(args.dump_correct) if args.dump_correct else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    def record(experiment: str, method: str, row: dict, correct: torch.Tensor | None, note: str) -> None:
        key = method
        n = 2
        while key in results[experiment]:
            key = f"{method}__dup{n}"
            n += 1
        results[experiment][key] = row
        status[f"{experiment}/{key}"] = note
        if dump_dir is not None and correct is not None:
            (dump_dir / f"{experiment}__{key}.json").write_text(json.dumps(correct.int().tolist()))

    base_correct = None
    if BASE_HASH_FILE.is_file() and BASE_POLICY.is_file():
        pinned = BASE_HASH_FILE.read_text().strip()
        if hashlib.sha256(BASE_POLICY.read_bytes()).hexdigest() == pinned:
            base_correct = greedy_correct(load_checkpoint(BASE_POLICY), prompts, answers)
        else:
            status["_base_policy"] = "data/base_policy.pt does not match the pinned hash; base row omitted"
    for experiment in EXPERIMENTS:
        if base_correct is not None:
            record(experiment, BASE_ROW, metrics_from(base_correct, hard, True), base_correct,
                   "ok (evaluator-owned starting policy)")

    runs_root = Path(args.runs_root)
    run_files = sorted(runs_root.rglob("run.json")) if runs_root.is_dir() else []
    for run_file in run_files:
        try:
            meta = json.loads(run_file.read_text())
            experiment = meta.get("experiment")
            method = meta.get("method")
        except Exception as exc:  # unreadable run.json: nothing to attribute it to
            status[str(run_file)] = f"skipped: unreadable run.json ({exc})"
            continue
        if experiment not in EXPERIMENTS or not isinstance(method, str) or not METHOD_RE.match(method):
            status[str(run_file)] = f"skipped: unknown experiment {experiment!r} or invalid method name {method!r}"
            continue
        if method == BASE_ROW:
            method = f"{method}_submitted"
        try:
            violation = operating_point_violation(meta, experiment)
        except ValueError as exc:
            record(experiment, method, dict(FAILED), None, f"failed: {exc}")
            continue
        try:
            model = load_checkpoint(run_file.parent / "policy.pt")
            correct = greedy_correct(model, prompts, answers)
        except Exception as exc:
            record(experiment, method, dict(FAILED), None, f"failed: cannot evaluate policy.pt ({exc})")
            continue
        note = "ok" if violation is None else f"failed operating point: {violation}"
        record(experiment, method, metrics_from(correct, hard, violation is None), correct, note)

    scores = {
        "experiments": {
            name: {"slice": SLICE, "evaluated_prompts": int(len(prompts)), "results": rows}
            for name, rows in results.items()
        },
        "_status": status,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    sys.exit(main())
