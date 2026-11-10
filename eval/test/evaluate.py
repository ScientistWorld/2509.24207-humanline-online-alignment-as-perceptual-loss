"""Held-out test evaluator for the stale-feedback post-training gym.

For every ``run.json`` found under ``--runs-root`` (default ``checkpoints/``)
this evaluator loads the run's ``policy.pt`` into its own copy of the fixed
policy architecture, runs greedy decoding itself on held-out prompts that only
this directory holds, and scores exact-match accuracy.  Nothing the method
wrote is trusted as a score: run.json is read only for the experiment name,
the method name, the training seed and the declared operating point, which can
only fail a run.

Seeds.  Training a 0.6M-parameter policy with RL is noisy from seed to seed,
so a method is scored over a fixed set of training seeds (``SEEDS``): each
(experiment, method) row is the mean over those seeds, and a seed with no
usable run contributes 0.  Several runs declaring the same seed score the
worst of them; runs declaring other seeds are ignored; a run whose greedy
outputs repeat an earlier seed's (a copied or near-copied policy) contributes
0.  Every method found is scored in every experiment (a missing experiment is
a failed row).  If any submitted run of a method declares a schedule outside
the experiment's operating point, or an impossible value, the whole row fails.

Self-contained on purpose: no imports from data/, method/ or eval/train/.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
PROMPTS_FILE = HERE / "test_prompts.json"
BASE_HASH_FILE = HERE / "base_policy.sha256"
BASE_POLICY = WORKSPACE / "data" / "base_policy.pt"
DEFAULT_OUTPUT = WORKSPACE / "scoring" / "scores_test.json"
BASE_ROW = "base_policy"

# Fixed operating points.  Budgets are per run (one run = one training seed).
EXPERIMENTS = {
    # sampling at most every 64 optimizer updates (64x less often than online)
    "stale64": {"max_generated": 131072, "max_updates": 512, "max_rounds": 8, "min_refresh": 64},
    # one sampling round drawn from the starting policy, then no new samples
    "offline": {"max_generated": 131072, "max_updates": 512, "max_rounds": 1, "min_refresh": 1},
}
SEEDS = (7, 8, 9, 10, 11, 12, 13, 14)  # required training seeds per (experiment, method)
COPY_AGREEMENT = 0.9   # identical outputs on >= 90% of the prompts where either run errs ...
COPY_MIN_ERRORS = 50   # ... counted over at least this many prompts
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

    def step(self, x: torch.Tensor, past, start: int, mask: torch.Tensor):
        """Causal self-attention over cached keys/values plus the new positions."""
        bsz, seq, dim = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(dim, dim=-1)
        dh = dim // self.n_heads
        q = q.view(bsz, seq, self.n_heads, dh).transpose(1, 2)
        k = k.view(bsz, seq, self.n_heads, dh).transpose(1, 2)
        v = v.view(bsz, seq, self.n_heads, dh).transpose(1, 2)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        total = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        att = att.masked_fill(~mask[start : start + seq, :total], float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(bsz, seq, dim)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)))), (k, v)


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
        """Full (uncached) forward pass: logits for every position."""
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        for block in self.blocks:
            x, _ = block.step(x, None, 0, self.causal_mask)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def greedy(self, prompts: torch.Tensor) -> torch.Tensor:
        """Greedy decoding of ANSWER_DIGITS tokens with a key/value cache."""
        plen = prompts.shape[1]
        x = self.tok(prompts) + self.pos(torch.arange(plen, device=prompts.device))[None]
        caches = []
        for block in self.blocks:
            x, kv = block.step(x, None, 0, self.causal_mask)
            caches.append(kv)
        nxt = self.head(self.ln_f(x[:, -1])).argmax(dim=-1)
        out = [nxt]
        for t in range(1, ANSWER_DIGITS):
            pos = plen + t - 1
            x = self.tok(nxt[:, None]) + self.pos.weight[pos][None, None]
            for j, block in enumerate(self.blocks):
                x, caches[j] = block.step(x, caches[j], pos, self.causal_mask)
            nxt = self.head(self.ln_f(x[:, -1])).argmax(dim=-1)
            out.append(nxt)
        return torch.stack(out, dim=1)


def load_checkpoint(data: bytes) -> _Policy:
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
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
def greedy_outputs(model: _Policy, prompts: torch.Tensor, device: torch.device, chunk: int = 4096) -> torch.Tensor:
    """Greedy answer tokens for every prompt, (P, ANSWER_DIGITS) uint8 on the CPU."""
    model = model.to(device)
    out = [model.greedy(prompts[i : i + chunk].to(device)).to("cpu", torch.uint8) for i in range(0, len(prompts), chunk)]
    return torch.cat(out)


@torch.no_grad()
def greedy_outputs_uncached(model: _Policy, prompts: torch.Tensor, device: torch.device, chunk: int = 4096) -> torch.Tensor:
    """Reference decoding without the cache (used only by --verify-decode)."""
    model = model.to(device)
    out = []
    for i in range(0, len(prompts), chunk):
        ids = prompts[i : i + chunk].to(device)
        for _ in range(ANSWER_DIGITS):
            ids = torch.cat([ids, model(ids)[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
        out.append(ids[:, PROMPT_LEN:].to("cpu", torch.uint8))
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


def copy_agreement(out_a: torch.Tensor, ok_a: torch.Tensor, out_b: torch.Tensor, ok_b: torch.Tensor):
    """Share of identical outputs among prompts where either run errs, and that prompt count."""
    either_wrong = ~(ok_a & ok_b)
    n = int(either_wrong.sum())
    if n == 0:
        return 0.0, 0
    same = ((out_a == out_b).all(dim=1) & either_wrong).sum()
    return float(same) / n, n


# ---------------------------------------------------------------- main ----
def load_prompts():
    pairs = json.loads(PROMPTS_FILE.read_text())
    prompts = torch.tensor([encode_prompt(a, b) for a, b in pairs], dtype=torch.long)
    answers = torch.tensor([encode_answer(a, b) for a, b in pairs], dtype=torch.uint8)
    hard = torch.tensor([carries(a, b) >= HARD_MIN_CARRIES for a, b in pairs], dtype=torch.bool)
    return prompts, answers, hard


FAILED = {"pass_at_1": 0.0, "hard_pass_at_1": 0.0, "unconstrained_pass_at_1": 0.0, "operating_point_ok": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default=str(WORKSPACE / "checkpoints"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dump-correct", default="", help="optional directory for per-prompt correctness vectors")
    parser.add_argument("--verify-decode", type=int, default=0,
                        help="check cached against uncached decoding on the first N prompts of every policy")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    prompts, answers, hard = load_prompts()
    n_prompts = len(prompts)
    status: dict[str, str] = {}
    results: dict[str, dict[str, dict[str, float]]] = {name: {} for name in EXPERIMENTS}
    per_seed: dict[str, dict[str, dict[str, dict]]] = {name: {} for name in EXPERIMENTS}
    dump_dir = Path(args.dump_correct) if args.dump_correct else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    decode_mismatch = {"prompts_checked": 0, "mismatched_outputs": 0}

    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def evaluate_policy(data: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        sha = hashlib.sha256(data).hexdigest()
        if sha not in cache:
            model = load_checkpoint(data)
            outputs = greedy_outputs(model, prompts, device)
            if args.verify_decode:
                m = min(args.verify_decode, n_prompts)
                ref = greedy_outputs_uncached(model, prompts[:m], device)
                decode_mismatch["prompts_checked"] += m
                decode_mismatch["mismatched_outputs"] += int((ref != outputs[:m]).any(dim=1).sum())
            cache[sha] = (outputs, (outputs == answers).all(dim=1))
        return cache[sha]

    def accuracy(correct: torch.Tensor) -> tuple[float, float]:
        return float(correct.float().mean()), float(correct[hard].float().mean())

    # evaluator-owned starting policy (the no-post-training baseline row)
    if BASE_HASH_FILE.is_file() and BASE_POLICY.is_file():
        data = BASE_POLICY.read_bytes()
        if hashlib.sha256(data).hexdigest() == BASE_HASH_FILE.read_text().strip():
            _, base_correct = evaluate_policy(data)
            acc, hard_acc = accuracy(base_correct)
            for experiment in EXPERIMENTS:
                results[experiment][BASE_ROW] = {"pass_at_1": acc, "hard_pass_at_1": hard_acc,
                                                 "unconstrained_pass_at_1": acc, "operating_point_ok": 1.0}
                status[f"{experiment}/{BASE_ROW}"] = "ok (evaluator-owned starting policy, one evaluation)"
                if dump_dir is not None:
                    (dump_dir / f"{experiment}__{BASE_ROW}.json").write_text(
                        json.dumps({"mean": base_correct.int().tolist()}))
        else:
            status["_base_policy"] = "data/base_policy.pt does not match the pinned hash; base row omitted"

    # ---- collect submitted runs: method -> experiment -> seed -> [entries]
    runs_root = Path(args.runs_root)
    run_files = sorted(runs_root.rglob("run.json")) if runs_root.is_dir() else []
    methods: set[str] = set()
    submitted: dict[str, dict[str, dict[int, list[dict]]]] = {}
    violations: dict[tuple[str, str], list[str]] = {}
    for run_file in run_files:
        try:
            meta = json.loads(run_file.read_text())
            if not isinstance(meta, dict):
                raise ValueError("not a JSON object")
        except Exception as exc:  # unreadable run.json: nothing to attribute it to
            status[str(run_file)] = f"skipped: unreadable run.json ({exc})"
            continue
        method = meta.get("method")
        experiment = meta.get("experiment")
        if not isinstance(method, str) or not METHOD_RE.match(method):
            status[str(run_file)] = f"skipped: invalid method name {method!r}"
            continue
        if method == BASE_ROW:
            method = f"{BASE_ROW}_submitted"
        methods.add(method)
        if experiment not in EXPERIMENTS:
            status[str(run_file)] = f"ignored: unknown experiment {experiment!r}"
            continue
        seed = meta.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
            status[str(run_file)] = f"ignored: seed {seed!r} is not one of the required seeds {list(SEEDS)}"
            continue
        try:
            violation = operating_point_violation(meta, experiment)
        except ValueError as exc:
            violation = str(exc)
        if violation is not None:
            violations.setdefault((experiment, method), []).append(f"seed {seed}: {violation}")
        entry = {"file": str(run_file), "violation": violation}
        try:
            entry["outputs"], entry["correct"] = evaluate_policy((run_file.parent / "policy.pt").read_bytes())
        except Exception as exc:
            entry["error"] = f"cannot evaluate policy.pt ({exc})"
        submitted.setdefault(method, {}).setdefault(experiment, {}).setdefault(seed, []).append(entry)

    # ---- aggregate: every method in every experiment, mean over the required seeds
    for method in sorted(methods):
        for experiment in EXPERIMENTS:
            by_seed = submitted.get(method, {}).get(experiment, {})
            kept: list[tuple[int, torch.Tensor, torch.Tensor]] = []
            seed_rows: dict[str, dict] = {}
            mean_correct = torch.zeros(n_prompts, dtype=torch.float64)
            per_seed_correct = {}
            for seed in SEEDS:
                entries = by_seed.get(seed, [])
                row = {"pass_at_1": 0.0, "hard_pass_at_1": 0.0, "accuracy": 0.0}
                if not entries:
                    row["note"] = "missing: no run declares this seed"
                elif any("error" in e for e in entries):
                    row["note"] = next(e["error"] for e in entries if "error" in e)
                else:
                    worst = min(entries, key=lambda e: accuracy(e["correct"]))
                    copy_of = None
                    for seed_b, out_b, ok_b in kept:
                        agree, n = copy_agreement(worst["outputs"], worst["correct"], out_b, ok_b)
                        if n >= COPY_MIN_ERRORS and agree >= COPY_AGREEMENT:
                            copy_of = (seed_b, agree, n)
                            break
                    if copy_of is not None:
                        row["note"] = (f"copy of seed {copy_of[0]}: identical outputs on {copy_of[1]:.3f} of the "
                                       f"{copy_of[2]} prompts where either run errs; scored 0")
                    else:
                        kept.append((seed, worst["outputs"], worst["correct"]))
                        acc, hard_acc = accuracy(worst["correct"])
                        row.update({"pass_at_1": acc, "hard_pass_at_1": hard_acc, "accuracy": acc})
                        row["note"] = "ok" if len(entries) == 1 else f"ok (worst of {len(entries)} runs declaring this seed)"
                        mean_correct += worst["correct"].double()
                        per_seed_correct[str(seed)] = worst["correct"].int().tolist()
                seed_rows[str(seed)] = row
            mean_correct /= len(SEEDS)
            unconstrained = sum(r["accuracy"] for r in seed_rows.values()) / len(SEEDS)
            failed = violations.get((experiment, method))
            if not by_seed:
                results[experiment][method] = dict(FAILED)
                status[f"{experiment}/{method}"] = "failed: no run submitted for this experiment"
            elif failed:
                results[experiment][method] = {**FAILED, "unconstrained_pass_at_1": unconstrained}
                status[f"{experiment}/{method}"] = "failed operating point: " + "; ".join(failed[:3]) + (
                    f" (+{len(failed) - 3} more)" if len(failed) > 3 else "")
            else:
                results[experiment][method] = {
                    "pass_at_1": sum(r["pass_at_1"] for r in seed_rows.values()) / len(SEEDS),
                    "hard_pass_at_1": sum(r["hard_pass_at_1"] for r in seed_rows.values()) / len(SEEDS),
                    "unconstrained_pass_at_1": unconstrained,
                    "operating_point_ok": 1.0,
                }
                scored = sum(1 for r in seed_rows.values() if r["note"].startswith("ok"))
                problems = [f"seed {s}: {r['note']}" for s, r in seed_rows.items() if not r["note"].startswith("ok")]
                status[f"{experiment}/{method}"] = (f"ok ({scored}/{len(SEEDS)} seeds scored)" if not problems else
                                                    f"{scored}/{len(SEEDS)} seeds scored; " + "; ".join(problems))
            per_seed[experiment][method] = {s: {k: r[k] for k in ("pass_at_1", "hard_pass_at_1", "note")}
                                            for s, r in seed_rows.items()}
            if dump_dir is not None:
                (dump_dir / f"{experiment}__{method}.json").write_text(json.dumps(
                    {"mean": [round(v, 4) for v in mean_correct.tolist()], "per_seed": per_seed_correct}))

    scores = {
        "experiments": {
            name: {"slice": SLICE, "evaluated_prompts": int(n_prompts), "required_seeds": list(SEEDS), "results": rows}
            for name, rows in results.items()
        },
        "_per_seed": per_seed,
        "_status": status,
    }
    if args.verify_decode:
        scores["_decode_check"] = {**decode_mismatch, "device": str(device)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps({"experiments": scores["experiments"], "_status": status}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
