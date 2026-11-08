"""Build the fixed starting policy (data/base_policy.pt).

The base policy plays the role of the pretrained instruction model in the
paper: a partially capable model that post-training starts from.  It is
trained once with supervised next-token prediction on a labelled pool that is
disjoint from the post-training prompt pool, and training stops as soon as
greedy exact-match accuracy on a small check set reaches ``--target``.

This is a gym asset, not part of any post-training method.  The committed
checkpoint is authoritative; rerunning this script on different hardware may
produce a slightly different file, whose hash would then not match the hash
pinned by the evaluators.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from data.arithmetic import answer_ids, base_check_set, base_sft_pool, prompt_ids
from data.policy import TinyTransformerPolicy, answer_logits, generate, save_policy


def to_tensors(problems, device):
    prompts = torch.tensor([prompt_ids(p) for p in problems], dtype=torch.long, device=device)
    answers = torch.tensor([answer_ids(p) for p in problems], dtype=torch.long, device=device)
    return prompts, answers


@torch.no_grad()
def greedy_accuracy(model, prompts, answers) -> float:
    model.eval()
    out = generate(model, prompts, temperature=0.0)
    model.train()
    return float((out == answers).all(dim=1).float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/base_policy.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    pool = base_sft_pool()
    pool_prompts, pool_answers = to_tensors(pool, device)
    check_prompts, check_answers = to_tensors(base_check_set(), device)

    model = TinyTransformerPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    history = []
    acc = 0.0
    step = 0
    for step in range(1, args.max_steps + 1):
        idx = torch.randint(0, len(pool), (args.batch_size,), generator=gen).to(device)
        logits = answer_logits(model, pool_prompts[idx], pool_answers[idx])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), pool_answers[idx].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.check_every == 0:
            acc = greedy_accuracy(model, check_prompts, check_answers)
            if step % (args.check_every * 50) == 0:
                print(json.dumps({"event": "base_sft", "step": step, "loss": float(loss), "check_acc": acc}), flush=True)
            history.append({"step": step, "loss": float(loss), "check_acc": acc})
            if acc >= args.target:
                break

    sha = save_policy(model, args.output)
    meta = {
        "sha256": sha,
        "sft_steps": step,
        "check_accuracy": acc,
        "target": args.target,
        "device": str(device),
        "config": vars(args),
        "history_tail": history[-20:],
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"event": "base_done", **{k: meta[k] for k in ("sha256", "sft_steps", "check_accuracy")}}), flush=True)


if __name__ == "__main__":
    main()
