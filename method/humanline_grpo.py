from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from data.synthetic_math import CHARS, EOS, PAD, decode, encode, make_balanced_examples, make_examples, reward_completion, target_text
from method.model import StructuredAnswerTokenPolicy, TinyGRUPolicy


@dataclass
class RunMetrics:
    method: str
    train_steps: int
    refresh_interval: int
    humanline_sync_k: int
    pass_at_1: float
    mean_reward: float
    format_rate: float
    stale_refresh_interval: int
    generated_examples: int
    initial_pass_at_1: float
    initial_mean_reward: float
    samples: list[dict[str, str | int | float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_sequences(seqs: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), PAD, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool, device=device)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, : len(seq)] = True
    return ids, mask


def token_logps(model: TinyGRUPolicy, prefixes: list[str], completions: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "completion_logps"):
        return model.completion_logps(prefixes, completions, device)
    seqs = [encode(p + c, add_eos=True) for p, c in zip(prefixes, completions)]
    prompt_lens = [len(encode(p, add_eos=False)) for p in prefixes]
    ids, _ = collate_sequences(seqs, device)
    logits = model(ids[:, :-1])
    labels = ids[:, 1:]
    all_logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(all_logps, dtype=torch.bool)
    for i, (seq, prompt_len) in enumerate(zip(seqs, prompt_lens)):
        start = max(prompt_len - 1, 0)
        end = len(seq) - 1
        mask[i, start:end] = True
    return all_logps, mask


@torch.no_grad()
def generate_one(
    model: TinyGRUPolicy,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> str:
    was_training = model.training
    model.eval()
    try:
        if hasattr(model, "generate_completion"):
            return model.generate_completion(prompt, device, temperature)
        ids = torch.tensor([encode(prompt, add_eos=False)], dtype=torch.long, device=device)
        out: list[int] = []
        for _ in range(max_new_tokens):
            logits = model(ids)[:, -1, :] / max(temperature, 1e-6)
            if temperature <= 0.05:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            if next_id == EOS:
                break
            out.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        return decode(out)
    finally:
        if was_training:
            model.train()


def supervised_warm_start(model: TinyGRUPolicy, examples, device: torch.device, steps: int, batch_size: int, lr: float) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        batch = random.sample(examples, batch_size)
        if hasattr(model, "supervised_loss"):
            loss = model.supervised_loss([ex.prompt for ex in batch], [ex.answer for ex in batch], device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            continue
        seqs = [encode(ex.prompt + target_text(ex.answer), add_eos=True) for ex in batch]
        prompt_lens = [len(encode(ex.prompt, add_eos=False)) for ex in batch]
        ids, _ = collate_sequences(seqs, device)
        logits = model(ids[:, :-1])
        labels = ids[:, 1:]
        loss_mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, (seq, prompt_len) in enumerate(zip(seqs, prompt_lens)):
            start = max(prompt_len - 1, 0)
            end = len(seq) - 1
            loss_mask[i, start:end] = True
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, len(CHARS)),
            labels.reshape(-1),
            ignore_index=PAD,
            reduction="none",
        ).view_as(labels)
        loss = per_token_loss[loss_mask].mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def collect_predictions(model: TinyGRUPolicy, examples, device: torch.device, max_new_tokens: int) -> list[dict[str, str | int | float]]:
    rows = []
    model.eval()
    for ex in examples:
        text = generate_one(model, ex.prompt, device, max_new_tokens, temperature=0.01)
        reward = reward_completion(text, ex.answer)
        rows.append(
            {
                "prompt": ex.prompt,
                "answer": ex.answer,
                "completion": text,
                "reward": reward,
            }
        )
    model.train()
    return rows


def score_prediction_rows(rows: list[dict[str, str | int | float]]) -> tuple[float, float, float]:
    rewards = [float(row["reward"]) for row in rows]
    correct = sum(int(float(row["reward"]) >= 1.0) for row in rows)
    formatted = sum(int("<answer>" in str(row["completion"]) and "</answer>" in str(row["completion"])) for row in rows)
    if not rows:
        return 0.0, 0.0, 0.0
    return correct / len(rows), sum(rewards) / len(rewards), formatted / len(rows)


@torch.no_grad()
def evaluate_policy(model: TinyGRUPolicy, examples, device: torch.device, max_new_tokens: int) -> tuple[float, float, float]:
    rows = collect_predictions(model, examples, device, max_new_tokens)
    return score_prediction_rows(rows)


@torch.no_grad()
def sample_predictions(model: TinyGRUPolicy, examples, device: torch.device, max_new_tokens: int, n: int = 8) -> list[dict[str, str | int | float]]:
    model.eval()
    rows = []
    for ex in examples[:n]:
        completion = generate_one(model, ex.prompt, device, max_new_tokens, temperature=0.01)
        rows.append(
            {
                "prompt": ex.prompt,
                "answer": ex.answer,
                "completion": completion,
                "reward": reward_completion(completion, ex.answer),
            }
        )
    model.train()
    return rows


@torch.no_grad()
def make_grpo_batch(
    behavior: TinyGRUPolicy,
    examples,
    device: torch.device,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, list]:
    prompts: list[str] = []
    completions: list[str] = []
    rewards: list[float] = []
    group_ids: list[int] = []
    for group_id, ex in enumerate(examples):
        for _ in range(group_size):
            completion = generate_one(behavior, ex.prompt, device, max_new_tokens, temperature)
            prompts.append(ex.prompt)
            completions.append(completion)
            rewards.append(reward_completion(completion, ex.answer))
            group_ids.append(group_id)
    old_logps, mask = token_logps(behavior, prompts, completions, device)
    return {
        "prompts": prompts,
        "completions": completions,
        "rewards": rewards,
        "group_ids": group_ids,
        "old_logps": old_logps.detach().cpu(),
        "mask": mask.detach().cpu(),
    }


def advantages(rewards: list[float], group_ids: list[int], device: torch.device) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32, device=device)
    gids = torch.tensor(group_ids, dtype=torch.long, device=device)
    adv = torch.zeros_like(values)
    for gid in torch.unique(gids):
        idx = gids == gid
        group = values[idx]
        std = group.std(unbiased=False).clamp_min(1e-6)
        adv[idx] = (group - group.mean()) / std
    return adv


def grpo_update(
    policy: TinyGRUPolicy,
    reference: TinyGRUPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, list],
    device: torch.device,
    epsilon: float,
    beta_kl: float,
    humanline: bool,
    log_epsilon_p: float,
    log_epsilon_r: float,
) -> float:
    prompts = batch["prompts"]
    completions = batch["completions"]
    mask = batch["mask"].to(device)
    adv = advantages(batch["rewards"], batch["group_ids"], device).unsqueeze(-1)

    policy_logps, _ = token_logps(policy, prompts, completions, device)
    with torch.no_grad():
        ref_logps, _ = token_logps(reference, prompts, completions, device)

    if humanline:
        # Paper's Humanline GRPO uses policy/reference token ratios,
        # clamped upstream before the standard GRPO surrogate clips them.
        logratio = (policy_logps - ref_logps).clamp(log_epsilon_p, log_epsilon_r)
    else:
        logratio = policy_logps - ref_logps

    ratio = torch.exp(logratio)
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    pg_terms = torch.minimum(ratio * adv, clipped_ratio * adv)
    approx_kl = torch.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1.0
    objective = pg_terms - beta_kl * approx_kl
    loss = -objective[mask].mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def run(args: argparse.Namespace) -> RunMetrics:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.balanced_train:
        train_examples = make_balanced_examples(args.train_size, args.seed, args.min_value, args.max_value)
    else:
        train_examples = make_examples(args.train_size, args.seed, args.min_value, args.max_value)
    eval_examples = make_examples(args.eval_size, args.seed + 10_000, args.min_value, args.max_value)

    if args.policy_type == "structured":
        policy = StructuredAnswerTokenPolicy(args.max_value, args.hidden_size).to(device)
    else:
        policy = TinyGRUPolicy(len(CHARS), args.hidden_size).to(device)
    supervised_warm_start(policy, train_examples, device, args.sft_steps, args.batch_size, args.sft_lr)
    initial_predictions = collect_predictions(policy, eval_examples, device, args.max_new_tokens)
    initial_pass_at_1, initial_mean_reward, _ = score_prediction_rows(initial_predictions)
    if args.policy_type == "structured":
        reference = StructuredAnswerTokenPolicy(args.max_value, args.hidden_size).to(device)
    else:
        reference = TinyGRUPolicy(len(CHARS), args.hidden_size).to(device)
    reference.load_state_dict(policy.state_dict())
    if args.policy_type == "structured":
        behavior = StructuredAnswerTokenPolicy(args.max_value, args.hidden_size).to(device)
    else:
        behavior = TinyGRUPolicy(len(CHARS), args.hidden_size).to(device)
    behavior.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.rl_lr, weight_decay=1e-4)

    batch = None
    batch_pool = None
    generated_examples = 0
    pending_reference_state = None
    for step in range(args.train_steps):
        if step % args.refresh_interval == 0 or batch is None:
            behavior.load_state_dict(policy.state_dict())
            if args.pool_stale_batches:
                # Match the paper's stale math protocol more closely:
                # generate a pool from the old policy, then train on one
                # stale mini-batch per step until the next refresh.
                remaining = min(args.refresh_interval, args.train_steps - step)
                batch_pool = []
                for _ in range(remaining):
                    sampled = random.choices(train_examples, k=args.batch_size)
                    batch_pool.append(make_grpo_batch(behavior, sampled, device, args.group_size, args.max_new_tokens, args.temperature))
                generated_examples += remaining * args.batch_size * args.group_size
            else:
                sampled = random.sample(train_examples, args.batch_size)
                batch = make_grpo_batch(behavior, sampled, device, args.group_size, args.max_new_tokens, args.temperature)
                generated_examples += len(sampled) * args.group_size
        if args.pool_stale_batches:
            assert batch_pool is not None
            batch = batch_pool[step % args.refresh_interval]
        use_humanline_sync = args.humanline and not args.disable_humanline_sync
        use_humanline_clipping = args.humanline and not args.disable_humanline_clipping
        if use_humanline_sync and (step + 1) % args.sync_k == 0:
            pending_reference_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        loss = grpo_update(
            policy,
            reference,
            optimizer,
            batch,
            device,
            args.grpo_epsilon,
            args.beta_kl,
            use_humanline_clipping,
            args.log_epsilon_p,
            args.log_epsilon_r,
        )
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite GRPO loss at step {step + 1}: {loss}")
        if args.log_every and ((step + 1) % args.log_every == 0 or step == 0):
            print(
                json.dumps(
                    {
                        "event": "train_step",
                        "method": args.name,
                        "step": step + 1,
                        "loss": loss,
                        "generated_examples": generated_examples,
                    }
                ),
                flush=True,
            )
        if pending_reference_state is not None:
            reference.load_state_dict(pending_reference_state)
            pending_reference_state = None

    final_predictions = collect_predictions(policy, eval_examples, device, args.max_new_tokens)
    pass_at_1, mean_reward, format_rate = score_prediction_rows(final_predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_dir / f"{args.name}.pt")
    with (out_dir / f"{args.name}_initial_predictions.jsonl").open("w") as f:
        for row in initial_predictions:
            f.write(json.dumps(row) + "\n")
    with (out_dir / f"{args.name}_predictions.jsonl").open("w") as f:
        for row in final_predictions:
            f.write(json.dumps(row) + "\n")
    metrics = RunMetrics(
        method=args.name,
        train_steps=args.train_steps,
        refresh_interval=args.refresh_interval,
        humanline_sync_k=args.sync_k if use_humanline_sync else 0,
        pass_at_1=pass_at_1,
        mean_reward=mean_reward,
        format_rate=format_rate,
        stale_refresh_interval=args.refresh_interval,
        generated_examples=generated_examples,
        initial_pass_at_1=initial_pass_at_1,
        initial_mean_reward=initial_mean_reward,
        samples=sample_predictions(policy, eval_examples, device, args.max_new_tokens),
    )
    with (out_dir / f"{args.name}.json").open("w") as f:
        json.dump(asdict(metrics), f, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", default="checkpoints/humanline_grpo")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--humanline", action="store_true")
    parser.add_argument("--disable-humanline-clipping", action="store_true")
    parser.add_argument("--disable-humanline-sync", action="store_true")
    parser.add_argument("--pool-stale-batches", action="store_true")
    parser.add_argument("--balanced-train", action="store_true")
    parser.add_argument("--policy-type", choices=["gru", "structured"], default="gru")
    parser.add_argument("--sync-k", type=int, default=12)
    parser.add_argument("--refresh-interval", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=256)
    parser.add_argument("--sft-steps", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sft-lr", type=float, default=2e-3)
    parser.add_argument("--rl-lr", type=float, default=5e-4)
    parser.add_argument("--grpo-epsilon", type=float, default=0.15)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--log-epsilon-p", type=float, default=-1.5)
    parser.add_argument("--log-epsilon-r", type=float, default=1.5)
    parser.add_argument("--min-value", type=int, default=0)
    parser.add_argument("--max-value", type=int, default=19)
    parser.add_argument("--log-every", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(asdict(result), indent=2))
