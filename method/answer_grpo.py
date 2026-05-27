from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.synthetic_math import Example, make_examples, reward_completion, target_text


PROMPT_RE = re.compile(r"^(\d+)\+(\d+)=\?$")


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


class AnswerPolicy(nn.Module):
    """Small conditional policy over complete answer strings.

    The paper's computation is GRPO over generated completions.  For this
    scaled arithmetic gym, each action is one formatted answer string, which
    keeps the humanline/stale-data computation intact while avoiding failures
    caused only by character-level sequence modeling.
    """

    def __init__(self, max_value: int, hidden_size: int) -> None:
        super().__init__()
        self.max_value = max_value
        self.num_answers = 2 * max_value + 1
        self.embed = nn.Embedding(max_value + 1, hidden_size)
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.num_answers),
        )

    def forward(self, operands: torch.Tensor) -> torch.Tensor:
        emb = self.embed(operands)
        return self.net(emb.flatten(start_dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def operands_from_prompt(prompt: str) -> tuple[int, int]:
    match = PROMPT_RE.match(prompt)
    if match is None:
        raise ValueError(f"Unexpected prompt format: {prompt!r}")
    return int(match.group(1)), int(match.group(2))


def operand_tensor(examples: list[Example], device: torch.device) -> torch.Tensor:
    return torch.tensor([operands_from_prompt(ex.prompt) for ex in examples], dtype=torch.long, device=device)


def action_text(action: int) -> str:
    return target_text(action)


@torch.no_grad()
def sample_actions(
    model: AnswerPolicy,
    examples: list[Example],
    device: torch.device,
    group_size: int,
    temperature: float,
) -> dict[str, list]:
    operands = operand_tensor(examples, device)
    logits = model(operands) / max(temperature, 1e-6)
    dist = torch.distributions.Categorical(logits=logits)
    prompts: list[str] = []
    actions: list[int] = []
    rewards: list[float] = []
    group_ids: list[int] = []
    old_logps: list[float] = []
    for group_id, ex in enumerate(examples):
        for _ in range(group_size):
            action = int(dist.sample()[group_id].item())
            prompts.append(ex.prompt)
            actions.append(action)
            rewards.append(reward_completion(action_text(action), ex.answer))
            group_ids.append(group_id)
            old_logps.append(float(dist.log_prob(torch.tensor(action, device=device))[group_id].detach().cpu()))
    return {
        "prompts": prompts,
        "actions": actions,
        "rewards": rewards,
        "group_ids": group_ids,
        "old_logps": torch.tensor(old_logps, dtype=torch.float32),
    }


def logps_for_actions(model: AnswerPolicy, prompts: list[str], actions: list[int], device: torch.device) -> torch.Tensor:
    examples = [Example(prompt=p, answer=0) for p in prompts]
    operands = operand_tensor(examples, device)
    action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    return F.log_softmax(model(operands), dim=-1).gather(-1, action_tensor.unsqueeze(-1)).squeeze(-1)


def advantages(rewards: list[float], group_ids: list[int], device: torch.device) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32, device=device)
    gids = torch.tensor(group_ids, dtype=torch.long, device=device)
    adv = torch.zeros_like(values)
    for gid in torch.unique(gids):
        idx = gids == gid
        group = values[idx]
        adv[idx] = (group - group.mean()) / group.std(unbiased=False).clamp_min(1e-6)
    return adv


def supervised_warm_start(policy: AnswerPolicy, examples: list[Example], device: torch.device, steps: int, batch_size: int, lr: float) -> None:
    if steps <= 0:
        return
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    for _ in range(steps):
        batch = random.sample(examples, batch_size)
        operands = operand_tensor(batch, device)
        targets = torch.tensor([ex.answer for ex in batch], dtype=torch.long, device=device)
        loss = F.cross_entropy(policy(operands), targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()


def grpo_update(
    policy: AnswerPolicy,
    reference: AnswerPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, list],
    device: torch.device,
    epsilon: float,
    beta_kl: float,
    humanline_clipping: bool,
    log_epsilon_p: float,
    log_epsilon_r: float,
) -> float:
    old_logps = batch["old_logps"].to(device)
    adv = advantages(batch["rewards"], batch["group_ids"], device)
    policy_logps = logps_for_actions(policy, batch["prompts"], batch["actions"], device)
    with torch.no_grad():
        ref_logps = logps_for_actions(reference, batch["prompts"], batch["actions"], device)

    if humanline_clipping:
        clipped_delta = (policy_logps - ref_logps).clamp(log_epsilon_p, log_epsilon_r)
        surrogate_policy_logps = ref_logps + clipped_delta
    else:
        surrogate_policy_logps = policy_logps

    ratio = torch.exp(surrogate_policy_logps - old_logps)
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    objective = torch.minimum(ratio * adv, clipped_ratio * adv)
    approx_kl = torch.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1.0
    loss = -(objective - beta_kl * approx_kl).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def evaluate_policy(policy: AnswerPolicy, examples: list[Example], device: torch.device) -> tuple[float, float, float]:
    operands = operand_tensor(examples, device)
    actions = policy(operands).argmax(dim=-1).tolist()
    rewards = [reward_completion(action_text(action), ex.answer) for action, ex in zip(actions, examples)]
    correct = sum(int(action == ex.answer) for action, ex in zip(actions, examples))
    return correct / len(examples), sum(rewards) / len(rewards), 1.0


@torch.no_grad()
def sample_predictions(policy: AnswerPolicy, examples: list[Example], device: torch.device, n: int = 8) -> list[dict[str, str | int | float]]:
    operands = operand_tensor(examples[:n], device)
    actions = policy(operands).argmax(dim=-1).tolist()
    return [
        {
            "prompt": ex.prompt,
            "answer": ex.answer,
            "completion": action_text(action),
            "reward": reward_completion(action_text(action), ex.answer),
        }
        for action, ex in zip(actions, examples[:n])
    ]


def run(args: argparse.Namespace) -> RunMetrics:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_examples = make_examples(args.train_size, args.seed, args.min_value, args.max_value)
    eval_examples = make_examples(args.eval_size, args.seed + 10_000, args.min_value, args.max_value)

    policy = AnswerPolicy(args.max_value, args.hidden_size).to(device)
    supervised_warm_start(policy, train_examples, device, args.sft_steps, args.batch_size, args.sft_lr)
    initial_pass_at_1, initial_mean_reward, _ = evaluate_policy(policy, eval_examples, device)

    reference = AnswerPolicy(args.max_value, args.hidden_size).to(device)
    reference.load_state_dict(policy.state_dict())
    behavior = AnswerPolicy(args.max_value, args.hidden_size).to(device)
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
                # Paper-style stale sampling for math: sample a block of data
                # from the old policy, then consume one stale mini-batch per
                # optimizer step until the next refresh.
                remaining = min(args.refresh_interval, args.train_steps - step)
                batch_pool = []
                for _ in range(remaining):
                    sampled = random.choices(train_examples, k=args.batch_size)
                    batch_pool.append(sample_actions(behavior, sampled, device, args.group_size, args.temperature))
                generated_examples += remaining * args.batch_size * args.group_size
            else:
                sampled = random.sample(train_examples, args.batch_size)
                batch = sample_actions(behavior, sampled, device, args.group_size, args.temperature)
                generated_examples += len(sampled) * args.group_size
        if args.pool_stale_batches:
            assert batch_pool is not None
            batch = batch_pool[step % args.refresh_interval]
        use_humanline_sync = args.humanline and not args.disable_humanline_sync
        use_humanline_clipping = args.humanline and not args.disable_humanline_clipping
        if use_humanline_sync and (step + 1) % args.sync_k == 0:
            pending_reference_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        grpo_update(
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
        if pending_reference_state is not None:
            reference.load_state_dict(pending_reference_state)
            pending_reference_state = None

    pass_at_1, mean_reward, format_rate = evaluate_policy(policy, eval_examples, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_dir / f"{args.name}.pt")
    metrics = RunMetrics(
        method=args.name,
        train_steps=args.train_steps,
        refresh_interval=args.refresh_interval,
        humanline_sync_k=args.sync_k if args.humanline and not args.disable_humanline_sync else 0,
        pass_at_1=pass_at_1,
        mean_reward=mean_reward,
        format_rate=format_rate,
        stale_refresh_interval=args.refresh_interval,
        generated_examples=generated_examples,
        initial_pass_at_1=initial_pass_at_1,
        initial_mean_reward=initial_mean_reward,
        samples=sample_predictions(policy, eval_examples, device),
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
    parser.add_argument("--sync-k", type=int, default=12)
    parser.add_argument("--refresh-interval", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=768)
    parser.add_argument("--sft-steps", type=int, default=80)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sft-lr", type=float, default=2e-3)
    parser.add_argument("--rl-lr", type=float, default=5e-4)
    parser.add_argument("--grpo-epsilon", type=float, default=0.15)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--log-epsilon-p", type=float, default=-1.5)
    parser.add_argument("--log-epsilon-r", type=float, default=1.5)
    parser.add_argument("--min-value", type=int, default=0)
    parser.add_argument("--max-value", type=int, default=19)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(asdict(result), indent=2))
