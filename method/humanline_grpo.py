"""Token-level GRPO with optional humanline syncing and clipping.

Implements the paper's design pattern (Sec. 5.1 and Appendix C.2) on the gym's
fixed tiny transformer policy:

* GRPO (Eq. 1): group-relative advantages from a verifiable reward, a
  token-wise policy/reference probability ratio inside the clipped surrogate,
  a k3 KL penalty to the reference, DAPO-style token-mean normalisation.
* Sampling schedule: a pool of completions is sampled from a frozen snapshot of
  the policy every ``refresh_interval`` optimizer steps and consumed one
  mini-batch per step.  ``refresh_interval=1`` is online GRPO; 64 is the
  paper's 64x-less-frequent sampling; ``refresh_interval=train_steps`` is a
  single offline pool drawn from the starting policy.
* Reference model: at the start of each sampling round the reference is set to
  the policy that generated the round (Algorithm 1, line 3).  Without
  humanline it stays there for the whole round (online/stale GRPO; with a
  single round it never moves, i.e. offline GRPO).
* Humanline syncing: every ``sync_k`` steps, after the loss is computed and
  before the optimizer step, the reference is synced to the current policy
  (it therefore holds the previous-step policy for the next update).
* Humanline clipping: token-wise log-ratios are clamped to
  [log_epsilon_P, log_epsilon_R] upstream of the GRPO surrogate (which clips a
  second time).  The KL term is not humanline-clipped.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch

from data.arithmetic import Problem, prompt_ids, reward, train_pool
from data.policy import answer_token_logps, file_sha256, generate, load_policy, save_policy


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def sample_round(
    behavior: torch.nn.Module,
    problems: list[Problem],
    group_size: int,
    temperature: float,
    generator: torch.Generator,
    device: torch.device,
    chunk: int,
) -> dict[str, torch.Tensor]:
    """Sample ``group_size`` completions per prompt from a frozen policy snapshot."""
    behavior.eval()
    prompts = torch.tensor([prompt_ids(p) for p in problems for _ in range(group_size)], dtype=torch.long, device=device)
    pieces = [generate(behavior, prompts[i : i + chunk], temperature, generator) for i in range(0, len(prompts), chunk)]
    completions = torch.cat(pieces, dim=0)
    flat = completions.tolist()
    rewards = torch.tensor(
        [reward(flat[i], problems[i // group_size]) for i in range(len(flat))], dtype=torch.float32, device=device
    )
    grouped = rewards.view(-1, group_size)
    adv = (grouped - grouped.mean(dim=1, keepdim=True)) / grouped.std(dim=1, correction=0, keepdim=True).clamp_min(1e-6)
    return {"prompts": prompts, "completions": completions, "rewards": rewards, "advantages": adv.reshape(-1)}


def grpo_loss(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    prompts: torch.Tensor,
    completions: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace,
    humanline_clip: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_logps = answer_token_logps(policy, prompts, completions, args.temperature)
    with torch.no_grad():
        reference_logps = answer_token_logps(reference, prompts, completions, args.temperature)
    raw_logratio = policy_logps - reference_logps
    if humanline_clip:
        # humanline clipping: clamp token-wise log-ratios upstream of the loss
        logratio = raw_logratio.clamp(args.log_epsilon_p, args.log_epsilon_r)
    else:
        logratio = raw_logratio
    ratio = logratio.exp()
    adv = advantages.unsqueeze(-1)
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1.0 - args.grpo_epsilon, 1.0 + args.grpo_epsilon) * adv)
    # k3 KL estimator to the reference model (not humanline-clipped)
    kl = torch.exp(reference_logps - policy_logps) - (reference_logps - policy_logps) - 1.0
    loss = (-surrogate + args.beta_kl * kl).mean()
    with torch.no_grad():
        r = raw_logratio.exp()
        trust_clipped = ((adv > 0) & (r > 1 + args.grpo_epsilon)) | ((adv < 0) & (r < 1 - args.grpo_epsilon))
        stats = {
            "grpo_clip_frac": float(trust_clipped.float().mean()),
            "humanline_clamp_frac": float(
                ((raw_logratio < args.log_epsilon_p) | (raw_logratio > args.log_epsilon_r)).float().mean()
            ),
            "mean_abs_logratio": float(raw_logratio.abs().mean()),
            "kl": float(kl.mean()),
        }
    return loss, stats


def run(args: argparse.Namespace) -> dict:
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.output_root) / args.experiment / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    pool = train_pool()
    prompt_rng = random.Random(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    policy = load_policy(args.base_policy, device)
    policy.train()
    reference = copy.deepcopy(policy).eval()
    behavior = copy.deepcopy(policy).eval()
    for module in (reference, behavior):
        for p in module.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
    )
    warmup = max(1, int(round(args.warmup_ratio * args.train_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))

    use_sync = args.humanline and not args.disable_humanline_sync
    use_clip = args.humanline and not args.disable_humanline_clipping
    batch = args.prompts_per_step * args.group_size

    sampling_rounds = 0
    generated = 0
    pool_data: dict[str, torch.Tensor] | None = None
    round_start = 0
    log_path = run_dir / "train_log.jsonl"
    log_file = log_path.open("w")
    round_rewards = []
    for step in range(args.train_steps):
        if step % args.refresh_interval == 0:
            behavior.load_state_dict(policy.state_dict())
            reference.load_state_dict(policy.state_dict())
            steps_this_round = min(args.refresh_interval, args.train_steps - step)
            problems = [pool[prompt_rng.randrange(len(pool))] for _ in range(steps_this_round * args.prompts_per_step)]
            pool_data = sample_round(behavior, problems, args.group_size, args.temperature, generator, device, args.gen_chunk)
            sampling_rounds += 1
            generated += len(problems) * args.group_size
            round_start = step
            round_rewards.append(float(pool_data["rewards"].mean()))
            print(
                json.dumps({"event": "sample_round", "method": args.name, "step": step, "round": sampling_rounds,
                            "sampled_reward": round_rewards[-1], "generated": generated}),
                flush=True,
            )
        assert pool_data is not None
        offset = (step - round_start) * batch
        sl = slice(offset, offset + batch)
        loss, stats = grpo_loss(
            policy, reference, pool_data["prompts"][sl], pool_data["completions"][sl],
            pool_data["advantages"][sl], args, use_clip,
        )
        pending_reference = None
        if use_sync and (step + 1) % args.sync_k == 0:
            # humanline syncing: after the loss, before the optimizer step
            pending_reference = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        if pending_reference is not None:
            reference.load_state_dict(pending_reference)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        record = {"step": step + 1, "loss": loss_value, "grad_norm": float(grad_norm),
                  "batch_reward": float(pool_data["rewards"][sl].mean()), **stats}
        log_file.write(json.dumps(record) + "\n")
        if args.log_every and ((step + 1) % args.log_every == 0):
            print(json.dumps({"event": "train_step", "method": args.name, **record}), flush=True)
    log_file.close()

    policy_sha = save_policy(policy, run_dir / "policy.pt")
    meta = {
        "method": args.name,
        "experiment": args.experiment,
        "seed": args.seed,
        "refresh_interval": args.refresh_interval,
        "sampling_rounds": sampling_rounds,
        "generated_completions": generated,
        "optimizer_updates": args.train_steps,
        "base_policy_sha256": file_sha256(args.base_policy),
        "policy_sha256": policy_sha,
        "sampled_reward_per_round": round_rewards,
        "config": {k: v for k, v in vars(args).items()},
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"event": "done", "method": args.name, "run_dir": str(run_dir)}), flush=True)
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--experiment", choices=["stale64", "offline"], default="stale64")
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--base-policy", default="data/base_policy.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--humanline", action="store_true")
    parser.add_argument("--disable-humanline-sync", action="store_true")
    parser.add_argument("--disable-humanline-clipping", action="store_true")
    parser.add_argument("--sync-k", type=int, default=12)
    parser.add_argument("--log-epsilon-p", type=float, default=-1.5)
    parser.add_argument("--log-epsilon-r", type=float, default=1.5)
    parser.add_argument("--refresh-interval", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=512)
    parser.add_argument("--prompts-per-step", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grpo-epsilon", type=float, default=0.15)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--gen-chunk", type=int, default=8192)
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
