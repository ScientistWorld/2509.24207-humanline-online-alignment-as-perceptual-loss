"""Run the original learning pipeline with an auditable scale/schedule change.

All seeds use the default seed-7 evaluation examples, so seed spread measures
training randomness rather than a different mixture of evaluation prompts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from method import humanline_grpo as original


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    argv = [
        "--name", "offline_humanline_grpo", "--cpu", "--humanline",
        "--pool-stale-batches", "--balanced-train", "--policy-type", "structured",
        "--refresh-interval", "64", "--sync-k", "12", "--train-steps", "256",
        "--sft-steps", "160", "--train-size", "800", "--eval-size", "160",
        "--batch-size", "32", "--group-size", "8", "--hidden-size", "128",
        "--temperature", "0.8", "--rl-lr", "5e-4", "--beta-kl", "0.01",
        "--log-epsilon-p", "-1.5", "--log-epsilon-r", "1.5",
        "--min-value", "0", "--max-value", "19", "--max-new-tokens", "24",
        "--log-every", "64", "--seed", "7",
        "--output-dir", "checkpoints/humanline_grpo",
    ]
    argv += spec.get("args", [])
    if spec.get("remove_humanline"):
        argv.remove("--humanline")
    # The only runner change: hold the test prompts fixed across training seeds.
    make_examples = original.make_examples
    original.make_examples = lambda n, seed, low, high: make_examples(n, 10007, low, high)
    sys.argv = [sys.argv[0]] + argv
    result = original.run(original.parse_args())
    print(json.dumps(original.asdict(result), indent=2))


if __name__ == "__main__":
    main()
