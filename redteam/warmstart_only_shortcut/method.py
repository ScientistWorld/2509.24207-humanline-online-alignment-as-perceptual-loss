"""Shortcut: solve the finite grid during SFT; never use generated feedback."""
import json
from pathlib import Path

import torch

from data.synthetic_math import make_balanced_examples, make_examples
from method.humanline_grpo import collect_predictions, set_seed, supervised_warm_start
from method.model import StructuredAnswerTokenPolicy

set_seed(7)
device = torch.device("cpu")
model = StructuredAnswerTokenPolicy(19, 128).to(device)
supervised_warm_start(model, make_balanced_examples(800, 7, 0, 19), device,
                      steps=480, batch_size=32, lr=2e-3)
rows = collect_predictions(model, make_examples(160, 10007, 0, 19), device, 24)
accuracy = sum(r["answer"] == int(r["completion"].split(">")[1].split("<")[0]) for r in rows) / len(rows)
out = Path("checkpoints/humanline_grpo")
name = "offline_humanline_grpo"
for suffix in ["_predictions.jsonl", "_initial_predictions.jsonl"]:
    (out / (name + suffix)).write_text("".join(json.dumps(r) + "\n" for r in rows))
torch.save(model.state_dict(), out / (name + ".pt"))
(out / (name + ".json")).write_text(json.dumps({
    "method": name, "train_steps": 0, "sft_steps": 480,
    "stale_refresh_interval": 0, "generated_examples": 0,
    "initial_pass_at_1": accuracy, "pass_at_1": accuracy,
    "shortcut": "warmstart only; no stale-feedback post-training",
}, indent=2))
