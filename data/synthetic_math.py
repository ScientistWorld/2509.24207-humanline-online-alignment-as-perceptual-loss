"""Synthetic arithmetic task with programmatic correctness rewards.

This module defines the problem data only; method code must not be imported
here.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


VOCAB = list("0123456789+=?<>/answer \n")
PAD = 0
EOS = 1
CHARS = ["<pad>", "<eos>"] + sorted(set(VOCAB))
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for ch, i in STOI.items()}
ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: int


def make_examples(n: int, seed: int, min_value: int = 10, max_value: int = 99) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for _ in range(n):
        a = rng.randint(min_value, max_value)
        b = rng.randint(min_value, max_value)
        examples.append(Example(prompt=f"{a}+{b}=?", answer=a + b))
    return examples


def make_balanced_examples(n: int, seed: int, min_value: int = 10, max_value: int = 99) -> list[Example]:
    """Cycle over the full arithmetic grid before repeating examples.

    The reduced arithmetic benchmark has a finite prompt space.  Balanced
    Balanced examples cover the finite prompt grid before repeating, while
    leaving evaluation on the same problem distribution independent of the
    method implementation.
    """
    rng = random.Random(seed)
    grid = [
        Example(prompt=f"{a}+{b}=?", answer=a + b)
        for a in range(min_value, max_value + 1)
        for b in range(min_value, max_value + 1)
    ]
    examples: list[Example] = []
    while len(examples) < n:
        block = grid[:]
        rng.shuffle(block)
        examples.extend(block)
    return examples[:n]


def target_text(answer: int) -> str:
    return f" <answer>{answer}</answer>"


def reward_completion(text: str, answer: int) -> float:
    """Formatting and exact-correctness reward.

    The task gives a small format bonus but no numeric-nearness reward,
    avoiding constant central answers that score well without solving the
    prompt.
    """
    match = ANSWER_RE.search(text)
    if not match:
        return 0.0
    pred = int(match.group(1))
    fmt = 0.05
    correct = 1.0 if pred == answer else 0.0
    return fmt + correct


def extract_answer(text: str) -> int | None:
    match = ANSWER_RE.search(text)
    return int(match.group(1)) if match else None


def encode(text: str, max_len: int | None = None, add_eos: bool = True) -> list[int]:
    ids = [STOI[ch] for ch in text if ch in STOI]
    if add_eos:
        ids.append(EOS)
    if max_len is not None:
        ids = ids[:max_len]
    return ids


def decode(ids: list[int]) -> str:
    out = []
    for idx in ids:
        if idx == EOS:
            break
        if idx > 1:
            out.append(ITOS[idx])
    return "".join(out)
