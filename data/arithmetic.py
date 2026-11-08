"""Fixed-width three-digit addition with a verifiable exact-match reward.

This module defines the problem only: vocabulary, prompt/answer encoding, the
visible training-prompt pool, and the reward used for training.  It has no
torch dependency and must never import method code.

Prompt tokens:  <bos> a2 a1 a0 + b2 b1 b0 =     (9 tokens, zero-padded)
Answer tokens:  s3 s2 s1 s0                      (4 digits, zero-padded, MSB first)

The scored evaluation lives in eval/ and does not import this module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

DIGITS = "0123456789"
PLUS = 10
EQUALS = 11
BOS = 12
VOCAB = list(DIGITS) + ["+", "=", "<bos>"]
VOCAB_SIZE = len(VOCAB)

OPERAND_DIGITS = 3
ANSWER_DIGITS = 4
MAX_OPERAND = 10**OPERAND_DIGITS - 1
PROMPT_LEN = 1 + OPERAND_DIGITS + 1 + OPERAND_DIGITS + 1  # 9
SEQ_LEN = PROMPT_LEN + ANSWER_DIGITS  # 13

# Visible pools.  The RL training pool is what post-training methods may
# sample prompts from.  The base-SFT pool and base-check set were used once to
# build the fixed starting policy (data/base_policy.pt) and are listed here so
# the build is reproducible.  Held-out evaluation prompts live only in eval/.
TRAIN_POOL_SEED = 20251
TRAIN_POOL_SIZE = 20000
BASE_SFT_POOL_SEED = 20252
BASE_SFT_POOL_SIZE = 50000
BASE_CHECK_SEED = 20253
BASE_CHECK_SIZE = 1000


@dataclass(frozen=True)
class Problem:
    a: int
    b: int

    @property
    def answer(self) -> int:
        return self.a + self.b

    @property
    def prompt_text(self) -> str:
        return f"{self.a:0{OPERAND_DIGITS}d}+{self.b:0{OPERAND_DIGITS}d}="

    @property
    def answer_text(self) -> str:
        return f"{self.answer:0{ANSWER_DIGITS}d}"


def prompt_ids(problem: Problem) -> list[int]:
    a = [int(c) for c in f"{problem.a:0{OPERAND_DIGITS}d}"]
    b = [int(c) for c in f"{problem.b:0{OPERAND_DIGITS}d}"]
    return [BOS] + a + [PLUS] + b + [EQUALS]


def answer_ids(problem: Problem) -> list[int]:
    return [int(c) for c in problem.answer_text]


def decode(ids: list[int]) -> str:
    return "".join(VOCAB[i] if 0 <= i < VOCAB_SIZE else "?" for i in ids)


def reward(completion_ids: list[int], problem: Problem) -> float:
    """Verifiable training reward: 1 for an exact four-digit answer, else 0."""
    return 1.0 if list(completion_ids) == answer_ids(problem) else 0.0


def num_carries(a: int, b: int) -> int:
    carry = 0
    count = 0
    for _ in range(OPERAND_DIGITS):
        s = a % 10 + b % 10 + carry
        carry = 1 if s >= 10 else 0
        count += carry
        a //= 10
        b //= 10
    return count


def sample_problems(n: int, seed: int, exclude: set[tuple[int, int]] | None = None) -> list[Problem]:
    """Distinct uniformly random operand pairs, skipping excluded pairs."""
    rng = random.Random(seed)
    exclude = exclude or set()
    seen: set[tuple[int, int]] = set()
    out: list[Problem] = []
    while len(out) < n:
        pair = (rng.randint(0, MAX_OPERAND), rng.randint(0, MAX_OPERAND))
        if pair in seen or pair in exclude:
            continue
        seen.add(pair)
        out.append(Problem(*pair))
    return out


def train_pool() -> list[Problem]:
    """Prompts that post-training methods may sample (labels are not provided)."""
    return sample_problems(TRAIN_POOL_SIZE, TRAIN_POOL_SEED)


def base_sft_pool() -> list[Problem]:
    """Labelled pairs used only to build the fixed starting policy."""
    exclude = {(p.a, p.b) for p in train_pool()}
    return sample_problems(BASE_SFT_POOL_SIZE, BASE_SFT_POOL_SEED, exclude)


def base_check_set() -> list[Problem]:
    """Small held-out set used only to decide when the base build stops."""
    exclude = {(p.a, p.b) for p in train_pool()} | {(p.a, p.b) for p in base_sft_pool()}
    return sample_problems(BASE_CHECK_SIZE, BASE_CHECK_SEED, exclude)


def visible_pairs() -> set[tuple[int, int]]:
    pools = train_pool() + base_sft_pool() + base_check_set()
    return {(p.a, p.b) for p in pools}
