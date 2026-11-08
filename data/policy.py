"""Fixed policy architecture for the stale-feedback post-training task.

Every submitted policy must be a ``state_dict`` of ``TinyTransformerPolicy``
with the default constructor arguments, loadable with ``strict=True``.  The
evaluators in eval/ keep their own copy of this architecture and run greedy
decoding themselves, so changing this file does not change how a checkpoint
is scored.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from data.arithmetic import ANSWER_DIGITS, PROMPT_LEN, VOCAB_SIZE

BASE_POLICY_PATH = Path(__file__).resolve().parent / "base_policy.pt"


class Block(nn.Module):
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
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyTransformerPolicy(nn.Module):
    """Decoder-only character-level transformer (about 0.6M parameters)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 16,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 512,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)), persistent=False
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        seq = ids.shape[1]
        positions = torch.arange(seq, device=ids.device)
        x = self.tok(ids) + self.pos(positions)[None]
        for block in self.blocks:
            x = block(x, self.causal_mask)
        return self.head(self.ln_f(x))


def answer_logits(model: nn.Module, prompts: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    """Teacher-forced logits for each answer token: (batch, ANSWER_DIGITS, vocab)."""
    full = torch.cat([prompts, answers], dim=1)
    logits = model(full[:, :-1])
    return logits[:, PROMPT_LEN - 1 :, :]


def answer_token_logps(
    model: nn.Module, prompts: torch.Tensor, answers: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Per-token log-probabilities of ``answers`` under ``model``: (batch, ANSWER_DIGITS)."""
    logits = answer_logits(model, prompts, answers) / temperature
    return torch.log_softmax(logits, dim=-1).gather(-1, answers.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompts: torch.Tensor,
    temperature: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressively generate ANSWER_DIGITS tokens; greedy when temperature <= 0."""
    ids = prompts
    for _ in range(ANSWER_DIGITS):
        logits = model(ids)[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[:, PROMPT_LEN:]


def load_policy(path: str | Path, device: torch.device | str = "cpu") -> TinyTransformerPolicy:
    model = TinyTransformerPolicy()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def save_policy(model: nn.Module, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()}
    torch.save(state, path)
    return file_sha256(path)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
