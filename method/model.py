from __future__ import annotations

import torch
from torch import nn


class TinyGRUPolicy(nn.Module):
    """Small causal language model for the scaled arithmetic RLVR task."""

    def __init__(self, vocab_size: int, hidden_size: int = 128, layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.embed(input_ids))
        return self.lm_head(hidden)
