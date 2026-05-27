from __future__ import annotations

import re

import torch
import torch.nn.functional as F
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
            dropout=0.0,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.embed(input_ids))
        return self.lm_head(hidden)


class StructuredAnswerTokenPolicy(nn.Module):
    """Prompt-conditioned token policy for formatted arithmetic answers.

    The reduced benchmark's valid completions all have the form
    ``" <answer>N</answer>"``.  This policy keeps generation token-level while
    avoiding a separate failure mode where a very small character GRU learns
    only the marginal answer prior.  It models a categorical distribution over
    numeric answer strings and exposes autoregressive character log-probs for
    the answer digits and the answer-ending token.
    """

    answer_re = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
    prompt_re = re.compile(r"(-?\d+)\+(-?\d+)=\?")

    def __init__(self, max_addend: int, hidden_size: int = 128):
        super().__init__()
        self.max_addend = max_addend
        self.max_answer = 2 * max_addend
        self.a_embed = nn.Embedding(max_addend + 1, hidden_size)
        self.b_embed = nn.Embedding(max_addend + 1, hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.max_answer + 1),
        )
        self.answer_strings = [str(i) for i in range(self.max_answer + 1)]

    def _parse_prompt(self, prompt: str) -> tuple[int, int]:
        match = self.prompt_re.search(prompt)
        if not match:
            return 0, 0
        a = max(0, min(self.max_addend, int(match.group(1))))
        b = max(0, min(self.max_addend, int(match.group(2))))
        return a, b

    def _prompt_tensors(self, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pairs = [self._parse_prompt(prompt) for prompt in prompts]
        a = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
        b = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
        scale = max(float(self.max_addend), 1.0)
        numeric = torch.stack([a.float() / scale, b.float() / scale], dim=-1)
        return a, b, numeric

    def answer_logits(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        a, b, numeric = self._prompt_tensors(prompts, device)
        features = torch.cat([self.a_embed(a), self.b_embed(b), numeric], dim=-1)
        return self.net(features)

    def supervised_loss(self, prompts: list[str], answers: list[int], device: torch.device) -> torch.Tensor:
        logits = self.answer_logits(prompts, device)
        labels = torch.tensor(
            [max(0, min(self.max_answer, int(answer))) for answer in answers],
            dtype=torch.long,
            device=device,
        )
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def generate_completion(self, prompt: str, device: torch.device, temperature: float) -> str:
        logits = self.answer_logits([prompt], device).squeeze(0)
        if temperature <= 0.05:
            answer = int(torch.argmax(logits).item())
        else:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            answer = int(torch.multinomial(probs, 1).item())
        return f" <answer>{answer}</answer>"

    def completion_logps(
        self,
        prompts: list[str],
        completions: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(self.answer_logits(prompts, device), dim=-1)
        max_len = max(max(1, len(completion)) for completion in completions)
        out = torch.zeros((len(completions), max_len), dtype=torch.float32, device=device)
        mask = torch.zeros((len(completions), max_len), dtype=torch.bool, device=device)

        for i, completion in enumerate(completions):
            match = self.answer_re.search(completion)
            if not match:
                out[i, 0] = -20.0
                mask[i, 0] = True
                continue

            answer_text = match.group(1)
            answer_start = match.start(1)
            answer_end = match.end(1)
            row = log_probs[i]
            row_terms: list[torch.Tensor] = []
            row_positions: list[int] = []

            for j in range(len(answer_text)):
                prefix = answer_text[:j]
                target_prefix = answer_text[: j + 1]
                denom_indices = [k for k, s in enumerate(self.answer_strings) if s.startswith(prefix)]
                num_indices = [k for k, s in enumerate(self.answer_strings) if s.startswith(target_prefix)]
                if not num_indices:
                    row_terms.append(row.new_tensor(-20.0))
                else:
                    denom = torch.logsumexp(row[denom_indices], dim=0) if denom_indices else row.new_tensor(0.0)
                    num = torch.logsumexp(row[num_indices], dim=0)
                    row_terms.append(num - denom)
                row_positions.append(answer_start + j)

            exact_indices = [k for k, s in enumerate(self.answer_strings) if s == answer_text]
            prefix_indices = [k for k, s in enumerate(self.answer_strings) if s.startswith(answer_text)]
            if exact_indices and prefix_indices and answer_end < max_len:
                denom = torch.logsumexp(row[prefix_indices], dim=0)
                num = torch.logsumexp(row[exact_indices], dim=0)
                row_terms.append(num - denom)
                row_positions.append(answer_end)

            for pos, term in zip(row_positions, row_terms):
                if pos < max_len:
                    out[i, pos] = term
                    mask[i, pos] = True

            if not row_terms:
                out[i, 0] = -20.0
                mask[i, 0] = True

        return out, mask
