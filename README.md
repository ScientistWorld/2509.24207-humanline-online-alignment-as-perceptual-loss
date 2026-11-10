# Humanline: Online Alignment as Perceptual Loss

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening). The runs listed below were made on that earlier version.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.24207"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.24207-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2509.24207](https://arxiv.org/abs/2509.24207) |
|:---|:---|
| **Domain** | LLM post-training / alignment |
| **TL;DR** | The paper argues that online alignment works partly because it approximates human probability perception, then turns that insight into humanline variants of DPO/KTO/GRPO. |

## Summary

The paper studies why online on-policy alignment often outperforms offline off-policy alignment. It connects PPO/GRPO-style clipping to prospect-theory probability distortion and proposes a general humanline design pattern: periodically sync the reference model from a lagged policy and clamp token-wise policy/reference log-ratios upstream of the loss. Experiments report that offline humanline variants close the online/offline gap on instruction following, and stale humanline GRPO preserves MATH500 performance while sampling much less often.

## Problem

Can language models be post-trained effectively when feedback data is stale or off-policy, rather than freshly sampled from the current model at every update?

## The paper's method

The paper introduces humanline variants of reference-model-based alignment objectives. The goal is to make offline or stale feedback data behave more like the human-perceived distribution of model outcomes.

## Key results of the paper

- Llama3-8B-Instruct offline+humanline GRPO reaches 25.24 length-controlled AlpacaEval2 win rate versus 15.52 for offline GRPO and 25.05 for online GRPO.
- On mathematical reasoning, both online GRPO and 64x-less-frequent humanline GRPO reach MATH500 Pass@1 of 0.593 +/- 0.019 after 1600 steps.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | +49.3 |
| `gpt-5.6-terra-xhigh` | -112.0 |
| `gpt-5.5-xhigh` | -112.5 |
| `gpt-5.6-luna-xhigh` | -173.0 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2509.24207 --mode innovate --agent claude-code
sw evaluate 2509.24207
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

