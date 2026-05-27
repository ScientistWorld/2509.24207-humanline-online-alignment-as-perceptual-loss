# Overview

- **Paper ID:** 2509.24207
- **Title:** Humanline: Online Alignment as Perceptual Loss
- **Domain:** LLM post-training / alignment
- **TL;DR:** The paper argues that online alignment works partly because it approximates human probability perception, then turns that insight into humanline variants of DPO/KTO/GRPO.

## Short Summary

The paper studies why online on-policy alignment often outperforms offline off-policy alignment. It connects PPO/GRPO-style clipping to prospect-theory probability distortion and proposes a general humanline design pattern: periodically sync the reference model from a lagged policy and clamp token-wise policy/reference log-ratios upstream of the loss. Experiments report that offline humanline variants close the online/offline gap on instruction following, and stale humanline GRPO preserves MATH500 performance while sampling much less often.

## Key Results

- Llama3-8B-Instruct offline+humanline GRPO reaches 25.24 length-controlled AlpacaEval2 win rate versus 15.52 for offline GRPO and 25.05 for online GRPO.
- On mathematical reasoning, both online GRPO and 64x-less-frequent humanline GRPO reach MATH500 Pass@1 of 0.593 +/- 0.019 after 1600 steps.
