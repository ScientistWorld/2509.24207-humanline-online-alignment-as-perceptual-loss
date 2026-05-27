# Progress

## What Works

Implemented a scaled reproduction of the paper's verifiable-reward humanline GRPO setting. The method code includes upstream token log-ratio clipping, standard GRPO surrogate clipping, stale/off-policy batch reuse, and lagged humanline reference syncing.

## Results

First CPU smoke job completed successfully and wrote `scoring/scores.json`. Both stale GRPO and offline humanline GRPO reached `format_rate=1.0` but `pass_at_1=0.0`, so the implementation ran but did not yet support the core claim.

Second CPU job completed successfully after adding a dense numeric reward. Exact answer accuracy improved from `0.025` initial to `0.04375` after GRPO, and mean reward improved from `0.4223` initial to `0.47025`. The proposed humanline run and stale non-humanline baseline were still identical, and saved samples showed both policies collapsed to a constant `<answer>16</answer>` response.

## Remaining

Run the revised answer-token-only warm start with a 64-step stale refresh interval. If the policy still collapses to a constant answer, inspect saved samples and increase conditional signal or task coverage before claiming `core_claim`.

## Issues

The first scaled task used a sparse exact-match reward and produced no numeric correctness. I added a bounded numeric-closeness term to the reward so GRPO receives a useful signal before exact answers are sampled frequently.

The second run showed that the model learned formatting but not enough prompt-conditioned arithmetic. I changed supervised warm start so cross-entropy is applied only to completion tokens, matching the RL/evaluation target and preventing prompt-copying tokens from dominating the loss.

## Deviations from Paper

- Paper used Qwen2.5-1.5B-Instruct and MATH500; this reproduction uses a tiny GRU and generated arithmetic tasks to fit a first smoke run. The GRPO and humanline computational steps are preserved.
- Paper used 64x stale sampling for math; the smoke script uses a 64-step stale refresh interval at reduced model/data scale.
