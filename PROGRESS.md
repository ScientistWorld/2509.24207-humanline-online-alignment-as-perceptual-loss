# Progress

## What Works

Implemented a scaled reproduction of the paper's verifiable-reward humanline GRPO setting. The method code includes upstream token log-ratio clipping, standard GRPO surrogate clipping, stale/off-policy batch reuse, and lagged humanline reference syncing.

## Results

First CPU smoke job completed successfully and wrote `scoring/scores.json`. Both stale GRPO and offline humanline GRPO reached `format_rate=1.0` but `pass_at_1=0.0`, so the implementation ran but did not yet support the core claim.

## Remaining

Run the revised dense-reward arithmetic setup with a 64-step stale refresh interval. If the humanline run still fails to improve exact answer accuracy, inspect saved samples and tune the reward/training scale again.

## Issues

The first scaled task used a sparse exact-match reward and produced no numeric correctness. I added a bounded numeric-closeness term to the reward so GRPO receives a useful signal before exact answers are sampled frequently.

## Deviations from Paper

- Paper used Qwen2.5-1.5B-Instruct and MATH500; this reproduction uses a tiny GRU and generated arithmetic tasks to fit a first smoke run. The GRPO and humanline computational steps are preserved.
- Paper used 64x stale sampling for math; the revised smoke script now defaults to a 64-step stale refresh interval at reduced model/data scale.
