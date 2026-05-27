# Progress

## What Works

Implemented a scaled reproduction of the paper's verifiable-reward humanline GRPO setting. The method code includes upstream token log-ratio clipping, standard GRPO surrogate clipping, stale/off-policy batch reuse, and lagged humanline reference syncing.

## Results

First CPU smoke job completed successfully and wrote `scoring/scores.json`. Both stale GRPO and offline humanline GRPO reached `format_rate=1.0` but `pass_at_1=0.0`, so the implementation ran but did not yet support the core claim.

Second CPU job completed successfully after adding a dense numeric reward. Exact answer accuracy improved from `0.025` initial to `0.04375` after GRPO, and mean reward improved from `0.4223` initial to `0.47025`. The proposed humanline run and stale non-humanline baseline were still identical, and saved samples showed both policies collapsed to a constant `<answer>16</answer>` response.

Third CPU job completed successfully after masking warm-start loss to answer tokens. It still did not support the core claim: stale GRPO reached `pass_at_1=0.0625`, while offline humanline GRPO reached `pass_at_1=0.05`; samples still collapsed to a constant `<answer>19</answer>` output.

Fourth CPU job completed successfully with the structured answer policy. It removed constant-output collapse, but still did not support the core claim: the warm-start policy began at `pass_at_1=0.18125`, stale GRPO ended at `0.14375`, and offline humanline GRPO ended at `0.10`.

## Remaining

Run the exact-dominant verifier reward with wider humanline log-ratio bounds for the structured answer-policy GRPO implementation. If this still degrades from the warm-start state, the next issue to inspect is whether stale-batch reuse needs an online comparison run or a smaller learning rate.

## Issues

The first scaled task used a sparse exact-match reward and produced no numeric correctness. I added a bounded numeric-closeness term to the reward so GRPO receives a useful signal before exact answers are sampled frequently.

The second run showed that the model learned formatting but not enough prompt-conditioned arithmetic. I changed supervised warm start so cross-entropy is applied only to completion tokens, matching the RL/evaluation target and preventing prompt-copying tokens from dominating the loss.

The third run confirmed that character-level generation, not the GRPO code path, was dominating the failure mode. I added a structured policy over complete formatted arithmetic answers so the next job tests the paper's online-alignment computation without sequence-format collapse.

The fourth run showed that the dense closeness reward can favor close but wrong sums. I reduced the format/closeness terms and made exact correctness dominate, and widened the humanline clipping range for the action-level scaled policy so the sync behavior is not masked by overly tight action-logit clipping.

## Deviations from Paper

- Paper used Qwen2.5-1.5B-Instruct and MATH500; this reproduction uses a tiny GRU and generated arithmetic tasks to fit a first smoke run. The GRPO and humanline computational steps are preserved.
- Paper used 64x stale sampling for math; the smoke script uses a 64-step stale refresh interval at reduced model/data scale.
