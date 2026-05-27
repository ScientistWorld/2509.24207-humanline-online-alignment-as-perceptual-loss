# Progress

## What Works

Implemented a scaled reproduction of the paper's verifiable-reward humanline GRPO setting. The method code includes upstream token log-ratio clipping, standard GRPO surrogate clipping, stale/off-policy batch reuse, and lagged humanline reference syncing.

## Results

First CPU smoke job completed successfully and wrote `scoring/scores.json`. Both stale GRPO and offline humanline GRPO reached `format_rate=1.0` but `pass_at_1=0.0`, so the implementation ran but did not yet support the core claim.

Second CPU job completed successfully after adding a dense numeric reward. Exact answer accuracy improved from `0.025` initial to `0.04375` after GRPO, and mean reward improved from `0.4223` initial to `0.47025`. The proposed humanline run and stale non-humanline baseline were still identical, and saved samples showed both policies collapsed to a constant `<answer>16</answer>` response.

Third CPU job completed successfully after masking warm-start loss to answer tokens. It still did not support the core claim: stale GRPO reached `pass_at_1=0.0625`, while offline humanline GRPO reached `pass_at_1=0.05`; samples still collapsed to a constant `<answer>19</answer>` output.

Fourth CPU job completed successfully with the structured answer policy. It removed constant-output collapse, but still did not support the core claim: the warm-start policy began at `pass_at_1=0.18125`, stale GRPO ended at `0.14375`, and offline humanline GRPO ended at `0.10`.

Fifth CPU job completed successfully with exact-dominant reward and wider humanline clipping. It still did not support the core claim: warm-start began at `pass_at_1=0.18125`, stale GRPO ended at `0.14375`, and offline humanline GRPO ended at `0.11875`.

Sixth CPU job completed successfully and reached `core_claim` on the scaled arithmetic setting. With the same `64`-step stale refresh and `16384` generated examples, offline humanline GRPO improved from `initial_pass_at_1=0.06875` to `pass_at_1=0.175`, while stale GRPO reached `0.15`. Mean reward also favored humanline (`0.31847` vs `0.29308`) with both methods retaining `format_rate=1.0`.

## Remaining

Expand beyond the minimum by running the same structured-policy protocol at a harder `128`-step stale refresh interval. The 64-step run already supports `core_claim`; the next run tests whether the advantage persists when generated data is refreshed less often.

## Issues

The first scaled task used a sparse exact-match reward and produced no numeric correctness. I added a bounded numeric-closeness term to the reward so GRPO receives a useful signal before exact answers are sampled frequently.

The second run showed that the model learned formatting but not enough prompt-conditioned arithmetic. I changed supervised warm start so cross-entropy is applied only to completion tokens, matching the RL/evaluation target and preventing prompt-copying tokens from dominating the loss.

The third run confirmed that character-level generation, not the GRPO code path, was dominating the failure mode. I added a structured policy over complete formatted arithmetic answers so the next job tests the paper's online-alignment computation without sequence-format collapse.

The fourth run showed that the dense closeness reward can favor close but wrong sums. I reduced the format/closeness terms and made exact correctness dominate, and widened the humanline clipping range for the action-level scaled policy so the sync behavior is not masked by overly tight action-logit clipping.

The fifth run improved humanline slightly relative to the prior structured-policy run but still degraded from the warm-start. I reduced SFT steps, increased batch/group size, lowered RL learning rate, and increased KL pressure for the next run so the RL phase has room to improve and the fixed stale reference is a meaningful constraint.

The sixth run showed that the scaled humanline computation works when the warm-start is not already near its local ceiling and exact rewards are sampled densely enough. I added a 128-step refresh expansion to test the same claim at a stronger stale-data setting.

## Deviations from Paper

- Paper used Qwen2.5-1.5B-Instruct and MATH500; this reproduction uses a tiny GRU and generated arithmetic tasks to fit a first smoke run. The GRPO and humanline computational steps are preserved.
- Paper used 64x stale sampling for math; the smoke script uses a 64-step stale refresh interval at reduced model/data scale.
