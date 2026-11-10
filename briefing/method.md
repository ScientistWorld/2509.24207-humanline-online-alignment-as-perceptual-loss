# Paper's Method

## Key Contribution

The paper introduces humanline variants of reference-model-based alignment objectives (DPO, KTO, GRPO). Motivated by prospect theory, it argues that online on-policy sampling works because it approximates the human-perceived distribution of model outcomes, and that PPO/GRPO clipping already recovers part of this perceptual bias. Making the bias explicit should let stale or offline data train as well as fresh online data.

## Approach (the design pattern)

1. **Humanline syncing.** Every `k` steps, after the loss is computed but before the optimizer step, the reference model is synced with the policy (so the next update is judged against the previous-step policy).
2. **Humanline clipping.** Token-wise log-ratios `log pi_theta / pi_ref` are clamped to `[log eps_P, log eps_R]` (the paper uses -1.5 and 1.5) upstream of the loss. For GRPO the ratio is therefore clipped twice: once by the humanline clamp and once by GRPO's `[1-eps, 1+eps]` surrogate. The KL term is not humanline-clipped.

## Implementation in This Gym

`method/humanline_grpo.py` implements GRPO (group-relative advantages from the 0/1 verifiable reward, the clipped surrogate on the token-wise policy/reference ratio, a k3 KL penalty to the reference, DAPO-style token-mean loss) on the fixed tiny transformer, with the paper's two humanline changes behind `--humanline` (and `--disable-humanline-sync` / `--disable-humanline-clipping` for the ablations). Sampling follows the paper's 64x-less-frequent protocol: every `refresh_interval` updates a pool of 64 batches is sampled from a frozen snapshot of the policy and consumed one batch per update; the reference is reset to the sampling policy at the start of each round. A single round (`refresh_interval = 512`) is the offline setting, where the reference never moves without humanline syncing.

Settings (paper values where they apply): 32 prompts x group size 8 = 256 completions per update, temperature 0.7, AdamW (beta 0.9/0.999, eps 1e-8, no weight decay), 10% warmup, max grad norm 1.0, GRPO eps 0.15 (offline GRPO baseline: 0.5, paper Table 3), KL beta 0.01, humanline log-clip range [-1.5, 1.5]. Learning rate and `k` were selected per method on the visible slice (the paper tunes LR per variant and `k` per setting): humanline GRPO lr 1e-4 with `k = 48` (stale64) and `k = 192` (offline); GRPO lr 1e-4; online GRPO lr 3e-4. Scaled down from the paper: a 0.6M-parameter model and three-digit addition instead of Qwen2.5-1.5B on MATH, and 512 updates instead of 1600.

`scripts/method.sh`, `scripts/baseline.sh` and `scripts/ablation.sh` train each configuration for seeds 7-14.

## Main Claims and What Reproduces Here

- **Primary (paper):** offline or 64x-stale humanline GRPO matches online GRPO. **Here:** humanline GRPO beats the stale/offline GRPO baseline at the same data volume (+0.016 and +0.011 hidden-test accuracy over seeds 7-14; +0.016 and +0.013 over 24 seeds, t = 3.4 and 2.6), but stays well below online GRPO (0.76): the direction reproduces, closing the gap does not at this scale.
- **Secondary (paper):** syncing gives most of the improvement; clipping alone is no better than the offline variant. **Here (offline):** sync-only ≈ full humanline (0.557 vs 0.558), clip-only well below the baseline (0.473). In stale64 both components alone land between the baseline and the full method.
- **Paper:** syncing too often (`k = 1`) collapses with small models. **Here:** `k = 4` collapses (0.19 stale64, 0.04 offline).
