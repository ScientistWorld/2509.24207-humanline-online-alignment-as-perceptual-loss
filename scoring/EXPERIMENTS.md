# Experiment Overview

Both experiments start from the same fixed starting policy and model, train on the model's own sampled completions scored by a verifiable exact-match reward, and are scored on held-out three-digit addition prompts as the mean over the eight training seeds 7-14. They differ only in how often new completions may be sampled.

## stale64

Establishes how much answer quality a post-training method can reach when completions may only be sampled every 64 optimizer updates (64x less often than online), with the same number of updates and the same completion volume that sampling at every update would use. Sampling at every update reaches about 0.76 accuracy under the same budget, so the experiment measures how much of the loss caused by staleness a method recovers. Scored: held-out accuracy and accuracy on the carry-heavy half; a run outside the sampling schedule fails.

## offline

Establishes the same for the fully offline case: one pool of completions, sampled once from the starting policy before training, is all the feedback a method gets for 512 updates. It measures how well a method learns from data that becomes increasingly off-policy as training proceeds. Scored the same way; a run that samples again fails.

Both rows include an evaluator-owned no-post-training row (the starting policy) and a fresh-sampling comparator that is scored as failed (its accuracy is shown only as a diagnostic).
