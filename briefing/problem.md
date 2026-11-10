# Problem Definition

## Research Question

Can a generative model be post-trained effectively from verifiable feedback when the training completions are stale or off-policy, rather than freshly sampled from the current model at every update?

## Why It Matters

Fresh online sampling is expensive: generation, labeling and training are tightly coupled, so every optimizer step waits for new samples from the current model. If training can reuse completions sampled much less often (or sampled once, offline) without losing quality, sampling, labeling and training can be decoupled and post-training becomes cheaper and more parallel.

## The Task in This Gym

- **Model.** A fixed decoder-only character transformer (about 0.6M parameters, `data/policy.py`). Every submission is a `state_dict` of exactly this architecture.
- **Starting point.** Every run starts from the committed policy `data/base_policy.pt`, which was supervised-trained once (on a pool disjoint from all evaluation prompts) to about 36% accuracy and then frozen.
- **Problem.** Three-digit addition, `a2a1a0+b2b1b0=` → four answer digits. The training feedback is a verifiable exact-match reward on the model's own sampled completions (`data/arithmetic.py`). Methods may sample prompts from the visible training pool of 20,000 operand pairs.
- **Operating points.** Two fixed training regimes, per training run:
  - `stale64`: completions may be sampled at most every 64 optimizer updates (64x less often than online): 512 updates, at most 8 sampling rounds, at most 131,072 generated completions.
  - `offline`: a single sampling round, drawn from the starting policy before training, then no new samples: 512 updates, at most 131,072 generated completions.
  Sampling at every update with the same 131,072-completion volume ("online") reaches about 76% held-out accuracy in 512 updates; it is outside both operating points and serves only as a reference for how much is lost to staleness.
- **Seeds.** Training from feedback is noisy from seed to seed, so each method is trained once for each of the eight training seeds 7-14 in each experiment and scored as the mean.

## Success Criteria

A successful method reaches higher exact-match accuracy on held-out addition prompts (and on the harder, carry-heavy half) at each fixed operating point, from the same starting policy and model, using the model's own sampled completions and the verifiable reward as its training signal, without sampling more often or generating more completions than the operating point allows.
