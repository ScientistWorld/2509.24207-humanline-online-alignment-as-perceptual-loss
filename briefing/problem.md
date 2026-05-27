# Problem Definition

## Research Question

Can language models be post-trained effectively when feedback data is stale or off-policy, rather than freshly sampled from the current model at every update?

## Why It Matters

Fresh online sampling is expensive because generation, labeling, and training are tightly coupled. A good solution should preserve downstream task quality while reducing the frequency of sampling and labeling.

## Success Criteria

A successful system improves verifiable task performance from model-generated feedback, maintains answer correctness and output format under stale/off-policy training data, and does not simply use more sampled examples to compensate.
