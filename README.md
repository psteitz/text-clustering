# text-clustering

Text clustering algorithms implemented using Huggingface models and frameworks.

1. Apply k-means classification to the representations from the top layer of a pre-trained transformer model after forward feeding texts through the model.
2. Compute "broad support" metric for each sentence. For each sentence $s$ in a collection $T$ of texts, estimate the amount of support that it has among all of the texts as follows.  For each text $t \in T$, let $u(s,t)$ be the entailment probability computed by an NLI model with $premise = s$ and $hypothesis = t$.  Then set $S(s, T) = \sum_{u(s,t) > k}u(s,t)$ where $k$ is a threshold (say, $0.5$).  So $S(s,T)$ is the sum of how much each text in $T$ implies $s$, restricting the sum to the texts that imply it at some minimal level.
3. Define a set of semantic dimensions, do nli-based zero-shot classification, then cluster softmax vectors

