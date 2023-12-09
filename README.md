# text-clustering

Text clustering algorithms implemented using Huggingface models and frameworks.

1. Apply k-means classification to the representations from the top layer of a pre-trained transformer model after forward feed of full input text fields
2. Compute "broad support" metric for each sentence.  Define $s(t, p_1, p_2)$ to be the largest $k$ such that there are at least $np_1$ responses that imply $t$ with probability $p_2$ where $n$ is the total number of sentences.
3. Define a set of semantic dimensions, do nli-based classification, then cluster softmax vectors


