# text-clustering

Text clustering algorithms implemented using Huggingface models and frameworks.

1. Apply k-means classification to the representations from the top layer of a pre-trained transformer model after forward feed of full input text fields
2. Take all the input texts and split them up into sentences. Compute classifications as in 1 but at the sentence level
3. Split input texts into sentences and rank the sentences by how much the other sentences imply them
4. Compute "broad support" metric for each sentence.  Define $s(t, p_1, p_2)$ to be the largest $k$ such that there are at least $p_1/n$ responses that imply $t$ with probability $p_2$ where $n$ is the total number of sentences.