# Adon

Adon is an all-in-one python framework for **Ad**-h**o**c I**n**formation Retrieval. It is highlighted for the following properties:

1. **All-in-one.** Adon efficiently implements the entire ad-hoc retrieval pipeline: including the data loading, model training, encoding, indexing, and evaluation. Based on the this, Adon implements various neural retrieval/reranking models (Sparse, Dense, the newly proposed Generative, and the cross-encoder). One can easily train/evaluate a model with one line of command.
2. **Elastic.** The components of Adon are carefully designed to be highly extendable and independent with one another. You can focus on developing a specific part (e.g. encoding model) without inspecting details about the others (e.g. implementation of ANN indexes). You can also develop your own model within the Adon's framework by only adding a configuration file and a model's implementation.

Please refer to [the docs](http://Adon.readthedocs.io/) for the detailed introduction and user guides.

