# Bi-Phase IVFPQ
This tutorial explains how to reproduce our paper [Bi-Phase IVFPQ](https://arxiv.org/abs/2210.05521) on MSMARCO passage collection. 

## Reproducing From Checkpoint
1. Make sure you finished the data processing steps in {doc}`data`. Then you should download the checkpoint and the necessary index files on [Google Drive](). The uncompressed files would look like:
   ```
   Bi-Phase-IVFPQ
   └── MSMARCO-passage
     ├── ckpts
     │   ├── DistillVQ_d-RetroMAE
     │   │   └── best
     │   ├── TokIVF
     │   │   └── best
     │   └── TopIVF
     │       └── best
     └── index
         └── RetroMAE
             └── faiss
                 ├── IVF10000,PQ64x8
                 └── OPQ96,PQ96x8
   ```
   Move the `ckpts/*` to `src/data/cache/MSMARCO-passage/ckpts/`. Move the `index/*` to `src/data/cache/MSMARCO-passage/index/`.

2. Since Bi-IVFPQ is a general IVFPQ framework. It relies on off-the-shelf embeddings to work. Here we use the [distilled RetroMAE](https://arxiv.org/abs/2205.12035) as the embedding model.
   We encode all documents and queries using RetroMAE and save the resulted embeddings:
   ```bash
   # uses four gpus
   torchrun --nproc_per_node=4 run.py RetroMAE ++mode=eval ++plm=retromae_distill ++save_encode
   ```
   The resulted file will be stored at `src/data/cache/MSMARCO-passage/encode/RetroMAE/`. The evaluation defaults to use the `Flat` index that scans the database for each query. The metrics should be similar to:
   |MRR@10|Recall@10|Recall@100|Recall@1000|
   |:-:|:-:|:-:|:-:|
   |0.4155|0.708|0.9268|0.9876|
3. **Prepare PQ module.**
   ```bash
   python run.py DistillVQ_d-RetroMAE ++mode=eval ++save_index
   ```
   This evaluates the performance of exaustive PQ with 96 subvectors, whose metrics should be similar to:
   |MRR@10|Recall@10|Recall@100|Recall@1000|
   |:-:|:-:|:-:|:-:|
   0.3993|0.6846|0.9207|0.9845|

4. **Prepare Topic IVF.**
   ```bash
   python run.py TopIVF ++mode=eval
   ```
   This evaluates the performance of topic-phase IVF followed by PQ verification when selecting `20` topics for each query. The metrics should be
   |MRR@10|Recall@10|Recall@100|Recall@1000|
   |:-:|:-:|:-:|:-:|
   0.355|0.5947|0.7917|0.8423|19900|

5. **Prepare Term IVF.**
   ```bash
   # you should use more gpus than TopIVF because TokIVF involves a BERT and hence heavier
   torchrun --nproc_per_node=4 run.py TokIVF ++mode=eval ++save_encode
   ```
   - If you encounter memory issues when building the inverted index, please run the above command with `++index_thread=5`. If it still won't work, run it with `++index_shard=64 ++index_thread=5`.
  
   This evaluates the performance of term-phase IVF followed by PQ verification when selecting `3` terms for each document. The metrics should be
   |MRR@10|Recall@10|Recall@100|Recall@1000|
   |:-:|:-:|:-:|:-:|
   0.3937|0.67|0.8801|0.9255|

6. **Chain Them Together.**
   ```bash
   python run.py BIVFPQ
   ```
   This evaluates Bi-phase IVFPQ under its default settings: **3 terms** and **1 topic** for each document to index, **all included terms** and **20 topics** for each query to search. The results should be:
   |MRR@10|Recall@10|Recall@100|Recall@1000|
   |:-:|:-:|:-:|:-:|
   0.3984|0.6808|0.9121|0.9713|

   You can easily try different configurations:
   ```bash
   # index 5 terms for each document
   python run.py BIVFPQ ++x_text_gate_k=5
   # search 10 topics for each query
   python run.py BIVFPQ ++y_query_gate_k=10
   ```
   You can inspect `src/data/config/BIVFPQ.yaml` for more details.