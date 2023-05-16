# Autoregressive Search Engine with Term-Set Generation (AutoTSG)

This repository contains the implementation of AutoTSG.

## Downloading Data
0. Clone the repository;
1. Download the Natual Questions 320k Dataset from [HERE]();
2. Download the MSMARCO Document 300k Dataset from [HERE]();
3. Untar the file at anywhere you like, e.g. `/data/AutoTSG`;
4. Go to `src/data/config/base/_default.yaml`, set 
   - `data_root: /data/AutoTSG`. This tells the program where to find the data.
   - `plm_root: /data/AutoTSG/huggingface_PLMs`. By default, the language model from hugginface will be permanently downloaded to this folder and hence can be directly loaded afterwards.
5. ```bash
   // pre-tokenize the dataset
   python -m scripts.preprocess base=NQ320k
   python -m scripts.preprocess base=MS300k
   ```

## Reproducing from Our Checkpoint
### NQ320k
1. Download the model checkpoint and identifier from [HERE](); Then untar it with 
   ```bash
   tar -xzvf autotsg.nq.tar.gz -C src/data/cache/
   ```
2. ```bash
   torchrun --nproc_per_node=4 run.py AutoTSG base=NQ320k mode=eval ++nbeam=100 ++eval_batch_size=20
   ```
   - `nproc_per_node` determines how many GPU to use
   - `nbeam` determines beam size
3. With 4 A100s, the above command should finish within 25 min and yield results very similar to:
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.757|0.760|0.690|0.875|0.932|

### MS300k
1. Download the model checkpoint and identifier from [HERE](); Then untar it with 
   ```bash
   tar -xzvf autotsg.ms.tar.gz -C src/data/cache/
   ```
2. ```bash
   torchrun --nproc_per_node=4 run.py AutoTSG base=MS300k mode=eval ++nbeam=100 ++eval_batch_size=20
   ```
   - `nproc_per_node` determines how many GPU to use
   - `nbeam` determines beam size
3. With 4 A100s, the above command should finish within seconds and yield results very similar to:
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.484|0.491|0.359|0.766|0.907|

<!-- ## Reproducing from Scratch
### Training Matching-oriented Term Selector
### Producing Terms
### Likelihood-adapted Sequence-to-sequence Learning -->
