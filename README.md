# Autoregressive Search Engine with Term-Set Generation (AutoTSG)

This repository contains the implementation of AutoTSG.

## Downloading Data
0. Clone the repository and create the environment
   ```bash
   export CUDA=11.6
   conda create -n autotsg python=3.9.12
   conda activate autotsg
   conda install pytorch==1.12.1 cudatoolkit=$CUDA -c conda-forge -c pytorch
   conda install faiss-gpu==1.7.2 -c conda-forge
   pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0+$CUDA.html
   pip install transformers==4.21.3 hydra-core==1.2.0 notebook ipywidgets psutil
   ```
1. Download the Natual Questions 320k Dataset from [HERE](https://1drv.ms/u/s!Aipk4vd2SBrtgv9bKdcHs4jH0PKJXw?e=uWBLwb);
2. Download the MSMARCO Document 300k Dataset from [HERE](https://1drv.ms/u/s!Aipk4vd2SBrtgv9YngXx1vJEE2VjZQ?e=fzMbDj);
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
   torchrun --nproc_per_node=2 run.py AutoTSG base=NQ320k mode=eval ++nbeam=100 ++eval_batch_size=20
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
   torchrun --nproc_per_node=2 run.py AutoTSG base=MS300k mode=eval ++code_length=34 ++nbeam=100 ++eval_batch_size=20
   ```
   - `nproc_per_node` determines how many GPU to use
   - `code_length` determines the length for the concatenation of terms, which is default to 26
   - `nbeam` determines beam size
3. With 4 A100s, the above command should finish within seconds and yield results very similar to:
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.484|0.491|0.359|0.766|0.907|

## Reproducing from Scratch
There are three procedures to train AutoTSG from scratch. One can conveniently set `base=NQ320k` or `base=MS300k` for different datasets. The default dataset is `NQ320k`.

### Train the Matching-oriented Term Selector
0. Install Java11 that is required by [Anserini](src/anserini/).
   ```bash
   cd /the/path/you/like
   wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
   tar -xvf openjdk-11.0.2_linux-x64_bin.tar.gz
   # just temperarily set; it is recommended that you store the setting in ~/.bashrc
   export JAVA_HOME=/the/path/you/like/jdk-11.0.2
   export PATH=$JAVA_HOME/bin:$PATH
   ```
1. Run BM25 for sanity check.
   ```bash
   python run.py BM25 base=NQ320k ++k1=1.5 ++b=0.75
   ```
   The results should be
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.467|0.475|0.365|0.690|0.876|
2. Run BM25 on training set to produce hard negatives.
   ```bash
   python run.py BM25 base=NQ320k ++k1=1.5 ++b=0.75 ++load_index ++eval_set=train ++hits=200
   ```
   Note that we set `hits=200` for efficiency. Then collect negatives
   ```bash
   python -m scripts.negative base=NQ320k ++neg_type=BM25
   ```
   This command creates a file at `src/data/cache/NQ320k/dataset/train/negatives_BM25.pkl` storing the top 200 negatives for each training query.
3. Train the matching-oriented term selector (UniCOIL) using BM25 negatives.
   ```bash
   torchrun --nproc_per_node=2 run.py UniCOIL base=NQ320k ++batch_size=5 ++fp32
   ```
   The model will be automatically evaluated at the end of each epoch. The model may converge after 1 or 2 epochs. The results should be similar to
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.717|0.721|0.625|0.880|0.955|

### Produce Term-set Identifiers
1. Generate the terms and their weights, then run sanity check.
   ```bash
   torchrun --nproc_per_node=2 run.py DeepImpact base=NQ320k mode=eval ++load_ckpt=UniCOIL/best
   ```
   The results should be similar to
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.527|0.533|0.421|0.744|0.882|
2. Generate document identifiers.
   ```bash
   python run.py DeepImpact base=NQ320k mode=code ++code_type=words_comma_plus_stem ++code_tokenizer=t5 ++code_length=26 ++stem_code ++code_sep='\,'
   ```
   This command will create a memmap file at `src/data/cache/NQ320k/codes/words_comma_plus_stem/t5/26/codes.mmp`. Each line in this file is the terms selected from the corresponding document, descendingly sorted by their weights, and separated with comma. You can interact with them in [autotsg.ipynb](src/notebooks/autotsg.ipynb).

### Train T5 with Likelihood-adapted Seq2Seq Learning
1. Train the Seq2Seq model with the default (descending weight) term order on dataset training queries and psuedo-queries:
   ```bash
   torchrun --nproc_per_node=2 run.py AutoTSG ++train_set=[train,doct5-miss,doc] ++save_ckpt=iter0

   # evaluate with beam_size=100
   torchrun --nproc_per_node=2 run.py AutoTSG mode=eval ++load_ckpt=iter0 ++nbeam=100 ++eval_batch_size=20
   ```
   The results should be similar to
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.743|0.745|0.671|0.865|0.927|
2. Sample plausible document identifiers from the trained model.
   ```bash
   torchrun --nproc_per_node=2 run.py AutoTSG mode=code ++load_ckpt=iter0 ++sort_code ++nbeam=3 ++eval_set=train ++code_src=greedy-sample-3-tau5 ++eval_batch_size=500 ++decode_do_sample ++sample_tau=5 ++decode_do_greedy
   torchrun --nproc_per_node=2 run.py AutoTSG mode=code ++load_ckpt=iter0 ++sort_code ++nbeam=3 ++eval_set=doct5-miss ++code_src=greedy-sample-3-tau5 ++eval_batch_size=500 ++decode_do_sample ++sample_tau=5 ++decode_do_greedy
   torchrun --nproc_per_node=2 run.py AutoTSG mode=code ++load_ckpt=iter0 ++sort_code ++nbeam=3 ++eval_set=doc ++code_src=greedy-sample-3-tau5 ++eval_batch_size=500 ++decode_do_sample ++sample_tau=5 ++decode_do_greedy
   ```
3. Iteratively train the Seq2Seq model.
   ```bash
   torchrun --nproc_per_node=2 run.py AutoTSG ++train_set=[train,doct5-miss,doc] ++return_query_code ++code_src=greedy-sample-tau5 ++load_ckpt=iter0 ++save_ckpt=iter1 ++learning_rate=1e-5 ++scheduler=constant ++eval_delay=0

   # evaluate with beam_size=100
   torchrun --nproc_per_node=2 run.py AutoTSG mode=eval ++load_ckpt=iter1 ++nbeam=100 ++eval_batch_size=20
   ```
   The results should be similar to
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.757|0.760|0.690|0.875|0.932|


## Quick Maps
- For training configurations of AutoTSG, check [train.yaml](src/data/config/mode/train.yaml) and [autotsg.yaml](src/data/config/autotsg.yaml)
- For the implementation of our proposed *constrained greedy search* algorithm, check [index.py](src/utils/index.py) line 1731 (we modify the `.generate()` method in huggingface transformers and implement it with a new class named `BeamDecoder`).
- For interacting with the data and selected terms, run [autotsg.ipynb](src/notebooks/autotsg.ipynb)