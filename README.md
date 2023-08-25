# Autoregressive Search Engine with Term-Set Generation (AutoTSG)

This repository contains the implementation of [AutoTSG](https://arxiv.org/abs/2305.13859).

## Quick Maps
- For training configurations of AutoTSG, check [train.yaml](src/data/config/mode/train.yaml) and [autotsg.yaml](src/data/config/autotsg.yaml)
- For the implementation of our proposed *constrained greedy search* algorithm, check [index.py](src/utils/index.py), from line 1731 (we modify the `.generate()` method in huggingface transformers and implement it with a new class named `BeamDecoder`).
- For interacting with the data and selected terms, download our data and model, then run [autotsg.ipynb](src/notebooks/autotsg.ipynb).

## Downloading Data
0. Clone the repository and create the environment
   ```bash
   export CUDA=11.6
   conda create -n autotsg python=3.9.12
   conda activate autotsg
   conda install pytorch==1.10.1 cudatoolkit=$CUDA -c conda-forge -c pytorch
   conda install faiss-gpu==1.7.2 -c conda-forge
   pip install torch_scatter -f https://data.pyg.org/whl/torch-1.10.0+$CUDA.html
   pip install transformers==4.21.3 hydra-core==1.2.0 notebook ipywidgets psutil
   ```
1. Download the Natual Questions 320k Dataset from [HERE](https://1drv.ms/u/s!AmgRICrhHL4_bh3wgA_e7ejudKQ?e=NvthFS);
2. Download the MSMARCO Document 300k Dataset from [HERE](https://1drv.ms/u/s!AmgRICrhHL4_bIeJu1oMopijuew?e=bXPPdu);
3. Untar the file at anywhere you like, e.g. `/data/AutoTSG`;
4. Go to `src/data/config/base/_default.yaml`, set 
   - `data_root: /data/AutoTSG`. This tells the program where to find the data.
   - `plm_root: /data/AutoTSG/huggingface_PLMs`. By default, the language model from hugginface will be permanently downloaded to this folder and hence can be directly loaded afterwards.
5. ```bash
   // pre-tokenize the dataset
   python -m scripts.preprocess base=NQ320k ++query_set=[train,dev,doct5-miss,doc]
   python -m scripts.preprocess base=MS300k ++query_set=[train,dev,doct5-3,doc]
   ```

## Reproducing from Our Checkpoint
### NQ320k
1. Download the model checkpoint and identifier from [HERE](https://1drv.ms/u/s!AmgRICrhHL4_b9DKN9jQw9kf6ds?e=3iBH0I); Then untar it with 
   ```bash
   tar -xzvf autotsg.nq.tar.gz -C src/data/cache/
   ```
2. ```bash
   torchrun --nproc_per_node=2 run.py AutoTSG base=NQ320k mode=eval ++nbeam=100 ++eval_batch_size=20
   ```
   - `nproc_per_node` determines how many GPU to use
   - `nbeam` determines beam size
3. With 2 A100s, the above command should finish within 45 min and yield results very similar to:
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.757|0.760|0.690|0.875|0.932|

### MS300k
1. Download the model checkpoint and identifier from [HERE](https://1drv.ms/u/s!AmgRICrhHL4_bfXHeSUeUpsi0qI?e=CrCICo); Then untar it with 
   ```bash
   tar -xzvf autotsg.ms.tar.gz -C src/data/cache/
   ```
2. ```bash
   torchrun --nproc_per_node=2 run.py AutoTSG base=MS300k mode=eval ++code_length=34 ++nbeam=100 ++eval_batch_size=20
   ```
   - `nproc_per_node` determines how many GPU to use
   - `code_length` determines the length for the concatenation of terms, which is default to 26
   - `nbeam` determines beam size
3. With 2 A100s, the above command should finish within 1 minute and yield results very similar to:
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.484|0.491|0.359|0.766|0.907|

## Reproducing from Scratch
There are three procedures to train AutoTSG from scratch. One can conveniently set `base=NQ320k` or `base=MS300k` for different datasets in command line. The default dataset is `NQ320k`.

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
1. Run BM25 on training set to produce hard negatives.
   ```bash
   python run.py BM25 base=NQ320k ++k1=1.5 ++b=0.75 ++eval_set=train ++hits=200
   # python run.py BM25 base=MS320k ++k1=1.5 ++b=0.75 ++eval_set=train ++hits=200
   ```
   Note that we set `hits=200` for efficiency. Then collect negatives
   ```bash
   python -m scripts.negative base=NQ320k ++neg_type=BM25
   # python -m scripts.negative base=MS300k ++neg_type=BM25
   ```
   This command creates a file at `src/data/cache/NQ320k/dataset/train/negatives_BM25.pkl` storing the top 200 negatives for each training query.
2. Train the matching-oriented term selector (UniCOIL) using BM25 negatives.
   ```bash
   torchrun --nproc_per_node=2 run.py UniCOIL base=NQ320k ++batch_size=5 ++fp32
   # torchrun --nproc_per_node=2 run.py UniCOIL base=MS300k ++batch_size=5 ++fp32
   ```
   The model will be automatically evaluated at the end of each epoch. The model may converge after 1 or 2 epochs. On NQ320k, the results should be similar to
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.717|0.721|0.625|0.880|0.955|

### Produce Term-set Identifiers
1. Generate the terms and their weights, then run sanity check.
   ```bash
   torchrun --nproc_per_node=2 run.py DeepImpact base=NQ320k mode=eval ++load_ckpt=UniCOIL/best
   # torchrun --nproc_per_node=2 run.py DeepImpact base=MS300k mode=eval ++load_ckpt=UniCOIL/best
   ```
   On NQ320k, the results should be similar to
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.527|0.533|0.421|0.744|0.882|
2. Generate document identifiers.
   ```bash
   python run.py DeepImpact base=NQ320k mode=code ++code_type=words_comma_plus_stem ++code_tokenizer=t5 ++code_length=26 ++stem_code ++code_sep='\,'

   # for MS300k, set longer code length
   python run.py DeepImpact base=MS300k mode=code ++code_type=words_comma_plus_stem ++code_tokenizer=t5 ++code_length=34 ++stem_code ++code_sep='\,'
   ```
   This command will create a memmap file at `src/data/cache/NQ320k/codes/words_comma_plus_stem/t5/26/codes.mmp`. Each line in this file is the terms selected from the corresponding document, descendingly sorted by their weights, and separated with comma. You can interact with them in [autotsg.ipynb](src/notebooks/autotsg.ipynb).

### Train T5 with Likelihood-adapted Seq2Seq Learning
1. Train the Seq2Seq model with the default (descending weight) term order on dataset training queries and psuedo-queries:
   ```bash
   torchrun --nproc_per_node=2 run.py AutoTSG ++train_set=[train,doct5-miss,doc] ++save_ckpt=iter0
   # evaluate with beam_size=100
   torchrun --nproc_per_node=2 run.py AutoTSG mode=eval ++load_ckpt=iter0 ++nbeam=100 ++eval_batch_size=20

   # for MS300k, specify code length and replace doct5-miss with doct5-3
   # torchrun --nproc_per_node=2 run.py AutoTSG base=MS300k ++train_set=[train,doct5-3,doc] ++save_ckpt=iter0 ++code_length=34
   # torchrun --nproc_per_node=2 run.py AutoTSG base=MS300k mode=eval ++load_ckpt=iter0 ++nbeam=100 ++eval_batch_size=15 ++code_length=34
   ```
   On NQ320k, the results should be similar to
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.743|0.745|0.671|0.865|0.927|
2. Sample `3` plausible document identifiers from the trained model (the more samples the better results).
   ```bash
   torchrun --nproc_per_node=2 run.py AutoTSG mode=code ++load_ckpt=iter0 ++sort_code ++nbeam=3 ++eval_set=train ++code_src=iter0 ++eval_batch_size=500 ++decode_do_sample ++sample_tau=5 ++decode_do_greedy
   torchrun --nproc_per_node=2 run.py AutoTSG mode=code ++load_ckpt=iter0 ++sort_code ++nbeam=3 ++eval_set=doct5-miss ++code_src=iter0 ++eval_batch_size=500 ++decode_do_sample ++sample_tau=5 ++decode_do_greedy
   torchrun --nproc_per_node=2 run.py AutoTSG mode=code ++load_ckpt=iter0 ++sort_code ++nbeam=3 ++eval_set=doc ++code_src=iter0 ++eval_batch_size=500 ++decode_do_sample ++sample_tau=5 ++decode_do_greedy

   # for MS300k, specify code length, replace doct5-miss with doct5-3, and disable sampling with temperature.
   # torchrun --nproc_per_node=2 run.py AutoTSG mode=code ++load_ckpt=iter0 ++sort_code ++nbeam=3 ++eval_set=train ++code_src=iter0 ++eval_batch_size=400 ++decode_do_greedy ++code_length=34
   # torchrun --nproc_per_node=2 run.py AutoTSG mode=code ++load_ckpt=iter0 ++sort_code ++nbeam=3 ++eval_set=doct5-3 ++code_src=iter0 ++eval_batch_size=400 ++decode_do_greedy ++code_length=34
   # torchrun --nproc_per_node=2 run.py AutoTSG mode=code ++load_ckpt=iter0 ++sort_code ++nbeam=3 ++eval_set=doc ++code_src=iter0 ++eval_batch_size=400 ++decode_do_greedy ++code_length=34
   ```
3. Iteratively train the Seq2Seq model.
   ```bash
   torchrun --nproc_per_node=2 run.py AutoTSG ++train_set=[train,doct5-miss,doc] ++return_query_code ++code_src=iter0 ++load_ckpt=iter0 ++save_ckpt=iter1 ++learning_rate=1e-5 ++scheduler=constant ++eval_delay=0 ++batach_size=100
   # evaluate with beam_size=100
   torchrun --nproc_per_node=2 run.py AutoTSG mode=eval ++load_ckpt=iter1 ++nbeam=100 ++eval_batch_size=20

   # for MS300k, specify code length and replace doct5-miss with doct5-3
   # torchrun --nproc_per_node=2 run.py AutoTSG base=MS300k ++train_set=[train,doct5-3,doc] ++return_query_code ++code_src=iter0 ++load_ckpt=iter0 ++save_ckpt=iter1 ++learning_rate=1e-5 ++scheduler=constant ++eval_delay=0 ++batach_size=100 ++code_length=34
   # torchrun --nproc_per_node=2 run.py AutoTSG base=MS300k mode=eval ++load_ckpt=iter1 ++nbeam=100 ++eval_batch_size=15 ++code_length=34
   ```
   On NQ320k, the results should be similar to
   |MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
   |:-:|:-:|:-:|:-:|:-:|
   |0.757|0.760|0.690|0.875|0.932|
