# Quick Start

In this tutorial, you will first learn to **reproduce** the result of a sparse retriever [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and a dense retriever [RetroMAE](https://arxiv.org/abs/2205.12035) on MSMARCO passage collection.

Then you will learn how to **train** a basic dense retriever [DPR](https://arxiv.org/abs/2004.04906) and sparse retriever [UniCOIL](https://arxiv.org/abs/2106.14807) on MSMARCO passage collection.


## Prepare Data
The very first thing you should do is to download the MSMARCO passage data. You can directly download the files from [OneDrive](https://1drv.ms/u/s!Aipk4vd2SBrtv3RttSFWXGsAb6bL?e=kPeVe5). More details are {doc}`here <data>`. 

The file is actually a `.tar.gz` file, and you should untar it wherever you like.  Remember to tell the program where to find your data:
- If you save all the uncompressed files in `/home/user/Data`, you should set `data_root: /home/user/Data` in `src/data/config/base/_default.yaml`. 
- Also, change `plm_root` to a valid location on your system, where the language models downloaded from huggingface will be stored.

*Adon* aggregates all configurations for scripts and models in `data/config` using [hydra](https://github.com/facebookresearch/hydra). So in the following, if you want to modify some settings, go to `data/config` and find the corresponding file. Find more details {doc}`here <configs>`.

*Adon* also needs to create necessary files based on the downloaded data, it also defaults to use the efficient `numpy.memmap` to save the tokenzied corpus, which can reduce memory usage and speed up data loading. Run:
```bash
# all the following commands are executed under the src folder
cd src
python -m scripts.preprocess
```
This should results in creating several files in `src/data/cache/MSMARCO-passage/dataset/`. You can interact with the data in `src/notebooks/data.ipynb`.

So far, we have finished all the preperation steps. Lets dive in.

## Reproducing BM25
Adon integrates the efficient Lucene searcher from [Anserini](https://github.com/castorini/anserini), which requires JDK11 to work. You should first install jdk11 by
```bash
cd /the/path/you/like
wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
tar -xvf openjdk-11.0.2_linux-x64_bin.tar.gz

# just temperarily set; it is recommended that you store the setting in ~/.bashrc
export JAVA_HOME=/the/path/you/like/jdk-11.0.2
export PATH=$JAVA_HOME/bin:$PATH
```

You can now run the following to reproduce BM25 with the default `k1=0.82` and `b=0.68`:
```bash
python run.py BM25
```
- To modify `k1`, `b`, just specify `++k1=1 ++b=0.5` when running the command. The default value of `k1` and `b` stores at `data/config/index/bm25.yaml`.

The indexing and evaluating should finish within 10 minutes. The metrics will be printed at console, and also logged at `performance.log`, which should be:
|MRR@10|Recall@10|Recall@100|Recall@1000|
|:-:|:-:|:-:|:-:|
|0.1874|0.3916|0.6701|0.8573|


## Reproducing RetroMAE
RetroMAE is a powerful pre-trained language model specifically designed for dense retrieval. For pre-trained models like BERT and RetroMAE from [huggingface](https://huggingface.co/docs/transformers/v4.21.3/en/index), Adon will permanently download them in a given folder and directly load from the folder afterwards. You should specify the place you want to store by specifying `plm_root` in `data/config/base/_default.yaml`.

Then, just run
```bash
torchrun --nproc_per_node=2 run.py RetroMAE
```
- `python` is replaced with `torchrun` because we start the process in distributed mode
- `--nproc_per_node` denotes the number of GPUs to be used

The result should be (or similar to):
|MRR@10|Recall@10|Recall@100|Recall@1000|
|:-:|:-:|:-:|:-:|
|0.4155|0.708|0.9268|0.9876|


## Train a DPR
The central task of a retriever is to discriminate relevant document from the irrlevant ones in response to a query. To train a dense retriever, one should expose the model to some [hard negatives](https://arxiv.org/abs/2104.08051), where the basic hard negative is the documents that were ranked high in BM25.

To collect BM25 hard negatives, first generate the BM25 top ranked result on the `train` set by
```bash
python run.py BM25 ++load_index ++eval_set=train ++hits=200
```
There are several arguments involved:
- `load_index`: load the index that was just built when reproducing BM25 results
- `eval_set`: evaluate the model on the training set instead of the default dev set
- `hits`: the number of hits per query, set it to 200 so that the retrieval process is faster; moreover, extracting negatives from top 200 is enough for effective training.

Then, collect the non-ground-truth documents from the top ranked result by
```bash
python -m scripts.negative ++hard_neg_type=BM25
```
This command automatically loads the retrieval result generated above (at `data/cache/MSMARCO-passage/retrieve/BM25/train/retrieval_result.pkl`), filters out the ground-truth passages and generates the dictionary mapping a query to its BM25 hard negatives, stored at `data/cache/MSMARCO-passage/dataset/query/train/negatives_BM25.pkl`.

Finally, launch the training for DPR model:
```bash
python run.py DPR

# use multiple gpus
torchrun --nproc_per_node=4 run.py DPR
```
Check `data/config/dpr.yaml` to see the default arguments for training DPR. Adon will evaluate the model's performance on the dev query set every epoch, and save the best checkpoint at `data/cache/MSMARCO-passage/ckpts/DPR/best`.

## Train a UniCOIL

UniCOIL is a sparse model relying on the contextualized weights of overlapping tokens between the query and the passage to perform ranking. Since we have collected the negatives, just use them:

```
python run.py UniCOIL

# use multiple gpus
torchrun --nproc_per_node=4 run.py UniCOIL
```
Again, check `data/config/unicoil.yaml` to see the default arguments of UniCOIL.

