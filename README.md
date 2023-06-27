# Hybrid Inverted Index Is a Robust Accelerator for Dense Retrieval

This repository contains the implementation of HI2.

## Downloading Data
0. Clone the repository and create the environment
   ```bash
   export CUDA=11.6
   conda create -n hi2 python=3.9.12
   conda activate hi2
   conda install pytorch==1.10.1 cudatoolkit=$CUDA -c conda-forge -c pytorch
   conda install faiss-gpu==1.7.2 -c conda-forge
   pip install torch_scatter -f https://data.pyg.org/whl/torch-1.10.0+$CUDA.html
   pip install transformers==4.21.3 hydra-core==1.2.0 notebook ipywidgets psutil
   ```
1. Download the MSMARCO Passage Dataset from [HERE](https://1drv.ms/u/s!Aipk4vd2SBrtg5oxt3WgMe5NhFeR9g?e=HxR0BE);
2. Download the Natual Questions Dataset from [HERE](https://1drv.ms/u/s!Aipk4vd2SBrtg5oyNGedFAptLP-9Gw?e=zgWa2L);
3. Untar the file at anywhere you like, e.g. `/data/HI2`;
4. Go to `src/data/config/base/_default.yaml`, set 
   - `data_root: /data/HI2`. This tells the program where to find the data.
   - `plm_root: /data/HI2/huggingface_PLMs`. By default, the language model from hugginface will be permanently downloaded to this folder and hence can be directly loaded afterwards.
5. ```bash
   // pre-tokenize the dataset
   python -m scripts.preprocess base=MSMARCO-passage
   python -m scripts.preprocess base=NQ-open
   ```
6. Install JAVA that's required by Anserini
   ```bash
   cd /the/path/you/like
   wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
   tar -xvf openjdk-11.0.2_linux-x64_bin.tar.gz

   # just temperarily set; it is recommended that you store the setting in ~/.bashrc
   export JAVA_HOME=/the/path/you/like/jdk-11.0.2
   export PATH=$JAVA_HOME/bin:$PATH
   ```

## Reproducing from Our Checkpoint
### MSMARCO Passage
#### HI$^2_{\text{sup}}$
1. Download the model checkpoint from [HERE](https://1drv.ms/u/s!Aipk4vd2SBrtg5ozi2bjeUn6IlGYyw?e=0NY5IH); Then untar it with 
   ```bash
   tar -xzvf hi2.msmarco.tar.gz -C src/data/cache/
   ```
2. Prepare [RetroMAE](https://arxiv.org/abs/2205.12035) embeddings:
   ```bash
   torchrun --nproc_per_node=4 run.py RetroMAE mode=eval ++save_encode ++plm=retromae_distill
   ```
   - `nproc_per_node` determines how many GPU to use
   - `save_encode` creates cached embeddings in `src/data/cache/MSMARCO-passage/encode/RetroMAE`
   - The results shold be similar to
      |MRR@10|Recall@100|Recall@1000|#Documents to Evaluate|
      |:-:|:-:|:-:|:-:|
      |0.416|0.927|0.988|8841823|
3. Prepare terms:
   ```bash
   torchrun --nproc_per_node=4 run.py UniCOIL_d-RetroMAE mode=encode ++save_encode
   ```
4. Prepare clusters:
   ```bash
   torchrun --nproc_per_node=4 run.py TopIVF_d-RetroMAE mode=encode ++save_encode
   ```
5. Run HI$^2$:
   ```bash
   torchrun --nproc_per_node=4 run.py HI2 ++y_load_encode
   ```
   - `nproc_per_node` determines how many processes to parallel (the more the faster). There is no need for GPU.
   - The searching process should finish within 2 minutes and yield results very similar to:
      |MRR@10|Recall@100|Recall@1000|#Documents to Evaluate|
      |:-:|:-:|:-:|:-:|
      |0.401|0.916|0.976|56652|

#### HI$^2_{\text{unsup}}$
1. (*Skip if you already have.*) Download the model checkpoint from [HERE](https://1drv.ms/u/s!Aipk4vd2SBrtg5ozi2bjeUn6IlGYyw?e=0NY5IH); Then untar it with 
   ```bash
   tar -xzvf hi2.msmarco.tar.gz -C src/data/cache/
   ```
2. (*Skip if you already have.*) Prepare [RetroMAE](https://arxiv.org/abs/2205.12035) embeddings:
   ```bash
   torchrun --nproc_per_node=4 run.py RetroMAE mode=eval ++save_encode ++plm=retromae_distill
   ```
   - `nproc_per_node` determines how many GPU to use
   - `save_encode` creates cached embeddings in `src/data/cache/MSMARCO-passage/encode/RetroMAE`
   - The results shold be similar to
      |MRR@10|Recall@100|Recall@1000|#Documents to Evaluate|
      |:-:|:-:|:-:|:-:|
      |0.416|0.927|0.988|8841823|
3. Prepare BM25 token-level index:
   ```bash
   torchrun --nproc_per_node=32 run.py BM25 ++pretokenize ++granularity=token
   ```
   - `nproc_per_node` determines how many processes to go parallel (there are no GPU requirements)
   - `pretokenize` tells anserini to skip tokenization
   - `granularity=token` tells the model to save token-level document vector
4. Prepare terms:
   ```bash
   torchrun --nproc_per_node=32 run.py BM25 mode=encode ++pretokenize ++granularity=token ++save_weight ++save_encode
   ```
5. Prepare clusters:
   ```bash
   python run.py IVF mode=encode ++save_encode
   ```
6. Run HI$^2$:
   ```bash
   torchrun --nproc_per_node=4 run.py HI2 ++x_model=BM25 ++x_text_gate_k=15 ++y_model=IVF ++y_query_gate_k=25 ++verifier_src=RetroMAE ++y_load_encode ++x_load_ckpt=inv
   ```
   - `nproc_per_node` determines how many processes to parallel (the more the faster). There is no need for GPU.
   - `x_text_gate_k` sets how many terms to index for each document
   - `y_query_gate_k` sets how many clusters to probe for each query
   - The searching process should finish within 2 minutes and yield results very similar to:
      |MRR@10|Recall@100|Recall@1000|#Documents to Evaluate|
      |:-:|:-:|:-:|:-:|
      |0.380|0.899|0.963|79918|


### Natural Questions
#### HI$^2_{\text{sup}}$
1. Download the model checkpoint from [HERE](https://1drv.ms/u/s!Aipk4vd2SBrtg5o0Gl2Mb71E6Ykn-Q?e=V0Fwjp); Then untar it with 
   ```bash
   tar -xzvf hi2.nq.tar.gz -C src/data/cache/
   ```
2. Prepare [AR2](https://arxiv.org/abs/2110.03611) embeddings:
   ```bash
   torchrun --nproc_per_node=4 run.py AR2 base=NQ-open mode=eval ++save_encode ++plm=ernie
   ```
   - `nproc_per_node` determines how many GPU to use
   - `save_encode` creates cached embeddings in `src/data/cache/NQ-open/encode/AR2`
   - The results shold be similar to
      |Recall@5|Recall@20|Recall@100|#Documents to Evaluate|
      |:-:|:-:|:-:|:-:|
      |0.779|0.861|0.908|21015324|
3. Prepare terms:
   ```bash
   torchrun --nproc_per_node=4 run.py UniCOIL_d-AR2 base=NQ-open mode=encode ++save_encode
   ```
4. Prepare clusters:
   ```bash
   torchrun --nproc_per_node=4 run.py TopIVF_d-RetroMAE base=NQ-open mode=encode ++save_encode ++embedding_src=AR2 ++vq_src=AR2
   ```
5. Run HI$^2$:
   ```bash
   torchrun --nproc_per_node=4 run.py HI2-NQ ++y_load_encode
   ```
   - `nproc_per_node` determines how many processes to parallel (the more the faster). There is no need for GPU.
   - The searching process should finish within 1 minute and yield results very similar to:
      |Recall@5|Recall@20|Recall@100|#Documents to Evaluate|
      |:-:|:-:|:-:|:-:|
      |0.779|0.861|0.907|136691|

#### HI$^2_{\text{unsup}}$
1. (*Skip if you already have.*) Download the model checkpoint from [HERE](https://1drv.ms/u/s!Aipk4vd2SBrtg5o0Gl2Mb71E6Ykn-Q?e=V0Fwjp); Then untar it with 
   ```bash
   tar -xzvf hi2.nq.tar.gz -C src/data/cache/
   ```
2. Prepare [AR2](https://arxiv.org/abs/2110.03611) embeddings:
   ```bash
   torchrun --nproc_per_node=4 run.py AR2 base=NQ-open mode=eval ++save_encode ++plm=ernie
   ```
   - `nproc_per_node` determines how many GPU to use
   - `save_encode` creates cached embeddings in `src/data/cache/NQ-open/encode/AR2`
   - The results shold be similar to
      |Recall@5|Recall@20|Recall@100|#Documents to Evaluate|
      |:-:|:-:|:-:|:-:|
      |0.779|0.861|0.908|21015324|
3. Prepare BM25 token-level index:
   ```bash
   torchrun --nproc_per_node=32 run.py BM25 base=NQ-open ++pretokenize ++granularity=token
   ```
   - `nproc_per_node` determines how many processes to go parallel (there are no GPU requirements)
   - `pretokenize` tells anserini to skip tokenization
   - `granularity=token` tells the model to save token-level document vector
4. Prepare terms:
   ```bash
   torchrun --nproc_per_node=32 run.py BM25 base=NQ-open mode=encode ++pretokenize ++granularity=token ++save_weight ++save_encode
   ```
5. Prepare clusters:
   ```bash
   python run.py IVF base=NQ-open mode=encode ++save_encode
   ```
6. Run HI$^2$:
   ```bash
   torchrun --nproc_per_node=4 run.py HI2-NQ ++x_model=BM25 ++x_text_gate_k=20 ++y_model=IVF ++verifier_src=AR2 ++y_load_encode ++x_load_ckpt=inv
   ```
   - `nproc_per_node` determines how many processes to parallel (the more the faster). There is no need for GPU.
   - `x_text_gate_k` sets how many terms to index for each document
   - `y_query_gate_k` sets how many clusters to probe for each query
   - The searching process should finish within 1 minute and yield results very similar to:
      |Recall@5|Recall@20|Recall@100|#Documents to Evaluate|
      |:-:|:-:|:-:|:-:|
      |0.767|0.853|0.896|135790|

