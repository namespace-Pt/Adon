# Autoregressive Search Engine with Term-Set Generation (AutoTSG)

This repository contains the implementation of HI2.

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
1. Download the MSMARCO Passage Dataset from [HERE]();
2. Download the Natual Questions Dataset from [HERE]();
3. Untar the file at anywhere you like, e.g. `/data/HI2`;
4. Go to `src/data/config/base/_default.yaml`, set 
   - `data_root: /data/HI2`. This tells the program where to find the data.
   - `plm_root: /data/HI2/huggingface_PLMs`. By default, the language model from hugginface will be permanently downloaded to this folder and hence can be directly loaded afterwards.
5. ```bash
   // pre-tokenize the dataset
   python -m scripts.preprocess base=MSMARCO-passage
   python -m scripts.preprocess base=NQ-open
   ```

## Reproducing from Our Checkpoint
### MSMARCO Passage
#### HI$^2_{\text{sup}}$
1. Download the model checkpoint and identifier from [HERE](); Then untar it with 
   ```bash
   tar -xzvf hi2.msmarco.tar.gz -C src/data/cache/
   ```
2. ```bash
   torchrun --nproc_per_node=4 run.py BIVFPQ
   ```
   - `nproc_per_node` determines how many GPU to use
3. With 4 A100s, the above command should finish within 1 minutes and yield results very similar to:
   |MRR@10|Recall@100|Recall@100|
   |:-:|:-:|:-:|
   |0.401|0.916|0.976|

### Natural Questions
#### HI$^2_{\text{sup}}$
1. Download the model checkpoint and identifier from [HERE](); Then untar it with 
   ```bash
   tar -xzvf hi2.nq.tar.gz -C src/data/cache/
   ```
2. ```bash
   torchrun --nproc_per_node=4 run.py BIVFPQ-NQ
   ```
   - `nproc_per_node` determines how many GPU to use
3. With 4 A100s, the above command should finish within 1 minutes and yield results very similar to:
   |Recall@5|Recall@20|Recall@100|
   |:-:|:-:|:-:|
   |0.779|0.861|0.906|

