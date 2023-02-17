# Data
The very first thing to make use of this library is to download data (e.g. MSMARCO). You can also leverage your own data by organizing them into a specific format.

## Downloading
### Download Data
Currently, we upload MSMARCO passage collection on [OneDrive](https://1drv.ms/u/s!Aipk4vd2SBrtv3RttSFWXGsAb6bL?e=wdKhWf). The corpus in the drive is slightly different from the offical one: it contains the **title** of each passage, which can always lead to better performance of sparse/dense models, see [RocketQA](https://arxiv.org/abs/2010.08191). To make use of the data:
1. Download the `.tar.gz` file from the drive.
2. Uncompress it wherever you like, for example, `/data/Adon/`.
3. Go to `src/data/config/base/_default.yaml`, set `data_root: /data/Adon`. This tells the program where to find the data. If you want to perform experiments with other datasets, make sure they also be stored in `data_root`.

*Adon* leverages [hydra](https://github.com/facebookresearch/hydra) to manage configurations of models and scripts in `src/data/config/`. You can learn more of it in {doc}`configs`. For example, `text_length` in `src/data/config/base/MSMARCO-passage.yaml` controls how many tokens to preserve for each passage.

### Download Language Models
*Adon* integrates [huggingface](https://huggingface.co/docs/transformers/v4.21.3/en/index) to load a variety of language models. By default, *Adon* will permanently download a specified language model and save it in a given folder and directly load it from the folder afterwards. You should specify the place you want to store by specifying `plm_root` in `data/config/base/_default.yaml`.

Note that downloading language models would be automatic.

(preprocessing)=
## Preprocessing
Adon defaults to pre-tokenize the documents and queries and store the tokenized results in the [numpy.memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) file, which supports efficient random access hence speeds data loading. Therefore, you should always run the following command to generate necessary files to run models when dealing with a new dataset for the first time:
```bash
cd src
python -m scripts.preprocess base=MSMARCO-passage
```
- `base=MSMARCO-passage` tells the program to load the configuration in `src/data/config/base/MSMARCO-passage.yaml`
- You can modify things in the configuration file or by **command line**.
  - For example, `python -m scripts.preprocess base=MSMARCO-passage ++max_text_length=512` tells the program to preserve 512 tokens per document in terms of the BERT tokenizer.


## Create Your Data
With *Adon*, it's easy to conduct experiments on your own dataset, by organizing your corpus and training pairs in a standard form. Following the common practice of TREC, *Adon* requires these things for a new dataset:
```
/your/data_root/
└── New-Dataset
    ├── collection.tsv
    ├── qrels.dev.tsv
    ├── qrels.train.tsv
    ├── queries.dev.tsv
    └── queries.train.tsv
```
1. **collection.tsv**: The corpus file. Each line represents a single document. The features of the document is separated by `\t`, where the first column is the document ID (unique). The other columns may be the document title, abstract, content, etc. You can add arbitrary number of features to a document.
2. **queries.train/dev.tsv**: The query file. Each line represents a query for training or evaluating. There are only two features for a query, with the first one being query ID (unique) and the second one being query string. The two features are also separated by `\t`.
3. **qrels.train/dev.tsv**: The query-document relevance pair. Each line has four columns separated with `\t`, the first column is **query ID**, the second column is always **0**, the third column is **the relevant document ID**, the last column is always **1**.

Besides, you must add a new configuration file to describe your new dataset, i.e. create a new `yaml` file at `src/data/config/base/New-Dataset.yaml`:
```yaml
defaults:
  - _default

# the name of the folder under data_root
dataset: New-Dataset
# how many tokens in one document
text_length: 128
# how many tokens in one query
query_length: 32
# how many tokens to preserve in the memmap file for one document
max_text_length: 256
# how many tokens to preserve in the memmap file for one query
max_query_length: 64
# which language model to use by default
plm: bert
# which column of features to use for a document
text_col: [2]
# how to separate different features if multiple columns are used
text_col_sep: "sep"
```
Again, run [preprocessing](preprocessing) for your new dataset and everything is ready!
