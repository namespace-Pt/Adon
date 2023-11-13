import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import *

DEVICE = Union[int,Literal["cpu"]]

RETRIEVAL_MAPPING = Union[dict[int, list[int]], dict[int, list[tuple[int,float]]]]
ID_MAPPING = dict[str, int]

DENSE_METRIC = Literal["ip", "cos", "l2"]
DATA_FORMAT = Literal["memmap", "raw"]

TENSOR = torch.Tensor
LOADERS = dict[str,DataLoader]
NN_MODULE = torch.nn.Module
INDICES = Union[np.ndarray,list,torch.Tensor]

PLM_MAP = {
    "bert": {
        # different model may share the same tokenizer, so we can load the same tokenized data for them
        "tokenizer": "bert",
        "load_name": "bert-base-uncased"
    },
    "distilbert": {
        "tokenizer": "bert",
        "load_name": "distilbert-base-uncased",
    },
    "ernie": {
        "tokenizer": "bert",
        "load_name": "nghuyong/ernie-2.0-en"
    },
    "contriever": {
        "tokenizer": "contriever",
        "load_name": "null"
    },
    "gtr": {
        "tokenizer": "gtr",
        "load_name": "null"
    },
    "bert-chinese": {
        "tokenizer": "bert-chinese",
        "load_name": "bert-base-chinese"
    },
    "bert-xingshi": {
        "tokenizer": "bert-xingshi",
        "load_name": "null"
    },
    "t5-small": {
        "tokenizer": "t5",
        "load_name": "t5-small"
    },
    "t5": {
        "tokenizer": "t5",
        "load_name": "t5-base"
    },
    "t5-large": {
        "tokenizer": "t5",
        "load_name": "t5-large"
    },
    "doct5": {
        "tokenizer": "t5",
        "load_name": "castorini/doc2query-t5-base-msmarco"
    },
    "distilsplade": {
        "tokenizer": "bert",
        "load_name": "null"
    },
    "splade": {
        "tokenizer": "bert",
        "load_name": "null"
    },
    "bart": {
        "tokenizer": "bart",
        "load_name": "facebook/bart-base"
    },
    "bart-large": {
        "tokenizer": "bart",
        "load_name": "facebook/bart-large"
    },
    "retromae": {
        "tokenizer": "bert",
        "load_name": "Shitao/RetroMAE"
    },
    "retromae_msmarco": {
        "tokenizer": "bert",
        "load_name": "Shitao/RetroMAE_MSMARCO"
    },
    "retromae_distill": {
        "tokenizer": "bert",
        "load_name": "Shitao/RetroMAE_MSMARCO_distill"
    },
    "deberta": {
        "tokenizer": "deberta",
        "load_name": "microsoft/deberta-base"
    },
    "keyt5": {
        "tokenizer": "t5",
        "load_name": "snrspeaks/KeyPhraseTransformer"
    },
    "seal": {
        "tokenizer": "bart",
        "load_name": "tuner007/pegasus_paraphrase"
    },
    "doct5-nq": {
        "tokenizer": "t5",
        "load_name": "namespace-Pt/doct5-nq320k"
    }
}