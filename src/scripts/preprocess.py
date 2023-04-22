import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
from transformers import AutoTokenizer
from utils.util import save_pickle, load_pickle, Config
from utils.static import *

import hydra
from pathlib import Path
from omegaconf import OmegaConf
@hydra.main(version_base=None, config_path="../data/config/", config_name=f"script/{Path(__file__).stem}")
def get_config(hydra_config: OmegaConf):
    config._from_hydra(hydra_config)


def init_text(collection_path:str, cache_dir:str) -> ID_MAPPING:
    """
    convert document ids to offsets
    """
    tid2index = {}
    os.makedirs(cache_dir, exist_ok=True)
    with open(collection_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Collecting Text IDs", ncols=100, leave=False)):
            tid = line.split('\t')[0].strip()
            tid2index[tid] = len(tid2index)
    save_pickle(tid2index, os.path.join(cache_dir, "id2index.pkl"))
    return tid2index


def init_query_and_qrel(query_path:str, qrel_path:str, cache_dir:str, tid2index:ID_MAPPING):
    """
    Tokenize query file and transfer passage/document/query id in qrel file to its index in the saved token-ids matrix.

    Args:
        query_path: query file path
        qrel_path: qrel file path
        cache_dir: the directory to save files
        tid2index: mapping from text ids to text indices
    """
    os.makedirs(cache_dir, exist_ok=True)

    valid_queries = set()
    for line in open(qrel_path, 'r', encoding='utf-8'):
        try:
            query_id, _, positive_text_id, _ = line.strip().split()
        except:
            raise ValueError(f"Invalid format: {line}")
        if query_id in valid_queries:
            pass
        else:
            valid_queries.add(query_id)

    print("valid query number: {}".format(len(valid_queries)))

    qid2index = {}
    has_invalid = False
    tmp_query_path = ".".join([*query_path.split(".")[:-1], "tmp", "tsv"])
    
    with open(query_path, "r", encoding="utf-8") as f, \
        open(tmp_query_path, "w", encoding="utf-8") as g:
        for i, line in enumerate(tqdm(f, desc="Removing Missing Queries", ncols=100, leave=False)):
            query_id = line.split('\t')[0]
            if query_id not in valid_queries:
                has_invalid = True
                continue
            qid2index[query_id] = len(qid2index)
            g.write(line)

    if has_invalid:
        # backup queries that appear in the query file but not in the qrel file
        backup_query_path = ".".join([*query_path.split(".")[:-1], "backup", "tsv"])
        print(f"There are queries that appear in the query file but not in the qrel file! The original query file is saved at {backup_query_path}")
        os.rename(query_path, backup_query_path)
    else:
        os.remove(query_path)
    os.rename(tmp_query_path, query_path)

    qrels = []
    positives = defaultdict(list)
    with open(qrel_path, "r") as g:
        for line in tqdm(g, ncols=100, leave=False, desc="Processing Qrels"):
            fields = line.split()
            if len(fields) == 4:
                query_id, _, text_id, _ = fields
            elif len(fields) == 2:
                query_id, text_id = fields
            else:
                raise NotImplementedError
            query_index = qid2index[query_id]
            text_index = tid2index[text_id]
            qrels.append((query_index, text_index))
            # there may be multiple positive samples correpsonding to one query
            positives[query_index].append(text_index)

    save_pickle(qid2index, os.path.join(cache_dir, "id2index.pkl"))
    save_pickle(qrels, os.path.join(cache_dir, "qrels.pkl"))
    save_pickle(dict(positives), os.path.join(cache_dir, "positives.pkl"))
    return qid2index


def tokenize_to_memmap(input_path:str, cache_dir:str, num_rec:int, max_length:int, tokenizer:Any, tokenizer_type:str, tokenize_thread:int, text_col:list[int]=None, text_col_sep:str=None, is_query:bool=False) -> ID_MAPPING:
    """
    tokenize the passage/document text in multiple threads

    Args:
        input_path: query/passage/document file path
        cache_dir: save the output token ids etc
        num_rec: the number of records
        max_length: max length of tokens
        tokenizer(transformers.Tokenizer)
        tokenizer_type: the actual tokenizer vocabulary used
        tokenize_thread
        text_col
        text_col_sep
        is_query: if the input is a query

    Returns:
        mapping from the id to the index in the saved token-id matrix
    """
    cache_dir_with_plm = os.path.join(cache_dir, tokenizer_type)
    os.makedirs(cache_dir_with_plm, exist_ok=True)

    print(f"tokenizing {input_path} in {tokenize_thread} threads, output file will be saved at {cache_dir_with_plm}")

    arguments = []

    memmap_path = os.path.join(cache_dir_with_plm, "token_ids.mmp")
    # remove old memmap file
    if os.path.exists(memmap_path):
        os.remove(memmap_path)

    # create memmap first
    token_ids = np.memmap(
        memmap_path,
        shape=(num_rec, max_length),
        mode="w+",
        dtype=np.int32
    )

    for i in range(tokenize_thread):
        start_idx = round(num_rec * i / tokenize_thread)
        end_idx = round(num_rec * (i+1) / tokenize_thread)
        arguments.append((input_path, cache_dir_with_plm, num_rec, start_idx, end_idx, tokenizer, max_length, text_col, text_col_sep, is_query))

    with Pool(tokenize_thread) as p:
        p.starmap(_tokenize_to_memmap, arguments)



def _tokenize_to_memmap(input_path:str, output_dir:str, num_rec:int, start_idx:int, end_idx:int, tokenizer:Any, max_length:int, text_col:list[int]=None, text_col_sep:str=None, is_query:bool=False):
    """
    #. Tokenize the input text;

    #. do padding and truncation;

    #. then save the token ids, token_lengths, text ids

    Args:
        input_path: input text file path
        output_dir: directory of output numpy arrays
        start_idx: the begining index to read
        end_idx: the ending index
        tokenizer: transformer tokenizer
        max_length: max length of tokens
        text_col:
        text_col_sep:
        is_query
    """
    # some models such as t5 doesn't have sep token, we use space instead
    separator = text_col_sep
    pad_token_id = -1

    token_ids = np.memmap(
        os.path.join(output_dir, "token_ids.mmp"),
        shape=(num_rec, max_length),
        mode="r+",
        dtype=np.int32
    )

    with open(input_path, 'r') as f:
        pbar = tqdm(total=end_idx-start_idx, desc="Tokenizing", ncols=100, leave=False)
        for idx, line in enumerate(f):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break

            columns = line.split('\t')

            if is_query:
                # query has only one textual column
                text = columns[-1].strip()
            else:
                text = []
                for col_idx in text_col:
                    text.append(columns[col_idx].strip())
                text = separator.join(text)

            # only encode text
            token_id = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
            token_length = len(token_id)

            token_ids[idx, :] = token_id + [pad_token_id] * (max_length - token_length)
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    # manually action="store_true" because hydra doesn't support it
    for i, arg in enumerate(sys.argv):
        if "=" not in arg:
            sys.argv[i] += "=true"

    config = Config()
    get_config()

    cache_dir = os.path.join(config.cache_root, "dataset")
    data_dir = os.path.join(config.data_root, config.dataset)
    text_dir = os.path.join(cache_dir, "text")

    tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)
    tokenizer_type = config.plm_tokenizer

    collection_path = os.path.join(data_dir, "collection.tsv")

    if config.do_text:
        tid2index = init_text(collection_path, text_dir)
        if config.pretokenize:
            tokenize_to_memmap(collection_path, os.path.join(text_dir, ','.join([str(x) for x in config.text_col])), len(tid2index), config.max_text_length, tokenizer, tokenizer_type, config.tokenize_thread, text_col=config.text_col, text_col_sep=config.text_col_sep)

    if config.do_query:
        tid2index = load_pickle(os.path.join(text_dir, "id2index.pkl"))

        for query_set in config.query_set:
            query_path = os.path.join(data_dir, f"queries.{query_set}.tsv")
            qrel_path = os.path.join(data_dir, f"qrels.{query_set}.tsv")
            qid2index = init_query_and_qrel(query_path, qrel_path, os.path.join(cache_dir, "query", query_set), tid2index)
            if config.pretokenize:
                tokenize_to_memmap(os.path.join(data_dir, f"queries.{query_set}.tsv"), os.path.join(cache_dir, "query", query_set), len(qid2index), config.max_query_length, tokenizer, tokenizer_type, config.tokenize_thread, is_query=True)
