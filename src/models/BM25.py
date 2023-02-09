import os
import math
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .BaseModel import BaseSparseModel
from transformers import AutoTokenizer
from utils.typings import *
from utils.util import BaseOutput, synchronize, save_pickle



class BM25(BaseSparseModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)
        self.special_token_ids = set([x[1] for x in config.special_token_ids.values()])

        if self.config.pretokenize:
            self.index_dir = os.path.join(config.cache_root, "index", self.name, "pretokenize")
        elif self.config.get("return_code"):
            self.index_dir = os.path.join(config.cache_root, "index", self.name, config.code_type)


    @synchronize
    def encode_text(self, loader_text: DataLoader, load_all_encode: bool = False):
        if self.config.pretokenize:
            text_token_id_path = os.path.join(self.encode_dir, "text_token_ids.mmp")
            text_embedding_path = os.path.join(self.encode_dir, "text_embeddings.mmp")

            if load_all_encode:
                text_embeddings = np.memmap(
                    text_embedding_path,
                    mode="r",
                    dtype=np.float32
                ).reshape(len(loader_text.dataset), self._text_length, self._output_dim).copy()
                text_token_ids = np.memmap(
                    text_token_id_path,
                    mode="r",
                    dtype=np.int32
                ).reshape(len(loader_text.dataset), self._text_length)

            elif self.config.load_encode:
                text_embeddings = np.memmap(
                    text_embedding_path,
                    mode="r",
                    dtype=np.float32
                ).reshape(len(loader_text.dataset), self._text_length, self._output_dim)[loader_text.sampler.start: loader_text.sampler.end].copy()
                text_token_ids = np.memmap(
                    text_token_id_path,
                    mode="r",
                    dtype=np.int32
                ).reshape(len(loader_text.dataset), self._text_length)[loader_text.sampler.start: loader_text.sampler.end]

            else:
                assert self.config.eval_batch_size == 1, "Document Frequencies must be computed one by one!"
                text_token_ids = np.zeros((len(loader_text.sampler), self._text_length), dtype=np.int32)
                text_embeddings = np.zeros((len(loader_text.sampler), self._text_length, self._output_dim), dtype=np.float32)
                self.logger.info(f"encoding {self.config.dataset} text...")

                # counting df
                df = defaultdict(int)
                for i, x in enumerate(tqdm(loader_text, leave=False, ncols=100, desc="Collecting DFs")):
                    text_token_id = x["text"]["input_ids"].squeeze(0).numpy()
                    for j, token_id in enumerate(np.unique(text_token_id)):
                        df[token_id] += 1
                df = dict(df)

                if self.config.is_distributed:
                    all_dfs = self._gather_objects(df)
                df = defaultdict(int)
                for x in tqdm(all_dfs, desc="Merging DFs", ncols=100, leave=False):
                    for k, v in x.items():
                        df[k] += v
                df = dict(df)

                for i, x in enumerate(tqdm(loader_text, leave=False, ncols=100, desc="Computing TFIDFs")):
                    text_token_id = x["text"]["input_ids"].squeeze(0).numpy()
                    text_attn_mask = x["text"]["attention_mask"].squeeze(0).numpy()
                    length = text_attn_mask.sum()

                    text_token_ids[i] = text_token_id

                    tf = defaultdict(int)
                    for j, token_id in enumerate(text_token_id):
                        if token_id in self.special_token_ids:
                            continue
                        tf[token_id] += 1 / length
                    # force to assign 0 to pad token
                    tf[self.tokenizer.pad_token_id] = 0
                    tf = dict(tf)

                    for j, token_id in enumerate(text_token_id):
                        if token_id in self.special_token_ids:
                            continue
                        idf = np.log(len(loader_text.dataset) / (1 + df[token_id]))
                        tfidf = idf * tf[token_id]
                        # mask the negative tfidf score, which implies the document frequency of the token is bigger than the collection size
                        if tfidf < 0:
                            tfidf = 0
                        text_embeddings[i, j, 0] = tfidf

                    if "text_first_mask" in x:
                        text_first_mask = x["text_first_mask"].squeeze(0).numpy().astype(bool)
                        text_embeddings[i,:,0][~text_first_mask] = 0

                if self.config.save_encode:
                    self.save_to_mmp(
                        path=text_token_id_path,
                        shape=(len(loader_text.dataset), self._text_length),
                        dtype=np.int32,
                        loader=loader_text,
                        obj=text_token_ids
                    )
                    self.save_to_mmp(
                        path=text_embedding_path,
                        shape=(len(loader_text.dataset), self._text_length, self._output_dim),
                        dtype=np.float32,
                        loader=loader_text,
                        obj=text_embeddings
                    )

            text_embeddings = self._gate_text(text_embeddings)
            return BaseOutput(embeddings=text_embeddings, token_ids=text_token_ids)

        elif self.config.get("return_code"):
            if load_all_encode:
                text_codes = loader_text.dataset.text_codes.copy()
            else:
                text_codes = loader_text.dataset.text_codes[loader_text.sampler.start: loader_text.sampler.end].copy()
            return BaseOutput(token_ids=text_codes)

        else:
            return BaseOutput()


    @synchronize
    def encode_query(self, loader_query: DataLoader, load_all_encode: bool = False):
        if self.config.pretokenize or self.config.get("return_code"):
            return super().encode_query(loader_query, load_all_encode)
        else:
            return BaseOutput()


    def rerank(self, loaders: dict):
        from pyserini.index.lucene import IndexReader
        from utils.util import load_pickle

        index_reader = IndexReader(self.index_dir)

        tid2index = load_pickle(os.path.join(os.path.join(self.config.cache_root, "dataset", "text", "id2index.pkl")))
        tindex2id = {v: k for k, v in tid2index.items()}

        loader_rerank = loaders["rerank"]
        retrieval_result = defaultdict(list)

        self.logger.info("reranking...")
        for i, x in enumerate(tqdm(loader_rerank.dataset, ncols=100, leave=False)):
            query_idx = x["query_idx"]
            seq_idx = x["seq_idx"]

            seq_id = tindex2id[seq_idx]
            query = self.tokenizer.decode(x["query_token_id"], skip_special_tokens=True)
            score = index_reader.compute_query_document_score(seq_id, query)

            retrieval_result[query_idx].append((seq_idx, score))

        retrieval_result = self._gather_retrieval_result(retrieval_result)
        return retrieval_result


    @torch.no_grad()
    def compute_flops(self, loaders, log=True):
        """ compute flops as stated in SPLADE
        """
        from pyserini.index.lucene import IndexReader

        # document side
        loader_text = loaders["text"]
        doc_num = len(loader_text.dataset)
        query_num = len(loaders["query"].dataset)

        index = IndexReader(self.index_dir)
        terms = {}
        for k in tqdm(index.terms(), ncols=100, desc="Collecting Vocabulary"):
            terms[k.term] = 0

        D = terms
        Q = terms.copy()

        for i in tqdm(range(doc_num), ncols=100, desc="Collecting Text Terms"):
            tid = str(i)
            base_doc = index.get_document_vector(tid)
            base_key = base_doc.keys()
            for k in base_key:
                D[k] += 1

        with open(f"{self.config.data_root}/{self.config.dataset}/queries.dev.small.tsv") as f:
            for line in tqdm(f, ncols=100, desc="Collecting Query Terms"):
                qid, text = line.strip().split("\t")
                analysed = index.analyze(text)
                for k in analysed:
                    try:
                        Q[k] += 1
                    except:
                        pass

        flops = 0
        for k in D.keys():
            flops += (D[k] / doc_num * Q[k] / query_num)

        flops = round(flops, 2)
        self.metrics.update({"FLOPs": flops})
        if log:
            self.log_result()
            self.logger.info(f"FLOPs: {flops}")

