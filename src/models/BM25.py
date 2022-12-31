import os
import math
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .BaseModel import BaseSparseModel
from transformers import AutoTokenizer
from utils.typings import *
from utils.util import BaseOutput



class BM25(BaseSparseModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)

        if self.config.pretokenize:
            self.collection_dir = os.path.join(self.collection_dir, "pretokenize")
            self.index_dir = os.path.join(self.index_dir, "pretokenize")
        elif self.config.get("return_code"):
            self.collection_dir = os.path.join(self.collection_dir, self.config.code_type)
            self.index_dir = os.path.join(self.index_dir, self.config.code_type)


    def encode_text(self, loader_text: DataLoader, load_all_encode: bool = False):
        if self.config.pretokenize:
            self._synchronize()
            text_embedding_path = os.path.join(self.encode_dir, "text_embeddings.mmp")

            if load_all_encode:
                text_embeddings = np.memmap(
                    text_embedding_path,
                    mode="r",
                    dtype=np.float32
                ).reshape(len(loader_text.dataset), self.config.text_length, 1).copy()
                text_token_ids = loader_text.dataset.text_token_ids[:, :self.config.text_length].copy()

            elif self.config.load_encode:
                text_embeddings = np.memmap(
                    text_embedding_path,
                    mode="r",
                    dtype=np.float32
                ).reshape(len(loader_text.dataset), self.config.text_length, 1)[loader_text.sampler.start: loader_text.sampler.end].copy()
                text_token_ids = loader_text.dataset.text_token_ids[loader_text.sampler.start: loader_text.sampler.end, :self.config.text_length].copy()

            else:
                text_token_ids = loader_text.dataset.text_token_ids[loader_text.sampler.start: loader_text.sampler.end, :self.config.text_length].copy()
                text_embeddings = np.zeros((len(loader_text.sampler), self.config.text_length, 1), dtype=np.float32)
                self.logger.info(f"encoding {self.config.dataset} text...")

                # counting df
                df = defaultdict(int)
                for i, x in enumerate(tqdm(loader_text.dataset, leave=False, ncols=100)):
                    pos_text_token_id = x["pos_text_token_id"]
                    pos_text_attn_mask = x["pos_text_attn_mask"]
                    for j, token_id in enumerate(np.unique(pos_text_token_id)):
                        df[token_id] += 1
                df = dict(df)

                for i, x in enumerate(tqdm(loader_text.dataset, leave=False, ncols=100)):
                    pos_text_token_id = x["pos_text_token_id"]
                    pos_text_attn_mask = x["pos_text_attn_mask"]
                    length = pos_text_attn_mask.sum()

                    tf = defaultdict(int)
                    for j, token_id in enumerate(pos_text_token_id):
                        if pos_text_attn_mask[j] == 0:
                            break
                        tf[token_id] += 1 / length
                    tf = dict(tf)

                    for j, token_id in enumerate(pos_text_token_id):
                        if pos_text_attn_mask[j] == 0:
                            break
                        idf = np.log(len(loader_text.dataset) / (1 + df[token_id]))
                        tfidf = idf * tf[token_id]
                        # mask the negative tfidf score, which implies the document frequency of the token is bigger than the collection size
                        if tfidf < 0:
                            tfidf = 0
                        text_embeddings[i, j, 0] = tfidf

                if self.config.save_encode:
                    self.save_to_mmp(
                        path=text_embedding_path,
                        shape=(len(loader_text.dataset), self.config.text_length, 1),
                        dtype=np.float32,
                        loader=loader_text,
                        obj=text_embeddings
                    )

            text_embeddings = self._gate_text(text_embeddings)
            return BaseOutput(embeddings=text_embeddings, token_ids=text_token_ids)

        elif self.config.get("return_code"):
            self._synchronize()
            text_codes = loader_text.dataset.text_codes[loader_text.sampler.start: loader_text.sampler.end].copy()
            return BaseOutput(token_ids=text_codes)

        else:
            return BaseOutput()


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

