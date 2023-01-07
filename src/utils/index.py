import os
import json
import torch
import faiss
import time
import shutil
import subprocess
import numpy as np
import multiprocessing as mp
# lazy import to avoid cuda error message
from torch_scatter import scatter_max
from copy import copy
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from .util import save_pickle, load_pickle, MasterLogger, makedirs, isempty, synchronize
from .typings import *

import concurrent.futures as ft



class BaseIndex(object):
    def __init__(self) -> None:
        self.logger = MasterLogger(type(self).__name__)



class BasePostVerifier(BaseIndex):
    """
    Basic class for post verifier.
    """
    def __init__(self) -> None:
        super().__init__()



class FaissIndex():
    """
    Faiss Index.

    Attributes:
        index(faiss.Index): the faiss index object
        onGPU(bool): if the index is moved to GPU in :func:`utils.index.FaissIndex.fit`
    """
    def __init__(self, d:int, index_type:str, metric:DENSE_METRIC, start_text_idx:int, device:DEVICE, save_dir:str, **kwargs):
        """
        Args:
            d: embedding dimension
            index_type: command for index_factory
            metric
            start_text_idx: when ``config.parallel=='text'``, each shard starts from different offset
            device: if number, the index will be transfered to this device
            save_dir: the directory to save the index
        """
        self.start_text_idx = start_text_idx
        self.device = device
        self.save_dir = save_dir

        self.name = index_type
        self.logger = MasterLogger(index_type)
        self.save_path = os.path.join(save_dir, index_type)

        if metric == "l2":
            metric = faiss.METRIC_L2
        elif metric in ["ip", "cos"]:
            metric = faiss.METRIC_INNER_PRODUCT
        index = faiss.index_factory(d, index_type, metric)

        self.index = index
        self.onGPU = False


    def load(self, load_path:Optional[str]=None):
        """
        Load index from load_path.

        Args:
            load_path: if ``None``, use ``self.save_path``
        """
        if load_path is None:
            load_path = self.save_path

        self.logger.info(f"loading index from {load_path}")
        index = faiss.read_index(load_path)
        assert self.index.metric_type == index.metric_type, "Make sure the metric_type of the loaded index matches the created one!"
        self.index = index


    def save(self, save_path:Optional[str]=None):
        """
        Save index to a given path.

        Args:
            save_path: if ``None``, use ``self.save_path``
        """
        if save_path is None:
            save_path = self.save_path
        makedirs(save_path)
        self.logger.info(f"saving index at {save_path}...")
        # check if the index is on gpu
        if self.onGPU:
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, save_path)


    def fit(self, text_embeddings:np.ndarray):
        """
        #. Move the index to gpu if needed;

        #. Train the index by ``text_embeddings``;

        #. Add ``text_embeddings`` to the index.
        """
        if self.device != "cpu":
            # it's better to use gpu flat index because it's far more efficient
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.logger.info("use float16 to store vectors on GPU!")
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, self.index, co)
            # self.index = faiss.index_cpu_to_gpu_multiple_py([faiss.StandardGpuResources(), faiss.StandardGpuResources()], self.index, gpus=[2,3], co=co)
            # self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            self.onGPU = True
        else:
            self.onGPU = False

        if not self.index.is_trained:
            self.logger.info(f"training index...")
            if self.onGPU:
                faiss.GpuParameterSpace().set_index_parameter(self.index, "verbose", 1)
            else:
                faiss.ParameterSpace().set_index_parameter(self.index, "verbose", 1)
            self.index.train(text_embeddings)

        if self.index.ntotal == 0:
            self.logger.info(f"adding embeddings...")
            self.index.add(text_embeddings)
            # batch_size = 10000
            # for start_idx in tqdm(range(0, len(text_embeddings), batch_size), ncols=100, leave=False):
            #     text_embedding = text_embeddings[start_idx: start_idx + batch_size]
            #     self.index.add(text_embedding)


    def search(self, query_embeddings:np.ndarray, hits:int, batch_size:int=500, query_start_idx:int=0, eval_posting_length:bool=False, verifier:Optional[BasePostVerifier]=None, **kwargs) -> tuple[RETRIEVAL_MAPPING,Optional[np.ndarray]]:
        """
        KNN search the ``query_embeddings`` in ``self.index``.

        Args:
            query_embeddings: array of [N, D]
            hits: top K
            batch_size: the batch size for searching
            query_start_idx: when ``config.parallel=='query'``, offset the query across different processes
            eval_posting_length: when using IVFxx indexes, count the average hitted document number
            verifier: the verifier to post rank the hitted results

        Returns:
            the retrieval result with scores
            the posting length array or None
        """
        assert hits > 0

        # set dynamic properties on both cpu and gpu index
        if "nprobe" in kwargs and kwargs["nprobe"] and "IVF" in self.name:
            if self.onGPU:
                faiss.GpuParameterSpace().set_index_parameter(self.index, "nprobe", kwargs["nprobe"])
            else:
                faiss.ParameterSpace().set_index_parameter(self.index, "nprobe", kwargs["nprobe"])
        if "efSearch" in kwargs and kwargs["efSearch"] and "HNSW" in self.name:
            if self.onGPU:
                faiss.GpuParameterSpace().set_index_parameter(self.index, "efSearch", kwargs["efSearch"])
            else:
                faiss.ParameterSpace().set_index_parameter(self.index, "efSearch", kwargs["efSearch"])

        retrieval_result = {}
        if eval_posting_length and "IVF" in self.name:
            if isinstance(self.index, faiss.IndexPreTransform):
                ivf_index = faiss.downcast_index(self.index.index)
                vt = faiss.downcast_VectorTransform(self.index.chain.at(0))
                # chain the vector transform and the quantizer
                ivf_quantizer = faiss.IndexPreTransform(vt, ivf_index.quantizer)
            else:
                ivf_index = self.index
                ivf_quantizer = ivf_index.quantizer

            nprobe = ivf_index.nprobe
            nlist = ivf_index.nlist
            total_ivf_hits = np.zeros(nlist)
            posting_list_length_list = np.zeros(nlist)

            if self.onGPU:
                ivf_index = faiss.index_gpu_to_cpu(ivf_index)
            for i in range(nlist):
                posting_list_length_list[i] = ivf_index.invlists.list_size(i)

        for start_idx in tqdm(range(0, len(query_embeddings), batch_size), ncols=100, leave=False):
            query_embedding = query_embeddings[start_idx: start_idx + batch_size]
            batch_tscores, batch_tindices = self.index.search(query_embedding, hits)
            for i, tscores in enumerate(batch_tscores):
                qidx = i + start_idx
                tindices = batch_tindices[i]
                if verifier is not None:
                    tindices, tscores = verifier(qidx, tindices)

                tids = tindices + self.start_text_idx
                # ignore -1
                retrieval_result[qidx] = [(int(tids[j]), float(tscores[j])) for j in range(len(tids)) if tindices[j] != -1]

            if eval_posting_length:
                _, ivfids = ivf_quantizer.search(query_embedding, nprobe)  # batch_size, nprobe
                # accumulate total hits over ivf entries
                for ivfid in ivfids:
                    total_ivf_hits[ivfid] += 1

        if eval_posting_length and "IVF" in self.name:
            # average all queries
            posting_lists_length = total_ivf_hits / len(query_embeddings) @ posting_list_length_list
        else:
            posting_lists_length = None

        return retrieval_result, posting_lists_length

    @staticmethod
    def get_xb(index:faiss.Index) -> np.ndarray:
        """
        get the database of the index
        """
        xb = faiss.rev_swig_ptr(faiss.downcast_index(index).get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)
        return xb

    @staticmethod
    def get_pq_codebook(pq:faiss.ProductQuantizer) -> np.ndarray:
        """
        get the codebook of the pq
        """
        return faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)



class BaseInvertedIndex(BaseIndex):
    """
    Base class of the Inverted Indexes
    """
    def __init__(self, text_num:int, token_num:int, device:DEVICE, rank:int, save_dir:str, special_token_ids:Optional[set]=None):
        """
        Args:
            text_num: the number of documents
            token_num: the number of posting entries
            posting_prune: the fraction of postings to keep
            start_text_idx: when ``config.parallel=='text'``, each shard starts from different offset
            device
            save_dir: the directory for inverted lists
            special_token_ids: the special token ids in PLM tokenizer, usually can be obtained by :py:obj:``utils.manager.Manager.config.special_token_ids``
        """
        super().__init__()
        self.text_num = text_num
        self.token_num = token_num
        self.device = torch.device(device)
        self.rank = rank
        self.save_dir = save_dir

        if special_token_ids is not None:
            self.special_token_ids = special_token_ids
        else:
            self.special_token_ids = set()

        self.text_idx_inverted_lists_path = os.path.join(save_dir, f"text_idx_inverted_lists_{self.rank}.pt")
        self.token_idx_inverted_lists_path = os.path.join(save_dir, f"token_idx_inverted_lists_{self.rank}.pt")


    def fit(self, text_token_ids:np.ndarray, text_embeddings:np.ndarray, rebuild_index:bool=False, load_index:bool=False, save_index:bool=True, threads:int=16, shards:int=32, posting_prune:float=0.0, start_text_idx:int=0):
        """
        1. Populate the inverted lists;

        2. Save the inverted lists if necessary;

        3. Move the posting lists to gpu if necessary;

        4. Sort each posting list by the descending token weights.

        Args:
            text_token_ids: the token ids of each document, array of [N, L]
            text_embeddings: the token weights/vectors, array of [N, L, D]
            rebuild_index: if ``True``, default to skip saving posting lists
            load_index: if ``True``, load the posting lists from ``save_dir``
            save_index: if ``True``, save the posting lists to ``save_dir``

        Attributes:
            text_embeddings
        """
        self.logger.info("fitting inverted index...")

        # load the posting lists when we specify load_index, or when the model do not need to rebuild index
        if (load_index or not rebuild_index) and os.path.exists(self.text_idx_inverted_lists_path) and not save_index:
            self.text_idx_inverted_lists = torch.load(self.text_idx_inverted_lists_path, map_location=self.device)
            self.token_idx_inverted_lists = torch.load(self.token_idx_inverted_lists_path, map_location=self.device)
        # construct posting lists when the model needs to rebuild index, or when the posting lists doesn't exist
        elif rebuild_index or not os.path.exists(self.text_idx_inverted_lists_path):
            text_num_per_thread = len(text_token_ids) / shards
            # tmp_dir = os.path.join(self.save_dir, "posting_lists_tmp", str(self.device.index))
            arguments = []
            for i in range(shards):
                start_idx = round(text_num_per_thread * i)
                end_idx = round(text_num_per_thread * (i+1))

                arguments.append((
                    text_token_ids[start_idx: end_idx],
                    # offset to this shard
                    start_idx,
                    self.token_num,
                    self.special_token_ids,
                    # os.path.join(tmp_dir, f"{i}.txt")
                ))

            # process parallel consumes extensive memory if use lists to store;
            with mp.Pool(threads) as p:
                # imap returns an iterator
                outputs = p.imap(build_inverted_lists, arguments)

                text_idx_inverted_lists = [[] for _ in range(self.token_num)]
                token_idx_inverted_lists = [[] for _ in range(self.token_num)]

                self.logger.info("merging shards...")
                for inv_lists_pair in tqdm(outputs, total=shards, ncols=100, leave=False):
                    text_idx_inverted_list = inv_lists_pair[0]
                    token_idx_inverted_list = inv_lists_pair[1]
                    for j in range(self.token_num):
                        text_idx_inverted_lists[j].extend(text_idx_inverted_list[j])
                        token_idx_inverted_lists[j].extend(token_idx_inverted_list[j])

            # thread parallel is so slow (concurrent.futures.ThreadPoolExecutor)
            # with ft.ThreadPoolExecutor(max_workers=shards) as executor:
            #     text_idx_inverted_lists = [[] for _ in range(self.token_num)]
            #     token_idx_inverted_lists = [[] for _ in range(self.token_num)]
            #     # submit job to the thread pool
            #     outputs = [executor.submit(build_inverted_lists, argument) for argument in arguments]
            #     self.logger.info("merging shards...")
            #     for inv_lists_pair in tqdm(ft.as_completed(outputs), total=shards, ncols=100, leave=False, desc="Merging Shards"):
            #         text_idx_inverted_list = inv_lists_pair[0]
            #         token_idx_inverted_list = inv_lists_pair[1]
            #         for j in range(self.token_num):
            #             text_idx_inverted_lists[j].extend(text_idx_inverted_list[j])
            #             token_idx_inverted_lists[j].extend(token_idx_inverted_list[j])

            synchronize()
            self.logger.info("packing posting lists...")

            for i in tqdm(range(self.token_num), ncols=100, leave=False):
                text_indices = text_idx_inverted_lists[i]
                if len(text_indices):
                    text_idx_inverted_lists[i] = torch.tensor(text_indices, device=self.device, dtype=torch.int32)
                    token_idx_inverted_lists[i] = torch.tensor(token_idx_inverted_lists[i], device=self.device, dtype=torch.int16)

            # save the posting lists when we specify save_index or when the model doesn't need to rebuild index every time
            if save_index or (not rebuild_index):
                self.logger.info(f"saving index at {self.save_dir}...")
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(text_idx_inverted_lists, self.text_idx_inverted_lists_path)
                torch.save(token_idx_inverted_lists, self.token_idx_inverted_lists_path)
            self.text_idx_inverted_lists = text_idx_inverted_lists
            self.token_idx_inverted_lists = token_idx_inverted_lists

            synchronize()

        else:
            raise NotImplementedError(f"Conflict Option: rebuild_index={rebuild_index}, load_index={load_index}, save_index={save_index}!")

        self.text_embeddings = text_embeddings
        # offset
        self.start_text_idx = start_text_idx
        self.posting_prune = posting_prune

        # in case the embedding of each token is a scalar
        if text_embeddings.shape[-1] == 1:
            # static posting list prune
            # first sort the posting lists w.r.t. the token weight
            for token_id in tqdm(range(self.token_num), desc="Organizing Posting Lists", ncols=100, leave=False):
                text_idx_posting_list = self.text_idx_inverted_lists[token_id]
                # skip empty postings
                if len(text_idx_posting_list) == 0:
                    continue

                text_idx_posting_list = text_idx_posting_list.long()    # N
                token_idx_posting_list = self.token_idx_inverted_lists[token_id].long() # N

                weight_posting_list = self.text_embeddings[text_idx_posting_list, token_idx_posting_list].squeeze(-1)   # N

                non_zero = weight_posting_list > 0
                text_idx_posting_list = text_idx_posting_list[non_zero]
                token_idx_posting_list = token_idx_posting_list[non_zero]

                if posting_prune > 0:
                    weight_posting_list = weight_posting_list[non_zero]
                    _, sorted_idx = weight_posting_list.sort(dim=-1, descending=True)
                    text_idx_inverted_list = text_idx_posting_list[sorted_idx]
                    token_idx_inverted_list = token_idx_posting_list[sorted_idx]

                self.text_idx_inverted_lists[token_id] = text_idx_posting_list
                self.token_idx_inverted_lists[token_id] = token_idx_posting_list

            if posting_prune > 0:
                posting_lengths = np.array([len(x) for x in self.text_idx_inverted_lists if len(x) > 0], dtype=np.float32)
                if self.posting_prune >= 1:
                    posting_prune = self.posting_prune
                else:
                    posting_prune = int(np.percentile(posting_lengths, self.posting_prune * 100))

                self.posting_prune = posting_prune
                self.logger.info(f"pruning postings by {posting_prune}...")


    def _prune_posting_list(self, posting_list):
        """
        Shortcut for posting pruning.
        """
        if self.posting_prune > 0:
            posting_list = posting_list[:self.posting_prune]
        return posting_list



class InvertedHitIndex(BaseInvertedIndex):
    """
    Inverted Hit Index with no scores
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def search(self, query_token_ids:np.ndarray, eval_posting_length:bool=False, query_start_idx:int=0, verifier:Optional[BasePostVerifier]=None, **kwargs) -> tuple[RETRIEVAL_MAPPING, Optional[np.array]]:
        """
        Search the inverted index. Recall all documents hit by query_token_ids in the inverted index.

        Args:
            query_token_ids: array of [M, LQ]
            query_embeddings: array of [M, LQ, D] or None (regard all tokens in the query as the same important)
            eval_posting_length: if ``True``, count the average **hitted** document number
            query_start_idx: when ``config.parallel=='query'``, offset the query across different processes
            verifier: the verifier to post rank the hitted results
        """
        retrieval_result = {}
        global_score = torch.zeros(self.text_num, device=self.device)
        if eval_posting_length:
            posting_lists_length = np.zeros(query_token_ids.shape[0])
        else:
            posting_lists_length = None

        for qidx, token_ids in enumerate(tqdm(query_token_ids, ncols=100, leave=False)):
            global_score[:] = 0.
            for j, token_id in enumerate(token_ids):
                if token_id == -1 or token_id in self.special_token_ids:
                    continue

                text_idx_posting_list = self.text_idx_inverted_lists[token_id]   # n
                if len(text_idx_posting_list) == 0:
                    continue

                text_idx_posting_list = text_idx_posting_list.long()
                text_idx_posting_list = self._prune_posting_list(text_idx_posting_list)

                if eval_posting_length:
                    posting_lists_length[qidx] += len(text_idx_posting_list)

                # map the text idx to global scale
                # count the hit frequency
                global_score[text_idx_posting_list] += 1

            tindices = global_score.nonzero().squeeze(-1)
            tscores = global_score[tindices]

            if verifier is not None:
                tindices, tscores = verifier(qidx, tindices)

            # offset the text idx by the starting idx, because each process maintains a segment of corpus
            retrieval_result[qidx + query_start_idx] = list(zip((tindices + self.start_text_idx).tolist(), tscores.tolist()))

            if eval_posting_length:
                posting_lists_length[qidx] = (global_score > 0).sum()

        return retrieval_result, posting_lists_length



class InvertedVectorIndex(BaseInvertedIndex):
    """
    Inverted Vector Index as described in `COIL <https://aclanthology.org/2021.naacl-main.241.pdf>`_.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def search(self, query_token_ids:np.ndarray, query_embeddings:Optional[np.ndarray], hits, eval_posting_length:bool=False, query_start_idx:int=0, verifier:Optional[BasePostVerifier]=None, **kwargs) -> tuple[RETRIEVAL_MAPPING, Optional[np.ndarray]]:
        """
        Search the inverted index. The max matching score of the same token is preserved.

        Args:
            query_token_ids: array of [M, LQ]
            query_embeddings: array of [M, LQ, D] or None (regard all tokens in the query as the same important)
            hits: top K
            eval_posting_length: if ``True``, count the average **hitted** document number
            query_start_idx: when ``config.parallel=='query'``, offset the query across different processes
            verifier: the verifier to post rank the hitted results
        """
        assert hits > 0
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = torch.as_tensor(query_embeddings, device=self.device)

        retrieval_result = {}

        global_score = torch.zeros(self.text_num, device=self.device)
        if eval_posting_length:
            posting_lists_length = np.zeros(query_token_ids.shape[0])
        else:
            posting_lists_length = None

        for qidx, token_ids in enumerate(tqdm(query_token_ids, ncols=100, leave=False)):
            global_score[:] = 0.
            for j, token_id in enumerate(token_ids):
                if token_id == -1 or token_id in self.special_token_ids:
                    continue

                text_idx_posting_list = self.text_idx_inverted_lists[token_id]   # n
                if len(text_idx_posting_list) == 0:
                    continue

                text_idx_posting_list = text_idx_posting_list.long()
                token_idx_posting_list = self.token_idx_inverted_lists[token_id].long()

                text_idx_posting_list = self._prune_posting_list(text_idx_posting_list)
                token_idx_posting_list = self._prune_posting_list(token_idx_posting_list)

                embedding_posting_list = self.text_embeddings[text_idx_posting_list, token_idx_posting_list]
                if query_embeddings is not None:
                    token_embedding = query_embeddings[qidx, j]
                    if token_embedding.size(0) == 1 and token_embedding == 0:
                        continue
                    score = embedding_posting_list @ token_embedding  # n
                else:
                    assert query_embeddings.shape[-1] == 1, "Found query_embeddings is None but output_dim > 1!"
                    score = embedding_posting_list.squeeze(-1)

                if embedding_posting_list.shape[-1] > 1:
                    score = scatter_max(score, index=text_idx_posting_list, dim_size=self.text_num)[0]
                    global_score += score
                else:
                    global_score[text_idx_posting_list] += score

            tscores, tindices = global_score.topk(hits)  # k

            if verifier is not None:
                tindices, tscores = verifier(qidx, tindices)

            retrieval_result[qidx + query_start_idx] = list(zip((tindices + self.start_text_idx).tolist(), tscores.tolist()))

            if eval_posting_length:
                posting_lists_length[qidx] = (global_score > 0).sum()

        return retrieval_result, posting_lists_length



class BaseAnseriniIndex(BaseIndex):
    """
    Wrapper of Anserini Indexes.
    """
    def __init__(self, collection_path:str, collection_dir:str, index_dir:str) -> None:
        """
        Args:
            collection_path: the collection file path
            collection_dir: the json collections folder generated by :func:`utils.index.AnseriniImpactIndex.fit` and :func:`utils.index.AnseriniBM25Index.fit`
            index_dir: the folder to store the lucene index file; usually :py:obj:`models.BaseModel.BaseModel.index_dir`
        """
        super().__init__()
        self.collection_path = collection_path
        self.collection_dir = collection_dir
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        os.makedirs(collection_dir, exist_ok=True)


    def convert_retrieval_result(self, tmp_retrieval_result_path:str, qid2index:ID_MAPPING, tid2index:ID_MAPPING, verifier:Optional[BasePostVerifier]=None) -> RETRIEVAL_MAPPING:
        """
        Anserini defaults to use docids in the collection. Convert the naive docids to our doc idx.

        Args:
            tmp_retrieval_result_path: the path to the temporary retrieval result file created by anserini Searcher
            qid2index: mapping from query id to query idx; generated by :py:mod:`scripts.preprocess`
            tid2index: mapping from document id to document idx
        """
        # overwrite all ids by idxs for the sake of unification
        retrieval_result = defaultdict(list)
        g = open(tmp_retrieval_result_path)
        for line in tqdm(g, ncols=100, leave=False, desc="Collecting Retrieval Results"):
            fields = line.strip().split("\t")
            qidx = qid2index[fields[0]]
            tidx = tid2index[fields[1]]
            retrieval_result[qidx].append(tidx)
        g.close()

        if verifier is not None:
            for qidx, tindices in retrieval_result.items():
                retrieval_result[qidx] = verifier(qidx, tindices)[0]

        return retrieval_result


    def generate_collection(self, collection_path=None, output_dir=None, text_cols=[-1], max_docs_per_file=1000000):
        self.logger.info("converting tsv to jsonl collection...")
        if collection_path is None:
            collection_path = self.collection_path
        if output_dir is None:
            output_dir = self.collection_dir

        file_index = 0
        with open(collection_path, encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, ncols=100)):
                columns = line.split('\t')
                doc_id = columns[0]

                text = []
                for col_idx in text_cols:
                    text.append(columns[col_idx].strip())
                doc_text = " ".join(text)

                if i % max_docs_per_file == 0:
                    if i > 0:
                        output_jsonl_file.close()
                    output_path = os.path.join(output_dir, 'docs{:02d}.json'.format(file_index))
                    output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                    file_index += 1
                output_dict = {'id': doc_id, 'contents': doc_text}
                output_jsonl_file.write(json.dumps(output_dict) + '\n')
        output_jsonl_file.close()


    def prepare_query(self, query_path, query_token_ids, query_token_weights, qid2index, tmp_query_dir):
        qids = []
        qcontents = []
        if query_token_ids is None:
            with open(query_path) as f:
                for i, line in enumerate(f):
                    qid, qcontent = line.split("\t")
                    qids.append(qid.strip())
                    qcontents.append(qcontent.strip())
        else:
            qindex2id = {v: k for k, v in qid2index.items()}
            if query_token_weights is not None:
                query_token_weights = self._quantize(query_token_weights)
                for i, token_id in enumerate(query_token_ids):
                    unique_token_id = defaultdict(list)
                    token = self.tokenizer.convert_ids_to_tokens(token_id)
                    for j, tok in enumerate(token):
                        if tok in self.stop_words:
                            continue
                        impact = query_token_weights[i, j]
                        if impact <= 0:
                            continue
                        unique_token_id[tok].append(impact)

                    new_token_list = []
                    for tok, weights in unique_token_id.items():
                        # use the max score
                        weight = max(weights)
                        # use replication to control token weight
                        new_token_list += [tok] * weight

                    qids.append(qindex2id[i].strip())
                    qcontents.append(" ".join(new_token_list))
            else:
                for i, token_id in enumerate(query_token_ids):
                    token = self.tokenizer.convert_ids_to_tokens(token_id)
                    new_token_list = []
                    for j, tok in enumerate(token):
                        if tok == -1:
                            continue
                        if tok in self.stop_words:
                            continue
                        new_token_list.append(tok)

                    qids.append(qindex2id[i].strip())
                    qcontents.append(" ".join(new_token_list))

        # needs splitting
        if len(qids) > 10000:
            query_split_dir = os.path.join(tmp_query_dir, "anserini_query_splits")
            if not isempty(query_split_dir):
                shutil.rmtree(query_split_dir)
            os.makedirs(query_split_dir, exist_ok=True)
            # split queries into shards because Anserini cannot deal with large query file
            for i, query in enumerate(qcontents):
                if i % 5000 == 0:
                    if i > 0:
                        g.close()
                    g = open(os.path.join(query_split_dir, f"{str(i // 5000)}.tsv"), "w")
                g.write("\t".join([qids[i], query]) + "\n")
            g.close()
            return True, query_split_dir

        else:
            if query_token_ids is not None:
                os.makedirs(tmp_query_dir, exist_ok=True)
                query_path = os.path.join(tmp_query_dir, "queries.token.tsv")
                with open(query_path, "w") as f:
                    for qid, qcontent in zip(qids, qcontents):
                        f.write("\t".join([qid, qcontent]) + "\n")

            return False, query_path



class AnseriniImpactIndex(BaseAnseriniIndex):
    """
    Anserini impact index.
    """
    def __init__(self, collection_path:str, collection_dir:str, index_dir:str) -> None:
        """
        Args:
            quantize_bit: the quantization bits
            tokenizer: transformers tokenizer
            subword_to_word: collect subwords to words, e.g. :func:`utils.index.subword_to_word_bert`
            collection_path: the collection file path
            collection_dir: the json collections folder generated by :func:`utils.index.AnseriniImpactIndex.fit` and :func:`utils.index.AnseriniBM25Index.fit`
            index_dir: the folder to store the lucene index file; usually :py:obj:`models.BaseModel.BaseModel.index_dir`
            reduce: method to aggregate the token weights to word weight; {first, mean, max}
        """
        super().__init__(collection_path, collection_dir, index_dir)


    def _quantize(self, token_weights, quantize_bit=None):
        token_weights = np.ceil(token_weights * 100).astype(int)
        return token_weights


    def fit(self, text_token_ids:np.ndarray, text_token_weights:np.ndarray, quantize_bit:int, tokenizer:Any, subword_to_word:Callable, stop_words:set, reduce:str, thread_num:int=32, enable_build_collection:bool=True, enable_build_index:bool=True, language:str="eng", **kwargs):
        """
        1. Collect tokens into words and create json collection.

        2. Construct the anserini index.

        Args:
            text_token_ids: array of [N, L]
            text_token_weights: array of [N, L, 1]
            enable_build_collection: if ``True``, rebuild the json collection
            enable_build_index: if ``True``, rebuild the anserini index
            lanugage: {eng, zh}
        """
        self.tokenizer = tokenizer
        self.stop_words = stop_words

        if enable_build_collection:
            # if quantize_bit > 0:
            #     max_impact = text_token_weights.max().item()
            #     # quantize
            #     scale = (1 << quantize_bit) / max_impact
            #     text_token_weights = (text_token_weights * scale).astype(np.int32)
            text_token_weights = self._quantize(text_token_weights, quantize_bit)

            # each thread creates one jsonl file
            text_num = len(text_token_ids)
            text_num_per_thread = text_num / thread_num

            arguments = []
            for i in range(thread_num):
                output_path = os.path.join(self.collection_dir, "docs{:02d}.json".format(i))
                start_idx = round(text_num_per_thread * i)
                end_idx = round(text_num_per_thread * (i+1))

                arguments.append((
                    text_token_ids[start_idx: end_idx],
                    text_token_weights[start_idx: end_idx],
                    start_idx,
                    output_path,
                    tokenizer,
                    subword_to_word,
                    stop_words,
                    reduce
                ))

            with mp.Pool(thread_num) as p:
                p.starmap(build_impact_collection, arguments)

        if enable_build_index:
            # subprocess.run(f"""
            #     python -m pyserini.index.lucene --collection JsonVectorCollection \
            #     --generator DefaultLuceneDocumentGenerator \
            #     --input {self.collection_dir} \
            #     --index {self.index_dir} \
            #     --threads 32 --impact --pretokenized \
            #     --language {language}
            #     """,
            #     shell=True
            # )
            subprocess.run(f"""
                sh anserini/target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
                -generator DefaultLuceneDocumentGenerator \
                -input {self.collection_dir} \
                -index {self.index_dir} \
                -threads 32 -impact -pretokenized \
                -language {language}
                """,
                shell=True
            )


    def _search(self, topics, output, hits, language):
        # subprocess.run(f"""
        #     python -m pyserini.search.lucene \
        #     --index {self.index_dir} \
        #     --topics {topics} \
        #     --output {output} --output-format msmarco \
        #     --threads 32 \
        #     --impact --pretokenized \
        #     --hits {hits} \
        #     --language {language}
        #     """,
        #     shell=True
        # )
        subprocess.run(f"""
            sh anserini/target/appassembler/bin/SearchCollection \
            -topicreader TsvString -format msmarco \
            -index {self.index_dir} \
            -topics {topics} \
            -output {output} \
            -threads 32 \
            -parallelism 4 \
            -impact -pretokenized \
            -hits {hits} \
            -language {language}
            """,
            shell=True
        )


    def search(self, query_token_ids, retrieval_result_path:str, hits:int, qid2index:ID_MAPPING, tid2index:ID_MAPPING, query_token_weights:Optional[np.ndarray]=None, query_path:Optional[str]=None, tmp_query_dir:Optional[str]=None, language:str="eng", verifier:Optional[BasePostVerifier]=None, **kwargs) -> RETRIEVAL_MAPPING:
        """
        Search by Anserini.

        Args:
            query_token_ids: the pretokenized query token ids
            query_token_weights: the weights of each token
            query_path: the raw query file path
            retrieval_result_path:
            hits: Top K
            qid2index: mapping from query id to query idx; generated by :py:mod:`scripts.preprocess`
            tid2index: mapping from document id to document idx
            query_path: the raw query file path
            tmp_query_path: the temperary file to save pretokenized query
            lanugage: {eng, zh}
            verifier: the verifier to post rank the hitted results
        """
        tmp_retrieval_result_path = f"{retrieval_result_path}.tmp"
        split, query_path_or_dir = self.prepare_query(query_path=query_path, query_token_ids=query_token_ids, query_token_weights=query_token_weights, qid2index=qid2index, tmp_query_dir=tmp_query_dir)

        if split:
            retrieval_result = {}
            for query_path in tqdm(os.listdir(query_path_or_dir), ncols=100, desc="Searching"):
                query_path = os.path.join(query_path_or_dir, query_path)
                self._search(query_path, tmp_retrieval_result_path, hits, language)
                res = self.convert_retrieval_result(tmp_retrieval_result_path, qid2index, tid2index, verifier=verifier)
                retrieval_result.update(res)

        else:
            self._search(query_path_or_dir, tmp_retrieval_result_path, hits, language)
            retrieval_result = self.convert_retrieval_result(tmp_retrieval_result_path, qid2index, tid2index, verifier=verifier)

        return retrieval_result



class AnseriniBM25Index(BaseAnseriniIndex):
    """
    Anserini BM25 index.
    """
    def __init__(self, collection_path, collection_dir, index_dir) -> None:
        """
        Args:
            collection_path: the collection file path
            collection_dir: the json collections folder generated by :func:`utils.index.AnseriniImpactIndex.fit` and :func:`utils.index.AnseriniBM25Index.fit`
            index_dir: the folder to store the lucene index file; usually :py:obj:`models.BaseModel.BaseModel.index_dir`
        """
        super().__init__(collection_path, collection_dir, index_dir)


    def fit(self, text_cols, text_token_ids:Optional[np.ndarray]=None, tokenizer:Any=None, thread_num:int=32, enable_build_collection:bool=True, enable_build_index:bool=True, language:str="eng", stop_words:Optional[set]=None, **kwargs):
        """
        1. Convert the TSV collection into json collection by :mod:`scripts.collection`.

        2. Construct the anserini index.

        Args:
            enable_build_collection: if ``True``, rebuild the json collection
            enable_build_index: if ``True``, rebuild the anserini index
            lanugage: {eng, zh}
        """
        self.tokenizer = tokenizer
        self.stop_words = stop_words

        if enable_build_collection:
            if not isempty(self.collection_dir):
                shutil.rmtree(self.collection_dir)
                os.makedirs(self.collection_dir)

            if text_token_ids is None:
                self.generate_collection(text_cols=text_cols)
            else:
                # build pretokenized collection
                # each thread creates one jsonl file
                text_num = len(text_token_ids)
                text_num_per_thread = text_num / thread_num

                arguments = []
                for i in range(thread_num):
                    output_path = os.path.join(self.collection_dir, "docs{:02d}.json".format(i))
                    start_idx = round(text_num_per_thread * i)
                    end_idx = round(text_num_per_thread * (i+1))

                    arguments.append((
                        text_token_ids[start_idx: end_idx],
                        start_idx,
                        output_path,
                        tokenizer,
                        stop_words,
                    ))

                with mp.Pool(thread_num) as p:
                    p.starmap(build_pretokenized_collection, arguments)

        if enable_build_index:
            if len(os.listdir(self.index_dir)) > 0:
                shutil.rmtree(self.index_dir)
                os.makedirs(self.index_dir)

            if text_token_ids is None:
                # subprocess.run(f"""
                #     python -m pyserini.index.lucene --collection JsonCollection \
                #     --generator DefaultLuceneDocumentGenerator \
                #     --input {self.collection_dir} \
                #     --index {self.index_dir} \
                #     --threads 32 \
                #     --storeDocvectors \
                #     --language {language}
                #     """,
                #     shell=True
                # )
                subprocess.run(f"""
                    sh anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
                    -generator DefaultLuceneDocumentGenerator \
                    -input {self.collection_dir} \
                    -index {self.index_dir} \
                    -threads 32 \
                    -storeDocvectors \
                    -language {language}
                    """,
                    shell=True
                )
            else:
                # subprocess.run(f"""
                #     python -m pyserini.index.lucene --collection JsonCollection \
                #     --generator DefaultLuceneDocumentGenerator \
                #     --input {self.collection_dir} \
                #     --index {self.index_dir} \
                #     --threads 32 \
                #     --storeDocvectors --pretokenized \
                #     --language {language}
                #     """,
                #     shell=True
                # )
                subprocess.run(f"""
                    sh anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
                    -generator DefaultLuceneDocumentGenerator \
                    -input {self.collection_dir} \
                    -index {self.index_dir} \
                    -threads 32 \
                    -storeDocvectors -pretokenized \
                    -language {language}
                    """,
                    shell=True
                )


    def _search(self, topics, output, k1, b, hits, language, pretokenized=False):
        if pretokenized:
            # subprocess.run(f"""
            #     python -m pyserini.search.lucene \
            #     --index {self.index_dir} \
            #     --topics {topics} \
            #     --output {output} --output-format msmarco \
            #     --threads 32 \
            #     --bm25 --k1 {k1} --b {b} \
            #     --hits {hits} --pretokenized \
            #     --language {language}
            #     """,
            #     shell=True
            # )
            subprocess.run(f"""
                sh anserini/target/appassembler/bin/SearchCollection \
                -topicreader TsvString -format msmarco \
                -index {self.index_dir} \
                -topics {topics} \
                -output {output} \
                -threads 32 \
                -parallelism 4 \
                -bm25 -bm25.k1 {k1} -bm25.b {b} \
                -hits {hits} -pretokenized \
                -language {language}
                """,
                shell=True
            )
        else:
            # subprocess.run(f"""
            #     python -m pyserini.search.lucene \
            #     --index {self.index_dir} \
            #     --topics {topics} \
            #     --output {output} --output-format msmarco \
            #     --threads 32 \
            #     --bm25 --k1 {k1} --b {b} \
            #     --hits {hits} \
            #     --language {language}
            #     """,
            #     shell=True
            # )
            subprocess.run(f"""
                sh anserini/target/appassembler/bin/SearchCollection \
                -topicreader TsvString -format msmarco \
                -index {self.index_dir} \
                -topics {topics} \
                -output {output} \
                -threads 32 \
                -parallelism 4 \
                -bm25 -bm25.k1 {k1} -bm25.b {b} \
                -hits {hits} \
                -language {language}
                """,
                shell=True
            )


    def search(self, query_path:str, retrieval_result_path:str, hits:int, qid2index:ID_MAPPING, tid2index:ID_MAPPING, query_token_ids:Optional[np.ndarray]=None, tmp_query_dir:Optional[str]=None, language:str="eng", k1:float=0.82, b:float=0.68, verifier:Optional[BasePostVerifier]=None, **kwargs) -> RETRIEVAL_MAPPING:
        """
        Search by Anserini.

        Args:
            query_path: the raw query file path
            retrieval_result_path:
            hits: Top K
            qid2index: mapping from query id to query idx; generated by :py:mod:`scripts.preprocess`
            tid2index: mapping from document id to document idx
            lanugage: {eng, zh}
            query_token_ids: the pretokenized query token ids
            tmp_query_path: the temperary file to save pretokenized query
            k1: BM25 parameter
            b: BM25 parameter
            verifier: the verifier to post rank the hitted results
        """
        tmp_retrieval_result_path = f"{retrieval_result_path}.tmp"
        split, query_path_or_dir = self.prepare_query(query_path=query_path, query_token_ids=query_token_ids, query_token_weights=None, qid2index=qid2index, tmp_query_dir=tmp_query_dir)

        if split:
            retrieval_result = {}
            for query_path in tqdm(os.listdir(query_path_or_dir), ncols=100, desc="Searching"):
                query_path = os.path.join(query_path_or_dir, query_path)
                self._search(query_path, tmp_retrieval_result_path, k1, b, hits, language, pretokenized=query_token_ids is not None)
                res = self.convert_retrieval_result(tmp_retrieval_result_path, qid2index, tid2index, verifier=verifier)
                retrieval_result.update(res)

        else:
            self._search(query_path_or_dir, tmp_retrieval_result_path, k1, b, hits, language, pretokenized=query_token_ids is not None)
            retrieval_result = self.convert_retrieval_result(tmp_retrieval_result_path, qid2index, tid2index, verifier=verifier)

        return retrieval_result



class BM25Index(BaseIndex):
    """
    Naive BM25 index.
    """
    def __init__(self) -> None:
        super().__init__()


    def fit(self, corpus, ids=None, stop_tokens:set=set(), verbose=True):
        if ids is None:
            ids = np.arange(len(corpus))
        elif isinstance(ids, list):
            ids = np.array(ids)

        dfs = defaultdict(int)
        tfs = []
        inverted_lists = defaultdict(list)
        doc_lengths = np.zeros(len(corpus), dtype=np.float32)

        if verbose:
            iterator = tqdm(corpus, ncols=100, leave=False, desc="Collecting DFs")
        else:
            iterator = corpus

        for i, doc in enumerate(iterator):
            if isinstance(doc, str):
                doc = doc.split(" ")

            df = {}
            tf = defaultdict(int)
            for token in doc:
                if token not in stop_tokens:
                    tf[token] += 1
                    df[token] = 1
            tfs.append(dict(tf))
            for token in df:
                dfs[token] += 1
                # store the doc offset in the inverted lists of the corresponding token
                inverted_lists[token].append(i)

            doc_lengths[i] = len(doc)

        self.dfs = dict(dfs)
        self.tfs = tfs
        self.doc_length = doc_lengths
        self.inverted_lists = {k: np.array(v) for k, v in inverted_lists.items()}
        self.ids = ids
        self.N = len(corpus)


    def search(self, queries, hits, k1=0.82, b=0.68):
        query_num, query_length = len(queries), len(queries[0])
        global_scores = np.zeros(self.N, dtype=np.float32)

        for i, query in enumerate(tqdm(queries, ncols=100, leave=False, desc="Searching")):
            if isinstance(query, str):
                query = query.split(" ")
            for token in query:
                try:
                    candidates = self.inverted_lists[token]
                # the token is not in the corpus
                except KeyError:
                    continue

                tfs = np.array([self.tfs[candidate][token] for candidate in candidates], dtype=np.float32)

                df = self.dfs[token]
                idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)

                candidate_scores = idf * (k1 + 1) * tfs / (tfs + k1 * (1 - b + b * self.doc_length[candidates]))
                global_scores[candidates] += candidate_scores

            topk_indices = np.argpartition(-global_scores, hits)[:hits]
            topk_doc_ids = self.ids[topk_indices]
            return topk_doc_ids



class BaseTrieIndexIndex(BaseIndex):
    """
    Basic class for trie-like index. Used in :class:`models.BaseModel.BaseGenerativeModel`.
    """
    def __init__(self, save_dir:str, pad_token_id:int) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, type(self).__name__.lower())
        self.pad_token_id = pad_token_id


    def fit(self, text_codes:np.ndarray, load_index:bool=False, save_index:bool=False, rebuild_index:bool=False, verbose:bool=True):
        """
        1. Build TrieIndex from codes;
        2. Save TrieIndex if necessary.

        Args:
            text_codes: the codes of texts
            load_index: if ``True``, load the existing trie
            save_index: if ``True``, force to save the new constructed trie
            rebuild_index: if ``False``, default to ``load_index=True`` and ``save_index=False``; if ``True``, default to ``load_index=False`` and ``save_index=False``
        """
        if load_index or not rebuild_index and os.path.exists(self.save_path):
            self.load()
            return
        elif rebuild_index or not os.path.exists(self.save_path):
            self.add(text_codes, verbose=verbose)
            if save_index or not rebuild_index:
                os.makedirs(self.save_dir, exist_ok=True)
                self.save()


class NULL:
    pass


class Node:
    def __init__(self, value=NULL):
        self.value = value
        self.children = dict()

    def __len__(self):
        """Return the number of keys in the subtree rooted at this node."""
        return int(self.value is not NULL) + sum(map(len, self.children.values()))

    def __repr__(self):
        return '(%s, {%s})' % (
            repr(self.value) if self.value is not NULL else "NULL",
            ', '.join('%r: %r' % t for t in self.children.items()))

    def __copy__(self):
        clone = self.__class__(self.value)
        clone_children = clone.children
        for key, child in self.children.items():
            clone_children[key] = child.__copy__()
        return clone

    def __getstate__(self):
        return self.value, self.children

    def __setstate__(self, state):
        self.value, self.children = state


class TrieIndex(BaseTrieIndexIndex):
    """
    TrieIndex Index.
    """
    def __init__(self, save_dir:str=".", pad_token_id:int=0):
        """
        Args:
            save_dir: the directory to save the trie index
            pad_token_id: will replace the -1 in the codes with ``pad_token_id``
        """
        super().__init__(save_dir, pad_token_id)
        self._root = Node()

    def _find(self, key):
        """
        Find the node corresponding to the key.

        Returns:
            Optional[Node]: a valid node if the key is stored, otherwise None
        """
        node = self._root
        for part in key:
            node = node.children.get(part)
            if node is None:
                break
        return node

    def __len__(self):
        return len(self._root)

    def __bool__(self):
        return self._root.value is not NULL or bool(self._root.children)

    def __iter__(self):
        return self.iterkeys()

    def __contains__(self, key):
        node = self._find(key)
        return node is not None and node.value is not NULL

    def __getitem__(self, key):
        node = self._find(key)
        if node is None or node.value is NULL:
            raise KeyError
        return node.value

    def __setitem__(self, key, value):
        """
        Insert Key in the trie; Append the value into the value list.
        """
        node = self._root
        factory = Node
        for part in key:
            next_node = node.children.get(part)
            if next_node is None:
                node = node.children.setdefault(part, factory())
            else:
                node = next_node
        if node.value is NULL:
            node.value = [value]
        else:
            node.value.append(value)

    def __delitem__(self, key):
        nodes_parts = []
        node = self._root
        for part in key:
            nodes_parts.append((node, part))
            node = node.children.get(part)
            if node is None:
                break
        if node is None or node.value is NULL:
            raise KeyError
        node.value = NULL
        while node.value is NULL and not node.children and nodes_parts:
            node, part = nodes_parts.pop()
            del node.children[part]

    def clear(self):
        self._root.children.clear()

    def copy(self):
        clone = copy(super(TrieIndex, self))
        clone._root = copy(self._root)  # pylint: disable=protected-access
        return clone

    def __repr__(self):
        return '%s({%s})' % (
            self.__class__.__name__,
            ', '.join('%r: %r' % t for t in self.iteritems()))

    #----- extended mapping API methods ----------------------------------------
    # pylint: disable=arguments-differ
    def keys(self, prefix=None):
        """Return a list of this trie's keys.
        :param prefix: If not None, return only the keys prefixed by ``prefix``.
        """
        return list(self.iterkeys(prefix))

    def values(self, prefix=None):
        """Return a list of this trie's values.
        :param prefix: If not None, return only the values associated with keys prefixed by ``prefix``.
        """
        return list(self.itervalues(prefix))

    def items(self, prefix=None):
        """Return a list of this trie's items (``(key,value)`` tuples).
        :param prefix: If not None, return only the items associated with keys prefixed by ``prefix``.
        """
        return list(self.iteritems(prefix))

    def iterkeys(self, prefix=None):
        """Return an iterator over this trie's keys.
        :param prefix: If not None, yield only the keys prefixed by ``prefix``.
        """
        return (key for key, value in self.iteritems(prefix))

    def itervalues(self, prefix=None):
        """Return an iterator over this trie's values.
        :param prefix: If not None, yield only the values associated with keys prefixed by ``prefix``.
        """
        def generator(node, null=NULL):
            if node.value is not null:
                yield node.value
            for child in node.children.values():
                for subresult in generator(child):
                    yield subresult
        if prefix is None:
            root = self._root
        else:
            root = self._find(prefix)
            if root is None:
                root = Node()
        return generator(root)

    def iteritems(self, prefix=None):
        """Return an iterator over this trie's items (``(key,value)`` tuples).
        :param prefix: If not None, yield only the items associated with keys prefixed by ``prefix``.
        """
        parts = []
        append = parts.append

        # pylint: disable=dangerous-default-value
        def generator(node, key_factory=tuple, parts=parts,
                      append=append, null=NULL):
            if node.value is not null:
                yield (key_factory(parts), node.value)
            for part, child in node.children.items():
                append(part)
                for subresult in generator(child):
                    yield subresult
                del parts[-1]

        root = self._root
        if prefix is not None:
            for part in prefix:
                append(part)
                root = root.children.get(part)
                if root is None:
                    root = Node()
                    break

        return generator(root)

    def get_next_keys(self, prefix:Union[list,np.ndarray]) -> list:
        """
        Returns valid key at the next position given the ``prefix``.
        """
        node = self._find(prefix)
        if node is None:
            return []
        keys = list(node.children.keys())
        return keys

    def add(self, sequences:Union[list[list],np.ndarray], ids:Union[list,np.ndarray]=None, verbose:bool=True):
        """
        Add a bulk of sequences into the trie.

        Args:
            sequences: usually the document codes generated by :func:`models.BaseModel.BaseModel.generate_code`
            ids: auxillary ids for each document
            verbose: print information
        """
        assert isinstance(sequences, np.ndarray)
        if verbose:
            self.logger.info("adding nodes to trie...")

        if verbose:
            bar = tqdm(sequences, ncols=100, leave=False)
        else:
            bar = sequences

        for i, sequence in enumerate(bar):
            if ids is not None:
                id = ids[i]
            else:
                id = i

            # replace -1 with pad token
            sequence[sequence == -1] = self.pad_token_id
            self[sequence] = id

    def save(self, save_path:Optional[str]=None):
        """
        Save the trie at ``save_path``.

        Args:
            save_path: if ``None``, the ``self.save_path`` will be used
        """
        os.makedirs(self.save_dir, exist_ok=True)
        if save_path is None:
            save_path = self.save_path
        self.logger.info(f"saving trie at {save_path}")
        save_pickle(self._root, save_path)

    def load(self, save_path:Optional[str]=None):
        """
        Load the trie from ``save_path``.

        Args:
            save_path: if ``None``, the ``self.save_path`` will be used
        """
        if save_path is None:
            save_path = self.save_path
        self.logger.info(f"loading trie from {save_path}")
        self._root = load_pickle(save_path)

    def inspect_structure(self, verbose=True) -> np.ndarray:
        """
        check the children number in each layer of the trie
        """
        key_list = [[0]]
        children_num_per_layer = []
        i = 0

        while len(key_list):
            children_num_per_layer.append(0)
            new_key_list = []
            for key in key_list:
                children = self.get_next_keys(key)
                if len(children):
                    children_num_per_layer[i] += len(children) / len(key_list)
                    for child in children:
                        new_key_list.append(key + [child])

            key_list = new_key_list.copy()
            i += 1

        children_num_per_layer = np.array(children_num_per_layer)
        if verbose:
            self.logger.info(f"children number per layer: {np.round(children_num_per_layer, 3)}")
            self.logger.info(f"all leaves count: {np.round(np.prod(children_num_per_layer[:-1]), 1)}")
        return children_num_per_layer



def count_stepwise_code_collision(codes, verbose=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Count code collision at each step, both as sequences and as sets.
    """
    import pandas as pd
    code_set_collision_number_per_step = np.zeros(codes.shape[-1] - 1)
    code_sequence_collision_number_per_step = np.zeros(codes.shape[-1] - 1)

    for cutoff in range(len(code_set_collision_number_per_step)):
        codes_sub = codes[:, :cutoff + 1]
        codes_sub_set = np.sort(codes_sub, axis=-1)

        codes_sub = pd.DataFrame(codes_sub)
        codes_sub_set = pd.DataFrame(codes_sub_set)

        overlap = codes_sub.duplicated().sum()
        overlap_set = codes_sub_set.duplicated().sum()

        code_sequence_collision_number_per_step[cutoff] = overlap
        code_set_collision_number_per_step[cutoff] = overlap_set

    if verbose:
        print(f"code sequence collision at each step: {np.round(code_sequence_collision_number_per_step, 3)}")
        print(f"code set collision at each step: {np.round(code_set_collision_number_per_step, 3)}")

    return code_sequence_collision_number_per_step, code_set_collision_number_per_step


def permute_code(codes:np.ndarray, rule="rotate", level:int=5, k:int=5) -> list[np.ndarray]:
    """
    Reorder the first ``level`` codes.

    Args:
        codes: text codes generated by :func:`models.BaseModel.BaseModel.generate_code`
        rule: how to permute codes
        level: how many steps to permute
        k: how many replicas to return

    Returns:
        list[np.ndarray]: a series of permuted codes
    """
    new_codes = []
    dest = codes.copy()
    if rule == "rotate":
        candidate = codes[:, 1: level + 1]
        for i in range(1, k + 1, 1):
            rotate_candidate = np.roll(candidate, i, axis=-1)
            dest[:, 1: level + 1] = rotate_candidate
            new_codes.append(dest.copy())
    return new_codes



class FlatVerifier(BasePostVerifier):
    """
    Post verify ``retrieval_result`` by the brute-force ranking from ``config.verifier_src``.
    """
    def __init__(self, query_embeddings:np.ndarray, text_embeddings:np.ndarray, hits:int=1000, device:DEVICE="cpu", **kwargs) -> None:
        super().__init__()
        self.query_embeddings = query_embeddings
        self.text_embeddings = text_embeddings
        self.hits = hits
        self.device = device


    def __call__(self, qidx:int, tindices:INDICES):
        if isinstance(tindices, torch.Tensor):
            tindices = tindices.cpu().numpy()
        elif isinstance(tindices, list):
            tindices = np.array(tindices)
        elif isinstance(tindices, np.ndarray):
            pass
        else:
            raise NotImplementedError(f"Unsupported tindices type {type(tindices)}")
        query_embedding = torch.as_tensor(self.query_embeddings[qidx], device=self.device)
        text_embedding = torch.as_tensor(self.text_embeddings[tindices], device=self.device)
        # using numpy arrays is not efficient
        # tindices = tindices.cpu().numpy()
        # query_embedding = self.query_embeddings[qidx]
        # text_embedding = self.text_embeddings[tindices]

        scores = text_embedding @ query_embedding
        if len(scores) < self.hits:
            topk_score, topk_idx = scores.sort(dim=-1, descending=True)
            # topk_idx = np.argsort(scores, axis=-1)[::-1]
            # topk_score = scores[topk_idx]

            topk_tindices = tindices[topk_idx.tolist()]
        else:
            topk_score, topk_idx = scores.topk(self.hits)  # k
            # topk_idx = np.argpartition(-scores, self.hits)[:self.hits]
            # topk_score = scores[topk_idx]

            topk_tindices = tindices[topk_idx.tolist()]

        return topk_tindices, topk_score



class PQVerifier(BasePostVerifier):
    """
    Post verify ``retrieval_result`` by the PQ ranking from ``config.verifier_src``.
    """
    def __init__(self, query_embeddings:np.ndarray, pq_index:faiss.Index, start_text_idx:int, end_text_idx:int, hits:int=1000, **kwargs) -> None:
        """
        Args:
            start_text_idx: the offset of this shard
            end_text_idx: the ending offset
        """
        super().__init__()
        self.hits = hits

        if isinstance(pq_index, faiss.IndexPreTransform):
            vector_transform = faiss.downcast_VectorTransform(pq_index.chain.at(0))
            vt = faiss.vector_to_array(vector_transform.A).reshape(vector_transform.d_out, vector_transform.d_in)
            query_embeddings = np.matmul(query_embeddings, vt.T)
            pq = faiss.downcast_index(pq_index.index)
        elif isinstance(pq_index, faiss.IndexPQ):
            pq = pq_index
        else:
            raise ValueError(f"Provide PQ index!")

        M, ksub, dsub = pq.pq.M, pq.pq.ksub, pq.pq.dsub
        self.M, self.dsub = M, dsub
        self.codebook = faiss.vector_to_array(pq.pq.centroids).reshape(M, ksub, dsub)
        self.codes = faiss.vector_to_array(pq.codes).reshape(pq.ntotal, pq.pq.code_size)[start_text_idx: end_text_idx]
        self.query_embeddings = query_embeddings


    def __call__(self, qidx:int, tindices:INDICES):
        if isinstance(tindices, torch.Tensor):
            tindices = tindices.cpu().numpy()
        elif isinstance(tindices, list):
            tindices = np.array(tindices)
        else:
            raise NotImplementedError(f"Unsupported tindices type {type(tindices)}")

        text_code = self.codes[tindices]    # N, M

        tmp_range = np.arange(text_code.shape[1])   # M
        tmp_range = np.tile(tmp_range, len(text_code)).reshape(-1)
        sim_table = np.matmul(self.codebook, self.query_embeddings[qidx].reshape(self.M, self.dsub, 1)).squeeze(-1)        # M, ksub
        sim = sim_table[tmp_range, text_code.reshape(-1)].reshape(text_code.shape)                       # N, M
        scores = sim.sum(axis=-1)

        if len(scores) <= self.hits:
            topk_idx = np.argsort(scores, axis=-1)[::-1]
            topk_score = scores[topk_idx]
            topk_tindices = tindices[topk_idx]
        else:
            topk_idx = np.argpartition(-scores, self.hits)[:self.hits]
            topk_score = scores[topk_idx]
            topk_tindices = tindices[topk_idx]

        return topk_tindices, topk_score



def merge_retrieval_result(start_query_idx:int, end_query_idx:int, text_num:int, retrieval_results:list[RETRIEVAL_MAPPING]) -> tuple[RETRIEVAL_MAPPING, np.ndarray, np.ndarray]:
    """
    Merge a list of retrieval results into one.

    Args:
        start_query_idx: the starting offset in ``retrieval_results[i]``
        end_query_idx: the ending offset in ``retrieval_results[i]``
        retrieval_results: a list of retrieval results

    Returns:
        the merged retrieval result
        the ids of all hitted text, array of [n]
        the inverse index of all hitted text, array of [N]
    """
    unified_retrieval_result = {}
    all_hitted_text_id = set()

    for qidx in tqdm(range(start_query_idx, end_query_idx), desc="Collecting Hits", ncols=100, leave=False, position=0):
        merged_res = set()
        for retrieval_result in retrieval_results:
            # return empty list if qidx not exists
            # only append id, discard score
            if isinstance(next(iter(retrieval_result.values()))[0], tuple):
                res_with_score = retrieval_result.get(qidx, [])
                merged_res.update([x[0] for x in res_with_score])
            else:
                merged_res.update(retrieval_result.get(qidx, []))
        # remove -1 from faiss index
        merged_res.discard(-1)
        # skip the query_idx if no retrieval result
        # useful when distributed verification
        if len(merged_res) == 0:
            continue

        all_hitted_text_id.update(merged_res)
        unified_retrieval_result[qidx] = list(merged_res)

    hitted_ids = np.asarray(list(all_hitted_text_id), dtype=np.int32)     # D
    id2index = np.zeros(text_num, dtype=np.int32) - 1              # N
    id2index[hitted_ids] = np.arange(len(hitted_ids))
    return unified_retrieval_result, hitted_ids, id2index


def pq_quantize(codes:np.ndarray, centroids:np.ndarray) -> np.ndarray:
    """
    Reconstruct the embedding from PQ.

    Args:
        codes: array of [B, M]
        centroids: array of [M, ksub, dsub]

    Returns:
        Reconstructed embedding of [B, M*dsub]
    """
    # codes: bs, M
    M = codes.shape[1]
    if isinstance(codes, torch.Tensor):
        assert isinstance(centroids, torch.Tensor)
        first_indices = torch.arange(M).to(codes.device)
        first_indices = first_indices.expand(*codes.shape).reshape(-1)
        quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)
    elif isinstance(codes, np.ndarray):
        if isinstance(centroids, torch.Tensor):
            centroids = centroids.detach().cpu().numpy()
        first_indices = np.arange(M)
        first_indices = np.tile(first_indices, len(codes))
        quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)
    else:
        raise NotImplementedError()
    return quant_embeds


def augment_xb(xb, phi=None):
    norms = (xb ** 2).sum(1)
    if phi is None:
        phi = norms.max()
    extracol = np.sqrt(phi - norms)
    return np.hstack((xb, extracol.reshape(-1, 1)))


def augment_xq(xq):
    extracol = np.zeros(len(xq), dtype=np.float32)
    return np.hstack((xq, extracol.reshape(-1, 1)))


def build_inverted_lists(args):
    """
    Build inverted lists for :class:`utils.index.BaseInvertedIndex`.
    """
    token_ids, start_text_idx, token_num, stop_token_ids = args

    text_idx_inverted_lists = [[] for _ in range(token_num)]
    token_idx_inverted_lists = [[] for _ in range(token_num)]

    for text_idx, token_ids in enumerate(tqdm(token_ids, ncols=100, leave=False)):
        for token_idx, token_id in enumerate(token_ids):
            if token_id == -1 or token_id in stop_token_ids:
                continue
            # save the token's position (text_idx, token_idx) in the inverted list

            text_idx_inverted_lists[token_id].append(start_text_idx + text_idx)
            token_idx_inverted_lists[token_id].append(token_idx)

    return text_idx_inverted_lists, token_idx_inverted_lists


def subword_to_word_bert(x):
    """
    Returns:
        is_subword: True for a subword and False for valid word
        word: the filtered subword or the word itself
    """
    if x.startswith("##"):
        return True, x[2:]
    else:
        return False, x


def convert_tokens_to_words(tokens, subword_to_word, scores=None, reduce="max"):
    """
    transform the tokens output by tokenizer to words (connecting subwords)
    Returns:
        words: list of words, without stop_words
    """
    words = []
    word_scores = []
    if scores is None:
        scores = [-1] * len(tokens)

    for i, tok in enumerate(tokens):
        is_subword, word = subword_to_word(tok)

        if is_subword:
            # joining subword with the previous word
            words[-1] += word
            # merging the token score
            word_scores[-1].append(scores[i])

        else:
            # before appending new tokens, finalize the last word score
            if len(words) > 0:
                if reduce == "max":
                    word_scores[-1] = max(word_scores[-1])
                elif reduce == "mean":
                    word_scores[-1] = sum(word_scores[-1]) / len(word_scores[-1])
                elif reduce == "sum":
                    word_scores[-1] = sum(word_scores[-1])
                elif reduce == "first":
                    word_scores[-1] = word_scores[-1][0]
                else:
                    raise NotImplementedError

            words.append(word)
            word_scores.append([scores[i]])

    # deal with the last unfinalized word score
    if reduce == "max":
        word_scores[-1] = max(word_scores[-1])
    elif reduce == "mean":
        word_scores[-1] = sum(word_scores[-1]) / len(word_scores[-1])
    elif reduce == "sum":
        word_scores[-1] = sum(word_scores[-1])
    elif reduce == "first":
        word_scores[-1] = word_scores[-1][0]
    else:
        raise NotImplementedError

    return words, word_scores


def build_impact_collection(token_ids, token_scores, text_start_idx, output_json_path, tokenizer, subword_to_word, stop_words, reduce):
    """
    generate jsonl files in multiple threads
    """
    with open(output_json_path, "w") as g:
        for idx, token_id in enumerate(tqdm(token_ids, desc="Generating JSONs", ncols=100, leave=False)):
            tokens = tokenizer.convert_ids_to_tokens(token_id)
            scores = token_scores[idx]

            dic = {}
            vector = {}
            word_score_dict = defaultdict(list)

            if subword_to_word is not None:
                words, scores = convert_tokens_to_words(tokens, subword_to_word, scores=scores, reduce=reduce)
            else:
                words = tokens

            for word, score in zip(words, scores):
                # only index the word with positive impact
                if score <= 0:
                    continue
                if word in stop_words:
                    continue
                word_score_dict[word].append(score)

            for word, score in word_score_dict.items():
                # use the max score of a word as its impact
                score = max(score)
                vector[word] = score

            dic["id"] = idx + text_start_idx
            dic["contents"] = ""
            dic["vector"] = vector

            g.write(json.dumps(dic) + "\n")


def build_pretokenized_collection(token_ids, text_start_idx, output_json_path, tokenizer, stop_words):
    """
    generate jsonl files in multiple threads
    """
    with open(output_json_path, "w") as g:
        for idx, token_id in enumerate(tqdm(token_ids, desc="Generating JSONs", ncols=100, leave=False)):
            filtered_token_id = token_id[token_id != -1]
            tokens = tokenizer.convert_ids_to_tokens(filtered_token_id, skip_special_tokens=True)
            if isinstance(stop_words, set):
                tokens = [x for x in tokens if x not in stop_words]

            doc = {}
            doc["id"] = idx + text_start_idx
            doc["contents"] = " ".join(tokens)
            g.write(json.dumps(doc) + "\n")



INVERTED_INDEX_MAP = {
    "invvec": InvertedVectorIndex,
    "invhit": InvertedHitIndex,
}


ANSERINI_INDEX_MAP = {
    "bm25": AnseriniBM25Index,
    "impact-tok": AnseriniImpactIndex,
    "impact-word": AnseriniImpactIndex,
}


VERIFIER_MAP = {
    "pq": PQVerifier,
    "flat": FlatVerifier,
}


TRIE_INDEX_MAP = {
    "trie": TrieIndex,
}

SUBWORD_TO_WORD_FN = {
    "bert": subword_to_word_bert,
}

