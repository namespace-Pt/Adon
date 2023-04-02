import os
import json
import torch
import faiss
import time
import shutil
import subprocess
import numpy as np
import multiprocessing as mp
from torch_scatter import scatter_max
from transformers import T5ForConditionalGeneration
from copy import copy
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict, OrderedDict
from .util import save_pickle, load_pickle, makedirs, isempty, synchronize, mean, mean_len, MasterLogger
from .typings import *



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
        Get the database of the index.
        """
        xb = faiss.rev_swig_ptr(faiss.downcast_index(index).get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)
        return xb

    @staticmethod
    def get_pq_codebook(pq:faiss.ProductQuantizer) -> np.ndarray:
        """
        Get the codebook of the pq.

        Args:
            pq: faiss ProductQuantizer instance
        """
        return faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)

    @staticmethod
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



class BaseInvertedIndex(BaseIndex):
    """
    Base class of the Inverted Indexes
    """
    def __init__(self, text_num:int, token_num:int, device:DEVICE, rank:int, save_dir:str, special_token_ids:set=set()):
        """
        Args:
            text_num: the number of documents
            token_num: the number of tokens
            device
            rank: current global process rank
            save_dir: the directory for inverted lists
            special_token_ids: the special token ids in PLM tokenizer, usually can be obtained by :py:obj:``utils.manager.Manager.config.special_token_ids``
        """
        super().__init__()
        self.text_num = text_num
        self.token_num = token_num
        self.device = torch.device(device)
        self.save_dir = save_dir

        self.special_token_ids = special_token_ids

        self.text_idx_inverted_lists_path = os.path.join(save_dir, f"text_idx_inverted_lists_{rank}.pt")
        self.token_idx_inverted_lists_path = os.path.join(save_dir, f"token_idx_inverted_lists_{rank}.pt")


    def fit(self, text_token_ids:np.ndarray, text_embeddings:np.ndarray, load_index:bool=True, save_index:bool=True, threads:int=16, shards:int=32, posting_prune:float=0.0, start_text_idx:int=0):
        """
        1. Populate the inverted lists;

        2. Save the inverted lists if necessary;

        3. Move the posting lists to gpu if necessary;

        4. Sort each posting list by the descending token weights if necessary.

        Args:
            text_token_ids: the token ids of each document, array of [N, L]
            text_embeddings: the token weights/vectors, array of [N, L, D]
            load_index: if ``True``, load the posting lists from ``save_dir``
            save_index: if ``True``, save the posting lists to ``save_dir``
            threads: how many processes to put in the pool for parallel building inverted index
            shards: how many pieces to shard the text_token_ids
            posting_prune: what percentile of the inverted lists are kept
            start_text_idx: the offset of the text_idx on current process
        """
        self.logger.info("fitting inverted index...")

        # load the posting lists when we specify load_index, or when the model do not need to rebuild index
        if load_index and os.path.exists(self.text_idx_inverted_lists_path):
            self.text_idx_inverted_lists = torch.load(self.text_idx_inverted_lists_path, map_location=self.device)
            self.token_idx_inverted_lists = torch.load(self.token_idx_inverted_lists_path, map_location=self.device)
        # construct posting lists when the model needs to rebuild index, or when the posting lists doesn't exist
        else:
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
                    # donot impose uniqueness in each inverted list
                    False
                ))

            text_idx_inverted_lists = [[] for _ in range(self.token_num)]
            token_idx_inverted_lists = [[] for _ in range(self.token_num)]
            with mp.Pool(threads) as p:
                # imap returns an iterator
                outputs = p.imap(build_inverted_lists, arguments)
                self.logger.info("merging shards...")
                for inv_lists_pair in tqdm(outputs, total=shards, ncols=100, leave=False):
                    text_idx_inverted_list = inv_lists_pair[0]
                    token_idx_inverted_list = inv_lists_pair[1]
                    for j in range(self.token_num):
                        text_idx_inverted_lists[j].extend(text_idx_inverted_list[j])
                        token_idx_inverted_lists[j].extend(token_idx_inverted_list[j])
                
            synchronize()
            self.logger.info("packing posting lists...")

            for i in tqdm(range(self.token_num), ncols=100, leave=False):
                text_indices = text_idx_inverted_lists[i]
                if len(text_indices):
                    text_idx_inverted_lists[i] = torch.tensor(text_indices, device=self.device, dtype=torch.int32)
                    token_idx_inverted_lists[i] = torch.tensor(token_idx_inverted_lists[i], device=self.device, dtype=torch.int16)

            # save the posting lists when we specify save_index or when the model doesn't need to rebuild index every time
            if save_index:
                self.logger.info(f"saving index at {self.save_dir}...")
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(text_idx_inverted_lists, self.text_idx_inverted_lists_path)
                torch.save(token_idx_inverted_lists, self.token_idx_inverted_lists_path)
            self.text_idx_inverted_lists = text_idx_inverted_lists
            self.token_idx_inverted_lists = token_idx_inverted_lists

            synchronize()

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def search(self, query_token_ids:np.ndarray, eval_posting_length:bool=False, query_start_idx:int=0, verifier:Optional[BasePostVerifier]=None, **kwargs) -> tuple[RETRIEVAL_MAPPING, Optional[np.array]]:
        """
        Search the inverted index. Recall all documents hit by query_token_ids in the inverted index.

        Args:
            query_token_ids: array of [M, LQ]
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
            if not query_embeddings.flags.writeable:
                query_embeddings = query_embeddings.copy()
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


    def prepare_query(self, query_path, query_token_ids, query_token_weights, qid2index, tmp_query_dir):
        """
        Generate temperary query file for anserini.

        #. convert tokens to words if impact-word index

        #. split query into multiple files if there are two many
        """
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
            # all weights is 1 means no weights; quantizing this case causes repetitive query tokens
            if query_token_weights is not None and not (query_token_weights == 1).all():
                query_token_weights = self._quantize(query_token_weights)

            for i, token_ids in enumerate(query_token_ids):
                word_score_dict = defaultdict(list)
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                if query_token_weights is not None:
                    scores = query_token_weights[i]
                else:
                    # in case only pretokenize, no weights
                    scores = [1] * len(tokens)

                if hasattr(self, "subword_to_word") and self.subword_to_word is not None:
                    words, scores = convert_tokens_to_words(tokens, self.subword_to_word, scores=scores, reduce=self.reduce)
                else:
                    words = tokens

                for word, score in zip(words, scores):
                    # only index the word with positive impact
                    if score <= 0:
                        continue
                    if word in self.stop_words:
                        continue
                    word_score_dict[word].append(score)

                weighted_words = []
                for word, score in word_score_dict.items():
                    # use the max score
                    score = max(score)
                    # use replication to control token weight
                    weighted_words += [word] * int(score)

                qids.append(qindex2id[i].strip())
                qcontents.append(" ".join(weighted_words))

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
        if quantize_bit is not None:
            scale = (1<<quantize_bit) / token_weights.max()
        else:
            scale = 100
        token_weights = np.ceil(token_weights * scale).astype(int)
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
        # involked in prepare_query
        self.subword_to_word = subword_to_word
        self.reduce = reduce

        if enable_build_collection:
            self.logger.info("building collections...")

            if not isempty(self.collection_dir):
                shutil.rmtree(self.collection_dir)
                os.makedirs(self.collection_dir)

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
            if len(os.listdir(self.index_dir)) > 0:
                shutil.rmtree(self.index_dir)
                os.makedirs(self.index_dir)

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
            self.logger.info("building collections...")

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
    def __init__(self) -> None:
        """
        Naive BM25 index.
        """
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


class TrieIndex(BaseIndex):
    """
    TrieIndex Index.
    """
    def __init__(self, rank:int=0, save_dir:str=".", save_name:Optional[str]=None, **kwargs):
        """
        Args:
            save_dir: the directory to save the trie index
            pad_token_id: will replace the -1 in the codes with ``pad_token_id``
        """
        super().__init__()
        self.rank = rank
        self.save_dir = save_dir
        if save_name is None:
            self.save_path = os.path.join(save_dir, type(self).__name__.lower())
        else:
            self.save_path = os.path.join(save_dir, save_name)
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

    def __iter__(self):
        return self.iterkeys()

    def __contains__(self, key):
        node = self._find(key)
        return node is not None and node.value is not NULL

    def __getitem__(self, key):
        node = self._find(key)
        if node is None or node.value is NULL:
            raise KeyError(f"Invalid key {key}")
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

        def generator(node, key_factory=list, parts=parts,
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

    def get_valid_tokens(self, prefix:Union[list,np.ndarray]) -> list:
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

            sequence = sequence[sequence != -1].copy()
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
    
    def fit(self, text_codes:np.ndarray, load_index:bool=False, save_index:bool=False, verbose:bool=True, **kwargs):
        """
        1. Build TrieIndex from codes;
        2. Save TrieIndex if necessary.

        Args:
            text_codes: the codes of texts
            load_index: if ``True``, load the existing trie
            save_index: if ``True``, force to save the new constructed trie
            rebuild_index: if ``False``, default to ``load_index=True`` and ``save_index=False``; if ``True``, default to ``load_index=False`` and ``save_index=False``
        """
        if load_index and os.path.exists(self.save_path):
            self.load()

        else:
            if self.rank == 0:
                self.add(text_codes, verbose=verbose)            
                os.makedirs(self.save_dir, exist_ok=True)
                self.save()
            synchronize()
            self.load()

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
                children = self.get_valid_tokens(key)
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



class WordSetIndex(BaseIndex):
    def __init__(self, save_dir, sep_token_id=None, pad_token_id=0, eos_token_id=1, rank=0, **kwargs) -> None:
        """
        Construct inverted index. Search with set intersection.

        Args:
            save_dir: folder to save inverted index
            sep_token_id: the token id that separates words
        """
        super().__init__()
        self.rank = rank
        self.sep_token_id = sep_token_id
        self.save_dir = save_dir
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
    

    def fit(self, text_codes:np.ndarray, tokenizer=None, separator=" ", load_index:bool=True, threads:int=10, shards:int=32, **kwargs):
        """ Build the inverted lists.
        """
        self.logger.info("fitting inverted index...")
        docs_path = os.path.join(self.save_dir, "docs.npy")
        inverse_vocab_path = os.path.join(self.save_dir, "inverse_vocab.npy")
        inverted_lists_path = os.path.join(self.save_dir, "inverted_lists.npy")
        vocab = TrieIndex(save_dir=self.save_dir, save_name="vocab")

        if load_index and os.path.exists(inverted_lists_path):
            self.docs = np.load(docs_path)
            self.inverse_vocab = np.load(inverse_vocab_path)
            self.inverted_lists = np.load(inverted_lists_path, allow_pickle=True)
            vocab.load()
            self.vocab = vocab

        else:
            if self.rank == 0:
                word_idx = 0
                max_subword_count = 0
                max_word_count = 0
                
                inverse_vocab = []

                docs = []
                inverted_lists = []

                # strip off the leading 0
                for i, text_code in enumerate(tqdm(text_codes[:, 1:], ncols=100)):
                    if separator == " ":
                        text = tokenizer.decode(text_code[text_code != -1], skip_special_tokens=True)
                        # currently only support space-separated languages
                        words = text.split(separator)

                        for j, word in enumerate(words):
                            words[j] = tokenizer.encode(word, add_special_tokens=False)
                    
                    else:
                        # collect subwords of a word (words are separated by sep_token_id in text_codes)
                        word = []
                        words = []
                        for c in text_code:
                            if c == self.eos_token_id:
                                break
                            word.append(c)
                            if c == self.sep_token_id:
                                words.append(word.copy())
                                word.clear()
                        
                    # collect word ids of a document
                    doc = []
                    for word in words:
                        if len(word) > max_subword_count:
                            max_subword_count = len(word)
                        try:
                            word_id = vocab[word][0]
                        except KeyError:
                            word_id = word_idx
                            vocab[word] = word_id
                            # very important to add the copy here to make sure the word is deep copied
                            inverse_vocab.append(word.copy())
                            # create new inverted lists for this word
                            inverted_lists.append([])
                            word_idx += 1
                        # since the trie automatically saves the word id in a list, just slice it out
                        doc.append(word_id)
                        # add to inverted lists
                        inverted_lists[word_id].append(i)
                        word.clear()

                    if len(doc) > max_word_count:
                        max_word_count = len(doc)
                    docs.append(doc)

                inverted_lists = np.array([np.array(x, dtype=np.int32) for x in inverted_lists], dtype=object)
                # pad the subwords so that all of them are equal in length
                inverse_vocab = np.array([x + [-1] * (max_subword_count - len(x)) for x in inverse_vocab], dtype=np.int32)
                # there are by default the same number of words within each document
                docs = np.array([x + [-1] * (max_word_count - len(x)) for x in docs], dtype=np.int32)

                # always save the inverted lists because other processes want to load it
                self.logger.info(f"saving index at {self.save_dir}...")
                os.makedirs(self.save_dir, exist_ok=True)

                np.save(docs_path, docs)
                np.save(inverse_vocab_path, inverse_vocab)
                np.save(inverted_lists_path, inverted_lists, allow_pickle=True)
                vocab.save()
                
            synchronize()
            self.docs = np.load(docs_path)
            self.inverse_vocab = np.load(inverse_vocab_path)
            self.inverted_lists = np.load(inverted_lists_path, allow_pickle=True)
            vocab.load()
            self.vocab = vocab
        
        # the number of subwords of each word
        self.subword_num = (self.inverse_vocab != -1).sum(-1)

    def get_valid_tokens(self, prev_text_indices:np.ndarray, prev_words:np.ndarray, prefix:Optional[np.ndarray]):
        """
        Get a list of valid tokens for next step generation.

        prev_text_indices: indices of the texts that contain all previously generated words
        prev_words: previously generated words
        prefix: the current word prefix
        """
        if prev_text_indices is not None:
            # find valid words
            valid_words = self.docs[prev_text_indices].reshape(-1)
            # get unique words
            valid_words = np.unique(valid_words[valid_words != -1])
            # forbid previously generated words
            valid_words = np.setdiff1d(valid_words, prev_words, assume_unique=True)
            # return eos in case there are no more words to be decoded
            if len(valid_words) == 0:
                return [self.eos_token_id]
            
            valid_word_tokens = self.inverse_vocab[valid_words]
        else:
            valid_word_tokens = self.inverse_vocab
        
        if prefix is None:
            valid_tokens = valid_word_tokens[:, 0]
        else:
            if prev_text_indices is not None:
                prefix_len = len(prefix)
                # find words with same prefix
                valid_word_idx = (valid_word_tokens[:, :prefix_len] == prefix).sum(-1) == prefix_len
                # these tokens are valid
                valid_tokens = valid_word_tokens[valid_word_idx][:, prefix_len]
            else:
                valid_tokens = self.vocab.get_valid_tokens(prefix)
        
        return valid_tokens


    def get_valid_tokens_from_doc(self, text_idx, prev_words, prefix):
        """
        Get a list of valid tokens for next step generation. The tokens are from a given text.
        """
        doc = self.docs[text_idx]
        # remove padding words
        valid_words = doc[doc != -1]
        # remove duplicated words
        if prev_words is not None:
            valid_words = np.setdiff1d(valid_words, prev_words, assume_unique=True)
            if len(valid_words) == 0:
                return [self.eos_token_id]

        valid_word_tokens = self.inverse_vocab[valid_words]

        if prefix is None:
            # forbid previously generated words
            # return empty token set in case there are no more valid words
            # get the first token of all valid words
            return valid_word_tokens[:, 0]
        else:
            prefix_len = len(prefix)
            # find words with same prefix
            valid_word_idx = (valid_word_tokens[:, :prefix_len] == prefix).sum(-1) == prefix_len
            valid_tokens = valid_word_tokens[valid_word_idx][:, prefix_len]
            return valid_tokens


    def get_word(self, tokens:np.ndarray):
        """
        Get the word corresponding to the tokens. If the tokens do not represent a valid word, return None instead.
        """
        # NOTE: add [0] because the word idxs are stored in list
        try: 
            word = self.vocab[tokens][0]
        except KeyError:
            word = None
        
        return word


    def get_text(self, word, prev_text_indices=None):
        """
        Get indices of the text that contain the word. If prev_text_indices is provided, the intersection is returned.
        """
        text_indices = self.inverted_lists[word]
        if prev_text_indices is not None:
            text_indices = np.intersect1d(text_indices, prev_text_indices, assume_unique=True)
        return text_indices
    

    def get_text_from_scratch(self, words):
        """
        Get text indices
        """
        for i, word in enumerate(words):
            if i == 0:
                text_indices = self.get_text(word)
            else:
                text_indices = np.intersect1d(text_indices, self.get_text(word))
        return text_indices

    def reconstruct(self, text_idx:int, prev_words:np.ndarray, prefix:Optional[np.ndarray]=None):
        """
        Reconstruct the keyword sequence of text_idx with the prev_words and prefix putting at the head.

        Returns:
            tokens(list): the reconstructed token sequence, with the prev_words placing at the head
            append_idx(int): the index of the start of the reconstructed tokens
        """
        words = self.docs[text_idx]
        words = words[words != -1]
        nonoverlap = (words[:, None] == prev_words).sum(-1) == 0
        tokens_nonoverlap = self.inverse_vocab[words[nonoverlap]]

        if prefix is not None:
            prefix_match = ((tokens_nonoverlap[:, :len(prefix)] == prefix).all(-1)).reshape(-1)
            # slice out the first matching word
            prefix_match_indices = np.argwhere(prefix_match)[:, 0]
            try:
                prefix_match_idx = prefix_match_indices[0]
            except IndexError:
                return None, None
            prefix_match[prefix_match_indices[1:]] = False

            tokens_prefix_match = tokens_nonoverlap[prefix_match_idx]
            tokens_prefix_match = tokens_prefix_match[tokens_prefix_match != -1].tolist()

            tokens_nonoverlap = tokens_nonoverlap[~prefix_match].reshape(-1)
            tokens_nonoverlap = tokens_nonoverlap[tokens_nonoverlap != -1].tolist()

            tokens_overlap = self.inverse_vocab[prev_words].reshape(-1)
            tokens_overlap = tokens_overlap[tokens_overlap != -1].tolist()
            tokens = [self.pad_token_id] + tokens_overlap + tokens_prefix_match + tokens_nonoverlap + [self.eos_token_id]

            return tokens, len(tokens_overlap) + len(prefix) + 1

        else:
            tokens_nonoverlap = tokens_nonoverlap[tokens_nonoverlap != -1].tolist()

            tokens_overlap = self.inverse_vocab[prev_words].reshape(-1)
            tokens_overlap = tokens_overlap[tokens_overlap != -1].tolist()
            tokens = [self.pad_token_id] + tokens_overlap + tokens_nonoverlap + [self.eos_token_id]

            return tokens, len(tokens_overlap) + 1



class BeamDecoder():
    """
    Minxin for beam search with threshold and eos_hidden_states cache. Based on code from transformers.
    """
    @property
    def batch_size(self):
        return len(self.batch_filter)

    @property
    def cur_new_tokens(self):
        return self.cur_len - self.input_len
    
    @property
    def num_left_batch(self):
        return self.batch_filter.sum()

    def global_batch_idx(self, local_idx):
        return self.batch_indices[local_idx]
        
    def _set_num_beams(self):
        if isinstance(self.nbeam, list):
            num_beams = self.nbeam[min(self.cur_new_tokens, len(self.nbeam) - 1)]
            prev_num_beams = self.nbeam[max(min(self.cur_new_tokens - 1, len(self.nbeam) - 1), 0)]
        else:
            num_beams = self.nbeam
            prev_num_beams = self.nbeam
        self.num_beams = num_beams
        self.prev_num_beams = prev_num_beams

    def _cut_off_beams(self, inputs):
        if isinstance(inputs, torch.Tensor):
            x = inputs.unflatten(0, (self.batch_size, -1))
            if x.shape[1] > self.num_beams:
                inputs = x[:, :self.num_beams].flatten(0, 1)
            elif x.shape[1] < self.num_beams:
                raise ValueError(f"Found inputs shape {x.shape}, but num_beams {self.num_beams}!")
        return inputs
    
    def _get_model_kwargs_for_one_batch_candidates(self, model_kwargs, batch_idx, repeat_times):
        new_model_kwargs = {}
        for k, v in model_kwargs.items():
            if k == "past":
                reordered_decoder_past = ()
                for layer_past_states in v:
                    reordered_layer_past_states = ()
                    for layer_past_state in layer_past_states:
                        # need to set correct `past` for each of the four key / value states
                        reordered_layer_past_states = reordered_layer_past_states + (
                            layer_past_state.unflatten(0, (self.batch_size, self.num_beams))[batch_idx].repeat_interleave(repeat_times, dim=0),
                        )
                    reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
                new_model_kwargs["past_key_values"] = reordered_decoder_past
            elif k == "encoder_outputs":
                new_model_kwargs[k] = type(v)({"last_hidden_state": v.last_hidden_state.unflatten(0, (self.batch_size, self.num_beams))[batch_idx].repeat_interleave(repeat_times, dim=0)})
            elif isinstance(v, torch.Tensor):
                new_model_kwargs[k] = v.unflatten(0, (self.batch_size, self.num_beams))[batch_idx].repeat_interleave(repeat_times, dim=0)
            else:
                new_model_kwargs[k] = v
        return new_model_kwargs

    def _get_model_kwargs_for_one_beam(self, past_key_values, model_kwargs, batch_beam_idx):
        # select pask_key_values based on beam_index
        filtered_past = ()
        for layer_past_key_values in past_key_values:
            filtered_layer_past_key_values = ()
            for layer_past_key_value in layer_past_key_values:
                filtered_layer_past_key_values += (layer_past_key_value[batch_beam_idx].unsqueeze(0),)
            filtered_past += (filtered_layer_past_key_values,)
        # slice attention_mask and encoder_outputs in model_kwargs
        new_model_kwargs = {"past_key_values": filtered_past}
        for k, v in model_kwargs.items():
            if k == "past":
                continue
            elif k == "encoder_outputs":
                new_model_kwargs[k] = type(v)({"last_hidden_state": v.last_hidden_state[batch_beam_idx].unsqueeze(0)})
            elif isinstance(v, torch.Tensor):
                new_model_kwargs[k] = v[batch_beam_idx].unsqueeze(0)
            else:
                new_model_kwargs[k] = v
        return new_model_kwargs

    def _add_beam(self, global_batch_idx:int, hypothesis:Union[list,np.ndarray], score:float, eos_hidden_state:Optional[torch.Tensor]=None, text_idx:Union[list,np.ndarray]=None, word_set:Union[list,np.ndarray]=None):
        """
        Update self.beams
        """        
        # self.batch_add_beam_call[global_batch_idx] += 1
        
        # automatically add eos token id
        if hypothesis is not None and hypothesis[-1] != self.eos_token_id:
            hypothesis.append(self.eos_token_id)
        
        beams = self.beams[global_batch_idx]
        beam = (hypothesis, score, eos_hidden_state, text_idx, word_set)

        index = -1
        if len(beams) == 0:
            # the first hypothesis in this batch
            beams.append(beam)
            index = 0
        else:
            for i, x in enumerate(beams):
                # in wordset setting, one text may be added to beams multiple times, only keep the best one
                if text_idx is not None and (text_idx == x[-2]).all():
                    if score > x[1]:
                        beams[i] = beam
                        index = i
                        break
                    else:
                        index = -1
                        break
                # when the current hypothesis's score is bigger than the stored one, insert it
                if score > x[1]:
                    beams.insert(i, beam)
                    index = i
                    # very important, find the right place and exit
                    # otherwise causes infinite loop
                    break
            else:
                # when the current hypothesis's score is the smallest, just append it at the tail
                beams.append(beam)
                index = i + 1
        
        return index
    
    def _finalize(self):
        """
        Break the self.beams into beams, scores, eos_hidden_states and text_indices;
        """
        has_eos_hidden_state = self.beams[0][0][2] is not None
        has_text_idx = self.beams[0][0][3] is not None
        for i, batch_beams in enumerate(self.beams):
            self.seq_scores[i] = [x[1] for x in batch_beams]
            if has_eos_hidden_state:
                self.eos_hidden_states[i] = [x[2] for x in batch_beams]
            if has_text_idx:
                self.prev_text_indices[i] = [x[3] for x in batch_beams]
            self.beams[i] = [x[0] for x in batch_beams]
        
        # print(mean(self.batch_add_beam_call))

    def prepare(self, nbeam, input_ids, eos_token_id, pad_token_id, constrain_index, rank_type, dedup, tokenizer):
        # set necessary attributes that will be involked extensively
        self.nbeam = nbeam
        self.input_len = input_ids.shape[1]
        self.cur_len = input_ids.shape[1]
        self.device = input_ids.device
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.rank_type = rank_type
        # for debug
        self.tokenizer = tokenizer
        # whether to deduplicate among beams
        self.dedup = dedup
        
        batch_size = input_ids.shape[0]
        # a beam list for each batch, whose element is (decoded_seq, score) pair
        self.beams = [[] for _ in range(batch_size)]
        # the last hidden state from the decoder
        self.eos_hidden_states = [None for _ in range(batch_size)]        
        # generation probability cummulation
        self.seq_scores = [0 for _ in range(batch_size)]

        # to determine which batch finished
        self.batch_filter = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        self.batch_indices = torch.arange(batch_size, device=self.device)

        self.batch_add_beam_call = [0 for _ in range(batch_size)]

        self._set_num_beams()

        self.constrain_index_type = type(constrain_index).__name__[:-5].lower()
        # store the text_indices corresponding to previously generated tokens
        self.prev_text_indices = [[None for _ in range(self.num_beams)] for _ in range(batch_size)]        

        if self.constrain_index_type == "wordset":
            # use numpy array to store previously generated words to overcome the shallow copy issue in python lists
            self.prev_words = [[None for _ in range(self.num_beams)] for _ in range(batch_size)]
            self.prefixes = [[None for _ in range(self.num_beams)] for _ in range(batch_size)]

        # to bias the beam_indices to the batch_beam_indices
        self.beam_idx_bias = torch.arange(self.batch_size, device=self.device).unsqueeze(-1)
      
    def update_parameters_by_batch(self, input_ids, beam_scores, model_kwargs):
        # discard model_kwargs corresponding to the finished batches
        if self.num_left_batch < self.batch_size:
            # update model_kwargs
            input_ids = input_ids.unflatten(0, (self.batch_size, self.num_beams))[self.batch_filter].flatten(0,1)
            beam_scores = beam_scores.unflatten(0, (self.batch_size, self.num_beams))[self.batch_filter].flatten(0,1)
            for k, v in model_kwargs.items():
                if k == "past":
                    filtered_past = ()
                    for layer_past_key_values in v:
                        filtered_layer_past_key_values = ()
                        for layer_past_key_value in layer_past_key_values:
                            filtered_layer_past_key_values += (layer_past_key_value.unflatten(0, (self.batch_size, self.num_beams))[self.batch_filter].flatten(0, 1),)
                        filtered_past += (filtered_layer_past_key_values,)
                    model_kwargs[k] = filtered_past
                elif k == "encoder_outputs":
                    model_kwargs[k]["last_hidden_state"] = v["last_hidden_state"].unflatten(0, (self.batch_size, self.num_beams))[self.batch_filter].flatten(0, 1)
                elif isinstance(v, torch.Tensor):
                    model_kwargs[k] = v.unflatten(0, (self.batch_size, self.num_beams))[self.batch_filter].flatten(0, 1)

            # update batch filter
            self.batch_indices = self.batch_indices[self.batch_filter]
            self.batch_filter = self.batch_filter[self.batch_filter]
        return input_ids, beam_scores, model_kwargs
    
    def update_beams(self, beam_tokens, beam_scores, beam_indices, input_ids, model, model_kwargs, past_key_values, max_new_tokens, constrain_index:Union[TrieIndex, WordSetIndex]):
        """
        1. update beams if eos is decoded for trie index
        2. update beams if the decoded words correspond to a unique doc for wordset index
        3. update input_ids by concatenation
        4. update model_kwargs by next_beam_indices

        Args:
            next_beam_tokens: (batch_size, num_beams * 2)
            next_beam_scores: (batch_size, num_beams * 2)
            next_beam_indices: (batch_size, num_beams * 2)
        """
        # when threshold is 0, perform original beam search
        next_beam_tokens = torch.zeros((self.batch_size, self.num_beams), dtype=beam_tokens.dtype, device=self.device)
        next_beam_scores = torch.zeros((self.batch_size, self.num_beams), device=self.device)
        next_beam_indices = torch.zeros((self.batch_size, self.num_beams), dtype=beam_indices.dtype, device=self.device)

        for batch_id, (batch_beam_tokens, batch_beam_scores, batch_beam_indices) in enumerate(zip(beam_tokens, beam_scores, beam_indices)):
            beam_idx = 0
            global_batch_idx = self.global_batch_idx(batch_id)

            # we need to mix slicing out by beam_index and updating
            # so just override the content of batch_idx 
            if self.constrain_index_type == "wordset":
                prev_words_this_batch = [None for _ in range(self.num_beams)]
                prev_text_indices_this_batch = [None for _ in range(self.num_beams)]
                prefixes_this_batch = [None for _ in range(self.num_beams)]

            for beam_rank, (beam_token, beam_score, beam_index) in enumerate(zip(batch_beam_tokens, batch_beam_scores, batch_beam_indices)):
                batch_beam_idx = batch_id * self.prev_num_beams + beam_index

                # skip invalid beams
                if beam_score == -float("inf"):
                    continue

                # t1 = time.time()

                # when next token is separator, update prev_text_indices and prev_words
                # if prev_text_indices is 1, just add to beams
                if self.constrain_index_type == "wordset":
                    # by default we just slice out the beam_index result
                    prev_words = self.prev_words[global_batch_idx][beam_index]
                    prev_text_indices = self.prev_text_indices[global_batch_idx][beam_index]
                    prefix = self.prefixes[global_batch_idx][beam_index]
                    
                    if prefix is None:
                        prefix = beam_token.unsqueeze(0).cpu().numpy()
                    else:
                        prefix = np.concatenate([prefix, beam_token.unsqueeze(0).cpu().numpy()], axis=-1)

                    if constrain_index.sep_token_id is None or beam_token == constrain_index.sep_token_id:
                        word = constrain_index.get_word(prefix)
                        if word is not None:
                            # update text indices corresponding to generated words
                            new_prev_text_indices = constrain_index.get_text(word, prev_text_indices)
                            # only update when ``tokens'' and ``prev_words'' are valid
                            if len(new_prev_text_indices):
                                prev_text_indices = new_prev_text_indices
                                if prev_words is not None:
                                    prev_words = np.concatenate([prev_words, np.array([word], dtype=np.int32)], axis=-1)
                                    # sort words so that we can make element-wise comparison with other words in the batch
                                    prev_words.sort()
                                else:
                                    # in case this is the first generated word
                                    prev_words = np.array([word], dtype=np.int32)
                                # reset prefix
                                prefix = None
                            
                            elif constrain_index.sep_token_id is not None:
                                raise ValueError(f"The text indices corresponding to words ({prev_words}, {word}) are empty!")
                    
                    # t2 = time.time()
                    
                    # when there is only one document containing the generated words, add to beam
                    # when there are duplicated docids, eos will be decoded at the end, add to beam2
                    if (prev_text_indices is not None and len(prev_text_indices) == 1) or beam_token == constrain_index.eos_token_id:
                        pos = self._add_beam(
                            global_batch_idx=global_batch_idx,
                            hypothesis=None,
                            score=beam_score.item(),
                            text_idx=prev_text_indices,
                            word_set=prev_words,
                        )
                        # print(pos)
                        if self.rank_type == "eos" and pos != -1:
                            # get the remaining tokens
                            if beam_token == constrain_index.sep_token_id:
                                reconstructed_tokens, reconstruct_idx = constrain_index.reconstruct(prev_text_indices[0], prev_words)
                                # minus one to include the previous sep_token_id
                                incoming_input_ids = torch.tensor(reconstructed_tokens[reconstruct_idx - 1:], device=self.device).unsqueeze(0)   # L
                                # incoming_input_ids = torch.tensor([constrain_index.sep_token_id, constrain_index.eos_token_id], device=self.device).unsqueeze(0)   # L
                            
                            elif beam_token == constrain_index.eos_token_id:
                                incoming_input_ids = beam_token.view(1,1) # 1

                            else:
                                raise ValueError(f"Found beam_token=={beam_token} is neither eos {constrain_index.eos_token_id} nor sep {constrain_index.sep_token_id}!")

                            # compute the eos_embedding based on past and encoder_outputs
                            new_model_kwargs = self._get_model_kwargs_for_one_beam(past_key_values, model_kwargs, batch_beam_idx)
                            outputs = model(decoder_input_ids=incoming_input_ids, **new_model_kwargs)
                            eos_hidden_states = outputs.decoder_hidden_states[-1][:, -1].squeeze(0) # 768

                            # t3 = time.time()

                            # assign eos_embedding
                            # tuple does not support assignment
                            beam = self.beams[global_batch_idx][pos]
                            beam = (beam[0], beam[1], eos_hidden_states, beam[3], beam[4])
                            self.beams[global_batch_idx][pos] = beam

                            # print(self.tokenizer.batch_decode(incoming_input_ids))
                            # print(prev_text_indices)
                            # print(beam)
                            # print(t3 - t2, t2 - t1)
                            # input()

                    else:
                        is_dup = False
                        if self.dedup and prev_words is not None:
                            max_dup_num = len(prev_text_indices)
                            dup_num = 0
                            smallest_idx = -1
                            smallest_score = beam_score
                            # compare with other beams in this batch one-by-one and remove duplications
                            for other_beam_idx, other_words in enumerate(prev_words_this_batch[:beam_idx]):
                                # when the words are the same, only leave the top k scored ones where k is prev_text_indices (the number of documents that contain these words), so as to give more chance to other hypotheses
                                # since we are generating new beams one by one, there will be at most one redundant beam
                                if other_words is not None and len(prev_words) == len(other_words) and (prev_words == other_words).all():
                                    dup_num += 1
                                    # print(other_words, other_beam_idx, beam_idx)
                                    other_score = next_beam_scores[batch_id, other_beam_idx]
                                    if other_score < smallest_score:
                                        smallest_idx = other_beam_idx
                                        smallest_score = other_score

                            if dup_num > max_dup_num:
                                # replace the smallest with the current
                                if smallest_idx != -1:
                                    # sanity check
                                    assert prefixes_this_batch[smallest_idx] == prefix, f"Conflict prefix: {prefixes_this_batch} and {prefix} at {smallest_idx}"
                                    assert next_beam_tokens[batch_id, smallest_idx] == beam_token, f"Conflict other beam tokens: {next_beam_tokens[batch_id]} and {beam_token} at {smallest_idx}"
                                    next_beam_scores[batch_id, smallest_idx] = beam_score
                                    next_beam_indices[batch_id, smallest_idx] = beam_index
                                # otherwise, discard the current
                                is_dup = True

                        if not is_dup:
                            prev_words_this_batch[beam_idx] = prev_words
                            prev_text_indices_this_batch[beam_idx] = prev_text_indices
                            prefixes_this_batch[beam_idx] = prefix

                            # otherwise, when the generated words correspond to many documents or the word is not finished, continue
                            next_beam_tokens[batch_id, beam_idx] = beam_token
                            next_beam_scores[batch_id, beam_idx] = beam_score
                            next_beam_indices[batch_id, beam_idx] = beam_index
                            beam_idx += 1

                # when next token is eos, add to beam
                # when cur_new_tokens has reached the limits, add to beam regardless of eos token
                elif self.constrain_index_type == "trie":
                    if beam_token == self.eos_token_id or self.cur_new_tokens == max_new_tokens - 1:
                        # <eos> is only meaningful when being decoded within the top num_beams
                        if beam_rank < self.num_beams:
                            if self.rank_type == "eos":
                                new_model_kwargs = self._get_model_kwargs_for_one_beam(past_key_values, model_kwargs, batch_beam_idx)
                                outputs = model(decoder_input_ids=beam_token.view(1,1), **new_model_kwargs)
                                eos_hidden_states = outputs.decoder_hidden_states[-1][:, -1].squeeze(0) # 768
                            
                            else:
                                eos_hidden_states = None

                            self._add_beam(
                                global_batch_idx=global_batch_idx, 
                                hypothesis=input_ids[batch_beam_idx].tolist(),
                                score=beam_score.item(),
                                eos_hidden_state=eos_hidden_states
                            )

                    else:
                        next_beam_tokens[batch_id, beam_idx] = beam_token
                        next_beam_scores[batch_id, beam_idx] = beam_score
                        next_beam_indices[batch_id, beam_idx] = beam_index
                        beam_idx += 1

                else:
                    raise NotImplementedError(f"Constrain index type {constrain_index} not implemented yet!")
                # jump out when beam_idx reaches the limit
                if beam_idx == self.num_beams:
                    break
            
            if beam_idx == 0:
                self.batch_filter[batch_id] = False

            if beam_idx < self.num_beams:
                next_beam_scores[batch_id, beam_idx:] = -float("inf")
            
            if self.constrain_index_type == "wordset":
                self.prev_words[global_batch_idx] = prev_words_this_batch
                self.prev_text_indices[global_batch_idx] = prev_text_indices_this_batch
                self.prefixes[global_batch_idx] = prefixes_this_batch

        # used to slice input_ids and model_kwargs
        batch_beam_indices = (next_beam_indices.view(self.batch_size, -1) + self.beam_idx_bias[:self.batch_size] * self.prev_num_beams).view(-1)

        input_ids = torch.cat([input_ids[batch_beam_indices, :], next_beam_tokens.view(-1, 1)], dim=-1) # B*K, L
        # one more token
        self.cur_len += 1

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, batch_beam_indices),
                )
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        model_kwargs["past"] = reordered_decoder_past

        # cut off beams for shrinking beam sizes
        for k, v in model_kwargs.items():
            if k == "encoder_outputs":
                model_kwargs[k]["last_hidden_state"] = self._cut_off_beams(v["last_hidden_state"])
            elif isinstance(v, torch.Tensor):
                model_kwargs[k] = self._cut_off_beams(v)

        return next_beam_scores.view(-1), input_ids, model_kwargs

    def handle_threshold(self, input_ids, beam_scores, model, model_kwargs, threshold, trsd_start_len, constrain_index:Union[TrieIndex,WordSetIndex]):
        if threshold > 0 and self.cur_new_tokens >= trsd_start_len:
            # check if all possible beams within each batch are less than threshold
            for batch_id, batch in enumerate(input_ids.unflatten(0, (self.batch_size, self.num_beams)).tolist()):
                skip = False
                global_batch_idx = self.global_batch_idx(batch_id)

                if self.constrain_index_type == "trie":
                    candidates = []
                    text_indices = []
                    # determine which prefix each candidate belongs
                    repeat_times = []
                    for seq in batch:
                        pairs = constrain_index.items(prefix=seq)
                        repeat_time = 0
                        for candidate, tindices in pairs:
                            # add to candidates
                            candidates.append(candidate)
                            # add to text_indices
                            text_indices.append(tindices)
                            repeat_time += 1
                        # maybe zero 
                        repeat_times.append(repeat_time)
                    max_length = max([len(x) for x in candidates])
                
                elif self.constrain_index_type == "wordset":
                    candidates = []
                    # a dict to track the candidate text indices so that duplicated beams will not be added twice
                    # keep insertion order
                    candidates_set = OrderedDict()
                    repeat_times = []

                    prev_text_indices = self.prev_text_indices[global_batch_idx]
                    prev_words = self.prev_words[global_batch_idx]
                    prefixes = self.prefixes[global_batch_idx]

                    for i, tindices in enumerate(prev_text_indices):
                        candidate = []
                        if tindices is not None:
                            if len(tindices) > threshold:
                                skip = True
                                break
                            
                            for j, tidx in enumerate(tindices):
                                if tidx not in candidates_set:
                                    words = prev_words[i]
                                    reconstructed, start_idx = constrain_index.reconstruct(tidx, words, prefixes[i])
                                    # sometimes the prefix may not be valid for tidx because tidx is only updated when sep_token_id is decoded
                                    if reconstructed is not None:
                                        # minus one because the leading 0
                                        assert start_idx == self.cur_len, f"{reconstructed}, {batch[i]}, {start_idx}, {self.cur_len}"
                                        candidates_set[tidx] = True
                                        candidate.append(reconstructed)
                            
                        # always modify repeat times because one beam coresponds to one repeat_time
                        candidates.extend(candidate)
                        repeat_times.append(len(candidate))
                    
                    if len(candidates) == 0:
                        skip = True
                        break

                    max_length = max([len(x) for x in candidates])
                    text_indices = list(candidates_set.keys())
 
                if len(candidates) <= threshold and not skip:
                    # this batch is finished
                    self.batch_filter[batch_id] = False
                    # next we will modify the contents in candidates
                    # backup for later saving
                    candidates_tensor = candidates.copy()
                    for j, candidate in enumerate(candidates):
                        # pad to max length
                        if len(candidate) < max_length:
                            candidates_tensor[j] = candidate + [self.pad_token_id] * (max_length - len(candidate))

                    try:
                        incoming_input_ids = torch.tensor(candidates_tensor)[:, self.cur_len - 1:].to(self.device)
                    except:
                        print(max_length, len(candidates), candidates_tensor)
                        raise

                    repeat_times = torch.tensor(repeat_times, device=self.device)
                    new_model_kwargs = self._get_model_kwargs_for_one_batch_candidates(model_kwargs, batch_id, repeat_times)

                    outputs = model(decoder_input_ids=incoming_input_ids, **new_model_kwargs)

                    # collect generation probability
                    target_token_ids = incoming_input_ids[:, 1:]    # N, max_length-cur_len
                    eos_pos = target_token_ids == self.eos_token_id
                    eos_index = torch.arange(target_token_ids.shape[-1], device=self.device).expand(eos_pos.shape)[eos_pos]   # N

                    scores = outputs.logits.log_softmax(dim=-1)
                    scores = scores.gather(dim=-1, index=target_token_ids[..., None]).squeeze(-1) # N, max_length-cur_len
                    # cumulate log probs
                    scores = torch.cumsum(scores, dim=-1)
                    scores = scores[eos_pos]  # N

                    prev_scores = beam_scores.view(self.batch_size, self.num_beams)[batch_id].repeat_interleave(repeat_times, dim=0)   # N
                    scores = (scores + prev_scores).tolist()

                    if self.rank_type == "eos":
                        last_hidden_states = outputs.decoder_hidden_states[-1] # N, max_length, D
                        # note that eos_index is 1 bigger than previous because our basis is incoming_token_ids now
                        eos_hidden_states = last_hidden_states[range(last_hidden_states.shape[0]), eos_index+1] # N, D
                    
                    for k, candidate in enumerate(candidates):
                        self._add_beam(
                            global_batch_idx=global_batch_idx,
                            hypothesis=candidate,
                            score=scores[k],
                            eos_hidden_state=eos_hidden_states[k] if self.rank_type == "eos" else None,
                            text_idx=np.array([text_indices[k]], dtype=np.int32)
                        )


    @torch.no_grad()
    def search(self, model:T5ForConditionalGeneration, nbeam, threshold, trsd_start_len, max_new_tokens, constrain_index:Union[TrieIndex,WordSetIndex], rank_type="prob", tokenizer=None, dedup=False, **inputs):
        """
        Perform beam search with constrain from trie_index or wordset_index.
        """
        bos_token_id = model.config.bos_token_id
        eos_token_id = model.config.eos_token_id
        pad_token_id = model.config.pad_token_id

        # prepare model inputs to the encoder
        # input_tensor: the input sequence of shape (B, L)
        # model_kwargs: attention_mask, token_type_id, 
        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(None, bos_token_id, model_kwargs=inputs)

        model_kwargs["output_attentions"] = None
        # output hidden states
        model_kwargs["output_hidden_states"] = True
        model_kwargs["use_cache"] = True

        batch_size = inputs_tensor.shape[0]

        # prepare encoder_outputs
        if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )
        # prepare input_ids for the decoder
        if model.config.is_encoder_decoder:
            input_ids = model._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=None,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
                device=inputs_tensor.device,
            )   # (B, 1)
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        self.prepare(nbeam, input_ids, eos_token_id, pad_token_id, constrain_index, rank_type, dedup, tokenizer)

        beam_scores = torch.zeros((self.batch_size, self.num_beams), device=self.device)
        # make sure only the first beam will be selected during the first beam search iteration, thereby avoiding repetitive input_ids across beams
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # expand model_kwargs and input_ids to the max size, which is batch_size * num_beams
        input_ids, model_kwargs = model._expand_inputs_for_generation(
            input_ids, expand_size=self.num_beams, is_encoder_decoder=model.config.is_encoder_decoder, **model_kwargs
        )

        while self.num_left_batch and self.cur_new_tokens < max_new_tokens:
            # num_beams is consistent in the loop
            self._set_num_beams()

            # 1. adjust input_ids by past_key_values
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # 2. compute logits
            outputs = model(**model_inputs, return_dict=True)
            logits = outputs.logits[:, -1, :]    # (batch_size * num_beams, vocab_size)
            scores = torch.log_softmax(logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            # 3. do masking based on trie or intersection
            # IMPORTANT! in this section we use prev_num_beams because input_ids are based on prev_num_beams
            mask = torch.full_like(scores, -float("inf"))
            if self.constrain_index_type == "trie":
                # mask eos_token when cur_len < min_len; mask non-valid tokens with trie
                for batch_id, beam_sent in enumerate(input_ids.view(self.batch_size, self.prev_num_beams, -1)):
                    for beam_id, sent in enumerate(beam_sent):
                        mask[batch_id * self.prev_num_beams + beam_id, constrain_index.get_valid_tokens(sent.tolist())] = 0
                scores = scores + mask
            elif self.constrain_index_type == "wordset":
                for batch_id, beam_tokens in enumerate(input_ids.view(self.batch_size, self.prev_num_beams, -1).cpu().numpy()):
                    global_batch_idx = self.global_batch_idx(batch_id)
                    
                    prev_text_indices = self.prev_text_indices[global_batch_idx] # prev_num_beams, *
                    prev_words = self.prev_words[global_batch_idx] # prev_num_beams
                    prefixes = self.prefixes[global_batch_idx]
                    for beam_id, tokens in enumerate(beam_tokens):
                        beam_score = beam_scores[batch_id * self.prev_num_beams + beam_id]
                        # wordsetindex cannot return an empty list for an invalid token sequence
                        # we manually skip to check valid tokens when beam score is -inf
                        if torch.isinf(beam_score):
                            # do not get_valid_tokens for nonsence beam
                            valid_tokens = []                        
                        else:
                            # find previously generated tokens
                            # at the first decoding step, there are no prev_text_indices because all documents are available                        
                            valid_tokens = constrain_index.get_valid_tokens(prev_text_indices[beam_id], prev_words[beam_id], prefixes[beam_id])
                            mask[batch_id * self.prev_num_beams + beam_id, valid_tokens] = 0
                scores = scores + mask
            
            # 4. find next token
            next_beam_scores = scores + beam_scores[:, None]
            # reshape for beam search
            vocab_size = next_beam_scores.shape[-1]
            next_beam_scores = next_beam_scores.view(self.batch_size, -1) # (batch_size, num_beams * vocab_size)
            # scale top k to 5 times larger
            # trie index only needs 2*k (may decode to <eos>)
            # give chance to more hypotheses in wordset index 
            next_beam_scores, next_beam_tokens = torch.topk(
                next_beam_scores, k=self.num_beams * 5 if self.dedup else self.num_beams * 2, dim=1, largest=True, sorted=True
            )
            next_beam_indices = torch.div(next_beam_tokens, vocab_size, rounding_mode="floor")
            next_beam_tokens = next_beam_tokens % vocab_size

            # 5. update beam hypotheses if eos is decoded
            beam_scores, input_ids, model_kwargs = self.update_beams(next_beam_tokens, next_beam_scores, next_beam_indices, input_ids, model, model_kwargs, outputs.past_key_values, max_new_tokens, constrain_index)

            # print(input_ids)
            # # print(beam_scores)
            # print(tokenizer.batch_decode(input_ids))
            # try:
            #     print(self.prev_words)
            #     # print(self.prefixes)
            # except:
            #     pass
            # print(self.beams)
            # input()

            # 6. handle the finished batches according to the threshold
            self.handle_threshold(input_ids, beam_scores, model, model_kwargs, threshold, trsd_start_len, constrain_index)

            # 7. update model kwargs in case some batches finish
            input_ids, beam_scores, model_kwargs = self.update_parameters_by_batch(input_ids, beam_scores, model_kwargs)
        
        # try:
        self._finalize()
        # except:
        #     print(self.beams[0])
        #     print(self.prev_words[0], input_ids[0])
        #     raise



class GreedyKeywordSorter():
    @property
    def batch_size(self):
        return len(self.batch_filter)

    @property
    def cur_new_tokens(self):
        return self.cur_len - self.input_len
    
    @property
    def num_left_batch(self):
        return self.batch_filter.sum()

    def global_batch_idx(self, local_idx):
        return self.batch_indices[local_idx]
        
    def prepare(self, input_ids, eos_token_id, pad_token_id):
        # set necessary attributes that will be involked extensively
        self.input_len = input_ids.shape[1]
        self.cur_len = input_ids.shape[1]
        self.device = input_ids.device
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        batch_size = input_ids.shape[0]
        # a beam list for each batch, whose element is (decoded_seq, score) pair
        self.res = [None for _ in range(batch_size)]
        # to determine which batch finished
        self.batch_filter = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        self.batch_indices = torch.arange(batch_size, device=self.device)

        self.prev_words = [None for _ in range(batch_size)]
        self.prefixes = [None for _ in range(batch_size)]

    def update_parameters_by_batch(self, input_ids, model_kwargs):
        # discard model_kwargs corresponding to the finished batches
        if self.num_left_batch < self.batch_size:
            # update model_kwargs
            input_ids = input_ids[self.batch_filter]
            for k, v in model_kwargs.items():
                if k == "past":
                    filtered_past = ()
                    for layer_past_key_values in v:
                        filtered_layer_past_key_values = ()
                        for layer_past_key_value in layer_past_key_values:
                            filtered_layer_past_key_values += (layer_past_key_value[self.batch_filter],)
                        filtered_past += (filtered_layer_past_key_values,)
                    model_kwargs[k] = filtered_past
                elif k == "encoder_outputs":
                    model_kwargs[k]["last_hidden_state"] = v["last_hidden_state"][self.batch_filter]
                elif isinstance(v, torch.Tensor):
                    model_kwargs[k] = v[self.batch_filter]

            # update batch filter
            self.batch_indices = self.batch_indices[self.batch_filter]
            self.batch_filter = self.batch_filter[self.batch_filter]
        return input_ids, model_kwargs

    @torch.no_grad()
    def search(self, model:T5ForConditionalGeneration, text_indices:np.ndarray, wordset_index:Union[TrieIndex,WordSetIndex], **inputs):
        """
        Perform beam search with constrain from trie_index or wordset_index.
        """
        bos_token_id = model.config.bos_token_id
        eos_token_id = model.config.eos_token_id
        pad_token_id = model.config.pad_token_id

        # prepare model inputs to the encoder
        # input_tensor: the input sequence of shape (B, L)
        # model_kwargs: attention_mask, token_type_id, 
        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(None, bos_token_id, model_kwargs=inputs)

        model_kwargs["output_attentions"] = None
        # output hidden states
        model_kwargs["output_hidden_states"] = True
        model_kwargs["use_cache"] = True

        batch_size = inputs_tensor.shape[0]

        # prepare encoder_outputs
        if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )
        # prepare input_ids for the decoder
        if model.config.is_encoder_decoder:
            input_ids = model._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=None,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
                device=inputs_tensor.device,
            )   # (B, 1)
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        self.prepare(input_ids, eos_token_id, pad_token_id)

        docs = wordset_index.docs[text_indices] # B, L

        # print(docs, text_indices)

        while self.num_left_batch:
            # 1. adjust input_ids by past_key_values
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # 2. compute logits
            outputs = model(**model_inputs, return_dict=True)
            logits = outputs.logits[:, -1, :]    # (batch_size, vocab_size)
            scores = torch.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # 3. do masking based on trie or intersection
            # IMPORTANT! in this section we use prev_num_beams because input_ids are based on prev_num_beams
            mask = torch.full_like(scores, -float("inf"))
            for batch_id in range(len(input_ids)):
                global_batch_idx = self.global_batch_idx(batch_id)
                prev_words = self.prev_words[global_batch_idx]
                prefixes = self.prefixes[global_batch_idx]
                valid_tokens = wordset_index.get_valid_tokens_from_doc(text_indices[global_batch_idx], prev_words, prefixes)
                mask[batch_id, valid_tokens] = 0
            scores = scores + mask
            
            next_tokens = torch.argmax(scores, dim=-1)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs["past"] = outputs.past_key_values
            self.cur_len += 1

            # update prev_words
            for batch_id, tokens in enumerate(input_ids.cpu().numpy()):
                global_batch_idx = self.global_batch_idx(batch_id)
                doc = docs[global_batch_idx]
                doc = doc[doc != -1]

                next_token = tokens[-1]

                prefix = self.prefixes[global_batch_idx]
                if prefix is None:
                    # expand 1 dimension
                    prefix = next_token[None]
                else:
                    prefix = np.concatenate([prefix, next_token[None]], axis=-1)

                if next_token == wordset_index.sep_token_id:
                    word = wordset_index.get_word(prefix)

                    prev_words = self.prev_words[global_batch_idx]
                    if prev_words is not None:
                        prev_words = np.concatenate([prev_words, np.array([word], dtype=np.int32)], axis=-1)
                    else:
                        # in case this is the first generated word
                        prev_words = np.array([word], dtype=np.int32)

                    self.prev_words[global_batch_idx] = prev_words
                    prefix = None
                
                elif next_token == wordset_index.eos_token_id:
                    assert len(self.prev_words[global_batch_idx]) == len(doc)
                    self.res[global_batch_idx] = tokens
                    self.batch_filter[batch_id] = False

                self.prefixes[global_batch_idx] = prefix
                
            # 7. update model kwargs in case some batches finish
            input_ids, model_kwargs = self.update_parameters_by_batch(input_ids, model_kwargs)

            # print(self.prev_words)
            # print(self.prefixes)
            # print(input_ids)
            # input()



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
    token_ids, start_text_idx, token_num, stop_token_ids, to_unique = args

    text_idx_inverted_lists = [[] for _ in range(token_num)]
    token_idx_inverted_lists = [[] for _ in range(token_num)]

    for text_idx, token_ids in enumerate(tqdm(token_ids, ncols=100, leave=False)):
        if to_unique:
            token_set = set()
        for token_idx, token_id in enumerate(token_ids):
            if token_id == -1 or token_id in stop_token_ids:
                continue
            if to_unique and token_id in token_set:
                continue
            # save the token's position (text_idx, token_idx) in the inverted list
            text_idx_inverted_lists[token_id].append(start_text_idx + text_idx)
            token_idx_inverted_lists[token_id].append(token_idx)
            if to_unique:
                token_set.add(token_id)

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
        words: list of words
    """
    words = []
    word_scores = []
    if scores is None:
        scores = [1] * len(tokens)

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


# def convert_tokens_to_words(token_ids:np.ndarray, tokenizer, token_weights:np.ndarray=None, reduce:str="max"):
#     """
#     Map token_ids to words, possibly pool the corresponding token weights.
#     """
#     assert tokenizer.is_fast, "Must use fast tokenizer!"

#     ## reconstruct string
#     decoded = tokenizer.decode(token_ids)
#     # convert to BatchEncoding instance
#     batchenc = tokenizer(decoded, add_special_tokens=False)
#     # assert (token_ids == batchenc.input_ids).all()
#     # get the word index for each token
#     # the first element must be 0
#     word_ids = batchenc.word_ids()

#     words = []
#     word_weights = []

#     prev_word_id = 0
#     if token_weights is not None:
#         word_weight = []

#     for j, word_id in enumerate(word_ids):
#         if word_id is None:
#             continue
#         if word_id == prev_word_id:
#             if token_weights is not None:
#                 word_weight.append(token_weights[j])
#             continue
#         elif word_id != prev_word_id:
#             # get prev_word
#             charspan = batchenc.word_to_chars(prev_word_id)
#             prev_word = decoded[charspan.start: charspan.end]

#             words.append(prev_word)
#             if token_weights is not None:
#                 # pool token weights to word weight
#                 if reduce == "max":
#                     word_weight = max(word_weight)
#                 elif reduce == "mean":
#                     word_weight = sum(word_weight) / len(word_weight)
#                 elif reduce == "sum":
#                     word_weight = sum(word_weight)
#                 elif reduce == "first":
#                     word_weight = word_weight[0]
#                 else:
#                     raise NotImplementedError
#                 word_weights.append(word_weight)
#                 # must add the current weight
#                 try:
#                     word_weight = [token_weights[j]]
#                 except:
#                     print(len(word_ids), len(token_ids), len(batchenc.input_ids), token_ids, batchenc.input_ids)
#                     save_pickle({"token_ids": token_ids, "input_ids": batchenc.input_ids, "decoded": decoded}, "token_ids.pkl")
#                     raise

#             prev_word_id = word_id

#     # handle the last word
#     charspan = batchenc.word_to_chars(prev_word_id)
#     prev_word = decoded[charspan.start: charspan.end]

#     words.append(prev_word)
#     if token_weights is not None:
#         # pool token weights to word weight
#         if reduce == "max":
#             word_weight = max(word_weight)
#         elif reduce == "mean":
#             word_weight = sum(word_weight) / len(word_weight)
#         elif reduce == "sum":
#             word_weight = sum(word_weight)
#         elif reduce == "first":
#             word_weight = word_weight[0]
#         else:
#             raise NotImplementedError
#         word_weights.append(word_weight)

#     if token_weights is not None:
#         return words, word_weights
#     else:
#         return words, None


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
                # json cannot parse numpy dtype
                vector[word] = float(score)

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


GENERATIVE_INDEX_MAP = {
    "trie": TrieIndex,
    "wordset": WordSetIndex,
}

SUBWORD_TO_WORD_FN = {
    "bert": subword_to_word_bert,
}

