import os
import json
import faiss
import torch
import pickle
import logging
import collections
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from datetime import timedelta
from dataclasses import dataclass
from contextlib import contextmanager
from collections import OrderedDict
from torch._six import string_classes
from .typings import *



def load_pickle(path:str):
    """
    Load pickle file from path.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path:str):
    """
    Save pickle file.
    """
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def load_from_previous(model:torch.nn.Module, path:str):
    """
    Load checkpoint from the older version of Uni-Retriever, only load model parameters and overrides the config by the current config.
    """
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    state_dict = ckpt["model"]
    try:
        metrics = ckpt["metrics"]
        model.metrics = metrics
    except:
        pass

    new_state_dict = {}
    for k, v in state_dict.items():
        if "bert" in k:
            new_state_dict[k.replace("bert", "plm")] = v
        else:
            new_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict)
    if len(missing_keys):
        print(f"Missing Keys: {missing_keys}")
    if len(unexpected_keys):
        print(f"Unexpected Keys: {unexpected_keys}")


def makedirs(path:str, exist_ok:bool=True):
    """
    Shortcut for creating parent directory for a file.

    Args:
        path
        exist_ok: ignore if parent folder already exists
    """
    dirname = os.path.dirname(os.path.abspath(path))
    os.makedirs(dirname, exist_ok=exist_ok)


def readlink(path:str):
    """
    Read permalink.
    """
    if os.path.islink(path):
        return os.readlink(path)
    else:
        return path


def isempty(path:str):
    """
    Check if a folder is empty.
    """
    if os.path.isdir(path):
        return len(os.listdir(path)) == 0
    elif os.path.isfile(path):
        return False
    else:
        return True


def update_hydra_config(config:dict):
    """
    update the hydra config at inner layer by the one defined in the _global_ package layer
    """
    src = {}
    dest = {}
    for k, v in config.items():
        if isinstance(v, dict):
            dest[k] = v
        else:
            src[k] = v

    outer_keys = list(src.keys())
    for k, v in dest.items():
        for sk in outer_keys:
            if sk in v:
                # override the same config in the inner config group
                dest[k][sk] = src.pop(sk)

    # if there is still remaining parameters
    if len(src):
        dest["extra"] = {}
        for sk, sv in src.items():
            dest["extra"][sk] = sv

    return dest


def flatten_hydra_config(config:dict):
    """
    Flatten a two-layer hydra config dict
    """
    res = {}
    for k, v in config.items():
        res.update(v)
    return res


def synchronize(func:Optional[callable]=None):
    """
    A function or a decorator to synchronize all processes on enterring and exiting the function.
    """
    if func is None:
        if dist.is_initialized():
            dist.barrier()
        else:
            pass
        return

    def wrapper(*args, **kwargs):
        # read initialization state inside the wrapper function, executed when the function is called
        distributed = dist.is_initialized()
        if distributed:
            dist.barrier()
            # getting the returned value
            returned_value = func(*args, **kwargs)
            dist.barrier()
        else:
            returned_value = func(*args, **kwargs)
        return returned_value
    return wrapper


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def mrr_score(candidate, target, cutoffs):
    score = np.zeros(len(cutoffs))
    jump = False
    for i, x in enumerate(candidate, 1):
        if x == -1:
            break
        if x in target:
            for k, cutoff in enumerate(cutoffs):
                if i <= cutoff:
                    score[k] += 1 / i
                    jump = True
        if jump:
            jump = False
            break
    return score

def recall_score(candidate, target, cutoffs):
    score = np.zeros(len(cutoffs))
    for i, cutoff in enumerate(cutoffs):
        recall = target.intersection(set(candidate[:cutoff]))
        score[i] += len(recall) / len(target)
    return score

def precision_score(candidate, target, cutoffs):
    score = np.zeros(len(cutoffs))
    for i, cutoff in enumerate(cutoffs):
        precision = target.intersection(set(candidate[:cutoff]))
        score[i] += len(precision) / cutoff
    return score

def ndcg_score(candidate, target, cutoffs):
    idcg = np.zeros(len(cutoffs))
    score = np.zeros(len(cutoffs))
    for i, x in enumerate(candidate, 1):
        if x in target:
            for k, cutoff in enumerate(cutoffs):
                if i <= cutoff:
                    score[k] += 1 / np.log2(i + 1)
    for j, y in enumerate(target, 1):
        for k, cutoff in enumerate(cutoffs):
            if j <= cutoff:
                idcg[k] += 1 / np.log2(j + 1)
    score /= idcg
    return score

def ap_score(candidate, target):
    score = 0
    hit = 0
    for i, x in enumerate(candidate, 1):
        if x == -1:
            break
        if x in target:
            hit += 1
            score += hit / i
    if hit:
        score /= hit
    else:
        score = 0
    return score

def compute_metrics(retrieval_result:RETRIEVAL_MAPPING, ground_truth:RETRIEVAL_MAPPING, cutoffs:list[int]=[10, 100, 1000], metrics:list[str]=["mrr", "recall"], return_each_query:bool=False):
    """
    Compute metrics given a ``retrieval_result`` and the ``ground_truth`` dict.

    Args:
        retrieval_result: mapping query_id to its retrieved document ids
        ground_truth: mapping query_id to its ground truth document ids
        cutoffs: the cutoff to compute metrics
        metrics: the metrics to compute
        return_each_query: if true, return each query's metric as :class:`np.array`
    """
    query_num = len(retrieval_result)

    # sort the cutoff ascendingly
    cutoffs.sort()
    cutoffs = np.array(cutoffs)
    if "mrr" in metrics:
        if return_each_query:
            MRRs = np.zeros((query_num, len(cutoffs)))
        else:
            MRRs = np.zeros(len(cutoffs))
    if "recall" in metrics:
        if return_each_query:
            Recalls = np.zeros((query_num, len(cutoffs)))
        else:
            Recalls = np.zeros(len(cutoffs))
    if "ndcg" in metrics:
        if return_each_query:
            nDCGs = np.zeros((query_num, len(cutoffs)))
        else:
            nDCGs = np.zeros(len(cutoffs))
    if "precision" in metrics:
        if return_each_query:
            Precisions = np.zeros((query_num, len(cutoffs)))
        else:
            Precisions = np.zeros(len(cutoffs))
    if "map" in metrics:
        if return_each_query:
            MAP = np.zeros(query_num)
        else:
            MAP = 0

    for i, qidx in enumerate(tqdm(retrieval_result, ncols=100, desc="Computing Metrics", leave=False)):
        if qidx in ground_truth:
            target = set(ground_truth[qidx])
            candidate = retrieval_result[qidx]

            if "mrr" in metrics:
                mrr = mrr_score(candidate, target, cutoffs)
                if return_each_query:
                    MRRs[i] = mrr
                else:
                    MRRs += mrr
            if "recall" in metrics:
                recall = recall_score(candidate, target, cutoffs)
                if return_each_query:
                    Recalls[i] = recall
                else:
                    Recalls += recall
            if "ndcg" in metrics:
                ndcg = ndcg_score(candidate, target, cutoffs)
                if return_each_query:
                    nDCGs[i] = ndcg
                else:
                    nDCGs += ndcg
            if "precision" in metrics:
                precision = precision_score(candidate, target, cutoffs)
                if return_each_query:
                    Precisions[i] = precision
                else:
                    Precisions += precision
            if "map" in metrics:
                _map = ap_score(candidate, target)
                if return_each_query:
                    MAP[i] = _map
                else:
                    MAP += _map
        else:
            print(qidx)

    return_metrics = {}
    if return_each_query:
        if "mrr" in metrics:
            for i, cutoff in enumerate(cutoffs):
                mrr = MRRs[:, i]
                return_metrics[f"MRR@{cutoff}"] = mrr
        if "recall" in metrics:
            for i, cutoff in enumerate(cutoffs):
                recall = Recalls[:, i]
                return_metrics[f"Recall@{cutoff}"] = recall
        if "precision" in metrics:
            for i, cutoff in enumerate(cutoffs):
                precision = Precisions[:, i]
                return_metrics[f"Precision@{cutoff}"] = precision
        if "map" in metrics:
            return_metrics[f"MAP"] = MAP
        if "ndcg" in metrics:
            for i, cutoff in enumerate(cutoffs):
                ndcg = nDCGs[:, i]
                return_metrics[f"nDCG@{cutoff}"] = ndcg
    else:
        if "mrr" in metrics:
            MRRs /= query_num
            for i, cutoff in enumerate(cutoffs):
                mrr = MRRs[i]
                return_metrics[f"MRR@{cutoff}"] = round(mrr, 4)
        if "recall" in metrics:
            Recalls /= query_num
            for i, cutoff in enumerate(cutoffs):
                recall = Recalls[i]
                return_metrics[f"Recall@{cutoff}"] = round(recall, 4)
        if "precision" in metrics:
            Precisions /= query_num
            for i, cutoff in enumerate(cutoffs):
                precision = Precisions[i]
                return_metrics[f"Precision@{cutoff}"] = round(precision, 4)
        if "map" in metrics:
            MAP /= query_num
            return_metrics[f"MAP"] = round(MAP, 4)
        if "ndcg" in metrics:
            nDCGs /= query_num
            for i, cutoff in enumerate(cutoffs):
                ndcg = nDCGs[i]
                return_metrics[f"nDCG@{cutoff}"] = round(ndcg, 4)

    return return_metrics


def compute_metrics_nq(retrieval_result:RETRIEVAL_MAPPING, query_answer_path:str, collection_path:str):
    """
    Compute recall on NQ-open dataset; Since there is no ground-truth file, take the passage containing the answer as relevant one.

    Args:
        retrieval_result: mapping query_id to its retrieved document ids
        query_answer_path: the file containing the answers
        collection_path: the collection file path
    """
    from scripts.evalnq import load_test_data, validate

    answers, collection = load_test_data(query_answer_path, collection_path)
    return validate(retrieval_result, answers, collection, num_workers=0, batch_size=16)


def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, list):
        return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
    raise TypeError("default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}".format(elem_type))


def _get_title_code(input_path:str, output_path:str, all_line_count:int, start_idx:int, end_idx:int, tokenizer:Any, max_length:int):
    """
    Generate code based on titles of the NQ dataset. Add a padding token at the head.

    Args:
        input_path: the collection file path
        output_path: the ``np.memmap`` file path to save the codes
        all_line_count: the total number of records in the collection
        start_idx: the starting offset
        end_idx: the ending offset
        tokenizer(transformers.AutoTokenizer)
        max_length: the maximum length of tokens

    Returns:
        the populated memmap file
    """
    token_ids = np.memmap(
        output_path,
        shape=(all_line_count, max_length),
        mode="r+",
        dtype=np.int32
    )
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    with open(input_path, 'r') as f:
        pbar = tqdm(total=end_idx-start_idx, desc="Tokenizing", ncols=100, leave=False)
        for idx, line in enumerate(f):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break

            title = line.strip().split('\t')[1]
            # there is always a leading 0
            # - 2 because the leading 0 and the trailing eos_token_id
            output = tokenizer.encode(title, max_length=max_length - 2, padding=False, truncation=True, add_special_tokens=False)
            output.append(eos_token_id)

            token_ids[idx, 1: 1 + len(output)] = output

            pbar.update(1)
        pbar.close()


def isnumber(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def _get_token_code_for_misaligned_tokenizer(input_path:str, output_path:str, all_line_count:int, start_idx:int, end_idx:int, tokenizer:Any, max_length:int, order:str, stop_words:set):
    """
    Generate code based on json files produced by :func:`models.BaseModel.BaseModel.anserini_index`.
    First reorder the words by ``order``, and tokenize the word sequence by ``tokenizer``.
    Add a padding token at the head. Force all codes to be the same length and enclose codes with a sep token.

    Args:
        input_path: the collection file path
        output_path: the ``np.memmap`` file path to save the codes
        all_line_count: the total number of records in the collection
        start_idx: the starting idx
        end_idx: the ending idx
        tokenizer(transformers.AutoTokenizer)
        max_length: the maximum length of tokens
        order: the word order {weight, lexical, orginal}
        stop_words: some words to exclude
    """
    codes = np.memmap(
        output_path,
        mode="r+",
        dtype=np.int32
    ).reshape(all_line_count, max_length)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    k = max_length - 2

    with open(input_path, 'r') as f:
        pbar = tqdm(total=end_idx-start_idx, desc="Tokenizing", ncols=100, leave=False)
        for line in f:
            doc = json.loads(line.strip())
            idx = doc["id"]
            assert start_idx <= idx < end_idx

            word_score_pairs = doc["vector"]

            if len(stop_words):
                filtered_word_score_pairs = {}
                if r"\d" in stop_words:
                    filter_number = True
                else:
                    filter_number = False

                for word, score in word_score_pairs.items():
                    if word in stop_words:
                        continue
                    if filter_number and isnumber(word):
                        continue
                    else:
                        filtered_word_score_pairs[word] = score
                word_score_pairs = filtered_word_score_pairs

            if order == "weight":
                sorted_word = [word for word, score in sorted(word_score_pairs.items(), key=lambda x: x[1], reverse=True)[:k]]
            elif order == "original":
                # get the k largest token score
                sorted_score = sorted(word_score_pairs.values(), reverse=True)
                threshold_score = sorted_score[k - 1] if k <= len(sorted_score) else sorted_score[-1]
                sorted_word = [word for word, score in word_score_pairs.items() if score >= threshold_score]
            elif order == "lexical":
                sorted_score = sorted(word_score_pairs.values(), reverse=True)
                threshold_score = sorted_score[k - 1] if k <= len(sorted_score) else sorted_score[-1]
                sorted_word = sorted([word for word, score in word_score_pairs.items() if score >= threshold_score])
            else:
                raise NotImplementedError

            # - 2 because the leading 0 and the trailing eos_token_id
            # force to pad to max_length - 2
            output = tokenizer.encode(" ".join(sorted_word), max_length=max_length - 2, padding=False, truncation=True, add_special_tokens=False)
            if len(output) < max_length - 2:
                output.extend([pad_token_id for _ in range(max_length - 2 - len(output))])
            output.append(eos_token_id)

            codes[idx, 1:] = output

            pbar.update(1)
        pbar.close()


def _get_token_code_for_aligned_tokenizer(output_path:str, token_ids:np.ndarray, token_weights:np.ndarray, all_line_count:int, start_idx:int, tokenizer:Any, max_length:int, order:str, stop_token_ids:set):
    """
    Generate code based on cached files in ``data/cache/config.dataset/encode/model_name/text_embeddings.mmp`` produced by :func:`models.BaseModel.BaseSparseModel.encode_text`.
    First reorder the words by ``order``, and tokenize the word sequence by ``tokenizer``.
    Add a padding token at the head. Force all codes to be the same length and enclose codes with a sep token.

    Args:
        output_path: the ``np.memmap`` file path to save the codes
        token_ids
        token_weights
        all_line_count: the total number of records in the collection
        start_idx: the starting idx
        end_idx: the ending idx
        tokenizer(transformers.AutoTokenizer)
        max_length: the maximum length of tokens
        order: the word order {weight, lexical, orginal}
        stop_token_ids: the token ids to exclude
    """
    codes = np.memmap(
        output_path,
        mode="r+",
        dtype=np.int32
    ).reshape(all_line_count, max_length)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    k = max_length - 2

    for i, token_id in enumerate(tqdm(token_ids, ncols=100, desc="Collecting Tokens", leave=False)):
        token_weight = token_weights[i]

        # collect all the unique tokens, maintain its insertion order
        unique_token_score_pairs = OrderedDict()
        for j, tok in enumerate(token_id):
            if tok in stop_token_ids:
                continue
            if tok not in unique_token_score_pairs:
                unique_token_score_pairs[tok] = 1e-5
            # maintain the max score of each unique token
            unique_token_score_pairs[tok] = max(token_weight[j], unique_token_score_pairs[tok])

        if order == "weight":
            sorted_token = [tok for tok, score in sorted(unique_token_score_pairs.items(), key=lambda x: x[1], reverse=True)[:k]]
        elif order == "original":
            # get the k largest token score
            threshold_score = sorted(unique_token_score_pairs.values(), reverse=True)[k - 1]
            sorted_token = [tok for tok, score in unique_token_score_pairs.items() if score >= threshold_score]
        elif order == "lexical":
            threshold_score = sorted(unique_token_score_pairs.values(), reverse=True)[k - 1]
            sorted_token = sorted([tok for tok, score in unique_token_score_pairs.items() if score >= threshold_score])
        else:
            raise NotImplementedError

        if len(sorted_token) < max_length - 2:
            sorted_token.extend([pad_token_id for _ in range(max_length - 2 - len(sorted_token))])

        sorted_token.append(eos_token_id)
        # the sorted token must be length of k
        codes[start_idx + i, 1:] = sorted_token


class Sequential_Sampler:
    """
    The sampler used in creating sequential dataloader.
    """
    def __init__(self, dataset_length:int, num_replicas:int, rank:int) -> None:
        """
        Args:
            dataset_length: length of the dataset
            num_replicas: number of splits
            rank: the current process id

        Attributes:
            start: the starting index
            end: the ending index
        """
        super().__init__()
        len_per_worker = dataset_length / num_replicas
        # force to set rank==0 because when world_size==1 the local_rank is -1 by default
        if num_replicas == 1:
            rank = 0
        self.start = round(len_per_worker * rank)
        self.end = round(len_per_worker * (rank + 1))

    def __iter__(self):
        start = self.start
        end = self.end
        return iter(range(start, end, 1))

    def __len__(self):
        return self.end - self.start



class Cluster():
    """
    Mixin for performing a variety of clustering tasks based on ``faiss``.
    """
    def __init__(self, device:DEVICE="cpu"):
        """
        Args:
            device: the gpu id or cpu

        Attributes:
            cluster(faiss.Clustering): the cluster object
            index(faiss.Index): the index to compute distance when clustering
        """
        self.device = device

        self.cluster = None
        self.index = None

        self.logger = MasterLogger("Cluster")


    def _kmeans(self, embeddings:np.ndarray, k:int, metric:str="l2", niter:int=25):
        """
        Perform KMeans over ``embeddings``.

        Args:
            embeddings
            k: the number of clusters
            metric: the metric to compute distance
            niter: number of iterations
        """
        cp = faiss.ClusteringParameters()
        cp.verbose = True
        cp.niter = niter

        kmeans = faiss.Clustering(embeddings.shape[1], k, cp)

        if metric == "ip":
            index = faiss.IndexFlatIP(embeddings.shape[1])
        elif metric == "l2":
            index = faiss.IndexFlatL2(embeddings.shape[1])

        if self.device != "cpu":
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index)

        kmeans.train(embeddings, index)

        self.cluster = kmeans
        self.index = index


    def kmeans(self, embeddings:np.ndarray, k:int, metric:str="l2", num_replicas:int=1, **kargs) -> np.ndarray:
        """
        Fit and predict by kmeans.

        Args:
            embeddings
            k: the number of clusters
            metric: the metric to compute distance
            num_replicas: how many nearest neighbor to record in the final assignmens
            niter: number of iterations

        Returns:
            assignments array of [num_samples, num_replicas]
        """
        self._kmeans(embeddings=embeddings, k=k, metric=metric, **kargs)

        assignments = np.zeros((embeddings.shape[0], num_replicas), dtype=np.int32)
        batch_size = 1000
        for i in range(0, embeddings.shape[0], batch_size):
            q = embeddings[i: i + batch_size]
            score, assignment = self.index.search(q, num_replicas)
            assignments[i: i + batch_size] = assignment

        # discard the 1 dim
        return assignments.squeeze()


    def hierarchical_kmeans(self, embeddings:np.ndarray, k:int, leaf_node_num:int=10, **kargs) -> np.ndarray:
        """
        Fit and predict by hierarchical kmeans.

        Args:
            embeddings
            k: the number of clusters
            leaf_node_num: the maximum number of nodes in the leaf

        Returns:
            assignments array of [num_samples, num_replicas]
        """
        assignments = []

        def classify_recursion(candidates):
            """ iteratively cluster a given candidate set
            Args:
                candidates: list of nodes belonging to a specific cluster, each element in it is an offset in the text_embeddings
            """
            if len(candidates) <= leaf_node_num:
                # when there are only one elements in this cluster, stop adding new ids since the prefix is enough
                if len(candidates) == 1:
                    return
                # when the node number in this cluster is below the threshold
                # just assign each one a unique id
                for idx, cand in enumerate(candidates):
                    assignments[cand].append(idx)
                return

            candidate_embedding = embeddings[candidates]   # C, D

            new_pred = self.kmeans(candidate_embedding, k)

            # nonlocal pre_candidate_num
            # candidate_num = len(candidates)
            # if candidate_num == pre_candidate_num:
            #     print(candidates)
            # pre_candidate_num = candidate_num

            for i in range(k):
                children = []
                for j, label in enumerate(new_pred):
                    if label == i:
                        children.append(candidates[j])
                        # update the hierarchical id of the corresponding node
                        assignments[candidates[j]].append(label)
                classify_recursion(children)
            return

        # create id list for each document
        pred = self.kmeans(embeddings, k, **kargs)
        for label in pred:
            assignments.append([label])

        print(pred)
        for i in tqdm(range(k), ncols=100, desc="Hierarchical Clustering"):
            # the children node in one specific cluster
            children = []
            for j, label in enumerate(pred):
                if label == i:
                    # include the node in the i-th cluster's children
                    children.append(j)
            # iteratively cluster the candidates in the i-th cluster
            classify_recursion(children)

        return assignments


    def get_centroids(self) -> np.ndarray:
        return faiss.vector_to_array(self.cluster.centroids).reshape(self.cluster.k, self.cluster.d)



class MasterLogger():
    """
    The logger only outputs on the master node.
    """
    def __init__(self, name:str) -> None:
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.logger = logging.getLogger(name)

    def info(self, content:str):
        if self.rank == 0:
            return self.logger.info(content)
        else:
            pass

    def warning(self, content:str):
        if self.rank == 0:
            return self.logger.warning(content)
        else:
            pass

    def debug(self, content:str):
        if self.rank == 0:
            return self.logger.debug(content)
        else:
            pass

    def error(self, content:str):
        if self.rank == 0:
            return self.logger.error(content)
        else:
            pass



class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # essential to make pickle saving and loading work
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)



class Config(DotDict):
    """
    Config object. A dot access OrderedDict.
    """
    def __init__(self, *args, **kwargs):
        """
        Launch distributed necessary parameters.
        """
        super().__init__(*args, **kwargs)
        self.setup_distributed()

    def __repr__(self) -> str:
        # skip hidden attributes
        return str({k: v for k, v in super().items() if k[:9] != "_Config__"})

    def items(self):
        # skip hidden attributes
        return [(k, v) for k, v in super().items() if k[:9] != "_Config__"]

    def setup_distributed(self):
        """
        Set up distributed nccl backend.
        """
        # set to hidden attributes
        self.__local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.__local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.__global_world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.__global_rank = int(os.environ.get("RANK", 0))

        # set by torchrun
        if self.world_size > 1 and not dist.is_initialized():
            os.environ["TOKENIZERS_PARALLELISM"] = "True"
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

            os.environ["NCCL_DEBUG"] = "WARN"
            # os.environ["MASTER_ADDR"] = "localhost"
            # os.environ["MASTER_PORT"] = str(12356 + self.config.port_offset)

            # initialize the process group
            # set a large timeout because sometimes we may make the main process do a lot jobs and keep other processes waiting
            dist.init_process_group("nccl", timeout=timedelta(0, 1000000))

            # manager.device will be invoked in the model
            # set the device to the local rank because we may use multi-node distribution
            self.device = self.__local_rank
            # essential to make all_gather_object work properly
            torch.cuda.set_device(self.device)

    @property
    def rank(self):
        # if multi-node distributed, only supports them sharing the same file system
        return self.__global_rank

    @property
    def world_size(self):
        return self.__global_world_size

    @property
    def is_main_proc(self):
        return self.rank == 0

    @property
    def is_distributed(self):
        return self.world_size > 1



@dataclass
class BaseOutput:
    """
    Basic output for :class:`models.BaseModel.BaseModel`
    """
    token_ids: np.ndarray = None
    embeddings: np.ndarray = None
    codes: np.ndarray = None
    index: Any = None
