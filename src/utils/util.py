import os
import json
import faiss
import torch
import pickle
import random
import logging
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from datetime import timedelta
from omegaconf import OmegaConf
from dataclasses import dataclass
from contextlib import contextmanager
from collections import OrderedDict
from transformers import AutoModel, AutoTokenizer
from .static import *


def mean_len(i:Iterable):
    return sum([len(x) for x in i]) / len(i)

def mean(i:Iterable):
    return sum(i) / len(i)


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


def load_attributes(self, foreign_dataset, key_prefix, exclude_keys=[]):
    for k, v in vars(foreign_dataset).items():
        if k.startswith(key_prefix) and k not in exclude_keys:
            setattr(self, k, v)


def load_from_previous(model:torch.nn.Module, path:str):
    """
    Load checkpoint from the older version of Uni-Retriever, only load model parameters and overrides the config by the current config.
    """
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    print(f"loading from {path}...")
    state_dict = ckpt["model"]
    try:
        metrics = ckpt["metrics"]
        model.metrics = metrics
    except:
        pass

    new_state_dict = {}
    for k, v in state_dict.items():
        if "plm" in k:
            new_state_dict[k.replace("plm", "textEncoder")] = v
            new_state_dict[k.replace("plm", "queryEncoder")] = v
        else:
            new_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
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
    return path


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
    remain_keys = []
    for ok in outer_keys:
        ov = src[ok]
        for k, v in dest.items():
            if ok in v:
                # override the same config in the inner config group
                dest[k][ok] = ov
        else:
            remain_keys.append(ok)

    # if there is still remaining parameters
    if len(remain_keys):
        dest["__remaining"] = {}
        for key in remain_keys:
            dest["__remaining"][key] = src[key]

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
    return validate(retrieval_result, answers, collection, num_workers=8, batch_size=8)


def _get_title_code(input_path:str, output_path:str, all_line_count:int, start_idx:int, end_idx:int, tokenizer:Any, max_length:int, title_col_idx=1, separator:str=" "):
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
        title_col_idx: which column is title?

    Returns:
        the populated memmap file
    """
    unk_token_id = tokenizer.unk_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    
    token_ids = np.memmap(
        output_path,
        shape=(all_line_count, max_length),
        mode="r+",
        dtype=np.int32
    )
    if separator != " ":
        sep_id = tokenizer.convert_tokens_to_ids(separator)
        assert sep_id != unk_token_id
        separator += " "

    with open(input_path, 'r') as f:
        pbar = tqdm(total=end_idx-start_idx, desc="Tokenizing", ncols=100, leave=False)
        for idx, line in enumerate(f):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break
            
            fields = [x.strip() for x in line.split("\t")]
            title = fields[title_col_idx]

            # remove extra spaces and lowercase
            title = tokenizer.decode(tokenizer.encode(title, add_special_tokens=False), skip_special_tokens=True).strip().lower()
 
            # some title only contains one word and will be tokenized as unk_token
            if len(title) == 0:
                title =  " ".join(fields[title_col_idx + 1].split(" ")[:5])
                # remove extra spaces and lowercase
                title = tokenizer.decode(tokenizer.encode(title, add_special_tokens=False), skip_special_tokens=True).strip().lower()

            words = title.split(" ")
            
            length = 0
            output = []
            for word in words:
                word = word.replace(",", "")
                encoded = tokenizer.encode(word)
                # ignore unk_tokens
                if unk_token_id in encoded:
                    continue
                encoded = tokenizer.encode(word, add_special_tokens=False)
                # append sep token id
                if separator != " ":
                    encoded += [sep_id]

                if 0 < len(encoded) < max_length - 2: 
                    if length + len(encoded) <= max_length - 2:
                        output.extend(encoded)
                        length += len(encoded)
                    else:
                        break

            output += [eos_token_id]
            token_ids[idx, 1: 1 + len(output)] = output
            pbar.update(1)
        pbar.close()


def isnumber(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def _get_token_code(input_path:str, output_path:str, all_line_count:int, start_idx:int, end_idx:int, tokenizer:Any, max_length:int, init_order:str, post_order:str, stop_words:set, separator:str=" ", stem=False):
    """
    Generate code based on json files produced by :func:`models.BaseModel.BaseModel.anserini_index`.
    First reorder the words by ``order``, and tokenize the word sequence by ``tokenizer``.

    Args:
        input_path: the collection file path
        output_path: the ``np.memmap`` file path to save the codes
        all_line_count: the total number of records in the collection
        start_idx: the starting idx
        end_idx: the ending idx
        tokenizer(transformers.AutoTokenizer)
        max_length: the maximum length of tokens
        model: {weight, first, rand}
        order: the word order {weight, lexical, orginal}
        stop_words: some words to exclude
        separator: used to separate words from words
    """
    unk_token_id = tokenizer.unk_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    codes = np.memmap(
        output_path,
        mode="r+",
        dtype=np.int32
    ).reshape(all_line_count, max_length)

    if separator != " ":
        sep_id = tokenizer.convert_tokens_to_ids(separator)
        assert sep_id != unk_token_id
        separator += " "
    
    if stem:
        from pyserini.analysis import Analyzer, get_lucene_analyzer
        # Default analyzer for English uses the Porter stemmer:
        analyzer = Analyzer(get_lucene_analyzer())

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
                    # if len(word) == 1:
                    #     continue
                    else:
                        filtered_word_score_pairs[word] = score
                word_score_pairs = filtered_word_score_pairs
            
            if stem:
                filtered_word_score_pairs = {}
                for word, score in word_score_pairs.items():
                    # some unicode character produces empty analyzer results
                    try:
                        stemmed_word = analyzer.analyze(word)[0]
                    except:
                        continue
                    
                    if stemmed_word in filtered_word_score_pairs:
                        # pick the maximum score
                        filtered_word_score_pairs[stemmed_word] = (word, max(score, filtered_word_score_pairs[stemmed_word][1]))
                    else:
                        filtered_word_score_pairs[stemmed_word] = (word, score)
                word_score_pairs = {v[0]: v[1] for v in filtered_word_score_pairs.values()}

            if init_order == "weight":
                word_score_pairs = sorted(word_score_pairs.items(), key=lambda x: x[1], reverse=True)
            elif init_order == "first":
                word_score_pairs = list(word_score_pairs.items())
            elif init_order == "rand":
                word_score_pairs = list(word_score_pairs.items())
                random.shuffle(word_score_pairs)
            else:
                raise NotImplementedError(f"Init code order {init_order} not implemented!")

            # NOTE: remove any word consisting of unk_token_id
            length = 0
            output = []
            for word, score in word_score_pairs:
                word = word.replace(",", "")
                encoded = tokenizer.encode(word)
                if unk_token_id in encoded:
                    continue
                encoded = tokenizer.encode(word, add_special_tokens=False)
                # append sep token id
                if separator != " ":
                    encoded += [sep_id]

                # forbid very long words to take up all code length
                # NOTE: try to fill all the blanks
                if 0 < len(encoded) < max_length - 2: 
                    if length + len(encoded) <= max_length - 2:
                        output.append(encoded)
                        length += len(encoded)
                    # else:
                    #     break

            # shuffle
            if post_order == "rand":
                assert init_order != "rand", "Find init_order and post_order both are rand!"
                # there will be at most split words in the final code
                random.shuffle(output)

            # concate all encode words
            output = sum(output, [])
            output += [eos_token_id]
            codes[idx, 1: 1+len(output)] = output
            pbar.update(1)
        pbar.close()


def _get_chatgpt_code(input_path:str, output_path:str, all_line_count:int, start_idx:int, end_idx:int, tokenizer:Any, max_length:int):
    """
    Generate code from chatgpt keywords.

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
        separator: used to separate words from words
    """
    unk_token_id = tokenizer.unk_token_id

    codes = np.memmap(
        output_path,
        mode="r+",
        dtype=np.int32
    ).reshape(all_line_count, max_length)

    separator = ","
    assert tokenizer.convert_tokens_to_ids(separator) != unk_token_id
    separator += " "

    with open(input_path, 'r') as f:
        pbar = tqdm(total=end_idx-start_idx, desc="Tokenizing", ncols=100, leave=False)
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            elif i >= end_idx:
                break

            keywords = line.strip()
            keywords = keywords.split(separator)
            # forbid very long words to take up all code length
            # filter out unk_token_id
            valid_keywords = []
            for keyword in keywords:
                token = tokenizer.encode(keyword, add_special_tokens=False)
                if len(token) < max_length - 2 and unk_token_id not in token:
                    valid_keywords.append(keyword)
            keywords = valid_keywords

            keywords = separator.join(keywords)
            # add separator at the end to make sure all words are of same pattern
            # minus 1 because there is always a leading 0
            output = tokenizer.encode(keywords, max_length=max_length - 1, padding=False, truncation=True)

            decoded = tokenizer.decode(output, skip_special_tokens=True)
            sep = separator[:-1]
            sep_pos = decoded.rfind(sep)
            decoded = decoded[:sep_pos + len(sep)]
            output = tokenizer.encode(decoded, max_length=max_length-1, padding=False)

            codes[i, 1: 1+len(output)] = output

            pbar.update(1)
        pbar.close()



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
        # disable too few points warning
        cp.min_points_per_centroid = 1
        # cp.verbose = True
        cp.niter = niter

        kmeans = faiss.Clustering(embeddings.shape[1], k, cp)

        if metric == "ip":
            index = faiss.IndexFlatIP(embeddings.shape[1])
        elif metric == "l2":
            index = faiss.IndexFlatL2(embeddings.shape[1])
        else:
            raise NotImplementedError(f"Cluster metric {metric}!")

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

    def __repr__(self) -> str:
        # skip hidden attributes
        return str({k: v for k, v in sorted(self.items())})

    def items(self):
        # skip hidden attributes
        return [(k, v) for k, v in super().items() if k[:9] != "_Config__"]

    def __post_init__(self):
        self._set_distributed()
        # attributed starting with __ and set inside the class method is invisible to the outside, and cannot be saved by pickle
        self.__logger = MasterLogger("Config")
        # the logger is involked in the following methods and hence should be initialized first
        self._set_seed()
        self._set_plm()

        # post initialize
        # some awkward dataset specific setting
        # TODO: use hydra optional config
        self.cache_root = os.path.join("data", "cache", self.dataset)

        if self.mode not in ["train", "script"] and self.load_ckpt is None:
            self.load_ckpt = "best"
        if self.mode in ["encode", "encode-query", "encode-text"]:
            self.save_encode = True
        if self.mode == "train" and self.debug:
            self.eval_step = 10
            self.eval_delay = 0
            self.save_ckpt = "debug"
            self.save_index = False
            self.save_encode = False
        
    def _set_distributed(self):
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

            # manager.device will be invoked in the model
            # set the device to the local rank because we may use multi-node distribution
            if self.device != "cpu":
                # initialize the process group; nccl backend for cuda training/inference
                # set a large timeout because sometimes we may make the main process do a lot jobs and keep other processes waiting
                dist.init_process_group("nccl", timeout=timedelta(0, 1000000))

                self.device = self.__local_rank
                # essential to make all_gather_object work properly
                torch.cuda.set_device(self.device)

            else:
                # initialize the process group; gloo backend for cpu training/inference
                # set a large timeout because sometimes we may make the main process do a lot jobs and keep other processes waiting
                dist.init_process_group("gloo", timeout=timedelta(0, 1000000))

    def _set_plm(self, plm:Optional[str]=None, already_on_main_proc=False):
        """
        Load huggingface plms; download it if it doesn't exist. One may add a new plm into the ``PLM_MAP`` object so that Manager knows how to
        download it (``load_name``) and where to store it cache files (``tokenizer``).

        Attributes:
            special_token_ids(Dict[Tuple]): stores the token and token_id of each special tokens
        """
        if plm is None:
            plm = self.plm

        self.logger.info(f"setting PLM to {plm}...")

        self.plm_dir = os.path.join(self.plm_root, plm)
        self.plm_tokenizer = PLM_MAP[plm]["tokenizer"]

        # download plm once
        if self.is_main_proc and not os.path.exists(os.path.join(self.plm_dir, "pytorch_model.bin")):
            print(f"downloading {PLM_MAP[plm]['load_name']}")
            os.makedirs(self.plm_dir, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(PLM_MAP[plm]["load_name"])
            tokenizer.save_pretrained(self.plm_dir)
            model = AutoModel.from_pretrained(PLM_MAP[plm]["load_name"])
            model.save_pretrained(self.plm_dir)
        if not already_on_main_proc:
            synchronize()

        tokenizer = AutoTokenizer.from_pretrained(self.plm_dir)
        # otherwise transofrmers library throws an error logging for some plm that lacks some special token
        with all_logging_disabled():
            self.special_token_ids = {
                "cls": (tokenizer.cls_token, tokenizer.cls_token_id),
                "pad": (tokenizer.pad_token, tokenizer.pad_token_id),
                "unk": (tokenizer.unk_token, tokenizer.unk_token_id),
                "sep": (tokenizer.sep_token, tokenizer.sep_token_id),
                "eos": (tokenizer.eos_token, tokenizer.eos_token_id),
            }
            # try:
            #     bos_token_id = model.config.bos_token_id
        self.vocab_size = tokenizer.vocab_size
        del tokenizer
        # map text_col_sep to special_token if applicable
        self.text_col_sep = self.special_token_ids[self.text_col_sep][0] if self.text_col_sep in self.special_token_ids else self.text_col_sep

    def _set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        self.logger.info(f"setting seed to {seed}...")
        # set seed
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _from_hydra(self, hydra_config:OmegaConf):
        hydra_config = update_hydra_config(OmegaConf.to_container(hydra_config))
        # flatten the config object
        config = flatten_hydra_config(hydra_config)
        for k, v in config.items():
            setattr(self, k, v)

        self.__post_init__()
        if "__remaining" in hydra_config:
            self.logger.info(f"Incoming configs will be set: {hydra_config['__remaining']}")

        self.logger.info(f"Config: {self}")

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

    @property
    def logger(self):
        return self.__logger




@dataclass
class BaseOutput:
    """
    Basic output for :class:`models.BaseModel.BaseModel`
    """
    token_ids: np.ndarray = None
    embeddings: np.ndarray = None
    codes: np.ndarray = None
    index: Any = None
