import os
import collections
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict
from torch._six import string_classes
from torch.utils.data import Dataset, IterableDataset, DataLoader
from random import sample, choice, shuffle
from transformers import AutoTokenizer
from .util import load_pickle, save_pickle, MasterLogger, Config
from .typings import *



class BaseDataset(Dataset):
    """
    Base class for all datasets.
    """
    def __init__(self, config:Config, load_text:bool=False, load_query:bool=False):
        """
        Args:
            config: the config object
            load_text: if ``True``, load text related data
            load_query: if ``True``, load query related data

        Attributes:
            config
            cache_dir(str): the cache folder for the dataset related data
            _rank(int): the current process ID
            logger(MasterLogger)
        """
        super().__init__()

        self.logger = MasterLogger("Dataset")
        self.config = config

        self.cache_dir = os.path.join(config.cache_root, "dataset")

        self.cls_token_id = config.special_token_ids["cls"][1]
        self.sep_token_id = config.special_token_ids["sep"][1]
        self.pad_token_id = config.special_token_ids["pad"][1]
        self.special_token_ids = [x[1] for x in config.special_token_ids.values() if x[0] is not None]

        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)

        if load_text:
            if load_text == "memmap":
                text_name = ",".join([str(x) for x in config.text_col])

                self.text_token_ids = np.memmap(
                    os.path.join(self.cache_dir, "text", text_name, config.plm_tokenizer, "token_ids.mmp"),
                    mode="r",
                    dtype=np.int32
                ).reshape(-1, config.max_text_length)[:, :self.config.text_length]
                self.text_num = len(self.text_token_ids)

            elif load_text == "raw":
                text_suffix = []
                text_name = "collection.tsv"

                corpus = []
                # the seperator may be special token or regular token
                seperator = config.text_col_sep
                with open(os.path.join(config.data_root, config.dataset, text_name)) as f:
                    for line in f:
                        columns = line.split("\t")
                        text = []
                        for col_idx in config.text_col:
                            text.append(columns[col_idx].strip())
                        text = seperator.join(text)
                        corpus.append(text)
                self.corpus = np.array(corpus, dtype=object)
                self.text_num = len(self.corpus)

            if config.get("return_code"):
                self.text_codes = np.memmap(
                    os.path.join(config.cache_root, "codes", config.code_type, config.code_tokenizer, str(config.code_length), "codes.mmp"),
                    # read-only mode
                    mode="r",
                    dtype=np.int32
                ).reshape(self.text_num, config.code_length)

            if config.get("return_embedding"):
                self.text_embeddings = np.memmap(
                    os.path.join(config.cache_root, "encode", config.embedding_src, "text_embeddings.mmp"),
                    mode="r",
                    dtype=np.float32
                ).reshape(self.text_num, -1)

        if load_query:
            assert load_query in ["train-memmap", "dev-memmap", "test-memmap", "train-raw", "dev-raw", "test-raw"]
            mode, data_format = load_query.split("-")

            if data_format == "memmap":
                self.query_token_ids = np.memmap(
                    os.path.join(self.cache_dir, mode, config.plm_tokenizer, "token_ids.mmp"),
                    mode="r",
                    dtype=np.int32
                ).reshape(-1, config.max_query_length)[:, :self.config.query_length]
                self.query_num = len(self.query_token_ids)

            elif data_format == "raw":
                queries = []
                load_query = load_query.split("-")[0]
                with open(os.path.join(config.data_root, config.dataset, f"queries.{load_query}.small.tsv")) as f:
                    for line in f:
                        parsed_input = line.strip().split("\t")[1]
                        queries.append(parsed_input)
                self.queries = np.array(queries, dtype=object)
                self.query_num = len(self.queries)

                if not hasattr(self, "tokenizer"):
                    self.tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)

            if config.get("return_embedding"):
                self.query_embeddings = np.memmap(
                    os.path.join(config.cache_root, "encode", config.embedding_src, mode, "query_embeddings.mmp"),
                    mode="r",
                    dtype=np.float32
                ).reshape(self.query_num, -1)


    def init_negative(self, qrel_path, negative_path):
        """
        Initialize qrels and negatives.
        """
        qrels = load_pickle(qrel_path)

        if self.config.hard_neg_type == "random":
            all_positives = set([x[1] for x in qrels])
            sample_range = list(set(range(self.text_num)) - all_positives)
            negatives = defaultdict(lambda:sample_range)
            new_qrels = qrels

        elif self.config.hard_neg_type == "none":
            negatives = None
            new_qrels = qrels

        else:
            all_positives = set([x[1] for x in qrels])
            sample_range = list(set(range(self.text_num)) - all_positives)

            negatives = load_pickle(negative_path)
            for k,v in negatives.items():
                neg_num = len(v)
                # use -1 to indicate padded text, which will be masked in __getitem__ function
                if neg_num < self.config.hard_neg_num:
                    negatives[k].extend(sample(sample_range, self.config.hard_neg_num - neg_num))
                else:
                    # only use the first 1000 negatives
                    negatives[k] = v[:200]

            new_qrels = []
            for x in qrels:
                if x[0] in negatives:
                    new_qrels.append(x)

        return new_qrels, negatives


    def init_pair(self, candidate_path, positive_path=None):
        self.logger.info(f"collecting candidates from {candidate_path}")
        qrels = []
        labels = []
        candidates = load_pickle(candidate_path)
        if positive_path is not None:
            positives = load_pickle(positive_path)

        for qidx, candidates in tqdm(candidates.items(), ncols=100, leave=False):
            if positive_path:
                candidates = candidates[:self.config.candidate_num_train] if self.mode == "train" else candidates[:self.config.candidate_num]
                has_pos = False
                for i, candidate in enumerate(candidates):
                    qrels.append([qidx, candidate])
                    if candidate in positives[qidx]:
                        has_pos = True
                        labels.append(1)
                    else:
                        labels.append(0)
                # if the retrieval result has no ground truth document, we manually add 1 pair in training
                if not has_pos and self.mode == "train":
                    qrels.append([qidx, positives[qidx][0]])
                    labels.append(1)
            else:
                for i, candidate in enumerate(candidates):
                    # for point-wise training
                    qrels.append([qidx, candidate[0]])
                    labels.append(candidate[1])

        return qrels, labels


    def prepare_for_model(self, token_id, max_length):
        assert isinstance(token_id, list)
        outputs = self.tokenizer.prepare_for_model(token_id, max_length=max_length, padding="max_length", truncation=True, return_token_type_ids=False)
        return outputs


    def tokenize(self, input, max_length):
        return self.tokenizer(input, padding="max_length", max_length=max_length, truncation=True, return_token_type_ids=False)



class TrainDataset(BaseDataset):
    """
    Training dataset for ``config.loader_train=='neg'``.

    Attributes:
        qrels(list[tuple]): the list of ``(query_idx, pos_text_idx)`` pairs
        negatives(dict[int, list[int]]): the mapping from query index to its hard negative document index list
    """
    def __init__(self, config, data_format="memmap"):
        super().__init__(config, load_text=data_format, load_query=f"train-{data_format}")
        self.logger.info(f"initializing {config.dataset} {data_format} Training dataset...")
        self.data_format = data_format

        qrel_path = os.path.join(self.cache_dir, "train", "qrels.pkl")
        negative_path = os.path.join(self.cache_dir, "train", f"negatives_{config.hard_neg_type}.pkl")
        self.qrels, self.negatives = self.init_negative(qrel_path, negative_path)

        if config.enable_distill == "bi":
            self.text_teacher_embeddings = np.memmap(
                os.path.join(config.cache_root, "encode", config.distill_src, "text_embeddings.mmp"),
                mode="r",
                dtype=np.float32
            ).reshape(self.text_num, -1)

            self.query_teacher_embeddings = np.memmap(
                os.path.join(config.cache_root, "encode", config.distill_src, "train", "query_embeddings.mmp"),
                mode="r",
                dtype=np.float32
            ).reshape(self.query_num, 768)


    def __len__(self):
        return len(self.qrels)


    def __getitem__(self, index) -> dict:
        """
        Returns:
            dictionary containing the query and its positive document and its hard negative documents
        """
        query_idx, pos_text_idx = self.qrels[index]
        neg_text_idx = sample(self.negatives[query_idx], self.config.hard_neg_num)
        text_idx = np.array([pos_text_idx] + neg_text_idx)  # 1 + N

        if self.data_format == "memmap":
            query_token_id = self.query_token_ids[query_idx]
            query_token_id = query_token_id[query_token_id != -1].tolist()
            query = self.prepare_for_model(query_token_id, self.config.query_length)

            input_ids = []
            attention_mask = []

            text_token_ids = self.text_token_ids[text_idx]
            for text_token_id in text_token_ids:
                text_token_id = text_token_id[text_token_id != -1].tolist()
                text_outputs = self.prepare_for_model(text_token_id, self.config.text_length)
                input_ids.append(text_outputs.input_ids)
                attention_mask.append(text_outputs.attention_mask)

            # wrap the data for transformers in a dictionary
            text = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

        elif self.data_format == "raw":
            query = self.tokenize(self.queries[query_idx], self.config.query_length)

            # convert array to list so that tokenizer can recognize
            text = self.corpus[text_idx].tolist()
            text = self.tokenize(text, self.config.text_length)

        # default to int64 so that it can be directly converted to long tensor
        return_dict = {
            "query_idx": query_idx,
            "text_idx": text_idx,
            "query": query,
            "text": text,
        }

        if self.config.get("return_sep_mask"):
            query_token_id = np.array(query["input_ids"])
            sep_pos = ((query_token_id == self.cls_token_id) + (query_token_id == self.sep_token_id)).astype(np.bool)
            query_sep_mask = np.array(query["attention_mask"], dtype=np.int64)
            # mask [SEP] and [CLS]
            query_sep_mask[sep_pos] = 0
            return_dict["query_sep_mask"] = query_sep_mask

        if self.config.enable_distill == "bi":
            return_dict["query_teacher_embedding"] = self.query_teacher_embeddings[query_idx].astype(np.float32)
            return_dict["text_teacher_embedding"] = self.text_teacher_embeddings[text_idx].astype(np.float32)
        elif self.config.enable_distill == "cross":
            # TODO
            pass

        if self.config.get("return_embedding"):
            return_dict["query_embedding"] = self.query_embeddings[query_idx].astype(np.float32)
            return_dict["text_embedding"] = self.text_embeddings[text_idx].astype(np.float32)

        if self.config.get("return_code"):
            text_code = self.text_codes[text_idx].astype(np.int64)
            return_dict["text_code"] = text_code

            if self.config.get("return_prefix_mask"):
                text_code_prefix_mask = np.ones((1 + self.config.hard_neg_num, self.config.code_length), dtype=np.int64)
                # ignore the leading 0
                pos_text_code = text_code[0]
                for i in range(1, len(text_code)):
                    # loop over negative texts
                    for j, code in enumerate(text_code[i]):
                        # when the neg_text_code shares the same prefix with the pos_text_code, then the mask would be 0
                        if code == pos_text_code[j]:
                            text_code_prefix_mask[i, j] = 0
                        # when they are different, the following code will no longer share the same prefix, so all the following mask would be 1
                        else:
                            break
                return_dict["text_code_prefix_mask"] = text_code_prefix_mask

        if self.config.get("return_pair"):
            if self.data_format == "raw":
                query = self.queries[query_idx]
                text = self.corpus[text_idx].tolist()    # 1+N
                query = [query] * len(text)

                max_length = min(self.tokenizer.model_max_length, self.config.text_length + self.config.query_length)
                pair = self.tokenizer(query, text, padding="max_length", max_length=max_length, truncation="only_second")
                return_dict["pair"] = pair

            else:
                raise ValueError("Use raw data format to enable return_pair!")

        return return_dict



class TextDataset(BaseDataset):
    """
    Iterating the collection.
    """
    def __init__(self, config:Config, data_format:DATA_FORMAT="memmap"):
        super().__init__(config, load_text=data_format)
        self.logger.info(f"initializing {config.dataset} {data_format} Text dataset...")
        self.data_format = data_format


    def __len__(self) -> int:
        return self.text_num


    def __getitem__(self, index:int) -> dict:
        """
        Returns:
            dictionary containing one piece of document
        """
        if self.data_format == "memmap":
            text_token_id = self.text_token_ids[index]
            text_token_id = text_token_id[text_token_id != -1].tolist()
            text = self.prepare_for_model(text_token_id, self.config.text_length)

        elif self.data_format == "raw":
            text = self.tokenize(self.corpus[index], self.config.text_length)

        return_dict = {
            "text_idx": index,
            "text": text
        }

        if self.config.get("return_code"):
            return_dict["text_code"] = self.text_codes[index].astype(np.int64)

        if self.config.get("return_embedding"):
            return_dict["text_embedding"] = self.text_embeddings[index].astype(np.float32)

        if self.config.get("return_first_mask"):
            text_first_mask = np.zeros(self.config.text_length, dtype=np.bool)
            token_set = set()
            for i, token_id in enumerate(text["input_ids"]):
                if token_id in self.special_token_ids:
                    continue
                if token_id in token_set:
                    continue
                text_first_mask[i] = 1
                token_set.add(token_id)
            return_dict["text_first_mask"] = text_first_mask
        return return_dict



class QueryDataset(BaseDataset):
    """
    Iterating the queries of ``config.eval_set``.
    """
    def __init__(self, config:Config, mode:str="dev", data_format:DATA_FORMAT="memmap"):
        """
        Args:
            mode: which set of queries to load {train, dev}
        """
        super().__init__(config, load_query=f"{mode}-{data_format}")
        self.logger.info(f"initializing {config.dataset} {data_format} Query dataset...")
        self.data_format = data_format


    def __len__(self):
        return self.query_num


    def __getitem__(self, index:int):
        """
        Returns:
            dictionary containing one piece of query
        """
        if self.data_format == "memmap":
            query_token_id = self.query_token_ids[index]
            query_token_id = query_token_id[query_token_id != -1].tolist()
            query = self.prepare_for_model(query_token_id, self.config.query_length)

        elif self.data_format == "raw":
            query = self.tokenize(self.queries[index], self.config.query_length)

        return_dict = {
            "query_idx": index,
            "query": query
        }

        if self.config.get("return_embedding"):
            return_dict["query_embedding"] = self.query_embeddings[index].astype(np.float32)

        return return_dict



class PairDataset(BaseDataset):
    """
    Dataset for ``config.loader_eval=='rerank'``.

    Attributes:
        qrels(list[tuple[int,int]]): list of ``(qidx,cidx)`` pairs
        labels(list[float]): list of labels for each pair
    """
    def __init__(self, config, mode:str="dev", data_format:DATA_FORMAT="memmap"):
        """
        concat query and sequence
        """
        super().__init__(config, load_text=data_format, load_query=f"{mode}-{data_format}")
        self.logger.info(f"initializing {config.dataset} {data_format} Rerank {mode} dataset...")
        self.data_format = data_format

        self.mode = mode
        if os.path.exists(os.path.join(self.cache_dir, mode, f"candidates_{config.candidate_type}.pkl")):
            candidate_path = os.path.join(self.cache_dir, mode, f"candidates_{config.candidate_type}.pkl")
            self.qrels, self.labels = self.init_pair(candidate_path)
        elif os.path.exists(os.path.join(config.cache_root, "retrieve", config.candidate_type, mode, "retrieval_result.pkl")):
            candidate_path = os.path.join(config.cache_root, "retrieve", config.candidate_type, mode, "retrieval_result.pkl")
            positive_path = os.path.join(config.cache_root, "dataset", mode, "positives.pkl")
            self.qrels, self.labels = self.init_pair(candidate_path, positive_path)
        else:
            raise FileNotFoundError(f"Invalid Candidate Type {config.candidate_type}")


    def __len__(self):
        return len(self.qrels)


    def __getitem__(self, index):
        """
        Returns:
            the dictionary containing the query and its candidate
        """
        query_idx, text_idx = self.qrels[index]
        label = self.labels[index]

        if self.data_format == "memmap":
            query_token_id = self.query_token_ids[query_idx]
            query_token_id = query_token_id[query_token_id != -1].tolist()
            query = self.prepare_for_model(query_token_id, self.config.query_length)

            text_token_id = self.text_token_ids[text_idx]
            text_token_id = text_token_id[text_token_id != -1].tolist()
            text = self.prepare_for_model(text_token_id, self.config.text_length)

        elif self.data_format == "raw":
            query = self.tokenize(self.queries[query_idx], self.config.query_length)
            text = self.tokenize(self.corpus[text_idx], self.config.text_length)

        return_dict = {
            "label": label,
            "text_idx": text_idx,
            "query_idx": query_idx,
            "query": query,
            "text": text
        }

        if self.config.get("return_code"):
            return_dict["text_code"] = self.text_codes[text_idx].astype(np.int64)

        if self.config.get("return_pair"):
            if self.data_format == "raw":
                query = self.queries[query_idx]
                text = self.corpus[text_idx]
                max_length = min(self.tokenizer.model_max_length, self.config.text_length + self.config.query_length)
                pair = self.tokenizer(query, text, padding="max_length", max_length=max_length, truncation="only_second")
                return_dict["pair"] = pair

            else:
                raise ValueError("Use raw data format to enable return_pair!")

        return return_dict



class RawTripleTrainDataset(IterableDataset):
    """
    Training dataset for ``config.loader_train=='triple-raw'``.
    Use :class:`utils.util.IterableDataloader` to load.

    Attributes:
        triple_path(str): file containing triples of ``(query, positive doc, negative doc)`` pairs.
    """
    def __init__(self, config) -> None:
        """
        iterably load the triples, tokenize and return
        """
        super().__init__()

        self.logger = MasterLogger("Dataset")

        self.logger.info(f"initializing {config.dataset} raw Triple Training dataset...")

        self.config = config

        self.triple_path = os.path.join(config.data_root, config.dataset, f"triples.{config.triple_type}.tsv")
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)
        self.special_token_ids = [x[1] for x in self.config.special_token_ids.values() if x is not None]


    def tokenize(self, input, max_length):
        return self.tokenizer(input, padding="max_length", max_length=max_length, truncation=True, return_token_type_ids=False)


    def __iter__(self):
        """
        Yields:
            dictionary containing the query and its positive document and its raw triple negative document
        """
        # reopen the file every time
        with open(self.triple_path, "r") as triples:
            for line in triples:
                fields = line.strip().split("\t")
                query, pos_text, neg_text = fields[:3]

                query = self.tokenize(query, self.config.query_length)

                text = self.tokenize([pos_text, neg_text], self.config.text_length)

                # default to int64 so that it can be directly converted to long tensor
                return_dict = {
                    "query": query,
                    "text": text
                }

                if self.config.get("return_sep_mask"):
                    query_token_id = np.array(query["input_ids"])
                    sep_pos = ((query_token_id == self.cls_token_id) + (query_token_id == self.sep_token_id)).astype(np.bool)
                    query_sep_mask = np.array(query["attention_mask"], dtype=np.int64)
                    # mask [SEP] and [CLS]
                    query_sep_mask[sep_pos] = 0
                    return_dict["query_sep_mask"] = query_sep_mask

                yield return_dict

            # when the file reaches its end, reset the cursor so that calling iter(dataset) would read the file from the beginning
            else:
                triples.seek(0)



class NMTTrainDataset(BaseDataset):
    """
    Training dataset for ``config.loader_train=='nmt'``.

    Attributes:
        qrels(list[tuple[int,int]]): list of ``(qidx,cidx)`` pairs

    """
    def __init__(self, config, data_format:DATA_FORMAT="memmap"):
        assert config.get("return_code")

        super().__init__(config, load_text=data_format, load_query=f"train-{data_format}")
        self.logger.info(f"initializing {config.dataset} NMT Train dataset...")

        positives = load_pickle(os.path.join(config.cache_root, "dataset", "train", "positives.pkl"))
        candidates = {}
        for k, v in positives.items():
            candidates[k] = v

        if config.get("return_doct5"):
            text_name = ",".join([str(x) for x in config.text_col])
            doct5_token_ids = np.memmap(
                os.path.join(self.cache_dir, "text", text_name, config.plm_tokenizer, "doct5.mmp"),
                mode="r",
                dtype=np.int32
            ).reshape(self.text_num, -1, config.query_length)[:, :config.query_per_doc]
            # one query is one row
            self.doct5_token_ids = doct5_token_ids.reshape(-1, config.query_length)

            offset = max(positives.keys())
            for i in range(self.text_num):
                # 1 is the positive label
                for j in range(config.query_per_doc):
                    candidates[i * config.query_per_doc + j + offset] = [i]

        qrels = []
        for qidx, candidates in tqdm(candidates.items(), ncols=100, desc="Collecting Qrels", leave=False):
            for candidate in candidates:
                # for point-wise training
                qrels.append([qidx, candidate])

        self.qrels = qrels


    def __len__(self):
        return len(self.qrels)


    def __getitem__(self, index):
        query_idx, text_idx = self.qrels[index]

        if query_idx >= self.query_num:
            new_query_idx = query_idx - self.query_num
            query_token_id = self.doct5_token_ids[new_query_idx]
        else:
            query_token_id = self.query_token_ids[query_idx]

        query_token_id = query_token_id[query_token_id != -1].tolist()
        query = self.prepare_for_model(query_token_id, self.config.query_length)

        # remove the leading padding token
        text_code = self.text_codes[text_idx].astype(np.int64)

        return_dict = {
            "text_idx": text_idx,
            "query_idx": query_idx,
            "query": query,
            "text_code": text_code
        }
        return return_dict



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




def prepare_data(config) -> LOADERS:
    """
    Prepare dataloader for training/evaluating.

    Returns:
        dict[str, DataLoader]:

            train: dataloader for trianing; including neg/triple/triple-raw...

            text: dataloader for text corpus if ``config.loader_text`` is not ``none``;

            query: dataloader for query in ``eval_set`` if ``config.loader_eval`` is ``retrieve``;

            rerank: dataloader for rerank if ``config.loader_eval`` is ``rerank``;

    """
    loaders = {}

    if config.loader_text != "none":
        dataset_passage = TextDataset(config, config.loader_text)
        if config.parallel == "text":
            sampler_passage = Sequential_Sampler(len(dataset_passage), num_replicas=config.world_size, rank=config.rank)
        else:
            sampler_passage = Sequential_Sampler(len(dataset_passage), num_replicas=1, rank=0)
        loaders["text"] = DataLoader(dataset_passage, batch_size=config.eval_batch_size, sampler=sampler_passage, num_workers=config.num_worker, collate_fn=default_collate)

    if config.loader_query != "none":
        dataset_query = QueryDataset(config, mode=config.eval_set, data_format=config.loader_query)
        if config.parallel == "query":
            sampler_query = Sequential_Sampler(len(dataset_query), num_replicas=config.world_size, rank=config.rank)
        else:
            sampler_query = Sequential_Sampler(len(dataset_query), num_replicas=1, rank=0)
        loaders["query"] = DataLoader(dataset_query, batch_size=config.eval_batch_size, sampler=sampler_query, num_workers=config.num_worker, collate_fn=default_collate)

    if config.get("loader_rerank") != "none":
        dataset_rerank = PairDataset(config, config.eval_set, data_format=config.loader_rerank)
        sampler_rerank = Sequential_Sampler(len(dataset_rerank), num_replicas=config.world_size, rank=config.rank)
        loaders["rerank"] = DataLoader(dataset_rerank, batch_size=config.eval_batch_size, sampler=sampler_rerank, num_workers=config.num_worker, collate_fn=default_collate)

    # import psutil
    # memory_consumption = round(psutil.Process().memory_info().rss / 1e6)
    # self.logger.info(f"Memory Usage of Curren Process is {memory_consumption} MB!")

    return loaders


def prepare_train_data(config, return_dataloader=False):
    if config.get("loader_train") == "neg":
        train_dataset = TrainDataset(config)
    elif config.get("loader_train") == "neg-raw":
        train_dataset = TrainDataset(config, data_format="raw")
    elif config.get("loader_train") == "triple-raw":
        train_dataset = RawTripleTrainDataset(config)
    elif config.get("loader_train") == "pair":
        train_dataset = PairDataset(config, mode="train")
    elif config.get("loader_train") == "pair-memmap":
        train_dataset = PairDataset(config, mode="train", data_format="memmap")
    elif config.get("loader_train") == "nmt":
        train_dataset = NMTTrainDataset(config)

    # only used in developing (dev.ipynb)
    if return_dataloader:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=default_collate)
        return train_loader

    return train_dataset

