import os
import collections
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch._six import string_classes
from torch.utils.data import Dataset, IterableDataset, DataLoader
from random import sample, choices, shuffle
from transformers import AutoTokenizer
from .util import load_pickle, save_pickle, load_attributes, MasterLogger, Config
from .static import *



class BaseDataset(Dataset):
    def __init__(self, config) -> None:
        super().__init__()

        self.logger = MasterLogger("Dataset")
        self.config = config

        self.cache_dir = os.path.join(config.cache_root, "dataset")

        self.special_token_ids = [x[1] for x in config.special_token_ids.values() if x[0] is not None]
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)
    
    def _to_numpy(self, data):
        if isinstance(data, Mapping):
            new_data = {}
            for k, v in data.items():
                if isinstance(v, list):
                    new_data[k] = np.array(v, dtype=np.int64)
                elif isinstance(v, Mapping):
                    new_data[k] = self._to_numpy(v)
            new_data = type(data)(new_data)
        elif isinstance(data, list):
            return np.array(data, dtype=np.int64)
        return new_data



class TextDataset(BaseDataset):
    """
    Iterating the collection.
    """
    def __init__(self, config:Config):
        super().__init__(config)
        self.logger.info(f"initializing {config.dataset} {config.data_format} Text dataset...")

        if config.data_format == "memmap":
            text_name = ",".join([str(x) for x in config.text_col])

            if config.text_type == "default":
                self.text_token_ids = np.memmap(
                    os.path.join(self.cache_dir, "text", text_name, config.plm_tokenizer, "token_ids.mmp"),
                    mode="r",
                    dtype=np.int32
                ).reshape(-1, config.max_text_length)[:, :self.config.text_length]
            elif config.text_type == "code":
                self.text_token_ids = np.memmap(
                    os.path.join(config.cache_root, "codes", config.code_type, config.plm_tokenizer, str(config.text_length), "codes.mmp"),
                    # read-only mode
                    mode="r",
                    dtype=np.int32
                ).reshape(-1, config.text_length)
            else:
                raise NotImplementedError(f"Invalid text type {config.text_type}!")
            self.text_num = len(self.text_token_ids)

        elif config.data_format == "raw":
            text_name = "collection.tsv"
            corpus = []
            # the seperator may be special token or regular token
            seperator = config.text_col_sep
            if config.text_type == "default":
                with open(os.path.join(config.data_root, config.dataset, text_name)) as f:
                    for line in f:
                        columns = line.split("\t")
                        text = []
                        for col_idx in config.text_col:
                            text.append(columns[col_idx].strip())
                        text = seperator.join(text)
                        corpus.append(text)
            else:
                raise NotImplementedError(f"Invalid text type {config.text_type} for data format {config.data_format}!")
            self.texts = np.array(corpus, dtype=object)
            self.text_num = len(self.texts)

        if config.get("return_code"):
            self.text_codes = np.memmap(
                os.path.join(config.cache_root, "codes", config.code_type, config.code_tokenizer, str(config.code_length), "codes.mmp"),
                # read-only mode
                mode="r",
                dtype=np.int32
            ).reshape(self.text_num, config.code_length)

        if config.get("return_embedding"):
            self.text_embeddings = np.memmap(
                os.path.join(config.cache_root, "encode", config.embedding_src, "text", config.text_type, "text_embeddings.mmp"),
                mode="r",
                dtype=np.float32
            ).reshape(self.text_num, -1)

        if config.get("enable_distill") == "bi":
            self.text_teacher_embeddings = np.memmap(
                os.path.join(config.cache_root, "encode", config.distill_src, "text", self.config.text_type, "text_embeddings.mmp"),
                mode="r",
                dtype=np.float32
            ).reshape(self.text_num, -1)


    def __len__(self) -> int:
        return self.text_num

    def _prepare_for_model(self, token_id:np.ndarray, max_length):
        if self.config.text_type == "default":
            token_id = token_id[token_id != -1].tolist()
            if self.config.get("text_prefix"):
                token_id = self.tokenizer.encode(self.config.text_prefix, add_special_tokens=False) + token_id
        elif self.config.text_type == "code":
            # neglect the first padding token
            # copy the code because the memmap is read-only
            token_id = token_id[1:].copy()
            valid_token_num = (token_id != -1).sum()
            # mask the last sep_token_id or eos_token_id
            token_id[valid_token_num - 1] = -1
            token_id = token_id[token_id != -1].tolist()
        outputs = self.tokenizer.prepare_for_model(token_id, max_length=max_length, padding="max_length", truncation=True)
        return outputs
    
    def _tokenize(self, inputs, max_length):
        if self.config.get("text_prefix"):
            if isinstance(inputs, str):
                inputs = self.config.text_prefix + inputs
            elif isinstance(inputs, list):
                for i, x in enumerate(inputs):
                    inputs[i] = self.config.text_prefix + x
        return self.tokenizer(inputs, padding="max_length", max_length=max_length, truncation=True)

    def __getitem__(self, index:Union[int,np.ndarray,list]) -> dict:
        """
        Returns:
            dictionary containing one piece of document
        """
        if self.config.data_format == "memmap":
            if isinstance(index, int) or isinstance(index, np.int_):
                text_token_id = self.text_token_ids[index]
                text = self._prepare_for_model(text_token_id, self.config.text_length)
            
            elif isinstance(index, Iterable):
                input_ids = []
                attention_mask = []
                text_token_ids = self.text_token_ids[index]
                # in this case, we need to load a bunch of texts once
                for text_token_id in text_token_ids:
                    text_outputs = self._prepare_for_model(text_token_id, self.config.text_length)
                    input_ids.append(text_outputs.input_ids)
                    attention_mask.append(text_outputs.attention_mask)
                text = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            
            else:
                raise NotImplementedError(f"Invalid index type {type(index)}")

        elif self.config.data_format == "raw":
            if isinstance(index, int) or isinstance(index, np.int_):
                text = self._tokenize(self.texts[index], self.config.text_length)
            
            elif isinstance(index, np.ndarray) or isinstance(index, list):
                text = self.texts[index].tolist()
                text = self._tokenize(text, self.config.text_length)

        return_dict = {
            "text_idx": index,
            "text": self._to_numpy(text)
        }

        if self.config.get("return_code"):
            return_dict["text_code"] = self.text_codes[index].astype(np.int64)

        if self.config.get("return_embedding"):
            return_dict["text_embedding"] = self.text_embeddings[index].astype(np.float32)

        if self.config.get("return_first_mask"):
            text_first_mask = np.zeros_like(text["attention_mask"], dtype=bool)
            token_set = set()
            if text_first_mask.ndim > 1:
                for i, token_ids in enumerate(text["input_ids"]):
                    token_set = set()
                    for j, token_id in enumerate(token_ids):
                        if token_id in self.special_token_ids:
                            continue
                        if token_id in token_set:
                            continue
                        text_first_mask[i, j] = 1
                        token_set.add(token_id)
            
            else:
                for i, token_id in enumerate(text["input_ids"]):
                    if token_id in self.special_token_ids:
                        continue
                    if token_id in token_set:
                        continue
                    text_first_mask[i] = 1
                    token_set.add(token_id)
            return_dict["text_first_mask"] = text_first_mask

        if self.config.get("enable_distill") == "bi":
            return_dict["text_teacher_embedding"] = self.text_teacher_embeddings[index]
            
        return return_dict



class QueryDataset(BaseDataset):
    """
    Iterating the queries of ``config.eval_set``.
    """
    def __init__(self, config:Config, query_set:str="dev"):
        """
        Args:
            mode: which set of queries to load {train, dev}
        """
        super().__init__(config)
        self.logger.info(f"initializing {config.dataset} {config.data_format} Query {query_set} dataset...")
        self.query_set = query_set

        if config.data_format == "memmap":
            self.query_token_ids = np.memmap(
                os.path.join(self.cache_dir, "query", query_set, config.plm_tokenizer, "token_ids.mmp"),
                mode="r",
                dtype=np.int32
            ).reshape(-1, config.max_query_length)[:, :self.config.query_length]
            self.query_num = len(self.query_token_ids)

        elif config.data_format == "raw":
            queries = []
            load_query = load_query.split("-")[0]
            with open(os.path.join(config.data_root, config.dataset, f"queries.{load_query}.tsv")) as f:
                for line in f:
                    query = line.strip().split("\t")[1]
                    queries.append(query)
            self.queries = np.array(queries, dtype=object)
            self.query_num = len(self.queries)

        if config.get("return_embedding"):
            self.query_embeddings = np.memmap(
                os.path.join(config.cache_root, "encode", config.embedding_src, "query", query_set, "query_embeddings.mmp"),
                mode="r",
                dtype=np.float32
            ).reshape(self.query_num, -1)

        if config.get("enable_distill") == "bi":
            self.query_teacher_embeddings = np.memmap(
                os.path.join(config.cache_root, "encode", config.distill_src, "query", query_set, "query_embeddings.mmp"),
                mode="r",
                dtype=np.float32
            ).reshape(self.query_num, 768)
        
    def __len__(self):
        return self.query_num

    def _prepare_for_model(self, token_id:np.ndarray, max_length):
        token_id = token_id[token_id != -1].tolist()
        if self.config.get("query_prefix"):
            token_id = self.tokenizer.encode(self.config.query_prefix, add_special_tokens=False) + token_id
        outputs = self.tokenizer.prepare_for_model(token_id, max_length=max_length, padding="max_length", truncation=True)
        return outputs

    def _tokenize(self, inputs, max_length):
        if self.config.get("text_prefix"):
            if isinstance(inputs, str):
                inputs = self.config.text_prefix + inputs
            elif isinstance(inputs, list):
                for i, x in enumerate(inputs):
                    inputs[i] = self.config.text_prefix + x
        return self.tokenizer(inputs, padding="max_length", max_length=max_length, truncation=True)

    def __getitem__(self, index:Union[int,np.ndarray,list]):
        """
        Returns:
            dictionary containing one piece of query
        """
        if self.config.data_format == "memmap":
            if isinstance(index, int) or isinstance(index, np.int_):
                query_token_id = self.query_token_ids[index]
                query = self._prepare_for_model(query_token_id, self.config.query_length)

            elif isinstance(index, np.ndarray) or isinstance(index, list):
                input_ids = []
                attention_mask = []
                query_token_ids = self.query_token_ids[index]
                # in this case, we need to load a bunch of texts once
                for query_token_id in query_token_ids:
                    query_outputs = self._prepare_for_model(query_token_id, self.config.query_length)
                    input_ids.append(query_outputs.input_ids)
                    attention_mask.append(query_outputs.attention_mask)
                query = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

        elif self.config.data_format == "raw":
            if isinstance(index, int) or isinstance(index, np.int_):
                query = self._tokenize(self.queries[index], self.config.query_length)
            
            elif isinstance(index, np.ndarray) or isinstance(index, list):
                query = self.queries[index].tolist()
                query = self._tokenize(query, self.config.query_length)

        return_dict = {
            "query_idx": index,
            "query": self._to_numpy(query)
        }

        if self.config.get("return_embedding"):
            return_dict["query_embedding"] = self.query_embeddings[index].astype(np.float32)
        
        if self.config.get("enable_distill") == "bi":
            return_dict["query_teacher_embedding"] = self.query_embeddings[index].astype(np.float32)
        
        if self.config.get("return_special_mask"):
            query_token_id = np.array(query["input_ids"])
            sep_pos = (query_token_id[:, None] == self.special_token_ids).any(-1)
            query_special_mask = np.array(query["attention_mask"], dtype=np.int64)
            # mask [SEP] and [CLS]
            query_special_mask[sep_pos] = 0
            return_dict["query_special_mask"] = query_special_mask

        return return_dict



class TrainDataset(BaseDataset):
    def __init__(self, config, text_dataset:TextDataset, query_datasets:list[QueryDataset]):
        super().__init__(config)
        self.logger.info(f"initializing {config.dataset} {config.data_format} Training dataset...")

        self.text_num = text_dataset.text_num
        self.query_num = sum([dataset.query_num for dataset in query_datasets])
        
        if config.get("return_query_code"):
            self.query_codes = []

        qrels = []
        negatives = {}
        for i, query_dataset in enumerate(query_datasets):
            query_set = query_dataset.query_set

            qrel_path = os.path.join(self.cache_dir, "query", query_set, "qrels.pkl")
            negative_path = os.path.join(self.cache_dir, "query", query_set, f"negatives_{config.neg_type}.pkl")
            qrel, negative = self.init_training(i, qrel_path, negative_path)
            # each qrel now have three elements: (query_set_idx, query_idx, text_idx)
            qrels.extend(qrel)
            # negatives are two-layered dictionary, the first layer for different query sets, the second for queries
            negatives[i] = negative
            
            if config.get("return_query_code"):
                self.query_codes.append(np.memmap(
                    os.path.join(config.cache_root, "codes", config.code_type, config.code_tokenizer, str(config.code_length), config.code_src, query_set, "codes.mmp"),
                    mode="r",
                    dtype=np.int32
                ).reshape(len(qrel), -1, config.code_length))
        
        if self.config.get("permute_code") and self.config.permute_code > 0:
            assert config.code_sep is not None
            assert config.neg_type == "none"
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.plm_root, config.code_tokenizer))
            code_sep_id = tokenizer.convert_tokens_to_ids(config.code_sep)
            self.code_sep_id = code_sep_id
            self.eos_token_id = tokenizer.eos_token_id
        
        if self.config.get("elastic_ce"):
            assert self.config.return_code and self.config.code_sep != " "
            assert config.neg_type == "none"
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.plm_root, config.code_tokenizer))
            code_sep_id = tokenizer.convert_tokens_to_ids(config.code_sep)
            self.code_sep_id = code_sep_id
            self.eos_token_id = tokenizer.eos_token_id

        self.qrels = qrels
        self.negatives = negatives
        self.text_dataset = text_dataset
        self.query_datasets = query_datasets


    def init_training(self, query_set_idx, qrel_path, negative_path):
        """
        Initialize qrels and negatives.
        """
        qrels = load_pickle(qrel_path)
        # append the query_set_idx to each qrel element
        # also append qrel_idx
        for i, x in enumerate(qrels):
            qrels[i] = (i, query_set_idx,) + x

        if self.config.neg_type == "random":
            all_positives = set([x[1] for x in qrels])
            sample_range = list(set(range(self.text_num)) - all_positives)
            negatives = defaultdict(lambda:sample_range)
            new_qrels = qrels

        elif self.config.neg_type == "none":
            negatives = {}
            new_qrels = qrels

        else:
            negatives = load_pickle(negative_path)
            for k, v in negatives.items():
                neg_num = len(v)
                if neg_num < self.config.neg_num:
                    negatives[k] = v + choices(v, k=self.config.neg_num - neg_num)
                else:
                    # only use the first 200 negatives
                    negatives[k] = v[:200]

            # discard records that do not have negatives
            new_qrels = []
            for x in qrels:
                # x[2] denotes query_idx
                if x[2] in negatives:
                    new_qrels.append(x)

        return new_qrels, negatives


    def __len__(self):
        return len(self.qrels)


    def __getitem__(self, index) -> dict:
        """
        Returns:
            dictionary containing the query and its positive document and its hard negative documents
        """
        qrel_idx, query_set_idx, query_idx, pos_text_idx = self.qrels[index]
        if self.config.neg_type != "none":
            neg_text_idx = sample(self.negatives[query_set_idx][query_idx], self.config.neg_num)
        else:
            neg_text_idx = []
        text_idx = np.array([pos_text_idx] + neg_text_idx)  # 1 + N

        text_outputs = self.text_dataset[text_idx]
        query_outputs = self.query_datasets[query_set_idx][query_idx]

        return_dict = {
            **text_outputs,
            **query_outputs,
            "qrel_idx": qrel_idx
        }

        if self.config.get("return_prefix_mask"):
            text_code = return_dict["text_code"]
            text_code_prefix_mask = np.ones((1 + self.config.neg_num, self.config.code_length), dtype=np.int64)
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
            raw_query = self.tokenizer.decode(query_outputs["query"]["input_ids"], skip_special_tokens=True)    # string
            raw_text = self.tokenizer.batch_decode(text_outputs["text"]["input_ids"], skip_special_tokens=True) # a list of strings

            raw_query = [raw_query] * len(raw_text)
            max_length = min(self.tokenizer.model_max_length, self.config.text_length + self.config.query_length)
            pair = self.tokenizer(raw_query, raw_text, padding="max_length", max_length=max_length, truncation="only_second")
            return_dict["pair"] = pair
        
        if self.config.get("return_query_code"):
            # FIXME: here is a displeasing workaround
            # we need to maintain a query-text table to generate multiple codes for multiple texts
            return_dict["query_code"] = self.query_codes[query_set_idx][qrel_idx].astype(np.int64)
        
        if self.config.get("permute_code") is not None and self.config.permute_code > 0:
            # there must be only one text
            if "query_code" in return_dict:
                text_code = return_dict["query_code"][0].tolist()
            else:
                text_code = return_dict["text_code"][0].tolist()
            text_codewords = []
            word = []
            # get each keyword
            for c in text_code:
                if c == 0:
                    continue
                if c == -1:
                    break
                word.append(c)
                if c == self.code_sep_id:
                    text_codewords.append(word)
                    word = []
            permuted_text_code = [text_code]
            for i in range(self.config.permute_code):
                order = np.random.permutation(len(text_codewords))
                # prepend 0 at the head
                code = sum([text_codewords[x] for x in order], [0]) + word
                code += [-1 for _ in range(len(text_code) - len(code))]
                # word keeps eos_token_id
                permuted_text_code.append(code)
            # int64
            if "query_code" in return_dict:
                return_dict["query_code"] = np.array(permuted_text_code)
            else:
                return_dict["text_code"] = np.array(permuted_text_code)
        
        if self.config.get("elastic_ce"):
            text_code = return_dict["text_code"][0][1:]
            
            from .index import TrieIndex
            trie = TrieIndex()
            word = []
            for i, c in enumerate(text_code):
                word.append(c)
                if c == self.eos_token_id:
                    break
                if c == self.code_sep_id:
                    trie[word] = True
                    word.clear()
            
            # defaults to -1
            elastic_labels = np.zeros((len(text_code), len(text_code)), dtype=np.int64) - 1
            j = 0
            prefix = []
            for i, c in enumerate(text_code):
                if c == self.eos_token_id:
                    elastic_labels[i, :] = c
                    break
                valid_tokens = trie.get_valid_tokens(prefix)
                elastic_labels[i, :len(valid_tokens)] = valid_tokens
                if len(valid_tokens) < len(text_code):
                    elastic_labels[i, len(valid_tokens):] = -1

                prefix.append(c)
                if c == self.code_sep_id:
                    del trie[prefix]
                    prefix.clear()

            return_dict["elastic_label"] = elastic_labels

        return return_dict



class PairDataset(BaseDataset):
    """
    Dataset for ``config.loader_eval=='rerank'``.

    Attributes:
        qrels(list[tuple[int,int]]): list of ``(qidx,cidx)`` pairs
        labels(list[float]): list of labels for each pair
    """
    def __init__(self, config, text_dataset:TextDataset, query_datasets:list[QueryDataset], mode:str="dev"):
        """
        concat query and sequence
        """
        super().__init__(config)
        self.logger.info(f"initializing {config.dataset} {config.data_format} Pair dataset...")
        # train or eval
        self.mode = mode

        qrels = []
        labels = []
        for i, query_dataset in enumerate(query_datasets):
            query_set = query_dataset.query_set
            if os.path.exists(os.path.join(self.cache_dir, "query", query_set, f"candidates_{config.candidate_type}.pkl")):
                candidate_path = os.path.join(self.cache_dir, "query", query_set, f"candidates_{config.candidate_type}.pkl")
                qrel, label = self.init_pair(i, candidate_path)
            elif os.path.exists(os.path.join(config.cache_root, "retrieve", config.candidate_type, query_set, "retrieval_result.pkl")):
                candidate_path = os.path.join(config.cache_root, "retrieve", config.candidate_type, query_set, "retrieval_result.pkl")
                positive_path = os.path.join(config.cache_root, "dataset", "query", query_set, "positives.pkl")
                qrel, label = self.init_pair(i, candidate_path, positive_path)
            else:
                raise FileNotFoundError(f"Invalid Candidate Type {config.candidate_type}")
            # each qrel now have three elements: (query_set_idx, query_idx, text_idx)
            qrels.extend(qrel)
            labels.extend(label)
        
        self.qrels = qrels
        self.labels = labels
        self.text_dataset = text_dataset
        self.query_datasets = query_datasets

    def init_pair(self, query_set_idx, candidate_path, positive_path=None):
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
                    qrels.append((query_set_idx, qidx, candidate))
                    if candidate in positives[qidx]:
                        has_pos = True
                        labels.append(1)
                    else:
                        labels.append(0)
                # if the retrieval result has no ground truth document, we manually add 1 pair in training
                if not has_pos and self.mode == "train":
                    qrels.append([query_set_idx, qidx, positives[qidx][0]])
                    labels.append(1)
            else:
                for i, candidate in enumerate(candidates):
                    # for point-wise training
                    qrels.append([query_set_idx, qidx, candidate[0]])
                    labels.append(candidate[1])
        return qrels, labels

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, index):
        """
        Returns:
            the dictionary containing the query and its candidate
        """
        query_set_idx, query_idx, text_idx = self.qrels[index]
        label = self.labels[index]

        text_outputs = self.text_dataset[text_idx]
        query_outputs = self.query_datasets[query_set_idx][query_idx]

        return_dict = {
            "label": float(label),
            **text_outputs,
            **query_outputs
        }

        if self.config.get("return_pair"):
            raw_query = self.tokenizer.decode(query_outputs["query"]["input_ids"], skip_special_tokens=True)    # string
            raw_text = self.tokenizer.decode(text_outputs["text"]["input_ids"], skip_special_tokens=True) # string

            max_length = min(self.tokenizer.model_max_length, self.config.text_length + self.config.query_length)
            pair = self.tokenizer(raw_query, raw_text, padding="max_length", max_length=max_length, truncation="only_second")
            return_dict["pair"] = pair

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
        return self.tokenizer(input, padding="max_length", max_length=max_length, truncation=True)


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
                    "query": self._to_numpy(query),
                    "text": self._to_numpy(text)
                }

                if self.config.get("return_special_mask"):
                    query_token_id = np.array(query["input_ids"])
                    sep_pos = (query_token_id[:, None] == self.special_token_ids).any(-1)
                    query_special_mask = np.array(query["attention_mask"], dtype=np.int64)
                    # mask [SEP] and [CLS]
                    query_special_mask[sep_pos] = 0
                    return_dict["query_special_mask"] = query_special_mask

                yield return_dict

            # when the file reaches its end, reset the cursor so that calling iter(dataset) would read the file from the beginning
            else:
                triples.seek(0)



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
        self.rank = rank

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
    Prepare dataloader for evaluation.

    Returns:
        dict[str, DataLoader]:

            text: dataloader for text corpus if ``config.loader_text`` is not ``none``;

            query: dataloader for query in ``eval_set`` if ``config.loader_eval`` is ``retrieve``;

            rerank: dataloader for rerank if ``config.loader_eval`` is ``rerank``;

    """
    loaders = {}
    text_dataset = TextDataset(config)
    query_dataset = QueryDataset(config, config.eval_set)

    if config.parallel == "text":
        sampler_text = Sequential_Sampler(len(text_dataset), num_replicas=config.world_size, rank=config.rank)
    else:
        sampler_text = Sequential_Sampler(len(text_dataset), num_replicas=1, rank=0)
    loaders["text"] = DataLoader(text_dataset, batch_size=config.eval_batch_size, sampler=sampler_text, num_workers=config.num_worker, collate_fn=default_collate)

    if config.parallel == "query":
        sampler_query = Sequential_Sampler(len(query_dataset), num_replicas=config.world_size, rank=config.rank)
    else:
        sampler_query = Sequential_Sampler(len(query_dataset), num_replicas=1, rank=0)
    loaders["query"] = DataLoader(query_dataset, batch_size=config.eval_batch_size, sampler=sampler_query, num_workers=config.num_worker, collate_fn=default_collate)

    if config.eval_mode == "rerank":
        # pass in a list with only one element
        rerank_dataset = PairDataset(config, text_dataset, [query_dataset])
        sampler_rerank = Sequential_Sampler(len(rerank_dataset), num_replicas=config.world_size, rank=config.rank)
        loaders["rerank"] = DataLoader(rerank_dataset, batch_size=config.eval_batch_size, sampler=sampler_rerank, num_workers=config.num_worker, collate_fn=default_collate)

    return loaders


def prepare_train_data(config, text_dataset=None, return_dataloader=False):
    """
    Args:
        text_dataset: the text dataset constructed in advance
        return_dataloader: also return dataloader
    """
    if text_dataset is None:
        text_dataset = TextDataset(config)

    query_datasets = []
    for query_set in config.train_set:
        query_datasets.append(QueryDataset(config, query_set))
    
    if config.get("loader_train") == "neg":
        train_dataset = TrainDataset(config, text_dataset, query_datasets)
    elif config.get("loader_train") == "triple-raw":
        train_dataset = RawTripleTrainDataset(config)
    elif config.get("loader_train") == "pair":
        train_dataset = PairDataset(config, text_dataset, query_datasets, mode="train")
    else:
        raise NotImplementedError(f"Train loader type {config.loader_train} not  implemented!")
    # only used in developing (dev.ipynb)
    if return_dataloader:
        sampler_train = Sequential_Sampler(len(train_dataset), num_replicas=config.world_size, rank=config.rank)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=default_collate, sampler=sampler_train)
        return train_loader
    return train_dataset
