import os
import time
import torch
import psutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from typing import Optional, Mapping
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils.util import load_pickle, save_pickle, compute_metrics, compute_metrics_nq, makedirs, readlink, synchronize, BaseOutput, MasterLogger, Config
from utils.index import *



class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    def __init__(self, config:Config):
        """
        Args:
            config: the configuration object intialized by :func:`utils.manager.Manager.setup`
            name: the name of the model

        Attributes:
            metrics(dict): the metric dictionary containing ``metric_type: metric_value`` pairs
            config
            name(str): the name of the model
            index_dir(str): the folder to save index e.g. :mod:`utils.index.FaissIndex` and :mod:`utils.index.InvertedVectorIndex`
            collection_dir(str): the folder to save json collections returned by e.g. :func:`utils.index.AnseriniBM25Index.fit`
            encode_dir(str): the folder to save text encoding memmap file returned by :func:`models.BaseModel.BaseSparseModel.encode_text`
            query_dir(str): the folder to save query encoding memmap file returned by :func:`models.BaseModel.BaseSparseModel.encode_query`
            retrieval_result_path(str): the path of the final retrieval result file returned by :func:`models.BaseModel.BaseModel.retrieve`
            _rank(int): the current process ID
            _world_size(int): the number of all processes
            _distributed(bool): if distributed training/evaluating is enabled
            logger(MasterLoger): the logger
        """
        super().__init__()

        # the model's performance, populated when evaluating
        self.metrics = {}
        self.config = config
        self.name = config.name

        # all the following attributes can be generated according to name
        self.index_dir = os.path.join(config.cache_root, "index", self.name, "index")
        self.collection_dir = os.path.join(config.cache_root, "index", self.name, "collection")
        self.retrieve_dir = os.path.join(config.cache_root, config.eval_mode, self.name, config.eval_set)
        self.encode_dir = os.path.join(config.cache_root, "encode", self.name)
        self.query_dir = os.path.join(self.encode_dir, config.eval_set)
        # refered by transformer trainer
        self.ckpt_dir = os.path.join(config.cache_root, "ckpts", self.name)

        self.retrieval_result_path = os.path.join(self.retrieve_dir, "retrieval_result.pkl")

        self.logger = MasterLogger(self.name)


    def _move_to_device(self, data, exclude_keys=["text_idx", "query_idx"]):
        """
        Move data to device.

        Args:
            exclude_keys: variables that should be kept unchanged
        """
        if isinstance(data, Mapping):
            # move the value to device if its key is not exluded
            return type(data)({k: self._move_to_device(v) if k not in exclude_keys else v for k, v in data.items()})
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.config.device)
        return data


    def _l2_distance(self, x1:TENSOR, x2:TENSOR) -> TENSOR:
        """
        Compute l2 similarity.

        Args:
            x1: tensor of [B, D]
            x2: tensor of [B, D]
        """
        ip = torch.matmul(x1, x2.transpose(-1, -2))  # B D
        norm_1 = torch.sum(x1 * x1, dim=-1, keepdim=False).unsqueeze(1).expand(-1, x2.size(0))  # B D
        norm_2 = torch.sum(x2 * x2, dim=-1, keepdim=False).unsqueeze(0).expand(x1.size(0), -1)  # B D
        return norm_1 + norm_2 - 2 * ip


    def _cos_sim(self, x1:TENSOR, x2:TENSOR, temperature:float=0.1) -> TENSOR:
        """
        Compute cosine similarity.

        Args:
            x1: tensor of [B, D]
            x2: tensor of [B, D]
            temperature: scale the similarity scores by dividing temperature
        """
        # x1 = F.normalize(x1, dim=-1)
        # x2 = F.normalize(x2, dim=-1)
        return x1.matmul(x2.transpose(-1,-2)) / temperature


    def _compute_teacher_score(self, x):
        """
        Compute teacher score in knowledge distillation; return None if training in contrastive mode.
        """
        if self.config.objective == "kd":
            if "teacher_score" in x:
                teacher_score = x["teacher_score"]  # B, 1+N

            elif "query_teacher_embedding" in x:
                query_teacher_embedding = x["query_teacher_embedding"]  # B, D
                text_teacher_embedding = x["text_teacher_embedding"]    # B, (1+N), D
                if self.config.is_distributed and self.config.enable_all_gather:
                    query_teacher_embedding = self._gather_tensors(query_teacher_embedding)
                    text_teacher_embedding = self._gather_tensors(text_teacher_embedding)
                B, D = query_teacher_embedding.shape
                teacher_score = query_teacher_embedding.matmul(text_teacher_embedding.view(-1, D).transpose(-1,-2)) # B, B * (1 + N)

                if not self.config.enable_inbatch_negative:
                    teacher_score = teacher_score.view(B, B, -1)[range(B), range(B)]    # B, 1 + N

            else:
                raise ValueError("At least teacher_score or query/text teacher embedding should be provided in knowledge distillation!")

        elif self.config.objective == "contra":
            teacher_score = None

        else:
            raise NotImplementedError(f"Objective type {self.config.objective} not implemented!")
        return teacher_score


    def _compute_loss(self, score:TENSOR, label:TENSOR, teacher_score:Optional[TENSOR]):
        """
        A general method to compute loss (contrastive cross-entropy or distillation)

        Args:
            score: tensor of [B, *]
            label: tensor of [B, *]
            x: the input data
        """
        if teacher_score is None:
            loss = F.cross_entropy(score, label)
        else:
            assert score.shape == teacher_score.shape, f"Teacher score {teacher_score.shape} and student score {score.shape} mismatch!"

            label = F.softmax(teacher_score, dim=-1)
            score = F.log_softmax(score, dim=-1)
            loss = torch.mean(-torch.sum(label * score, dim=-1))
        return loss


    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer; Subclass may override this function to create custom optimizers. Return None to use the default optimizer created by Trainer.

        Returns:
            optimizer
        """
        return None


    def _gather_objects(self, local_object:object) -> list[object]:
        """
        Gather common python objects across processes.

        .. note::
            This function implicitly consumes GPU.

        Args:
            local_object: python object to collect
        """
        all_objects = [None for _ in range(self.config.world_size)]
        dist.all_gather_object(all_objects, local_object)
        return all_objects


    def _gather_tensors(self, local_tensor:TENSOR) -> TENSOR:
        """
        Gather tensors from all gpus on each process.

        Args:
            local_tensor: the tensor that needs to be gathered

        Returns:
            concatenation of local_tensor in each process
        """
        if local_tensor is None:
            return None
        all_tensors = [torch.empty_like(local_tensor) for _ in range(self.config.world_size)]
        dist.all_gather(all_tensors, local_tensor.contiguous())
        all_tensors[self.config.rank] = local_tensor
        return torch.cat(all_tensors, dim=0)


    def save_to_mmp(self, path:str, shape:tuple, dtype:np.dtype, loader:DataLoader, obj:np.ndarray, batch_size:int=1000):
        """
        #. Create a ``np.memmap`` file of ``shape`` with ``dtype``;

        #. Create lock;

        #. Save the ``obj`` to the offset :attr:`utils.util.Sequential_Sampler.start`;

        #. Release lock.

        Args:
            path: the memmap file path
            shape: the shape of the memmap file to be created
            dtype:
            loader: the dataloader for the data
            obj: the array to be stored
            batch_size: saving in batch
        """
        if self.config.is_main_proc:
            save_dir = os.path.split(path)[0]

            if os.path.exists(path):
                os.remove(path)
            else:
                os.makedirs(save_dir, exist_ok=True)

            lock_path = os.path.join(save_dir, "lock")
            i = 0
            while os.path.exists(lock_path):
                if i == 0:
                    self.logger.info("found lock, waiting for other programs...")
                time.sleep(3)
                i = 1

            save_pickle("this is a lock", lock_path)
            mmp = np.memmap(
                path,
                shape=shape,
                mode="w+",
                dtype=dtype
            )
            del mmp
        # make sure the memmap file has been created
        synchronize()

        self.logger.info(f"saving at {path}")
        mmp = np.memmap(
            path,
            shape=shape,
            mode="r+",
            dtype=dtype
        )

        start_idx = loader.sampler.start
        end_idx = loader.sampler.end
        max_length = shape[0]
        # add in batch
        if max_length > batch_size:
            for i in tqdm(range(start_idx, end_idx, batch_size), leave=False, ncols=100):
                mmp[i: min(i + batch_size, end_idx)] = obj[i - start_idx: i - start_idx + batch_size]
        else:
            mmp[start_idx: end_idx] = obj

        if self.config.is_main_proc:
            # remove the lock
            os.remove(lock_path)


    def gather_retrieval_result(self, retrieval_result:RETRIEVAL_MAPPING, hits: Optional[int]=None, retrieval_result_path: Optional[str]=None) -> RETRIEVAL_MAPPING:
        """
        #. Gather ``retrieval_result`` across processes;

        #. Returning the reordered result cut off to top k;

        #. Create a lock;

        #. Save the result at :attr:`models.BaseModel.BaseModel.retrieval_result_path`.

        #. Release the lock.

        Args:
            retrieval_result: each tuple is a document id-score pair
        Returns:
            processed retrieval result
        """
        if hits is None:
            hits = self.config.hits if self.config.get("verifier_type", "none") == "none" else self.config.verifier_hits
        if retrieval_result_path is None:
            retrieval_result_path = self.retrieval_result_path

        retrieval_result_name = Path(retrieval_result_path).stem

        if self.config.is_main_proc:
            makedirs(retrieval_result_path)

        # create lock for saving and reading the temporary retrieval result
        # check if the lock exists, wait until the lock is released
        lock_path = os.path.join(self.retrieve_dir, f"lock")
        i = 0
        while os.path.exists(lock_path):
            if i == 0:
                self.logger.info("found lock, waiting for other programs...")
            time.sleep(3)
            i += 1

        # make sure every process jump out of the detecting-lock loop before creating a new lock
        synchronize()

        if self.config.is_main_proc:
            save_pickle("this is a lock", lock_path)

        self.logger.info("saving retrieval results...")

        if self.config.is_distributed:
            local_retrieval_result_path = f"{retrieval_result_path}.{self.config.rank}"
            save_pickle(retrieval_result, local_retrieval_result_path)
            # make sure all processes save the retrieval_result
            synchronize()

            # collect the retrieval result only on master node
            retrieval_result = defaultdict(list)
            if self.config.is_main_proc:
                for i in tqdm(range(self.config.world_size), desc="Merging Retrieval Results", ncols=100, leave=False):
                    tmp_path = f"{retrieval_result_path}.{i}"
                    output = load_pickle(tmp_path)
                    for k, v in output.items():
                        retrieval_result[k].extend(v)
                    os.remove(tmp_path)

        if self.config.is_main_proc:
            # the value of a retrieval_result key is a list of tuple (id, score) or just an id
            with_score = isinstance(next(iter(retrieval_result.values()))[0], tuple)

            if self.config.save_score:
                if not with_score:
                    self.logger.warning("The retrieval result has no score attached, ignoring save_score!")
                retrieval_result_with_scores = defaultdict(list)

            # sort retrieval result
            for qidx, res in retrieval_result.items():
                if hits > 0:
                    if with_score:
                        res = sorted(res, key=lambda x: x[1], reverse=True)
                    reorder_result = res[:hits]
                else:
                    reorder_result = res

                retrieval_result[qidx] = [item[0] if with_score else item for item in reorder_result]

                if self.config.save_score and with_score:
                    retrieval_result_with_scores[qidx] = reorder_result

            # save result
            save_pickle(retrieval_result, retrieval_result_path)
            if self.config.save_score:
                save_pickle(retrieval_result_with_scores, os.path.join(self.retrieve_dir, f"{retrieval_result_name}_with_scores.pkl"))

            # remove the lock
            os.remove(lock_path)

        return retrieval_result


    def init_verifier(self, loaders:LOADERS, load_all_verifier:bool=False):
        """
        Initialize post verifier defined in :pyobj:``utils.index.VERIFIER_MAP``.

        Args:
            loaders
            load_all_verifier: if ``True``, load all the verifier embeddings/codes
        """
        if self.config.get("verifier_type", "none") == "none":
            return None

        else:
            loader_query = loaders["query"]
            loader_text = loaders["text"]

            if load_all_verifier:
                start_text_idx = start_query_idx = 0
                end_text_idx = len(loader_text.dataset)
                end_query_idx = len(loader_query.dataset)
            else:
                start_text_idx = loader_text.sampler.start
                end_text_idx = loader_text.sampler.end
                start_query_idx = loader_query.sampler.start
                end_query_idx = loader_query.sampler.end

            self.logger.info(f"initilizing verifier {self.config.verifier_src}:{self.config.verifier_type}...")

            query_embeddings = np.memmap(
                # the embedding file may be a symbolic link
                readlink(os.path.join(self.config.cache_root, "encode", self.config.verifier_src, self.config.eval_set, "query_embeddings.mmp")),
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), -1)[start_query_idx: end_query_idx].copy()

            text_embeddings = pq_index = None
            if self.config.verifier_type == "flat":
                text_embeddings = np.memmap(
                    # the embedding file may be a symbolic link
                    readlink(os.path.join(self.config.cache_root, "encode", self.config.verifier_src, "text_embeddings.mmp")),
                    mode="r",
                    dtype=np.float32
                ).reshape(len(loader_text.dataset), -1)[start_text_idx: end_text_idx].copy()
            elif self.config.verifier_type == "pq":
                pq_index = faiss.read_index(os.path.join(self.config.cache_root, "index", self.config.verifier_src, "index", self.config.verifier_index))

            verifier = VERIFIER_MAP[self.config.verifier_type](
                query_embeddings=query_embeddings,
                text_embeddings=text_embeddings,
                pq_index=pq_index,
                hits=self.config.verifier_hits,
                device=self.config.device,
                start_text_idx=start_text_idx,
                end_text_idx=end_text_idx,
            )
            return verifier


    def encode(self, loaders):
        """
        Shotcut for encoding both text and query.
        """
        self.encode_text(loaders["text"])
        self.encode_query(loaders["query"])


    def index(self, loaders:LOADERS):
        """
        The index method. Subclass should override this function.
        """
        pass


    def retrieve(self, loaders:LOADERS):
        """
        The retrieve method. Subclass should override this function.
        """
        pass


    @synchronize
    def rerank(self, loaders: dict):
        """
        Rerank the candidates in :mod:`utils.dataset.PairDataset`.
        """
        loader_rerank = loaders["rerank"]
        retrieval_result = defaultdict(list)

        self.logger.info("reranking...")
        for i, x in enumerate(tqdm(loader_rerank, ncols=100, leave=False)):
            query_idx = x["query_idx"].tolist()	# B
            text_idx = x["text_idx"].tolist()	# B
            score = self.compute_score(x).cpu().tolist()	# B
            for j, qidx in enumerate(query_idx):
                retrieval_result[qidx].append((text_idx[j], score[j]))

            if self.config.get("debug") and i > 5:
                break
        return retrieval_result


    @synchronize
    @torch.no_grad()
    def evaluate(self, loaders:LOADERS, log:bool=True):
        """
        #. Evaluate the model on ``config.eval_set``;

        #. Log the metrics;

        #. Save the checkpoint if necessary.

        Args:
            log: call `log_result()`?
        """
        self.eval()

        if self.config.load_result:
            retrieval_result = load_pickle(self.retrieval_result_path)

        else:
            if self.config.eval_mode == "rerank":
                retrieval_result = self.rerank(loaders)
            elif self.config.eval_mode == "retrieve":
                # all models should override the retrieve method
                retrieval_result = self.retrieve(loaders)
            else:
                raise ValueError(f"Unsupported Mode {self.config.mode}!")

            # merge retrieval result from all processes if _distributed
            # save retrieval result
            retrieval_result = self.gather_retrieval_result(retrieval_result)

        if self.config.is_main_proc:
            self.logger.info("evaluating...")

            if self.config.dataset == "NQ-open":
                if self.config.eval_set == "train":
                    ground_truth_path = os.path.join(self.config.cache_root, "dataset", self.config.eval_set, "positives.pkl")
                    ground_truth = load_pickle(ground_truth_path)
                    metrics = compute_metrics(retrieval_result, ground_truth, cutoffs=self.config.eval_metric_cutoff)
                elif self.config.eval_set == "dev":
                    metrics = compute_metrics_nq(retrieval_result, os.path.join(self.config.data_root, self.config.dataset, "nq-test.qa.csv"), os.path.join(self.config.data_root, self.config.dataset, "collection.tsv"))
                else:
                    raise NotImplementedError
            else:
                ground_truth_path = os.path.join(self.config.cache_root, "dataset", self.config.eval_set, "positives.pkl")
                ground_truth = load_pickle(ground_truth_path)
                metrics = compute_metrics(retrieval_result, ground_truth, metrics=self.config.eval_metric, cutoffs=self.config.eval_metric_cutoff)

            self.metrics.update(metrics)
            # the model will add some metrics such as FLOPs
            all_metrics = {k: v for k, v in self.metrics.items() if k != "_best"}
            self.logger.info(f"{self.name}: {all_metrics}")

            if log:
                self.log_result()

            # other processes automatically return None
            return all_metrics


    def log_result(self):
        """
            Save the model metrics and configurations in ``performance.log``.
        """
        name = self.name
        metrics = {k:v for k, v in self.metrics.items() if k != "_best"}

        with open("performance.log", "a+") as f:
            d = vars(self.config)
            line = "{} : {}\n{}\n".format(name, str(d), str(metrics))
            f.write(line)

            try:
                if self.config.dataset == "NQ-open":
                    markdown_format_metric = "|".join([str(metrics["Recall@5"]), str(metrics["Recall@10"]), str(metrics["Recall@20"]), str(metrics["Recall@100"])]) + "|"
                elif self.config.dataset in ["NQ", "Top300k", "Rand300k"]:
                    markdown_format_metric = "|".join([str(metrics["MRR@10"]), str(metrics["Recall@5"]), str(metrics["Recall@10"]), str(metrics["Recall@100"]), str(metrics["Recall@1000"])]) + "|"
                else:
                    markdown_format_metric = "|".join([str(metrics["MRR@10"]), str(metrics["Recall@10"]), str(metrics["Recall@100"]), str(metrics["Recall@1000"])]) + "|"
            except:
                markdown_format_metric = ""
            if "FLOPs" in metrics:
                markdown_format_metric += str(metrics["FLOPs"]) + "|"
            if "Posting_List_Length" in metrics:
                markdown_format_metric += str(metrics["Posting_List_Length"]) + "|"
            if "X Posting_List_Length" in metrics and "Y Posting_List_Length" in metrics:
                markdown_format_metric += str(metrics["X Posting_List_Length"] + metrics["Y Posting_List_Length"]) + "|"
            f.write(markdown_format_metric + "\n")
            f.write("\n")


    # no synchronize here because maybe we only call this function on main process
    def save(self, checkpoint:Optional[Union[str,int]]=None):
        """
        Save the model at ``checkpoint``.
        """
        # set to eval mode when saving
        if self.training:
            self.eval()
            training = True
        else:
            training = False

        if checkpoint is None:
            checkpoint = self.config.save_ckpt

        save_path = f"{self.config.cache_root}/ckpts/{self.name}/{checkpoint}"

        os.makedirs(os.path.split(save_path)[0], exist_ok=True)

        self.logger.info("saving model at {}...".format(save_path))
        model_dict = self.state_dict()

        if self.config.is_main_proc:
            save_dict = {}
            save_dict["config"] = self.config
            save_dict["model"] = model_dict
            save_dict["metrics"] = {k: v for k, v in self.metrics.items() if k != "_best"}
            torch.save(save_dict, save_path)

        if training:
            self.train()


    @synchronize
    def load(self):
        """
        Load the model with current config from ``config.load_ckpt``.
        """
        checkpoint = self.config.load_ckpt

        if checkpoint == "none" or checkpoint is None:
            return
        elif os.path.isfile(checkpoint):
            save_path = checkpoint
        elif os.path.isfile(f"{self.config.cache_root}/ckpts/{checkpoint}"):
            save_path = f"{self.config.cache_root}/ckpts/{checkpoint}"
        else:
            save_path = f"{self.config.cache_root}/ckpts/{self.name}/{checkpoint}"

        if not os.path.exists(save_path):
            self.logger.warning(f"Checkpoint {checkpoint} not found, not loading any checkpoints!")
            return

        self.logger.info("loading model from {}...".format(save_path))

        state_dict = torch.load(save_path, map_location=torch.device(self.config.device))
        missing_keys, unexpected_keys = self.load_state_dict(state_dict["model"], strict=False)

        current_config = self.config
        for k, v in state_dict["config"].items():
            try:
                if v != current_config[k]:
                    self.logger.info(f"model config {k} of the checkpoint is {v}, while it's {current_config[k]} in current config!")
            except KeyError:
                pass

        if len(missing_keys):
            self.logger.warning(f"Missing Keys: {missing_keys}")
        if len(unexpected_keys):
            self.logger.warning(f"Unexpected Keys: {unexpected_keys}")


    @classmethod
    def from_pretrained(cls, ckpt_path:str, device:DEVICE="cpu"):
        """
        Load model and its config from ``ckpt_path``.

        Args:
            cls: the model class
            ckpt_path: the path to the model checkpoint
            device: the device to load the model

        Returns:
            the loaded model
        """
        model_name_current = ckpt_path.split(os.sep)[-2]
        state_dict = torch.load(ckpt_path, map_location=torch.device(device))
        config = state_dict["config"]
        model_name_ckpt = config.name
        assert model_name_ckpt == model_name_current, f"The model in the checkpoint is {model_name_ckpt}, while it's {model_name_current} in the current setting!"

        # transfer the model to a specific device
        config.device = device
        model = cls(config).to(config.device)
        assert model.name == config.name
        model.logger.info(f"loading model from {ckpt_path} with checkpoint config...")
        model.load_state_dict(state_dict["model"])
        model.metrics = state_dict["metrics"]

        model.eval()
        return model


    def generate_code(self, loaders):
        """
        Generate text codes used in :doc:`/sources/experiments/Model-Based IR`.
        """
        assert not self.config.is_distributed
        # the code is bind to the plm_tokenizer
        code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "codes.mmp")
        # all codes are led by 0 and padded by -1
        self.logger.info(f"generating codes from {self.config.code_type} with code_length: {self.config.code_length}, saving at {code_path}...")

        loader_text = loaders["text"]
        text_num = len(loader_text.dataset)

        if self.config.code_type == "title":
            from utils.util import _get_title_code

            collection_path = os.path.join(self.config.data_root, self.config.dataset, "collection.tsv")
            makedirs(code_path)

            # load all saved token ids
            text_codes = np.memmap(
                code_path,
                dtype=np.int32,
                mode="w+",
                shape=(text_num, self.config.code_length)
            )
            # the codes are always led by 0 and padded by -1
            text_codes[:, 0] = tokenizer.pad_token_id
            text_codes[:, 1:] = -1

            preprocess_threads = 32
            all_line_count = text_num
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.data_root, "PLM", self.config.code_tokenizer))

            arguments = []
            for i in range(preprocess_threads):
                start_idx = round(all_line_count * i / preprocess_threads)
                end_idx = round(all_line_count * (i+1) / preprocess_threads)
                arguments.append((collection_path, code_path, all_line_count, start_idx, end_idx, tokenizer, self.config.code_length))
            with mp.Pool(preprocess_threads) as p:
                p.starmap(_get_title_code, arguments)

        else:
            raise NotImplementedError



class BaseSparseModel(BaseModel):
    """
    Base class for all models that rely on token weights to rank documents.
    """
    def __init__(self, config:Config):
        super().__init__(config)

        # add the following sparse-model-specific attributes to the config
        # set to true to rebuild the inverted index every time
        self._rebuild_index = False
        # set to false to include special tokens into the inverted index
        self._skip_special_tokens = True
        # set to 1 by default, meaning the token weight is stored
        self._output_dim = 1
        # posting list number, may be extended for latent topics
        self._posting_entry_num = self.config.vocab_size
        # valid text length for indexing and searching
        self._text_length = self.config.text_length
        self._query_length = self.config.query_length


    def _compute_overlap(self, query_token_id:TENSOR, text_token_id:TENSOR, cross_batch=True) -> TENSOR:
        """
        Compute overlapping mask between the query tokens and positive sequence tokens across batches.

        Args:
            query_token_id: [B1, LQ]
            text_token_id: [B2, LS]

        Returns:
            overlapping_mask: [B, LQ, B, LS] if cross_batch, else [B, LQ, LS]
        """
        if cross_batch:
            query_token_id = query_token_id.unsqueeze(-1).unsqueeze(-1) # B, LQ, 1, 1
            text_token_id = text_token_id.unsqueeze(0).unsqueeze(0)   # 1, 1, B, LS
        else:
            query_token_id = query_token_id.unsqueeze(-1)   # B, LQ, 1
            seq_token_id = seq_token_id.unsqueeze(-2)   # B, 1, LS

        overlapping_mask = text_token_id == query_token_id
        return overlapping_mask


    def _gate_query(self, query_embeddings:np.ndarray, query_token_ids:np.ndarray, k:Optional[int]=None):
        """
        Gate the query token weights so that only the top ``config.query_gate_k`` tokens are valid. Moreover, change the gated token ids to -1.

        Args:
            query_embeddings: [N, L, 1]
        """
        if k is None:
            k = self.config.query_gate_k
        if k > 0:
            self.logger.info(f"gating query by {k}...")
            assert query_embeddings.shape[-1] == 1
            query_embeddings = query_embeddings.squeeze(-1)
            non_topk_indices = np.argpartition(-query_embeddings, k)[:, k:]
            np.put_along_axis(query_embeddings, non_topk_indices, values=0, axis=-1)
            np.put_along_axis(query_token_ids, non_topk_indices, values=-1, axis=-1)
            # append the last dimension
            query_embeddings = np.expand_dims(query_embeddings, axis=-1)
        return query_embeddings, query_token_ids


    def _gate_text(self, text_embeddings:np.ndarray, k:Optional[int]=None):
        """
        Gate the text token weights so that only the top ``config.query_gate_k`` tokens are valid. Keep the text_token_ids because we will use it to construct the entire inverted lists.

        Args:
            query_embeddings: [N, L, 1]
        """
        if k is None:
            k = self.config.text_gate_k
        if k > 0:
            self.logger.info(f"gating text by {k}...")
            assert text_embeddings.shape[-1] == 1
            text_embeddings = text_embeddings.squeeze(-1)
            non_topk_indices = np.argpartition(-text_embeddings, k)[:, k:]
            np.put_along_axis(text_embeddings, non_topk_indices, values=0, axis=-1)
            # np.put_along_axis(text_token_ids, non_topk_indices, values=-1, axis=-1)
            # append the last dimension
            text_embeddings = np.expand_dims(text_embeddings, axis=-1)
        return text_embeddings


    def encode_text_step(self, x):
        """
        One step in encode_text.

        Args:
            x: a data record.

        Returns:
            the text token id for indexing, array of [B, L]
            the text token embedding for indexing, array of [B, L, D]
        """
        text_token_id = x["text"]["input_ids"].numpy()
        text_token_embedding = np.ones((*text_token_id.shape, self._output_dim), dtype=np.float32)
        return text_token_id, text_token_embedding


    def encode_query_step(self, x):
        """
        One step in encode_text.

        Args:
            x: a data record.

        Returns:
            the query token id for searching, array of [B, L]
            the query token embedding for indexing, array of [B, L, D]
        """
        query_token_id = x["query"]["input_ids"].numpy()
        query_token_embedding = np.ones((*query_token_id.shape, self._output_dim), dtype=np.float32)
        return query_token_id, query_token_embedding


    @synchronize
    @torch.no_grad()
    def encode_text(self, loader_text:DataLoader, load_all_encode:bool=False):
        """
        Encode texts into token weights or token vecs.

        Args:
            load_all_encode: bool, set to true to load the entire cache file

        Returns:
            BaseOutput:
                text_embeddings: array of [N, L, D]
                text_token_ids: array of [N, L]
        """
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
            ).reshape(len(loader_text.dataset), self._text_length).copy()

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
            ).reshape(len(loader_text.dataset), self._text_length)[loader_text.sampler.start: loader_text.sampler.end].copy()

        else:
            text_token_ids = np.zeros((len(loader_text.sampler), self._text_length), dtype=np.int32)
            text_embeddings = np.zeros((len(loader_text.sampler), self._text_length, self._output_dim), dtype=np.float32)

            start_idx = end_idx = 0
            self.logger.info(f"encoding {self.config.dataset} text...")
            for i, x in enumerate(tqdm(loader_text, leave=False, ncols=100)):
                text_token_id, text_embedding = self.encode_text_step(x)

                end_idx += text_embedding.shape[0]
                text_token_ids[start_idx: end_idx] = text_token_id
                text_embeddings[start_idx: end_idx] = text_embedding
                start_idx = end_idx
                if self.config.debug:
                    if i > 10:
                        break

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


    @synchronize
    @torch.no_grad()
    def encode_query(self, loader_query:DataLoader, load_all_encode:bool=False):
        """
        Encode each query into a weight or a vector.

        Args:
            load_all_encode: if ``True``, load all cached memmap from :attr:`models.BaseModel.BaseModel.encode_dir`

        Returns:
            BaseOutput:
                embeddings: np.ndarray of [N, L, D]
                token_ids: np.ndarray of [N, L]
        """
        query_token_id_path = os.path.join(self.query_dir, "query_token_ids.mmp")
        query_embedding_path = os.path.join(self.query_dir, "query_embeddings.mmp")

        if load_all_encode:
            query_embeddings = np.memmap(
                query_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._query_length, self._output_dim).copy()
            query_token_ids = np.memmap(
                query_token_id_path,
                mode="r+",
                dtype=np.int32
            ).reshape(len(loader_query.dataset), self._query_length).copy()
        elif self.config.load_encode:
            query_embeddings = np.memmap(
                query_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._query_length, self._output_dim)[loader_query.sampler.start: loader_query.sampler.end].copy()
            query_token_ids = np.memmap(
                query_token_id_path,
                mode="r+",
                dtype=np.int32
            ).reshape(len(loader_query.dataset), self._query_length)[loader_query.sampler.start: loader_query.sampler.end].copy()
        else:
            query_token_ids = np.zeros((len(loader_query.sampler), self._query_length), dtype=np.int32)
            query_embeddings = np.zeros((len(loader_query.sampler), self._query_length, self._output_dim), dtype=np.float32)

            start_idx = end_idx = 0
            self.logger.info(f"encoding {self.config.dataset} {self.config.eval_set} query...")
            for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
                query_token_id, query_embedding = self.encode_query_step(x)

                end_idx += query_embedding.shape[0]
                query_token_ids[start_idx: end_idx] = query_token_id
                query_embeddings[start_idx: end_idx] = query_embedding
                start_idx = end_idx
                if self.config.debug:
                    if i > 10:
                        break

            if self.config.save_encode:
                self.save_to_mmp(
                    path=query_token_id_path,
                    shape=(len(loader_query.dataset), self._query_length),
                    dtype=np.int32,
                    loader=loader_query,
                    obj=query_token_ids
                )
                self.save_to_mmp(
                    path=query_embedding_path,
                    shape=(len(loader_query.dataset), self._query_length, self._output_dim),
                    dtype=np.float32,
                    loader=loader_query,
                    obj=query_embeddings
                )

        query_embeddings, query_token_ids = self._gate_query(query_embeddings, query_token_ids)
        return BaseOutput(embeddings=query_embeddings, token_ids=query_token_ids)


    def inverted_index(self, loader_text:DataLoader):
        """
        Construct :class:`utils.index.BaseInvertedIndex`.
        """
        encode_output = self.encode_text(loader_text)

        text_embeddings = encode_output.embeddings
        text_token_ids = encode_output.token_ids
        text_embeddings_tensor = torch.as_tensor(text_embeddings, device=self.config.device)

        if not self._rebuild_index:
            dir_names = [str(self._text_length)]
            if self.config.get("expand_title") and self.config.dataset == "MSMARCO-passage":
                dir_names.append("title")
            # if self.config.expand_doct5:
            #     dir_names.append("expand")
            save_dir = os.path.join(self.config.cache_root, "index", "InvList", self.config.plm_tokenizer, "_".join(dir_names), str(self.config.world_size))
        else:
            save_dir = os.path.join(self.index_dir, str(self.config.world_size))

        special_token_ids = set()
        if self._skip_special_tokens:
            # add all the special_token_ids
            special_token_ids.update([x[1] for x in self.config.special_token_ids.values() if x[0] is not None])
        # add stop tokens
        # special_token_ids.update(self.stop_token_ids)

        index = INVERTED_INDEX_MAP[self.config.index_type](
            text_num=text_embeddings_tensor.shape[0],
            token_num=self._posting_entry_num,
            # in composited models e.g. UniRetriever, it is possible that the config has no posting_prune key
            posting_prune=self.config.get("posting_prune", 0),
            start_text_idx=loader_text.sampler.start,
            device=self.config.device,
            save_dir=save_dir,
            special_token_ids=special_token_ids
        )
        index.fit(
            text_token_ids=text_token_ids,
            text_embeddings=text_embeddings_tensor,
            rebuild_index=self._rebuild_index,
            load_index=self.config.load_index,
            save_index=self.config.save_index,
            threads=self.config.get("index_thread", 16) // self.config.world_size
        )

        if self.config.eval_flops:
            return BaseOutput(
                embeddings=text_embeddings,
                token_ids=text_token_ids,
                index=index
            )
        else:
            return BaseOutput(index=index)


    def anserini_index(self, loader_text:DataLoader):
        """
        Construct :class:`utils.index.BaseAnseriniIndex`.
        """
        if self.config.index_type == "impact":
            if self.config.is_distributed:
                assert self.config.save_encode

        # compute embeddings
        encode_output = self.encode_text(loader_text)

        if self.config.is_main_proc:
            if self.config.index_type == "impact" and self.config.is_distributed:
                # load cache only on the master node
                encode_output = self.encode_text(loader_text, load_all_encode=True)

            text_token_ids = encode_output.token_ids
            text_token_weights = encode_output.embeddings

            stop_words = set()
            # include plm special tokens
            stop_words = stop_words | set(x[0] for x in self.config.special_token_ids.values() if x[0] is not None)

            if self.config.index_type == "impact":
                punctuations = set([x for x in ";:'\\\"`~[]<>()\{\}/|?!@$#%^&*…-_=+,."])
                nltk_stop_words = set(["a", "about", "also", "am", "to", "an", "and", "another", "any", "anyone", "are", "aren't", "as", "at", "be", "been", "being", "but", "by", "despite", "did", "didn't", "do", "does", "doesn't", "doing", "done", "don't", "each", "etc", "every", "everyone", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's", "him", "himself", "his", "however", "i", "i'd", "if", "i'll", "i'm", "in", "into", "is", "isn't", "it", "its", "it's", "itself", "i've", "just", "let's", "like", "lot", "may", "me", "might", "mightn't", "my", "myself", "no", "nor", "not", "of", "on", "onto", "or", "other", "ought", "oughtn't", "our", "ours", "ourselves", "out", "over", "shall", "shan't", "she", "she'd", "she'll", "she's", "since", "so", "some", "something", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "tht", "to", "too", "usually", "very", "via", "was", "wasn't", "we", "we'd", "well", "we'll", "were", "we're", "weren't", "we've", "will", "with", "without", "won't", "would", "wouldn't", "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've"])
                # include punctuations
                stop_words = stop_words | punctuations
                # include nltk stop words
                stop_words = stop_words | nltk_stop_words
                # include numbers in stopwords
                stop_words.add(r"\d")

            collection_path = os.path.join(self.config.data_root, self.config.dataset, "collection.tsv")

            index = ANSERINI_INDEX_MAP[self.config.index_type](
                collection_path=collection_path,
                collection_dir=self.collection_dir,
                index_dir=self.index_dir
            )

            if self.config.load_collection:
                enable_build_collection = False
            else:
                enable_build_collection = True

            if self.config.load_index:
                # if load index, then load the collection as well
                enable_build_index = False
                enable_build_collection = False
            else:
                enable_build_index = True

            index.fit(
                text_cols=self.config.text_col,
                text_token_ids=text_token_ids,
                text_token_weights=text_token_weights.squeeze(-1) if text_token_weights is not None else None,
                quantize_bits=self.config.quantize_bits,
                tokenizer=AutoTokenizer.from_pretrained(self.config.plm_dir),
                subword_to_word=SUBWORD_TO_WORD_FN.get(self.config.plm_tokenizer),
                stop_words=stop_words,
                reduce=self.config.word_reduce,
                thread_num=self.config.index_thread,
                enable_build_collection=enable_build_collection,
                enable_build_index=enable_build_index,
                language=self.config.get("language")
            )

        else:
            index = None

        return BaseOutput(index=index)


    @synchronize
    def index(self, loaders:LOADERS):
        """
        Wrapper to construct a variety of sparse indexes. Subclass may override this function to create customized index.
        """
        if self.config.index_type in INVERTED_INDEX_MAP:
            return self.inverted_index(loaders["text"])
        elif self.config.index_type in ANSERINI_INDEX_MAP:
            return self.anserini_index(loaders["text"])
        else:
            raise NotImplementedError


    @synchronize
    def retrieve(self, loaders:LOADERS) -> RETRIEVAL_MAPPING:
        """
        #. Retrieve by the index;

        #. Compute auxillary metrics if necessary;

        #. Post verify if necessary.

        Returns:
            retrieval result
        """
        loader_query = loaders["query"]

        output = self.index(loaders)
        index = output.index

        encode_output = self.encode_query(loader_query)
        query_embeddings = encode_output.embeddings
        query_token_ids = encode_output.token_ids

        # use anserini to retrieve
        if isinstance(index, BaseAnseriniIndex):
            # anserini index only on the main process
            if self.config.is_main_proc:
                os.makedirs(self.retrieve_dir, exist_ok=True)

                tid2index = load_pickle(os.path.join(self.config.cache_root, "dataset", "text", "id2index.pkl"))
                qid2index = load_pickle(os.path.join(self.config.cache_root, "dataset", self.config.eval_set, "id2index.pkl"))

                query_path = f"{self.config.data_root}/{self.config.dataset}/queries.{self.config.eval_set}.small.tsv"

                # load all verifier embeddings on the master node
                verifier = self.init_verifier(loaders, load_all_verifier=True)

                t1 = time.time()
                retrieval_result = index.search(
                    query_token_ids=query_token_ids,
                    query_path=query_path,
                    tmp_query_dir=os.path.join(self.config.cache_root, "index", self.name, "query"),
                    retrieval_result_path=self.retrieval_result_path,
                    hits=self.config.hits,
                    qid2index=qid2index,
                    tid2index=tid2index,
                    language=self.config.language,
                    k1=self.config.k1,
                    b=self.config.b,
                    verifier=verifier
                )

            else:
                retrieval_result = {}

            t2 = time.time()

        # inverted index and None
        elif isinstance(index, BaseInvertedIndex):
            verifier = self.init_verifier(loaders)

            t1 = time.time()
            self.logger.info("searching...")
            retrieval_result, posting_list_length = index.search(
                query_token_ids=query_token_ids,
                query_embeddings=query_embeddings,
                # this is useful when performing query side parallel
                query_start_idx=loader_query.sampler.start,
                hits=self.config.hits,
                eval_posting_length=self.config.eval_posting_length,
                verifier=verifier
            )
            # retrieval_result = {0:[(0,1.)]}
            t2 = time.time()
            # manually delete the index
            del index

            if self.config.eval_posting_length:
                if self.config.is_distributed:
                    posting_list_length = np.asarray(self._gather_objects(posting_list_length.mean())).sum()
                else:
                    posting_list_length = posting_list_length.mean()

                self.metrics["Posting_List_Length"] = int(np.round(posting_list_length))
                self.logger.info(f"Average Posting Length is {self.metrics['Posting_List_Length']}!")

        else:
            raise NotImplementedError

        if self.config.eval_efficiency and self.config.is_main_proc:
            time_consumption = round(t2-t1, 2)
            memory_consumption = round(psutil.Process().memory_info().rss / 1e6)
            self.metrics["Time"] = time_consumption
            self.metrics["Memory"] = memory_consumption
            self.logger.info(f"Total Search Time is {time_consumption} seconds!")
            self.logger.info(f"Memory Usage of Curren Process is {memory_consumption} MB!")

        if self.config.eval_flops:
            self.compute_flops(loaders, output.token_ids, output.embeddings, query_token_ids, query_embeddings, log=False)

        return retrieval_result


    @synchronize
    @torch.no_grad()
    def compute_flops(self, loaders:LOADERS, text_token_ids:np.ndarray, text_token_weights:np.ndarray, query_token_ids:np.ndarray, query_token_weights:np.ndarray, log:bool=True):
        """
        Compute flops as stated in `SPLADE <https://arxiv.org/pdf/2109.10086.pdf>`_;

        .. note::
            This function uses the cached embedding to compute flops.
        """
        assert self._output_dim == 1
        loader_text = loaders["text"]
        loader_query = loaders["query"]

        self.logger.info("computing flops...")

        text_token_weights = text_token_weights.squeeze(-1)
        if query_token_weights is not None:
            query_token_weights = query_token_weights.squeeze(-1)
        else:
            query_token_weights = np.ones(query_token_ids.shape, dtype=np.float32)

        D = np.zeros(self._posting_entry_num)
        Q = np.zeros(self._posting_entry_num)

        for i, text_token_id in enumerate(tqdm(text_token_ids, ncols=100, desc="Collecting Tokens in Text", leave=False)):
            # ignore the token id whose weight is 0
            text_token_id = text_token_id[text_token_weights[i] != 0]
            D[text_token_id] += 1

        for i, query_token_id in enumerate(tqdm(query_token_ids, ncols=100, desc="Collecting Tokens in Query", leave=False)):
            # ignore the token id whose weight is 0
            query_token_id = query_token_id[query_token_weights[i] != 0]
            Q[query_token_id] += 1

        if self._skip_special_tokens:
            special_token_ids = [x[1] for x in self.config.special_token_ids.values() if x[0] is not None]
            D[special_token_ids] = 0
            Q[special_token_ids] = 0

        D /= len(loader_text.sampler)
        Q /= len(loader_query.sampler)
        flops = Q @ D

        # when distributed, compute flops of each shard and merge by average
        if self.config.is_distributed:
            all_flops = self._gather_objects(flops)
            flops = np.asarray(all_flops).mean()

        flops = round(flops, 2)

        self.metrics.update({"FLOPs": flops})
        if log:
            self.log_result()
            self.logger.info(f"FLOPs: {flops}")


    @torch.no_grad()
    def generate_code(self, loaders:LOADERS):
        """
        Generate codes from the cache embedding files.
        """
        assert not self.config.is_distributed

        if self.config.code_type == "title":
            return super().generate_code(loaders)
        else:
            # the code is bind to the code_tokenizer
            code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "codes.mmp")
            self.logger.info(f"generating codes from {self.config.code_type} with code_length: {self.config.code_length}, saving at {code_path}...")

            loader_text = loaders["text"]
            text_num = len(loader_text.dataset)
            makedirs(code_path)

            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.data_root, "PLM", self.config.code_tokenizer))

            # load all saved token ids
            # all codes are led by 0 and padded by -1
            text_codes = np.memmap(
                code_path,
                dtype=np.int32,
                mode="w+",
                shape=(text_num, self.config.code_length)
            )
            # the codes are always led by 0 and padded by -1
            text_codes[:, 0] = tokenizer.pad_token_id
            text_codes[:, 1:] = -1

            code_order = self.config.code_type.split("-")[-1]

            # in this case, we just filter out unique tokens and select top k important tokens
            if self.config.plm_tokenizer == self.config.code_tokenizer:
                from utils.util import _get_token_code_for_aligned_tokenizer

                # set the number of tokens to be selected
                encode_output = self.encode_text(loaders["text"], load_all_encode=True)
                text_token_ids = encode_output.token_ids
                text_token_embeddings = encode_output.embeddings.squeeze(-1)

                stop_token_ids = set([x[1] for x in self.config.special_token_ids.values()])

                thread_num = 32
                # each thread creates one jsonl file
                text_num_per_thread = text_num / thread_num

                arguments = []
                # re-tokenize words in the collection folder
                for i in range(thread_num):
                    start_idx = round(text_num_per_thread * i)
                    end_idx = round(text_num_per_thread * (i+1))

                    arguments.append((
                        code_path,
                        text_token_ids[start_idx: end_idx],
                        text_token_embeddings[start_idx: end_idx],
                        text_num,
                        start_idx,
                        tokenizer,
                        self.config.code_length,
                        code_order,
                        stop_token_ids
                    ))

                with mp.Pool(thread_num) as p:
                    p.starmap(_get_token_code_for_aligned_tokenizer, arguments)

            else:
                from utils.util import _get_token_code_for_misaligned_tokenizer

                thread_num = 0
                for path in os.listdir(self.collection_dir):
                    # check if current path is a file
                    if os.path.isfile(os.path.join(self.collection_dir, path)):
                        thread_num += 1

                # each thread creates one jsonl file
                text_num_per_thread = text_num / thread_num

                arguments = []
                # re-tokenize words in the collection folder
                for i in range(thread_num):
                    input_path = os.path.join(self.collection_dir, "docs{:02d}.json".format(i))
                    start_idx = round(text_num_per_thread * i)
                    end_idx = round(text_num_per_thread * (i+1))

                    arguments.append((
                        input_path,
                        code_path,
                        text_num,
                        start_idx,
                        end_idx,
                        tokenizer,
                        self.config.code_length,
                        code_order,
                    ))

                # the collection has no special_tokens so we don't need to filter them out
                with mp.Pool(thread_num) as p:
                    p.starmap(_get_token_code_for_misaligned_tokenizer, arguments)



class BaseDenseModel(BaseModel):
    """
    Base class for all models that rely on sequence embeddings to rank documents.
    """
    def __init__(self, config):
        super().__init__(config)


    def encode_text_step(self, x):
        text = self._move_to_device(x["text"])
        embedding = self.textEncoder(**text)[0][:, 0]

        if self.config.dense_metric == "cos":
            embedding = F.normalize(embedding, dim=-1)
        return embedding.cpu().numpy()


    def encode_query_step(self, x):
        query = self._move_to_device(x["query"])
        embedding = self.queryEncoder(**query)[0][:, 0]

        if self.config.dense_metric == "cos":
            embedding = F.normalize(embedding, dim=-1)
        return embedding.cpu().numpy()


    @synchronize
    @torch.no_grad()
    def encode_text(self, loader_text:DataLoader, load_all_encode:bool=False):
        """
        Encode each text into a vector.

        Args:
            load_all_encode: bool, set to true to load the entire cache file

        Returns:
            BaseOutput:
                text_embeddings: array of [N, D]
        """
        text_embedding_path = os.path.join(self.encode_dir, "text_embeddings.mmp")

        if load_all_encode:
            text_embeddings = np.memmap(
                text_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_text.dataset), self._output_dim)

        elif self.config.load_encode:
            text_embeddings = np.memmap(
                text_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_text.dataset), self._output_dim)[loader_text.sampler.start: loader_text.sampler.end]

        else:
            text_embeddings = np.zeros((len(loader_text.sampler), self._output_dim), dtype=np.float32)
            start_idx = end_idx = 0
            self.logger.info(f"encoding {self.config.dataset} text...")
            for i, x in enumerate(tqdm(loader_text, leave=False, ncols=100)):
                text_embedding = self.encode_text_step(x)

                end_idx += text_embedding.shape[0]
                text_embeddings[start_idx: end_idx] = text_embedding
                start_idx = end_idx
                if self.config.debug:
                    if i > 10:
                        break

            if self.config.save_encode:
                self.save_to_mmp(
                    path=text_embedding_path,
                    shape=(len(loader_text.dataset), self._output_dim),
                    dtype=np.float32,
                    loader=loader_text,
                    obj=text_embeddings
                )

        return BaseOutput(embeddings=text_embeddings)


    @synchronize
    @torch.no_grad()
    def encode_query(self, loader_query:DataLoader, load_all_encode:bool=False):
        """
        Encode each query into a vector.

        Args:
            load_all_encode: bool, set to true to load the entire cache file

        Returns:
            BaseOutput:
                query_embeddings: array of [N, D]
        """
        query_embedding_path = os.path.join(self.query_dir, "query_embeddings.mmp")

        if load_all_encode:
            query_embeddings = np.memmap(
                query_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._output_dim)

        elif self.config.load_encode:
            query_embeddings = np.memmap(
                query_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._output_dim)[loader_query.sampler.start: loader_query.sampler.end]

        else:
            query_embeddings = np.zeros((len(loader_query.sampler), self._output_dim), dtype=np.float32)
            start_idx = end_idx = 0
            self.logger.info(f"encoding {self.config.dataset} {self.config.eval_set} query...")
            for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
                query_embedding = self.encode_query_step(x) # B, D

                end_idx += query_embedding.shape[0]
                query_embeddings[start_idx: end_idx] = query_embedding
                start_idx = end_idx
                if self.config.debug:
                    if i > 10:
                        break

            if self.config.save_encode:
                self.save_to_mmp(
                    path=query_embedding_path,
                    shape=(len(loader_query.dataset), self._output_dim),
                    dtype=np.float32,
                    loader=loader_query,
                    obj=query_embeddings
                )

        return BaseOutput(embeddings=query_embeddings)


    def faiss_index(self, loader_text:DataLoader):
        """
        Construct :class:`utils.index.FaissIndex`
        """
        text_embeddings = None
        if not self.config.load_index:
            text_embeddings = self.encode_text(loader_text).embeddings

        if self.config.index_type != "Flat" and not self.config.is_main_proc > 0:
            index = None
        else:
            if self.config.device != "cpu":
                # release temperary gpu cache so that faiss can use it
                torch.cuda.empty_cache()

            index = FaissIndex(
                index_type=self.config.index_type,
                d=self._output_dim,
                metric=self.config.dense_metric,
                start_text_idx=loader_text.sampler.start,
                device=self.config.device,
                save_dir=self.index_dir,
                by_residual=self.config.get("by_residual", True)
            )
            if self.config.load_index:
                index.load()

            index.fit(text_embeddings)

            if self.config.save_index:
                index.save()

        return BaseOutput(index=index)


    @synchronize
    def index(self, loaders):
        """
        Wrapper to construct a variety of faiss indexes. Subclass may override this function to create customized index.
        """
        return self.faiss_index(loaders["text"])


    @synchronize
    def retrieve(self, loaders:LOADERS) -> RETRIEVAL_MAPPING:
        """
        #. Retrieve by the index;

        #. Compute auxillary metrics if necessary;

        #. Post verify if necessary.

        Returns:
            retrieval result
        """
        loader_query = loaders["query"]

        index = self.index(loaders).index

        # place the encode_query outside of the if condition because there is a synchronize step inside encode_query function
        encode_output = self.encode_query(loader_query)
        query_embeddings = encode_output.embeddings

        if index is not None:
            t1 = time.time()
            self.logger.info("searching...")

            if "Flat" in index.name:
                verifier = self.init_verifier(loaders)
            else:
                # load all verifier for ANN indexes like IVFPQ, since it only stores at rank==0
                verifier = self.init_verifier(loaders, load_all_verifier=True)

            retrieval_result, posting_list_length = index.search(
                query_embeddings=query_embeddings,
                hits=self.config.hits,
                eval_posting_length=self.config.eval_posting_length and "IVF" in self.config.index_type,
                # the following config are index-specific, may be missing
                nprobe=self.config.get("nprobe"),
                efSearch=self.config.get("hnswef"),
                verifier=verifier
            )
            t2 = time.time()
            # manually delete the index
            del index

            if self.config.eval_efficiency and self.config.is_main_proc:
                time_consumption = round(t2-t1, 2)
                memory_consumption = round(psutil.Process().memory_info().rss / 1e6)
                self.metrics["Time"] = time_consumption
                self.metrics["Memory"] = memory_consumption
                self.logger.info(f"Total Search Time is {time_consumption} seconds!")
                self.logger.info(f"Memory Usage of Curren Process is {memory_consumption} MB!")

            if self.config.eval_posting_length and posting_list_length:
                # ANN index does not support parallel
                # if self.config.is_distributed:
                #     posting_list_length = np.asarray(self._gather_objects(posting_list_length)).sum()
                self.metrics["Posting_List_Length"] = int(np.round(posting_list_length))
                self.logger.info(f"Average Posting Length is {self.metrics['Posting_List_Length']}!")
        else:
            retrieval_result = defaultdict(list)

        return retrieval_result


    @torch.no_grad()
    def cluster(self, loaders:LOADERS):
        """Perform clusering over cached embeddings.
        """
        from utils.util import Cluster
        assert not self.config.is_distributed, "Clustering only available when not distributed!"
        assert "-" in self.config.cluster_type, "Use hyphen to separate cluster type and cluster metric"
        cluster_type, cluster_metric = self.config.cluster_type.split("-")

        if cluster_type == "flat":
            self.logger.info(f"{self.config.cluster_type} clustering text embeddings...")
            cluster_num = self.config.ncluster
            cluster_dir = os.path.join(self.config.cache_root, "cluster", self.name, self.config.cluster_type, str(cluster_num))
            os.makedirs(cluster_dir, exist_ok=True)

            loader_text = loaders["text"]
            encode_output = self.encode_text(loader_text)
            text_embeddings = encode_output.embeddings
            num_replicas = 50

            cluster = Cluster(device=self.config.device)

            assignments = cluster.kmeans(text_embeddings, cluster_num, num_replicas=num_replicas, metric=cluster_metric)
            centroids = cluster.get_centroids()

            np.save(os.path.join(cluster_dir, "centroids.npy"), centroids)

            assignments_mmp = np.memmap(
                os.path.join(cluster_dir, "assignments.mmp"),
                shape=(text_embeddings.shape[0], num_replicas),
                mode="w+",
                dtype=np.int32
            )
            assignments_mmp[:] = assignments

            # compute node number per cluster
            cluster_node_num = [0]*cluster_num
            for x in assignments[:, 0]:
                cluster_node_num[x] += 1
            cluster_node_num = np.asarray(cluster_node_num)
            self.logger.info(f"clustered {len(text_embeddings)} nodes into {len(centroids)} clusters, average cluster node number is {cluster_node_num.mean()}, max cluster node number is {cluster_node_num.max()}, min cluster node number is {cluster_node_num.min()}")

        elif cluster_type == "hier":
            loader_text = loaders["text"]
            encode_output = self.encode_text(loader_text)
            text_embeddings = encode_output.embeddings

            cluster_dir = os.path.join(self.config.cache_root, "cluster", self.name, self.config.cluster_type)
            os.makedirs(cluster_dir, exist_ok=True)
            cluster = Cluster(device=self.config.device)

            cluster_num = self.config.ncluster
            assignments = cluster.hierarchical_kmeans(text_embeddings, cluster_num, self.config.leaf_node_num, metric=cluster_metric,)
            # assignments = load_pickle("assignments.pkl")
            all_code_length = np.array([len(x) for x in assignments])
            self.logger.info(f"average code length is {all_code_length.mean()}, max code length is {all_code_length.max()}, min code length is {all_code_length.min()}")
            # save_pickle(assignments, "assignments.pkl")

            assignments_mmp = np.memmap(
                os.path.join(cluster_dir, "assignments.mmp"),
                shape=(text_embeddings.shape[0], all_code_length.max()),
                mode="w+",
                dtype=np.int32
            )
            assignments_mmp[:] = -1
            for i, x in enumerate(assignments):
                assignments_mmp[i, :len(x)] = x
            del assignments_mmp

        elif cluster_type == "ivf":
            self.logger.info(f"{self.config.cluster_type} clustering text embeddings...")
            cluster_num = self.config.ncluster
            cluster_dir = os.path.join(self.config.cache_root, "cluster", self.name, self.config.cluster_type, str(cluster_num))
            os.makedirs(cluster_dir, exist_ok=True)

            loader_text = loaders["text"]
            encode_output = self.encode_text(loader_text)
            text_embeddings = encode_output.embeddings
            num_replicas = 50

            ivf = faiss.index_factory(self._output_dim, f"IVF{cluster_num},Flat", faiss.METRIC_INNER_PRODUCT if cluster_metric == "ip" else faiss.METRIC_L2)
            if self.config.device != "cpu":
                ivf = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.config.device, ivf)

            ivf.train(text_embeddings)
            quantizer = faiss.downcast_index(ivf.quantizer)

            if self.config.device != "cpu":
                centroids = faiss.rev_swig_ptr(faiss.index_gpu_to_cpu(quantizer).get_xb(), quantizer.ntotal * quantizer.d).reshape(quantizer.ntotal, quantizer.d)
            else:
                centroids = faiss.rev_swig_ptr(quantizer.get_xb(), quantizer.ntotal * quantizer.d).reshape(quantizer.ntotal, quantizer.d)

            np.save(os.path.join(cluster_dir, "centroids.npy"), centroids)

            assignments = np.memmap(
                os.path.join(cluster_dir, "assignments.mmp"),
                shape=(text_embeddings.shape[0], num_replicas),
                mode="w+",
                dtype=np.int32
            )

            batch_size = 1000
            for i in range(0, text_embeddings.shape[0], batch_size):
                q = text_embeddings[i: i + batch_size]
                score, assignment = quantizer.search(q, num_replicas)
                assignments[i: i + batch_size] = assignment

            # compute node number per cluster
            cluster_node_num = [0]*cluster_num
            for x in assignments[:, 0]:
                cluster_node_num[x] += 1
            cluster_node_num = np.asarray(cluster_node_num)
            self.logger.info(f"clustered {len(text_embeddings)} nodes into {len(centroids)} clusters, average cluster node number is {cluster_node_num.mean()}, max cluster node number is {cluster_node_num.max()}, min cluster node number is {cluster_node_num.min()}")


    @torch.no_grad()
    def generate_code(self, loaders:LOADERS):
        """
        Generate codes from the cached clusering assignments.
        """
        if self.config.code_type == "title":
            return super().generate_code(loaders)

        else:
            # the code is bind to the code_tokenizer
            code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "codes.mmp")
            # all codes are led by 0 and padded by -1
            self.logger.info(f"generating codes from {self.config.code_type} with code_length: {self.config.code_length}, saving at {code_path}...")

            loader_text = loaders["text"]
            text_num = len(loader_text.dataset)
            assignment_path = os.path.join(self.config.cache_root, "cluster", self.name, self.config.cluster_type, "assignments.mmp")

            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.data_root, "PLM", self.config.code_tokenizer))

            # generate codes from pre-defined cluster assignments
            if os.path.exists(assignment_path):
                makedirs(code_path)
                assignments = np.memmap(
                    assignment_path,
                    mode="r+",
                    dtype=np.int32,
                ).reshape(text_num, -1)

                assert self.config.code_length >= assignments.shape[1] + 2, "The code_length must be greater than the assignment length by 2 because we have a leading 0 and an eos_token_id!"
                codes = np.memmap(
                    code_path,
                    # plus one because the code should be lead with the padding token id
                    shape=(text_num, self.config.code_length),
                    mode="w+",
                    dtype=np.int32
                )
                codes[:, 0] = tokenizer.pad_token_id
                codes[:, 1:] = -1
                bias = tokenizer.vocab_size
                # another bias to distinguish the same cluster id in different layer
                if self.config.code_type.split("-")[-1] == "bias":
                    bias += np.arange(codes.shape[1]) * (assignments.max() + 1)

                for i, x in enumerate(assignments):
                    length = (x != -1).sum()
                    codes[i, 1: length + 1] = x[:length]
                    if isinstance(bias, np.ndarray):
                        codes[i, 1: length + 1] += bias[:length]
                    else:
                        codes[i, 1: length + 1] += bias

                    # assign eos_token_id
                    codes[i, length + 1] = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id

            else:
                raise FileNotFoundError(f"{assignment_path} not found!")



class BaseGenerativeModel(BaseModel):
    """
    Base class for generative models e.g. DSI, WebUltron.
    """
    def __init__(self, config:Config):
        super().__init__(config)

        #: str: we separate the saving folder of generative model
        self.code_dir = os.path.join(self.config.cache_root, "codes", self.name if self.config.code_type == "self" else self.config.code_type, self.config.code_tokenizer, str(self.config.code_length))


    @synchronize
    @torch.no_grad()
    def encode_text(self, loader_text:DataLoader):
        """
        Encode each text into its code.
        """
        text_codes = loader_text.dataset.text_codes[loader_text.sampler.start: loader_text.sampler.end].copy()
        return BaseOutput(codes=text_codes)


    def trie_index(self, loader_text:DataLoader):
        """
        Construct :class:`utils.index.TrieIndex`.
        """
        if not self.config.load_index:
            encode_output = self.encode_text(loader_text)    # N, L
            text_codes = encode_output.codes
        else:
            text_codes = None

        index = TRIE_INDEX_MAP[self.config.index_type](
            save_dir=os.path.join(self.code_dir, "tries"),
            pad_token_id=self.config.special_token_ids["pad"][1]
        )

        index.fit(
            text_codes=text_codes,
            rebuild_index=self.config.code_type == "self",
            load_index=self.config.load_index,
            # only save at rank==0 because tries across processes are always the same
            save_index=self.config.is_main_proc and self.config.save_index
        )

        return BaseOutput(index=index)


    @synchronize
    def index(self, loaders:LOADERS):
        """
        Wrapper to construct a variety of trie indexes. Subclass may override this function to create customized index.
        """
        if self.config.index_type in TRIE_INDEX_MAP:
            return self.trie_index(loaders["text"])


    @synchronize
    @torch.no_grad()
    def retrieve(self, loaders:LOADERS) -> RETRIEVAL_MAPPING:
        """
        #. Retrieve by the index;

        #. Save the generated query codes if necessary;

        #. Compute auxillary metrics if necessary;

        #. Post verify if necessary.

        Returns:
            retrieval result
        """
        loader_query = loaders["query"]

        trie = self.index(loaders).index
        loader_query = loaders["query"]

        def prefix_allowed_tokens_fn(batch_id, sent):
            valid = trie.get_next_keys(sent.cpu().numpy())
            return valid

        retrieval_result = {}

        self.logger.info("searching...")
        start_idx = 0
        # in case the query is parallel
        query_start_idx = loader_query.sampler.start

        query_codes = np.full((len(loader_query.sampler), self.config.hits, self.config.code_length), trie.pad_token_id, dtype=np.int32)

        for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
            query = self._move_to_device(x["query"])
            outputs = self._generate(
                **query,
                min_length=None,
                max_length=self.config.code_length,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                num_return_sequences=self.config.hits,
                num_beams=self.config.nbeam,
                length_penalty=self.config.length_penalty,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )

            B, N = query["input_ids"].shape[0], self.config.hits
            codes = outputs.sequences.view(B, N, -1).cpu().numpy()   # B, n, L; n is the num_return_sequences
            scores = outputs.sequences_scores.view(B, N).cpu().numpy()    # B, n
            # cache the codes
            end_idx = start_idx + B
            query_codes[start_idx: end_idx, :, :codes.shape[-1]] = codes

            for j in range(start_idx, end_idx):
                res = defaultdict(list)
                for k, c in enumerate(query_codes[j]):
                    ids = trie[c]
                    for id in ids:
                        res[id].append(scores[j - start_idx, k])
                # may be duplicated doc ids (1 doc with 2 codes)
                retrieval_result[j + query_start_idx] = [(k, max(v)) for k, v in res.items()]

            start_idx = end_idx

            if self.config.debug:
                if i > 2:
                    break


        if self.config.save_encode:
            self.save_to_mmp(
                os.path.join(self.query_dir, "codes.mmp"),
                shape=(len(loader_query.dataset), self.config.hits, self.config.code_length),
                dtype=query_codes.dtype,
                loader=loader_query,
                obj=query_codes
            )

        return retrieval_result


    @synchronize
    @torch.no_grad()
    def rerank(self, loaders:LOADERS):
        """Evaluate by reranking.
        """
        assert self.config.batch_size_eval == 1, "Reranking must be performed with batch_size_eval=1!"

        loader_query = loaders["query"]
        loader_text = loaders["text"]

        def prefix_allowed_tokens_fn(batch_id, sent):
            valid = trie.get_next_keys(sent.cpu().numpy())
            return valid

        # get all the text codes
        encode_output = self.encode_text(loader_text)    # N, L
        text_codes = encode_output.codes

        self.logger.info(f"reranking...")

        if os.path.exists(self.config.candidate_type):
            candidate_path = self.config.candidate_type
        elif os.path.exists(os.path.join(self.config.cache_root, "retrieve", self.config.candidate_type, self.config.eval_set, "retrieval_result.pkl")):
            candidate_path = os.path.join(self.config.cache_root, "retrieve", self.config.candidate_type, self.config.eval_set, "retrieval_result.pkl")
        else:
            raise FileNotFoundError(f"{self.config.candidate_type} Not Found!")

        # pad retrieval result
        candidates = load_pickle(candidate_path)

        # save the generated codes for each query
        query_codes = np.full((len(loader_query.sampler), self.config.hits, self.config.code_length), self.config.special_token_ids["pad"][1], dtype=np.int32)

        retrieval_result = {}

        for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
            query = self._move_to_device(x["query"])
            candidate = candidates[x["query_idx"].tolist()[0]]

            trie = TRIE_INDEX_MAP[self.config.index_type](
                save_dir=os.path.join(self.code_dir, "tries"),
                pad_token_id=self.config.special_token_ids["pad"][1]
            )
            # use the candidate idxs as ids
            trie.add(text_codes[candidate], ids=candidate, verbose=False)

            outputs = self._generate(
                **query,
                min_length=None,
                max_length=self.config.code_length,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                num_return_sequences=self.config.hits,
                num_beams=self.config.nbeam,
                length_penalty=self.config.length_penalty,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )

            codes = outputs.sequences.view(self.config.hits, -1).cpu().numpy()   # n, L; n is the num_return_sequences
            scores = outputs.sequences_scores.view(self.config.hits).cpu().numpy()    # n

            # cache the codes
            query_codes[i, :, :codes.shape[-1]] = codes

            res = defaultdict(list)
            for k, c in enumerate(query_codes[i]):
                ids = trie[c]
                for id in ids:
                    res[id].append(scores[k])
            retrieval_result[i] = [(k, max(v)) for k, v in res.items()]

            if self.config.debug:
                if i > 2:
                    break

        if self.config.save_encode:
            self.save_to_mmp(
                os.path.join(self.query_dir, "codes.mmp"),
                shape=(len(loader_query.dataset), self.config.hits, self.config.code_length),
                dtype=query_codes.dtype,
                loader=loader_query,
                obj=query_codes
            )

        retrieval_result = self.gather_retrieval_result(retrieval_result)
        return retrieval_result


