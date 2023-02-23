import os
import faiss
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel
from .BaseModel import BaseSparseModel
from .UniCOIL import UniCOIL
from utils.util import BaseOutput, synchronize
from utils.index import FaissIndex
from utils.typings import *



class TopIVF(BaseSparseModel):
    """
    Topic IVF.
    """
    def __init__(self, config):
        super().__init__(config)

        path = os.path.join(config.cache_root, "index", config.vq_src, "faiss", config.vq_index)
        self.logger.info(f"loading index from {path}...")
        index = faiss.read_index(path)
        assert isinstance(index, faiss.IndexIVFPQ), "Make sure the index is IVFPQ index!"

        quantizer = faiss.downcast_index(index.quantizer)
        ivf_centroids = FaissIndex.get_xb(quantizer)
        self.ivfCentroids = nn.parameter.Parameter(torch.tensor(ivf_centroids))

        pq = index.pq
        pq_centroids = FaissIndex.get_pq_codebook(pq)
        if config.freeze_pq:
            self.register_buffer("pqCentroids", torch.tensor(pq_centroids))
        else:
            self.pqCentroids = nn.parameter.Parameter(torch.tensor(pq_centroids))

        invlists = index.invlists
        cs = invlists.code_size

        pq_codes = np.zeros((index.ntotal, pq.M), dtype=np.int16)
        ivf_codes = np.zeros(index.ntotal, dtype=np.int32)
        for i in range(index.nlist):
            ls = invlists.list_size(i)
            list_ids = faiss.rev_swig_ptr(invlists.get_ids(i), ls)
            list_codes = faiss.rev_swig_ptr(invlists.get_codes(i), ls * cs).reshape(ls, cs)
            for j, docid in enumerate(list_ids):
                pq_codes[docid] = list_codes[j]
                ivf_codes[docid] = i

        self._ivf_codes = ivf_codes
        self._pq_codes = pq_codes

        if config.train_encoder:
            self.queryEncoder = AutoModel.from_pretrained(f"{config.plm_root}/retromae_distill")
            self.queryEncoder.pooler = None

        self._rebuild_index = True
        self._posting_entry_num = index.nlist
        self._skip_special_tokens = False
        self._text_length = 1
        self._query_length = config.query_gate_k


    def create_optimizer(self) -> torch.optim.Optimizer:
        ivf_parameter_names = ["ivfCentroids"]
        pq_parameter_names = ["pqCentroids"]

        ivf_parameters = []
        pq_parameters = []
        encoder_parameters = []

        for name, param in self.named_parameters():
            if any(x in name for x in ivf_parameter_names):
                ivf_parameters.append(param)
            elif any(x in name for x in pq_parameter_names):
                pq_parameters.append(param)
            else:
                encoder_parameters.append(param)

        optimizer_grouped_parameters = [
            {
                "params": ivf_parameters,
                "lr": self.config.learning_rate_ivf
            },
            {
                "params": pq_parameters,
                "lr": self.config.learning_rate_pq
            },
            {
                "params": encoder_parameters,
                "lr": self.config.learning_rate
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        return optimizer


    def _encode_query(self, **kwargs):
        """
        encode tokens with bert
        """
        embedding = self.queryEncoder(**kwargs)[0][:, 0]
        return embedding


    def _quantize_ivf_query(self, query_embedding:TENSOR, k:Optional[int]=None) -> TENSOR:
        if k is None:
            k = self._query_length
        ivf_quantization = query_embedding.matmul(self.ivfCentroids.transpose(-1, -2)) # B, ncluster
        ivf_weight, ivf_id = ivf_quantization.topk(dim=-1, k=k, sorted=True)
        return ivf_id, ivf_weight


    def _quantize_ivf_doc(self, embedding:Optional[TENSOR]=None, text_idx:Optional[np.ndarray]=None):
        if self.config.quantize_ivf == "dym":
            ivf_quantization = embedding.matmul(self.ivfCentroids.transpose(-1, -2)) # B, ncluster
            ivf_assign_soft = torch.softmax(ivf_quantization, dim=-1)    # B, nlist
            ivf_weight, ivf_id = ivf_assign_soft.max(dim=-1, keepdim=True)    # B, 1
            if self.training:
                ivf_assign_hard = torch.zeros_like(ivf_assign_soft, device=ivf_assign_soft.device, dtype=ivf_assign_soft.dtype).scatter_(-1, ivf_id, 1.0)
                # straight-through trick
                ivf_assign = ivf_assign_hard.detach() - ivf_assign_soft.detach() + ivf_assign_soft  # B, C
                quantized_embedding = ivf_assign.matmul(self.ivfCentroids)
            else:
                quantized_embedding = None
            return quantized_embedding, ivf_id, ivf_weight

        elif self.config.quantize_ivf == "fixed":
            ivf_id = torch.as_tensor(self._ivf_codes[text_idx], device=self.config.device, dtype=torch.long)
            ivf_weight = torch.ones(ivf_id.shape, device=ivf_id.device)
            if self.training:
                quantized_embedding = self.ivfCentroids[ivf_id]
            else:
                quantized_embedding = None
            # unsqueeze to match the _text_length (1)
            return quantized_embedding, ivf_id.unsqueeze(-1), ivf_weight.unsqueeze(-1)


    def _quantize_pq(self, text_idx:TENSOR):
        # pq_id = self._pq_codes[text_idx].long()
        pq_id = torch.as_tensor(self._pq_codes[text_idx], device=self.config.device, dtype=torch.long)
        quantized_embedding = FaissIndex.pq_quantize(pq_id, self.pqCentroids)
        return quantized_embedding


    def forward(self, x):
        # do not transfer text_idx because the codes are on cpu
        x = self._move_to_device(x)

        text_idx = x["text_idx"].view(-1).numpy()   # B*(1+N)
        text_embedding = x["text_embedding"].flatten(0, 1)   # B*(1+N), D

        if self.config.train_encoder:
            query_embedding = self._encode_query(**x["query"])
        else:
            query_embedding = x["query_embedding"]

        text_ivf_quantization, _, _ = self._quantize_ivf_doc(text_embedding, text_idx)  # B*(1+N), D
        text_pq_quantization = self._quantize_pq(text_idx) + text_ivf_quantization  # B*(1+N), D

        if self.config.is_distributed and self.config.enable_all_gather:
            query_embedding = self._gather_tensors(query_embedding)
            text_embedding = self._gather_tensors(text_embedding)
            text_ivf_quantization = self._gather_tensors(text_ivf_quantization)
            text_pq_quantization = self._gather_tensors(text_pq_quantization)

        score_ivf = query_embedding.matmul(text_ivf_quantization.transpose(-1,-2))	# B, B*(1+N)
        score_pq = query_embedding.matmul(text_pq_quantization.transpose(-1, -2))  # B, B*(1+N)
        if self.config.train_encoder:
            score_dense = query_embedding.matmul(text_embedding.transpose(-1, -2)) # B, B*(1+N)

        B = query_embedding.size(0)
        if self.config.enable_inbatch_negative:
            label = torch.arange(B, device=self.config.device)
            label = label * (text_embedding.shape[0] // query_embedding.shape[0])
        else:
            label = torch.zeros(B, dtype=torch.long, device=self.config.device)
            score_ivf = score_ivf.view(B, B, -1)[range(B), range(B)]    # B, 1+N
            score_pq = score_pq.view(B, B, -1)[range(B), range(B)]    # B, 1+N
            if self.config.train_encoder:
                score_dense = score_dense.view(B, B, -1)[range(B), range(B)]    # B, 1+N

        teacher_score = self._compute_teacher_score(x)
        loss_ivf = self._compute_loss(score_ivf, label, teacher_score)
        loss_pq = self._compute_loss(score_pq, label, teacher_score)
        if self.config.train_encoder:
            loss_dense = self._compute_loss(score_dense, label, teacher_score)
        else:
            loss_dense = 0

        return loss_pq + loss_ivf + loss_dense


    def encode_text_step(self, x):
        """
        Only handle quantize_ivf == dym
        """
        text_embedding = x["text_embedding"].to(self.config.device)
        _, text_ivf_id, text_ivf_weight = self._quantize_ivf_doc(embedding=text_embedding)
        # unsqueeze to match the _output_dim (1)
        return text_ivf_id.cpu().numpy(), text_ivf_weight.unsqueeze(-1).cpu().numpy()


    def encode_query_step(self, x):
        if hasattr(self, "queryEncoder"):
            query = self._move_to_device(x["query"])
            query_embedding = self._encode_query(**query) # B, D
        else:
            query_embedding = x["query_embedding"].to(self.config.device)

        query_ivf_id, query_ivf_weight = self._quantize_ivf_query(query_embedding)
        return query_ivf_id.cpu().numpy(), query_ivf_weight.unsqueeze(-1).cpu().numpy()


    @synchronize
    @torch.no_grad()
    def encode_text(self, loader_text:DataLoader, load_all_encode:bool=False):
        """
        Encode texts into token weights or token vecs. Specially handle the quantize_ivf==`fixed` because we can directly read all the ivf ids instead of encoding them.

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
            if self.config.quantize_ivf == "fixed":
                text_token_ids = np.expand_dims(self._ivf_codes[loader_text.sampler.start: loader_text.sampler.end], -1)
                text_embeddings = np.ones((*text_token_ids.shape, 1), dtype=np.float32)
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



class TokIVF(UniCOIL):
    """
    Uses explicit tokens as IVF entries.
    """
    def __init__(self, config):
        super().__init__(config)
        self.queryEncoder = None


    def _encode_query(self, **kwargs):
        token_embedding = torch.ones_like(kwargs["input_ids"], dtype=torch.float).unsqueeze(-1)
        return token_embedding


    def encode_query_step(self, x):
        return BaseSparseModel.encode_query_step(self, x)



class IVF(BaseSparseModel):
    def __init__(self, config):
        super().__init__(config)

        path = os.path.join(config.cache_root, "index", config.vq_src, "faiss", config.vq_index)
        self.logger.info(f"loading index from {path}...")
        index = faiss.read_index(path)
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
            vt = faiss.downcast_VectorTransform(index.chain.at(0))
            opq = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in).T
            self.register_buffer("vt", torch.tensor(opq))
        else:
            ivf_index = index

        ivf_centroids = FaissIndex.get_xb(index.quantizer)
        self.register_buffer("ivfCentroids", torch.tensor(ivf_centroids))

        invlists = ivf_index.invlists
        ivf_codes = np.zeros(index.ntotal, dtype=np.int32)
        for i in range(index.nlist):
            ls = invlists.list_size(i)
            list_ids = faiss.rev_swig_ptr(invlists.get_ids(i), ls)
            for j, docid in enumerate(list_ids):
                ivf_codes[docid] = i

        self._ivf_codes = ivf_codes

        self._rebuild_index = True
        self._posting_entry_num = ivf_index.nlist
        self._skip_special_tokens = False
        self._text_length = 1
        self._query_length = self.config.query_gate_k


    def _quantize_ivf_query(self, embedding:Optional[TENSOR], k:Optional[int]) -> TENSOR:
        if k is None:
            k = self._query_length

        ivf_quantization = embedding.matmul(self.ivfCentroids.transpose(-1, -2)) # B, ncluster
        ivf_weight, ivf_id = ivf_quantization.topk(dim=-1, k=k)

        return ivf_id, ivf_weight


    def encode_query_step(self, x):
        query_embedding = x["query_embedding"].to(self.config.device)
        if hasattr(self, "vt"):
            query_embedding = query_embedding.matmul(self.vt)

        query_ivf_id, query_ivf_weight = self._quantize_ivf_query(embedding=query_embedding, k=self._query_length)
        return query_ivf_id.cpu().numpy(), query_ivf_weight.unsqueeze(-1).cpu().numpy()


    @synchronize
    @torch.no_grad()
    def encode_text(self, loader_text, load_all_encode=False):
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
            self.logger.info(f"encoding {self.config.dataset} text...")
            text_token_ids = np.expand_dims(self._ivf_codes[loader_text.sampler.start: loader_text.sampler.end], -1)
            text_embeddings = np.ones((*text_token_ids.shape, 1), dtype=np.float32)

            if self.config.save_encode:
                self.save_to_mmp(
                    path=text_embedding_path,
                    shape=(len(loader_text.dataset), self._text_length, self._output_dim),
                    dtype=np.float32,
                    loader=loader_text,
                    obj=text_embeddings
                )
                self.save_to_mmp(
                    path=text_token_id_path,
                    shape=(len(loader_text.dataset), self._text_length),
                    dtype=np.int32,
                    loader=loader_text,
                    obj=text_token_ids
                )

        return BaseOutput(embeddings=text_embeddings, token_ids=text_token_ids)

