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

        self._posting_entry_num = ivf_index.nlist
        self._skip_special_tokens = False
        self._text_length = 1
        self._query_length = config.query_gate_k


    def _encode_query(self, embedding:Optional[TENSOR]) -> TENSOR:
        ivf_quantization = embedding.matmul(self.ivfCentroids.transpose(-1, -2)) # B, ncluster
        ivf_weight, ivf_id = ivf_quantization.topk(dim=-1, k=self._query_length)
        return ivf_id, ivf_weight


    def encode_query_step(self, x):
        query_embedding = x["query_embedding"].to(self.config.device)
        if hasattr(self, "vt"):
            query_embedding = query_embedding.matmul(self.vt)
        query_ivf_id, query_ivf_weight = self._encode_query(query_embedding)
        return query_ivf_id.cpu().numpy(), query_ivf_weight.unsqueeze(-1).cpu().numpy()


    @synchronize
    @torch.no_grad()
    def encode_text(self, loader_text, load_all_encode=False):
        text_token_id_path = os.path.join(self.text_dir, "text_token_ids.mmp")
        text_embedding_path = os.path.join(self.text_dir, "text_embeddings.mmp")

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

        elif self.config.load_encode or self.config.load_text_encode:
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


class TopIVF(IVF):
    """
    Fix IVF assignments, and optimize centroid embeddings.
    """
    def __init__(self, config):
        super().__init__(config)
        self.ivfCentroids = nn.parameter.Parameter(self.ivfCentroids)

    def _encode_text(self, embedding, text_idx=None):
        ivf_assign = embedding.matmul(self.ivfCentroids.transpose(-1, -2)) # B, ncluster
        if text_idx is None:
            ivf_weight, ivf_id = ivf_assign.max(dim=-1)    # B
            ivf_assign_soft = torch.softmax(ivf_assign, dim=-1)    # B, nlist
            ivf_assign_hard = torch.zeros_like(ivf_assign_soft).scatter_(-1, ivf_id.unsqueeze(-1), 1.0)
            # straight-through trick
            ivf_assign_st = ivf_assign_hard.detach() - ivf_assign_soft.detach() + ivf_assign_soft  # B, nlist
            quantized_embedding = ivf_assign_st.matmul(self.ivfCentroids)

        else:
            ivf_id = torch.as_tensor(self._ivf_codes[text_idx], device=self.config.device, dtype=torch.long)
            ivf_weight = torch.ones(ivf_id.shape, device=ivf_id.device)
            quantized_embedding = self.ivfCentroids[ivf_id]

        returns = (ivf_id, ivf_weight)
        if self.training:
            returns = (quantized_embedding, ivf_assign) + returns
        return returns

    def forward(self, x):
        x = self._move_to_device(x)

        text_idx = x["text_idx"].view(-1).cpu().numpy()
        text_embedding = x["text_embedding"].flatten(0, 1)    # B*(1+N), D
        query_embedding = x["query_embedding"]   # B, D

        quantized_text_embedding, ivf_assign, ivf_id, _ = self._encode_text(text_embedding, text_idx)    # B*(1+N), D; B*(1+N), nlist; B*(1+N)

        if self.config.is_distributed and self.config.enable_all_gather:
            query_embedding = self._gather_tensors(query_embedding)
            text_embedding = self._gather_tensors(text_embedding)
            quantized_text_embedding = self._gather_tensors(quantized_text_embedding)
            ivf_assign = self._gather_tensors(ivf_assign)
            ivf_id = self._gather_tensors(ivf_id)

        score_ivf = query_embedding.matmul(quantized_text_embedding.transpose(-1,-2))	# B, B*(1+N)

        B = query_embedding.size(0)
        if self.config.enable_inbatch_negative:
            label = torch.arange(B, device=self.config.device)
            label = label * (text_embedding.shape[0] // query_embedding.shape[0])
            # mask_ind = ivf_id.view(B, -1)   # B, 1+N
            # # if the negative's label equals to the positive's
            # mask_ind = (mask_ind.T == mask_ind[:, 0]).T # B, 1+N
            # mask_ind[:, 0] = False
            # # mask the conflicting labels
            # label[mask_ind.view(-1)] = -100
        else:
            label = torch.zeros(B, dtype=torch.long, device=self.config.device)
            score_ivf = score_ivf.view(B, B, -1)[range(B), range(B)]    # B, 1+N

        loss = self._compute_loss(score_ivf, label)

        if self.config.enable_commit_loss:
            score_commit = ivf_assign   # B*(1+N), nlist
            label_commit = ivf_id   # B*(1+N)
            loss += self._compute_loss(score_commit, label_commit)

        return loss


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

