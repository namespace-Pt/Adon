import os
import faiss
import torch
import subprocess
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel
from .BaseModel import BaseDenseModel
from utils.util import BaseOutput, readlink
from utils.index import pq_quantize, FaissIndex
from utils.typings import *



class DistillVQ(BaseDenseModel):
    """
    The model is proposed in `this paper <https://arxiv.org/abs/2204.00185>`_. The implementation here follows its own `git repository <https://github.com/staoxiao/LibVQ>`_.
    """
    def __init__(self, config):
        assert "PQ" in config.index_type, "DistillVQ is intended for PQ based methods!"

        super().__init__(config)

        index = faiss.read_index(os.path.join(config.cache_root, "index", config.embedding_src, "index", config.index_type))
        if isinstance(index, faiss.IndexPreTransform):
            vt = faiss.downcast_VectorTransform(index.chain.at(0))
            opq = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in).T
            self.register_buffer("vt", torch.tensor(opq))

        if "IVF" in config.index_type:
            if isinstance(index, faiss.IndexPreTransform):
                ivf_index = faiss.downcast_index(index.index)
            else:
                ivf_index = index
            assert ivf_index.by_residual, "The IVF index should be encoded by residual!"

            quantizer = faiss.downcast_index(ivf_index.quantizer)
            ivf_centroids = FaissIndex.get_xb(quantizer)
            self.ivfCentroids = nn.parameter.Parameter(torch.tensor(ivf_centroids))

            pq = ivf_index.pq
            pq_centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
            if config.freeze_pq:
                self.register_buffer("pqCentroids", torch.tensor(pq_centroids))
            else:
                self.pqCentroids = nn.parameter.Parameter(torch.tensor(pq_centroids))

            invlists = ivf_index.invlists
            cs = invlists.code_size
            pq_codes = np.zeros((ivf_index.ntotal, pq.M), dtype=np.float32)
            ivf_codes = np.zeros(ivf_index.ntotal, dtype=np.float32)
            for i in tqdm(range(ivf_index.nlist), ncols=100, desc="Collecting IVF Codes"):
                ls = invlists.list_size(i)
                list_ids = faiss.rev_swig_ptr(invlists.get_ids(i), ls)
                list_codes = faiss.rev_swig_ptr(invlists.get_codes(i), ls * cs).reshape(ls, cs)
                for j, docid in enumerate(list_ids):
                    pq_codes[docid] = list_codes[j]
                    ivf_codes[docid] = i

            self._ivf_codes = ivf_codes
            self._pq_codes = pq_codes

        # load pq centroids
        elif "PQ" in config.index_type:
            if isinstance(index, faiss.IndexPreTransform):
                pq_index = faiss.downcast_index(index.index)
            else:
                pq_index = index
            # for both ivfpq and pq index, the product quantizer can be accessed by index.pq
            pq = pq_index.pq
            pq_centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
            self.pqCentroids = nn.parameter.Parameter(torch.tensor(pq_centroids))

            pq_codes = faiss.vector_to_array(pq_index.codes).reshape(-1, pq.M)
            pq_codes = torch.tensor(pq_codes)   # uint8 (64) or uint16

            self._pq_codes = pq_codes

        else:
            raise NotImplementedError(f"{config.index_type} not implemented!")

        if config.train_encoder:
            self.queryEncoder = AutoModel.from_pretrained(f"{config.plm_root}/retromae_distill")
            self.queryEncoder.pooler = None

        self._output_dim = index.d


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


    def _quantize_ivf(self, text_idx:np.ndarray, text_embedding:TENSOR) -> TENSOR:
        """
        Args:
            text_idx: the indices of the input documents, used to look up ``self._ivf_codes``; tensor of [B]
            embedding: the embedding of the documents, used to dynamically compute ivf assignments when ``self.config.train_ivf_assign==True``

        Returns:
            quantized ivf embedding (the closest ivf centroid)
        """
        if self.config.train_ivf_assign:
            ivf_sim = text_embedding.matmul(self.ivfCentroids.T)
            ivf_assign_soft = torch.softmax(ivf_sim, dim=-1)    # B, nlist
            _, max_index = ivf_assign_soft.max(dim=-1, keepdim=True)    # B, 1
            ivf_assign_hard = torch.zeros_like(ivf_assign_soft, device=ivf_assign_soft.device, dtype=ivf_assign_soft.dtype).scatter_(-1, max_index, 1.0)
            # straight-through trick
            ivf_assign = ivf_assign_hard.detach() - ivf_assign_soft.detach() + ivf_assign_soft  # B, C
            quantized_embedding = ivf_assign.matmul(self.ivfCentroids)

        else:
            ivf_id = torch.as_tensor(self._ivf_codes[text_idx.numpy()], device=self.config.device, dtype=torch.long) # B
            quantized_embedding = self.ivfCentroids[ivf_id]

        return quantized_embedding


    def _quantize_pq(self, text_idx:np.ndarray, embedding:TENSOR) -> TENSOR:
        """
        Args:
            text_idx: the indices of the input documents, used to look up ``self._pq_codes``; tensor of [B]
            embedding: the embedding of the documents, used to dynamically compute pq assignments when ``self.config.train_pq_assign==True``

        Returns:
            quantized pq embedding (the closest pq centroid)
        """
        if self.config.train_pq_assign:
            M, ksub, dsub = self.pqCentroids.shape
            B = embedding.shape[0]
            embedding = embedding.view(B, M, dsub)
            codebook = self.pqCentroids
            distance = - torch.sum((embedding.unsqueeze(-2) - codebook) ** 2, -1)
            pq_assign_soft = torch.softmax(distance, dim=-1)   # B, M, ksub
            _, max_index = pq_assign_soft.max(dim=-1, keepdim=True)
            pq_assign_hard = torch.zeros_like(pq_assign_soft, device=pq_assign_soft.device, dtype=pq_assign_soft.dtype).scatter_(-1, max_index, 1.0)
            # straight-through trick
            pq_assign = pq_assign_hard.detach() - pq_assign_soft.detach() + pq_assign_soft
            pq_assign = pq_assign.unsqueeze(-2)   # B, M, 1, ksub

            codebook = codebook.unsqueeze(0).expand(B, -1, -1, -1)    # B, M, ksub, dsub
            quantized_embedding = torch.matmul(pq_assign, codebook).view(B, -1)    # B, D

        else:
            pq_id = torch.as_tensor(self._pq_codes[text_idx.numpy()], device=self.config.device, dtype=torch.long)
            quantized_embedding = pq_quantize(pq_id, self.pqCentroids)

        return quantized_embedding


    def _encode_query(self, **kwargs):
        """
        encode tokens with bert
        """
        embedding = self.queryEncoder(**kwargs)[0][:, 0]
        return embedding


    def forward(self, x:dict) -> TENSOR:
        x = self._move_to_device(x)

        text_idx = x["text_idx"].view(-1).numpy()
        text_embedding = x["text_embedding"].flatten(0, 1)   # B*(1+N), D

        if self.config.train_encoder:
            query_embedding = self._encode_query(**x["query"])
        else:
            query_embedding = x["query_embedding"]

        if hasattr(self, "vt"):
            rotate_query_embedding = query_embedding.matmul(self.vt)
            rotate_text_embedding = text_embedding.matmul(self.vt)
        else:
            rotate_query_embedding = query_embedding
            rotate_text_embedding = text_embedding

        if hasattr(self, "ivfCentroids"):
            text_ivf_quantization = self._quantize_ivf(text_idx, rotate_text_embedding)
            quantize_text_embedding = rotate_text_embedding - text_ivf_quantization
            text_pq_quantization = self._quantize_pq(text_idx, quantize_text_embedding) + text_ivf_quantization
        else:
            text_ivf_quantization = None
            text_pq_quantization = self._quantize_pq(text_idx, rotate_text_embedding)

        if self.config.is_distributed and self.config.enable_all_gather:
            rotate_query_embedding = self._gather_tensors(rotate_query_embedding)
            rotate_text_embedding = self._gather_tensors(rotate_text_embedding)
            text_ivf_quantization = self._gather_tensors(text_ivf_quantization)
            text_pq_quantization = self._gather_tensors(text_pq_quantization)

        score_pq = query_embedding.matmul(text_pq_quantization.transpose(-1, -2))  # B, B*(1+N)
        if text_ivf_quantization is not None:
            score_ivf = query_embedding.matmul(text_ivf_quantization.transpose(-1,-2))	# B, B*(1+N)
        if self.config.train_encoder:
            score_dense = query_embedding.matmul(text_embedding.transpose(-1, -2)) # B, B*(1+N)

        B = query_embedding.size(0)
        if self.config.enable_inbatch_negative:
            label = torch.arange(B, device=self.config.device)
            label = label * (text_embedding.shape[0] // query_embedding.shape[0])
        else:
            label = torch.zeros(B, dtype=torch.long, device=self.config.device)
            score_pq = score_pq.view(B, B, -1)[range(B), range(B)]    # B, 1+N
            if text_ivf_quantization is not None:
                score_ivf = score_ivf.view(B, B, -1)[range(B), range(B)]    # B, 1+N
            if self.config.train_encoder:
                score_dense = score_dense.view(B, B, -1)[range(B), range(B)]    # B, 1+N

        teacher_score = self._compute_teacher_score(x)
        loss_pq = self._compute_loss(score_pq, label, teacher_score)
        if text_ivf_quantization is not None:
            loss_ivf = self._compute_loss(score_ivf, label, teacher_score)
        else:
            loss_ivf = 0
        if self.config.train_encoder:
            loss_dense = self._compute_loss(score_dense, label, teacher_score)
        else:
            loss_dense = 0

        if text_ivf_quantization is not None:
            # rescale the ivf loss
            loss_ivf = loss_ivf * float(loss_pq / (loss_ivf + 1e-6))

        return loss_pq + loss_ivf + loss_dense


    @torch.no_grad()
    def encode_text(self, loader_text:DataLoader, load_all_encode:bool=False):
        if load_all_encode:
            text_embeddings = loader_text.dataset.text_embeddings
            return BaseOutput(embeddings=text_embeddings)
        else:
            # create soft link to the embedding_src
            if self.config.is_main_proc and self.config.save_encode:
                os.makedirs(self.encode_dir, exist_ok=True)
                subprocess.run(
                    f"ln -sf {os.path.join(self.config.cache_root, 'encode', self.config.embedding_src, 'text_embeddings.mmp')} {os.path.join(self.encode_dir, 'text_embeddings.mmp')}",
                    shell=True
                )

            text_embeddings = loader_text.dataset.text_embeddings[loader_text.sampler.start: loader_text.sampler.end]
            return BaseOutput(embeddings=text_embeddings)


    @torch.no_grad()
    def encode_query(self, loader_query:DataLoader, load_all_encode:bool=False):
        self._synchronize()
        query_embedding_path = os.path.join(self.query_dir, "query_embeddings.mmp")

        if load_all_encode:
            query_embeddings = np.memmap(
                readlink(query_embedding_path),
                mode="r+",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._output_dim)

        elif self.config.load_encode:
            query_embeddings = np.memmap(
                readlink(query_embedding_path),
                mode="r+",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._output_dim)[loader_query.sampler.start: loader_query.sampler.end]

        else:
            if hasattr(self, "queryEncoder"):
                query_embeddings = np.zeros((len(loader_query.sampler), self._output_dim), dtype=np.float32)
                start_idx = end_idx = 0
                self.logger.info(f"encoding {self.config.dataset} {self.config.eval_set} query...")
                for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
                    query = self._move_to_device(x["query"])
                    query_embedding = self._encode_query(**query).cpu().numpy() # B, LS, D
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
            else:
                # create soft link to the embedding_src
                if self.config.is_main_proc and self.config.save_encode:
                    os.makedirs(self.query_dir, exist_ok=True)
                    subprocess.run(
                        f"ln -sf {os.path.join(self.config.cache_root, 'encode', self.config.embedding_src, self.config.eval_set, 'query_embeddings.mmp')} {os.path.join(self.query_dir, 'query_embeddings.mmp')}",
                        shell=True
                    )

                query_embeddings = loader_query.dataset.query_embeddings[loader_query.sampler.start: loader_query.sampler.end]
            return BaseOutput(embeddings=query_embeddings)


    def index(self, loaders:LOADERS):
        if self.config.load_index:
            return super().index(loaders)

        loader_text = loaders["text"]
        text_embeddings = self.encode_text(loader_text).embeddings

        if self.config.index_type != "Flat" and not self.config.is_main_proc:
            index = None
        else:
            index = FaissIndex(
                index_type=self.config.index_type,
                d=text_embeddings.shape[1],
                metric=self.config.dense_metric,
                start_text_idx=loader_text.sampler.start,
                device=self.config.device,
                save_dir=self.index_dir,
            )
            index.load(os.path.join(self.config.cache_root, "index", self.config.embedding_src, "index", self.config.index_type))

            # load opq transformation
            if isinstance(index.index, faiss.IndexPreTransform):
                vt = faiss.downcast_VectorTransform(index.index.chain.at(0))
                faiss.copy_array_to_vector(self.vt.T.cpu().numpy().ravel(), vt.A)
                vt.is_trained = True

            if "IVF" in self.config.index_type:
                # Important! Don't move the index to gpu; otherwise may trigger wierd performance issue of Faiss
                index.device = "cpu"

                if isinstance(index.index, faiss.IndexPreTransform):
                    ivfpq = faiss.downcast_index(index.index.index)
                else:
                    ivfpq = index.index

                quantizer = faiss.downcast_index(ivfpq.quantizer)
                # remove all embeddings from the quantizer
                quantizer.reset()
                quantizer.add(self.ivfCentroids.cpu().numpy())

                if self.config.train_ivf_assign and not self.config.train_pq_assign:
                    ivfpq.reset()
                    index.index.ntotal = 0
                    index.fit(text_embeddings)
                    # copy pq centroids after adding embeddings
                    pq = ivfpq.pq
                    faiss.copy_array_to_vector(self.pqCentroids.cpu().numpy().ravel(), pq.centroids)

                elif self.config.train_pq_assign:
                    # copy pq centroids before adding embeddings
                    pq = ivfpq.pq
                    faiss.copy_array_to_vector(self.pqCentroids.cpu().numpy().ravel(), pq.centroids)
                    ivfpq.reset()
                    index.index.ntotal = 0
                    index.fit(text_embeddings)

                else:
                    pq = ivfpq.pq
                    faiss.copy_array_to_vector(self.pqCentroids.cpu().numpy().ravel(), pq.centroids)
                    # do nothing
                    index.fit(text_embeddings)

            elif "PQ" in self.config.index_type:
                if isinstance(index.index, faiss.IndexPreTransform):
                    pq_index = faiss.downcast_index(index.index.index)
                else:
                    pq_index = index.index
                pq = pq_index.pq
                faiss.copy_array_to_vector(self.pqCentroids.cpu().numpy().ravel(), pq.centroids)
                pq_index.is_trained = True
                if self.config.train_pq_assign:
                    pq_index.reset()
                    index.index.ntotal = 0

                index.fit(text_embeddings)
            else:
                raise NotImplementedError

            if self.config.save_index:
                index.save()

        return BaseOutput(index=index)

