import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .BaseModel import BaseDenseModel



class DPR(BaseDenseModel):
    """
    The basic dense retriever. `Paper <https://arxiv.org/abs/2004.04906>`_.
    """
    def __init__(self, config):
        super().__init__(config)

        if self.config.get("untie_encoder"):
            self.queryEncoder = AutoModel.from_pretrained(config.plm_dir)
            self.textEncoder = AutoModel.from_pretrained(config.plm_dir)
            if hasattr(self.queryEncoder, "pooler"):
                self.queryEncoder.pooler = None
                self.textEncoder.pooler = None
            self._output_dim = self.textEncoder.config.hidden_size

        else:
            plm = AutoModel.from_pretrained(config.plm_dir)
            if hasattr(plm, "pooler"):
                plm.pooler = None
            self.textEncoder = plm
            self.queryEncoder = plm
            self._output_dim = plm.config.hidden_size


    def _encode_text(self, **kwargs):
        """
        encode tokens with bert
        """
        for k, v in kwargs.items():
            # B, 1+N, L -> B * (1+N), L
            if v.dim() == 3:
                kwargs[k] = v.view(-1, v.shape[-1])

        embedding = self.textEncoder(**kwargs)[0][:, 0]
        if self.config.dense_metric == "cos":
            embedding = F.normalize(embedding, dim=-1)
        return embedding


    def _encode_query(self, **kwargs):
        embedding = self.queryEncoder(**kwargs)[0][:, 0]
        if self.config.dense_metric == "cos":
            embedding = F.normalize(embedding, dim=-1)
        return embedding


    def forward(self, x):
        x = self._move_to_device(x)
        query_embedding = self._encode_query(**x["query"])	# B, D
        text_embedding = self._encode_text(**x["text"])	# *, D

        if self.config.is_distributed and self.config.enable_all_gather:
            query_embedding = self._gather_tensors(query_embedding)
            text_embedding = self._gather_tensors(text_embedding)

        if self.config.dense_metric == "ip":
            score = query_embedding.matmul(text_embedding.transpose(-1,-2))	# B, B*(1+N)
        elif self.config.dense_metric == "cos":
            score = self._cos_sim(query_embedding, text_embedding)
        elif self.config.dense_metric == "l2":
            score = self._l2_sim(query_embedding, text_embedding)
        else:
            raise NotImplementedError

        B = query_embedding.size(0)
        # in batch negative
        if self.config.enable_inbatch_negative:
            label = torch.arange(B, device=self.config.device)
            label = label * (text_embedding.shape[0] // query_embedding.shape[0])
        else:
            label = torch.zeros(B, dtype=torch.long, device=self.config.device)
            score = score.view(B, B, -1)[range(B), range(B)]    # B, 1+N

        loss = self._compute_loss(score, label, self._compute_teacher_score(x))
        return loss


    def rerank_step(self, x):
        """
        given a query and a sequence, output the sequence's score
        """
        query_embedding = self._encode_query(**x["query"])	# B, D
        text_embedding = self._encode_text(**x["text"]) # B, D
        B = query_embedding.size(0)
        score = query_embedding.matmul(text_embedding.transpose(-1, -2))[range(B), range(B)]
        return score


    def deploy(self):
        deploy_dir = os.path.join(self.config.cache_root, "deploy", self.name)
        os.makedirs(deploy_dir, exist_ok=True)

        AutoTokenizer.from_pretrained(self.config.plm_dir).save_pretrained(deploy_dir)
        if self.config.untie_encoder:
            self.queryEncoder.save_pretrained(os.path.join(deploy_dir, "query"))
            self.textEncoder.save_pretrained(os.path.join(deploy_dir, "text"))
        else:
            self.logger.info(f"saving plm model and tokenizer at {deploy_dir}...")
            self.plm.save_pretrained(deploy_dir)