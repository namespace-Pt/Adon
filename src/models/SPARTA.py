import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from .BaseModel import BaseSparseModel
from utils.util import BaseOutput


class SPARTA(BaseSparseModel):
    def __init__(self, config):
        super().__init__(config)

        self.plm = AutoModel.from_pretrained(config.plm_dir)
        self.plm.pooler = None

        self._skip_special_tokens = True
        self._text_length = self.config.text_decode_k


    def _encode_text(self, **kwargs):
        for k, v in kwargs.items():
            # B, 1+N, L -> B * (1+N), L
            if v.dim() == 3:
                kwargs[k] = v.view(-1, v.shape[-1])

        token_embedding = self.plm(**kwargs)[0]   # B, L, D
        return token_embedding


    def _encode_query(self, token_id):
        return self.plm.embeddings.word_embeddings(token_id)


    def forward(self, x):
        x = self._move_to_device(x)

        query_token_embedding = self._encode_query(x["query"]["input_ids"])    # B, L, D
        text_token_embedding = self._encode_text(**x["text"])    # B*(1+N), L, D

        if self.config.is_distributed and self.config.enable_all_gather:
            query_token_embedding = self._gather_tensors(query_token_embedding)
            text_token_embedding = self._gather_tensors(text_token_embedding)

        query_text_score = torch.einsum('qin,tjn->qitj', query_token_embedding, text_token_embedding)
        query_text_score = query_text_score.max(dim=-1)[0]    # B, LQ, B*(1+N)
        query_text_score = torch.log(torch.relu(query_text_score) + 1)
        score = query_text_score.sum(dim=1) # B, B*(1+N)

        B = score.shape[0]
        if self.config.enable_inbatch_negative:
            label = torch.arange(B, device=self.config.device)
            label = label * (text_token_embedding.shape[0] // query_token_embedding.shape[0])
        else:
            label = torch.zeros(B, dtype=torch.long, device=self.config.device)
            score = score.view(B, B, -1)[range(B), range(B)]    # B, 1+N

        loss = self._compute_loss(score, label, self._compute_teacher_score(x))
        return loss


    def encode_text_step(self, x):
        """
        Pre-compute interactions of all possible tokens with each text token, keep the most matching text token; then only index the topk decoded tokens (top k important tokens in the sense that they will contribute most to the final text score)
        """
        text = self._move_to_device(x["text"])
        text_token_embedding = self._encode_text(**text)    # B, L, D
        vocab_embedding = self.plm.embeddings.word_embeddings.weight   # V, D
        text_token_embedding = torch.einsum("vd,...ld->...lv", vocab_embedding, text_token_embedding)   # B, L, V
        text_embedding = torch.log(torch.relu(text_token_embedding.max(1)[0]) + 1)    # B, V

        text_token_id, text_token_weight = text_embedding.topk(k=self._text_length)
        return text_token_id.cpu().numpy(), text_token_weight.unsqueeze(-1).cpu().numpy()

