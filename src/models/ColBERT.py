import torch
import torch.nn as nn
from transformers import AutoModel
from .BaseModel import BaseModel



class ColBERT(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self._set_encoder()

        self.tokenProject = nn.Linear(self.textEncoder.config.hidden_size, config.token_dim)


    def _encode_text(self, **kwargs):
        for k, v in kwargs.items():
            # B, 1+N, L -> B * (1+N), L
            if v.dim() == 3:
                kwargs[k] = v.view(-1, v.shape[-1])

        token_all_embedding = self.textEncoder(**kwargs)[0]
        token_embedding = self.tokenProject(token_all_embedding)
        return token_embedding


    def _encode_query(self, **kwargs):
        token_all_embedding = self.queryEncoder(**kwargs)[0]
        token_embedding = self.tokenProject(token_all_embedding)
        return token_embedding


    def forward(self, x):
        x = self._move_to_device(x)

        query_token_embedding = self._encode_query(**x["query"])	# B, LQ, D
        text_token_embedding = self._encode_text(**x["text"])	    # B*(1+N), LS, D

        if self.config.is_distributed and self.config.enable_all_gather:
            query_token_embedding = self._gather_tensors(query_token_embedding)
            text_token_embedding = self._gather_tensors(text_token_embedding)

        query_text_score = torch.einsum('qin,tjn->qitj', query_token_embedding, text_token_embedding)
        query_text_score = query_text_score.max(dim=-1)[0]    # B, LQ, B*(1+N)
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


    def rerank_step(self, x):
        """
        given a query and a sequence, output the sequence's score
        """
        x = self._move_to_device(x)
        query_token_embedding = self._encode_query(**x["query"])	# B, LQ, D
        text_token_embedding = self._encode_text(**x["text"])	# B, LS, D

        query_text_score = query_token_embedding.matmul(text_token_embedding.transpose(-1,-2))
        score = query_text_score.max(dim=-1)[0].sum(dim=-1) # B
        return score


    def retrieve(self, manager, loaders):
        self.logger.error("currently we do not support retrieval with ColBERT, instead we evaluate it by reranking task")
        raise

