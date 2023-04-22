import torch
import torch.nn as nn
from .BaseModel import BaseSparseModel
from transformers import AutoModel



class COIL(BaseSparseModel):
    def __init__(self, config):
        super().__init__(config)

        self._set_encoder()

        self.tokenProject = nn.Linear(self.textEncoder.config.hidden_size, config.token_dim)

        self._output_dim = config.token_dim


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
        text_token_embedding = self._encode_text(**x["text"])	# B * (1+N), LS, D

        query_token_id = x["query"]["input_ids"]
        query_special_mask = x["query_special_mask"]
        text_token_id = x["text"]["input_ids"].view(text_token_embedding.shape[:-1])
        if self.config.is_distributed and self.config.enable_all_gather:
            query_token_id = self._gather_tensors(query_token_id)
            query_special_mask = self._gather_tensors(query_special_mask)
            text_token_id = self._gather_tensors(text_token_id)
            query_token_embedding = self._gather_tensors(query_token_embedding)
            text_token_embedding = self._gather_tensors(text_token_embedding)

        B, LQ, D = query_token_embedding.shape
        LS = text_token_id.shape[1]

        query_text_overlap = self._compute_overlap(query_token_id, text_token_id)   # B, LQ, B * (1+N), LS
        query_text_score = query_token_embedding.view(-1, D).matmul(text_token_embedding.view(-1, D).transpose(0, 1)).view(B, LQ, -1, LS)
        # only keep the overlapping tokens
        query_text_score = query_text_score * query_text_overlap
        # max pooling
        query_text_score = query_text_score.max(dim=-1)[0] # B, LQ, B * (1+N)
        # mask [CLS] and [SEP] and [PAD]
        query_text_score = query_text_score * query_special_mask.unsqueeze(-1)
        score = query_text_score.sum(dim=1) # B, B * (1+N)

        if self.config.enable_inbatch_negative:
            label = torch.arange(B, device=self.config.device)
            label = label * (text_token_embedding.shape[0] // query_token_embedding.shape[0])

        else:
            label = torch.zeros(B, dtype=torch.long, device=self.config.device)
            score = score.view(B, B, -1)[range(B), range(B)]    # B, 1+N

        loss = self._compute_loss(score, label, self._compute_teacher_score(x))
        return loss


    def encode_text_step(self, x):
        # only move text because others are not needed
        text = self._move_to_device(x["text"])
        text_token_id = text["input_ids"]
        text_token_embedding = self._encode_text(**text)
        text_token_embedding *= text["attention_mask"].unsqueeze(-1)
        return text_token_id.cpu().numpy(), text_token_embedding.cpu().numpy()


    def encode_query_step(self, x):
        # only move query because others are not needed
        query = self._move_to_device(x["query"])
        query_token_id = query["input_ids"]
        query_token_embedding = self._encode_query(**query)
        query_token_embedding *= query["attention_mask"].unsqueeze(-1)
        return query_token_id.cpu().numpy(), query_token_embedding.cpu().numpy()


    def rerank_step(self, x):
        """
        Given a query and a sequence, output the sequence's score
        """
        x = self._move_to_device(x)

        query_token_embedding = self._encode_query(**x["query"])	# B, LQ, D
        text_token_embedding = self._encode_text(**x["text"])	# B, LS, D

        overlap = self._compute_overlap(x["query"]["input_ids"], x["text"]["input_ids"], cross_batch=False)   # B, LQ, LS
        query_text_score = query_token_embedding.matmul(text_token_embedding.transpose(-1,-2)) * overlap  # B, LQ, LS
        # mask the [SEP] token and [CLS] token
        query_text_score = query_text_score.max(dim=-1)[0] * x["query_special_mask"]    # B, LQ
        score = query_text_score.sum(dim=1)
        return score

