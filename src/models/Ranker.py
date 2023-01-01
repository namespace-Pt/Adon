import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from .BaseModel import BaseModel



class CrossEncoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.plm = AutoModelForSequenceClassification.from_pretrained(config.plm_dir, num_labels=1)
        # self.plm.pooler = None
        if config.code_size > 0:
            self.plm.resize_token_embeddings(config.vocab_size + config.code_size)


    def _compute_score(self, **kwargs):
        """ concate the query and the input text;
        Args:
            query_token_id: B, LQ
            text_token_id: B, LS
        Returns:
            tensor of [B]
        """
        for k, v in kwargs.items():
            # B, 1+N, L -> B * (1+N), L
            if v.dim() == 3:
                kwargs[k] = v.view(-1, v.shape[-1])

        score = self.plm(**kwargs).logits.squeeze(-1)
        return score


    def compute_score(self, x):
        x = self._move_to_device(x)
        if "text_code" in x:
            # concate query and text code as inputs
            query_token_id = x["query"]["input_ids"]    # B, L
            query_attn_mask = x["query"]["attention_mask"]

            text_code = x["text_code"]  # B, 1+N, LC or B, LC
            if text_code.dim() == 3:
                text_code = text_code.flatten(0, 1) # B*(1+N) or B, LC

            M, L = text_code.shape[0] // query_token_id.shape[0], query_token_id.shape[-1]

            pair_token_id = torch.zeros((text_code.shape[0], text_code.shape[-1] + query_token_id.shape[-1] - 1), device=text_code.device)
            pair_token_id[:, :L] = query_token_id.repeat_interleave(M, 0)
            # remove the leading 0
            pair_token_id[:, L:] = text_code[:, 1:]

            pair_attn_mask = torch.zeros_like(pair_token_id)
            pair_attn_mask[:, :L] = query_attn_mask.repeat_interleave(M, 0)
            pair_attn_mask[:, L:] = (text_code != -1).float()

            pair = {
                "input_ids": pair_token_id,
                "attention_mask": pair_attn_mask
            }
            if "token_type_ids" in x["query"]:
                pair_type_id = torch.zeros_like(pair_attn_mask)
                pair_type_id[:, L:] = 1
                pair["token_type_ids"] = pair_type_id
        else:
            pair = x["pair"]

        score = self._compute_score(**pair) # B or B*(1+N)
        return score


    def forward(self, x):
        pair = x["pair"]
        score = self.compute_score(x) # B*(1+N)

        if pair["input_ids"].dim() == 3:
            # use cross entropy loss
            score = score.view(x["pair"]["input_ids"].shape[0], -1)
            label = torch.zeros(score.shape[0], dtype=torch.long, device=self.config.device)
            loss = F.cross_entropy(score, label)

        elif pair["input_ids"].dim() == 2:
            label = x["label"]
            loss = F.binary_cross_entropy(torch.sigmoid(score), label)

        return loss


