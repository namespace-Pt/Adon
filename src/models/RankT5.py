import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from .BaseModel import BaseModel



class RankT5(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.plm = T5ForConditionalGeneration.from_pretrained(config.plm_dir)


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

        batch_size = kwargs["input_ids"].shape[0]
        score = self.plm(**kwargs, decoder_input_ids=torch.zeros((batch_size, 1), dtype=torch.long, device=self.config.device)).logits[:, 0, self.config.ranking_token]
        return score


    def rerank_step(self, x):
        x = self._move_to_device(x)
        pair = x["pair"]
        score = self._compute_score(**pair)
        return score


    def forward(self, x):
        pair = x["pair"]
        score = self.rerank_step(x) # B*(1+N)

        if pair["input_ids"].dim() == 3:
            # use cross entropy loss
            score = score.view(x["pair"]["input_ids"].shape[0], -1)
            label = torch.zeros(score.shape[0], dtype=torch.long, device=self.config.device)
            loss = F.cross_entropy(score, label)

        elif pair["input_ids"].dim() == 2:
            label = x["label"]
            loss = F.binary_cross_entropy(torch.sigmoid(score), label)

        return loss


