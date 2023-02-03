import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from .UniCOIL import UniCOIL
from .BaseModel import BaseSparseModel



class DeepImpact(UniCOIL):
    def __init__(self, config):
        """
        `DeepImpact model <https://arxiv.org/abs/2104.12016>`_
        """
        super().__init__(config)

        plm_dim = self.textEncoder.config.hidden_size
        self.tokenProject = nn.Sequential(
            nn.Linear(plm_dim, plm_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(plm_dim, 1),
            nn.ReLU()
        )


    def encode_text_step(self, x):
        text = self._move_to_device(x["text"])
        text_token_id = text["input_ids"]
        text_token_weight = self._encode_text(**text).squeeze(-1)

        if "text_first_mask" in x:
            # mask the duplicated tokens' weight
            text_first_mask = self._move_to_device(x["text_first_mask"])
            text_token_weight = text_token_weight.masked_fill(~text_first_mask, 0)

        return text_token_id.cpu().numpy(), text_token_weight.unsqueeze(-1).cpu().numpy()



    def encode_query_step(self, x):
        """
        not contextualized
        """
        return BaseSparseModel.encode_query_step(self, x)
