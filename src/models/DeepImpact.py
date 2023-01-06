import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from .UniCOIL import UniCOIL



class DeepImpact(UniCOIL):
    def __init__(self, config):
        """
        `DeepImpact model <https://arxiv.org/abs/2104.12016>`_
        """
        super().__init__(config)

        plm_dim = self.plm.config.hidden_size
        self.tokenProject = nn.Sequential(
            nn.Linear(plm_dim, plm_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(plm_dim, 1),
            nn.ReLU()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)


    def encode_text_step(self, x):
        text = self._move_to_device(x["text"])
        text_embedding = self._encode_text(**text)
        return x["text"]["input_ids"].cpu().numpy(), text_embedding.cpu().numpy()


    def encode_query_step(self, x):
        """
        not contextualized
        """
        query_token_id = x["query"]["input_ids"].numpy()
        query_token_embedding = np.ones((*query_token_id.shape, self._output_dim), dtype=np.float32)
        return query_token_id, query_token_embedding
