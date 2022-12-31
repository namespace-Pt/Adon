import os
import torch
from collections import OrderedDict
from transformers import T5ForConditionalGeneration
from .BaseModel import BaseGenerativeModel


class DSI(BaseGenerativeModel):
    def __init__(self, config):
        super().__init__(config)

        self.plm = T5ForConditionalGeneration.from_pretrained(config.plm_dir)
        if config.code_size > 0:
            self.plm.resize_token_embeddings(config.vocab_size + config.code_size)


    def _generate(self, **kargs):
        codes = self.plm.generate(
            **kargs
        )
        return codes


    def forward(self, x):
        x = self._move_to_device(x)
        query = x["query"]
        # strip off the leading 0
        text_code = x["text_code"]
        # the code has a leading 0, shift left one position so t5 can shift it back
        labels = torch.zeros_like(text_code)
        labels[:, :-1] = text_code[:, 1:]
        # replace the padding code with the -100 (ignored when computing loss)
        labels = labels.masked_fill(labels == -1, -100)

        loss = self.plm(**query, labels=labels).loss
        return loss
