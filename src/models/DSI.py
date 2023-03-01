import os
import torch
import numpy as np
from transformers import T5ForConditionalGeneration
from .BaseModel import BaseGenerativeModel


class DSI(BaseGenerativeModel):
    def __init__(self, config):
        super().__init__(config)

        self.plm = T5ForConditionalGeneration.from_pretrained(config.plm_dir)
        if config.code_size > 0:
            self.plm.resize_token_embeddings(config.vocab_size + config.code_size)

        if config.code_type == "UniCOIL-weight-align":
            self.logger.info("initializing new tokens embeddings...")
            new_token_embeds = np.load(os.path.join(config.cache_root, "codes", "UniCOIL-weight-align", config.code_tokenizer, str(config.code_length), "new_token_embeds.npy"))
            self.plm.encoder.embed_tokens.weight.data[-config.code_size:] = torch.tensor(new_token_embeds)


    def forward(self, x):
        x = self._move_to_device(x)
        query = x["query"]
        # squeeze the auxillary dimension
        text_code = x["text_code"].squeeze(1)
        # the code has a leading 0, shift left one position so t5 can shift it back
        labels = torch.zeros_like(text_code)
        labels[:, :-1] = text_code[:, 1:]
        # replace the padding code with the -100 (ignored when computing loss)
        labels = labels.masked_fill(labels == -1, -100)

        loss = self.plm(**query, labels=labels).loss
        return loss


    def rerank_step(self, x):
        """
        Rerank using the log sum of the generation probabilities.
        """
        x = self._move_to_device(x)
        query = x["query"]
        # starts with 0
        text_code = x["text_code"]
        logits = self.plm(**query, decoder_input_ids=text_code).logits
        logits = logits.log_softmax(-1)
        score = logits.gather(dim=-1, index=text_code[:, 1:, None]).squeeze(-1).sum(-1)
        return score
