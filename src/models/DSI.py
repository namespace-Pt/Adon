import os
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from .BaseModel import BaseGenerativeModel


class DSI(BaseGenerativeModel):
    def __init__(self, config):
        super().__init__(config)

        self.plm = AutoModelForSeq2SeqLM.from_pretrained(config.plm_dir)
        if config.code_size > 0:
            self.plm.resize_token_embeddings(config.vocab_size + config.code_size)
        
        # NOTE: set hidden size to be involked when using deepspeed
        self.config.hidden_size = self.plm.config.hidden_size

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


class GENRE(DSI):
    def __init__(self, config):
        super().__init__(config)
    
    def generate_code(self, loaders):
        import multiprocessing as mp
        from transformers import AutoTokenizer

        assert self.config.code_type == "title"
        if self.config.is_main_proc:
            # the code is bind to the plm_tokenizer
            code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "codes.mmp")
            # all codes are led by 0 and padded by -1
            self.logger.info(f"generating codes from {self.config.code_type} with code_length: {self.config.code_length}, saving at {code_path}...")

            loader_text = loaders["text"]
            text_num = len(loader_text.dataset)

            from utils.util import _get_title_code, makedirs

            collection_path = os.path.join(self.config.data_root, self.config.dataset, "collection.tsv")
            makedirs(code_path)
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))

            # load all saved token ids
            text_codes = np.memmap(
                code_path,
                dtype=np.int32,
                mode="w+",
                shape=(text_num, self.config.code_length)
            )
            # the codes are always led by 0 and padded by -1
            text_codes[:, 0] = tokenizer.pad_token_id
            text_codes[:, 1:] = -1

            preprocess_threads = 10
            all_line_count = text_num

            arguments = [] 
            for i in range(preprocess_threads):
                start_idx = round(all_line_count * i / preprocess_threads)
                end_idx = round(all_line_count * (i+1) / preprocess_threads)
                arguments.append((collection_path, code_path, all_line_count, start_idx, end_idx, tokenizer, self.config.code_length, self.config.get("title_col", 0), self.config.get("code_sep", " ")))
            with mp.Pool(preprocess_threads) as p:
                p.starmap(_get_title_code, arguments)


class DSIQG(DSI):
    def __init__(self, config):
        super().__init__(config)
    
    def generate_code(self, loaders):
        from transformers import AutoTokenizer
        from utils.util import makedirs
        if self.config.is_main_proc:
            code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "codes.mmp")
            # all codes are led by 0 and padded by -1
            self.logger.info(f"generating codes from {self.config.code_type} with code_length: {self.config.code_length}, saving at {code_path}...")
            makedirs(code_path)

            loader_text = loaders["text"]
            text_num = len(loader_text.dataset)

            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))
            eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

            # load all saved token ids
            text_codes = np.memmap(
                code_path,
                dtype=np.int32,
                mode="w+",
                shape=(text_num, self.config.code_length)
            )
            # the codes are always led by 0 and padded by -1
            text_codes[:, 0] = tokenizer.pad_token_id
            text_codes[:, 1:] = -1

            for i in range(text_num):
                code = tokenizer.encode(str(i), add_special_tokens=False)
                code.append(eos_token_id)
                text_codes[i, 1: len(code) + 1] = code


