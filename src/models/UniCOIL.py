import torch
import torch.nn as nn
from .COIL import COIL



class UniCOIL(COIL):
    def __init__(self, config):
        config.token_dim = 1
        super().__init__(config)

        plm_dim = self.textEncoder.config.hidden_size
        self.tokenProject = nn.Sequential(
            nn.Linear(plm_dim, plm_dim),
            nn.ReLU(),
            nn.Linear(plm_dim, self._output_dim),
            nn.ReLU()
        )

        self.special_token_ids = [x[1] for x in config.special_token_ids.values() if x[0] is not None]


    def _to_bow(self, token_ids, token_weights):
        """
        Convert the token sequence (maybe repetitive tokens) into BOW (no repetitive tokens except pad token)

        Args:
            token_ids: tensor of B, L
            token_weights: tensor of B, L, 1

        Returns:
            bow representation of B, V
        """
        # create the src
        dest = torch.zeros((*token_ids.shape, self.config.vocab_size), device=token_ids.device) - 1   # B, L, V
        bow = torch.scatter(dest, dim=-1, index=token_ids.unsqueeze(-1), src=token_weights)
        bow = bow.max(dim=1)[0]    # B, V
        # only pad token and the tokens with positive weights are valid
        bow[:, self.special_token_ids] = 0
        return bow


    def encode_text_step(self, x):
        text = self._move_to_device(x["text"])
        text_token_id = text["input_ids"]
        text_token_weight = self._encode_text(**text)

        if "text_first_mask" in x:
            text_bow = self._to_bow(text_token_id, text_token_weight)
            text_token_weight = text_bow.gather(index=text_token_id, dim=-1)

            # mask the duplicated tokens' weight
            text_first_mask = self._move_to_device(x["text_first_mask"])
            # mask duplicated tokens' id
            text_token_id = text_token_id.masked_fill(~text_first_mask, 0)
            text_token_weight = text_token_weight.masked_fill(~text_first_mask, 0).unsqueeze(-1)

        # unsqueeze to map it to the _output_dim (1)
        return text_token_id.cpu().numpy(), text_token_weight.cpu().numpy()


    def encode_query_step(self, x):
        query = self._move_to_device(x["query"])
        query_token_id = query["input_ids"]

        query_token_weight = self._encode_query(**query)
        query_token_weight *= query["attention_mask"].unsqueeze(-1)

        # unsqueeze to map it to the _output_dim (1)
        return query_token_id.cpu().numpy(), query_token_weight.cpu().numpy()


    @torch.no_grad()
    def generate_code(self, loaders):
        if self.config.get("sort_code"):
            import os
            import numpy as np
            from tqdm import tqdm
            from transformers import AutoTokenizer
            from utils.index import convert_tokens_to_words, subword_to_word_bert
            from utils.data import prepare_train_data
            from utils.util import synchronize, makedirs

            text_dataset = loaders["text"].dataset
            # set necessary attributes to enable loader_train
            self.config.loader_train = "neg"
            self.config.train_set = [self.config.eval_set]
            self.config.hard_neg_type = "none"
            self.config.batch_size = self.config.eval_batch_size

            loader_train = prepare_train_data(self.config, text_dataset, return_dataloader=True)
            query_dataset = loader_train.dataset.query_datasets[0]

            code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), self.config.code_src, query_dataset.query_set, "codes.mmp")
            makedirs(code_path)

            self.logger.info(f"sorting keywords from {self.config.code_type} and saving at {code_path}...")

            text_tokenizer = AutoTokenizer.from_pretrained(self.config.plm_dir)
            code_tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))

            special_tokens = set([x[0] for x in self.config.special_token_ids.values()])

            # load all encoded results
            # text_outputs = self.encode_text(loaders["text"], load_all_cache=True)
            # query_outputs = self.encode_query(loaders["query"], load_all_cache=True)
            # text_token_ids = text_outputs.token_ids
            # text_token_weights = text_outputs.embeddings.squeeze(-1)
            # query_token_ids = query_outputs.token_ids
            # query_token_weights = query_outputs.embeddings.squeeze(-1)

            if self.config.is_main_proc:
                query_codes = np.memmap(
                    code_path,
                    dtype=np.int32,
                    shape=(len(loader_train.dataset), self.config.code_length),
                    mode="w+"
                )
                # default to -1 to be used as padding
                query_codes[:, 1:] = -1
            synchronize()
            query_codes = np.memmap(
                code_path,
                dtype=np.int32,
                mode="r+"
            ).reshape(len(loader_train.dataset), self.config.code_length)

            for i, x in enumerate(tqdm(loader_train, leave=False, ncols=100)):
                qrel_idx = x["qrel_idx"].numpy()

                # squeeze the second dimension (text_num)
                for k, v in x.items():
                    if k == "text":
                        for sk, sv in v.items():
                            x[k][sk] = sv.squeeze(1)
                    elif "text" in k:
                        x[k] = v.squeeze(1)

                text_code = x["text_code"]

                x = self._move_to_device(x)
                text_token_id = x["text"]["input_ids"]  # B, LS
                query_token_id = x["query"]["input_ids"]    # B, LQ

                text_token_weight = self._encode_text(**x["text"]) # B, LS, 1
                query_token_weight = self._encode_query(**x["query"])  # B, LQ, 1

                overlap = query_token_id.unsqueeze(-1) == text_token_id.unsqueeze(1)    # B, LS, LQ
                # accumulate query_token_weight when there is overlap
                text_token_weight = text_token_weight.squeeze(-1) + (query_token_weight * overlap).sum(1)   # B, LS
                text_token_weight = text_token_weight.cpu().numpy()

                for j, token_id in enumerate(text_token_id):
                    token_weight = text_token_weight[j] # LS
                    tokens = text_tokenizer.convert_ids_to_tokens(token_id)
                    words, weights = convert_tokens_to_words(tokens, subword_to_word_bert, scores=token_weight, reduce="max")

                    # a dict mapping the compressed phrase tokens to the phrase weight
                    word_weight_dict = {}
                    phrase_weight = 0
                    phrase_weights = []
                    # collect the accumulated weights for each phrase (comma separated)
                    for word, weight in zip(words, weights):
                        if word in special_tokens:
                            continue
                        # comma will be a standalone token
                        elif word == self.config.code_sep:
                            # compress all tokens to overcome the space issues from different tokenizers
                            phrase_weights.append(phrase_weight)
                            phrase_weight = 0
                        else:
                            phrase_weight += weight

                    phrase_weights = np.array(phrase_weights)
                    # sort the words by their weights descendingly
                    sorted_indices = np.argsort(phrase_weights)[::-1]

                    src_code = text_code[j]
                    src_phrases = code_tokenizer.decode(src_code[src_code != -1], skip_special_tokens=True)
                    src_phrases = [prs.strip() for prs in src_phrases.split(self.config.code_sep) if len(prs.strip())]

                    dest_phrases = []
                    try:
                        for idx in sorted_indices:
                            dest_phrases.append(src_phrases[idx])
                    except:
                        print(x["text_idx"][j], tokens, words)
                        raise
                    
                    # add separator at the end of each word
                    words = [prs + self.config.code_sep + " " for prs in dest_phrases]

                    query_code = "".join(words)
                    # the query code must be less than code_length - 1
                    query_code = code_tokenizer.encode(query_code)
                    assert len(query_code) < self.config.code_length

                    query_codes[qrel_idx[j], 1: len(query_code) + 1] = query_code

        else:
            return super().generate_code(loaders)
