import torch
import torch.nn as nn
from .DSI import DSI

class BOW(DSI):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        x = self._move_to_device(x)
        query = x["query"]
        # squeeze the auxillary dimension
        if "query_code" in x:
            text_code = x["query_code"].squeeze(1)
        else:
            text_code = x["text_code"].squeeze(1)
        
        if text_code.dim() == 2:                
            # the code has a leading 0, shift left one position so t5 can shift it back
            # default to -1
            if self.config.get("elastic_ce"):
                elastic_labels = x["elastic_label"]
                decoder_input_ids = text_code.masked_fill(text_code == -1, 0)
                logits = self.plm(**query, decoder_input_ids=decoder_input_ids).logits
                logits = torch.log_softmax(logits, dim=-1)

                # find invalid elastic_labels
                indicator = elastic_labels == -1    # B, LC-1, LC-1
                # only keep valid ones
                masked_labels = elastic_labels.masked_fill(indicator, 0)
                # gather logits
                res = logits.gather(dim=-1, index=masked_labels)
                # mask invalid positions
                res[indicator] = 0
                # forbids dividing zero
                denominator = (~indicator).sum(-1)
                denominator[denominator == 0] = 1
                # variable valid number per row
                loss = -(res.sum(-1) / denominator).mean()
                # if torch.isnan(loss):
                #     print(x["qrel_idx"], x["elastic_label"], text_code, res)
                #     return

            else:
                labels = torch.zeros_like(text_code) - 1
                labels[:, :-1] = text_code[:, 1:]
                # replace the padding code with the -100 (ignored when computing loss)
                labels = labels.masked_fill(labels == -1, -100)
                loss = self.plm(**query, labels=labels).loss
        else:
            # in case there are multiple codes for one text (config.permute_code > 0)
            encoder_outputs = self.plm.encoder(**x["query"])
            query_attn_mask = x["query"]["attention_mask"]
            M = text_code.shape[1]
            for k, v in encoder_outputs.items():
                encoder_outputs[k] = v.repeat_interleave(M, 0)
            query_attn_mask = query_attn_mask.repeat_interleave(M, 0)   # M*B, L

            text_code = text_code.flatten(0,1)  # M*B, CL
            # the code has a leading 0, shift left one position so t5 can shift it back
            # default to -1
            labels = torch.zeros_like(text_code) - 1
            labels[:, :-1] = text_code[:, 1:]
            # replace the padding code with the -100 (ignored when computing loss)
            labels = labels.masked_fill(labels == -1, -100)
            loss = self.plm(attention_mask=query_attn_mask, encoder_outputs=encoder_outputs, labels=labels).loss

        return loss
    
    def generate_code(self, loaders):
        """
        Greedily decode the keywords.
        """
        import os
        import numpy as np
        from utils.util import synchronize, makedirs
        assert self.config.index_type == "wordset", "Only wordset index can be used when sorting keywords!"

        if self.config.get("sort_code"):
            from tqdm import tqdm
            from utils.index import GreedyKeywordSorter
            from utils.data import prepare_train_data
            index = self.index(loaders).index

            text_dataset = loaders["text"].dataset
            # set necessary attributes to enable loader_train
            self.config.loader_train = "neg"
            self.config.train_set = [self.config.eval_set]
            self.config.hard_neg_type = "none"
            self.config.batch_size = self.config.eval_batch_size

            loader_train = prepare_train_data(self.config, text_dataset, return_dataloader=True)
            query_dataset = loader_train.dataset.query_datasets[0]

            code_path = os.path.join(self.code_dir, self.config.code_src, query_dataset.query_set, "codes.mmp")
            makedirs(code_path)

            self.logger.info(f"sorting keywords from {self.config.code_type} and saving at {code_path}...")

            if self.config.is_main_proc:
                query_codes = np.memmap(
                    code_path,
                    dtype=np.int32,
                    shape=(len(loader_train.dataset), self.config.code_length),
                    mode="w+"
                )
                # default to -1 to be used as padding
                query_codes[:] = -1
            synchronize()
            query_codes = np.memmap(
                code_path,
                dtype=np.int32,
                mode="r+"
            ).reshape(len(loader_train.dataset), self.config.code_length)

            greedy_sorter = GreedyKeywordSorter()

            start_idx = 0
            for i, x in enumerate(tqdm(loader_train, leave=False, ncols=100)):
                # if i < 49:
                #     continue
                qrel_idx = x["qrel_idx"]
                query = self._move_to_device(x["query"])

                encoder_outputs = self.plm.encoder(**query)
                B = query["input_ids"].shape[0]
                end_idx = start_idx + B

                greedy_sorter.search(
                    model=self.plm,
                    text_indices=x["text_idx"].squeeze(1).numpy(),
                    wordset_index=index,
                    **query, 
                    encoder_outputs=encoder_outputs
                )
                res = greedy_sorter.res

                # assign query_codes
                for j, y in enumerate(res):
                    query_codes[qrel_idx[j], :len(y)] = y

                start_idx = end_idx
                if self.config.debug:
                    if i > 2:
                        break

        else:
            if self.config.code_type == "chat":
                import multiprocessing as mp
                from utils.util import _get_chatgpt_code
                from transformers import AutoTokenizer

                # the code is bind to the code_tokenizer
                code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "codes.mmp")
                self.logger.info(f"generating codes from {self.config.code_type} with code_length: {self.config.code_length}, saving at {code_path}...")

                loader_text = loaders["text"]
                text_num = len(loader_text.dataset)
                makedirs(code_path)

                tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))

                # load all saved token ids
                # all codes are led by 0 and padded by -1
                text_codes = np.memmap(
                    code_path,
                    dtype=np.int32,
                    mode="w+",
                    shape=(text_num, self.config.code_length)
                )
                # the codes are always led by 0 and padded by -1
                text_codes[:, 0] = tokenizer.pad_token_id
                text_codes[:, 1:] = -1

                thread_num = 10
                # each thread creates one jsonl file
                text_num_per_thread = text_num / thread_num

                arguments = []
                # re-tokenize words in the collection folder
                for i in range(thread_num):
                    input_path = os.path.join(self.config.data_root, self.config.dataset, "keywords.tsv")
                    start_idx = round(text_num_per_thread * i)
                    end_idx = round(text_num_per_thread * (i+1))

                    arguments.append((
                        input_path,
                        code_path,
                        text_num,
                        start_idx,
                        end_idx,
                        tokenizer,
                        self.config.code_length,
                    ))

                # the collection has no special_tokens so we don't need to filter them out
                with mp.Pool(thread_num) as p:
                    p.starmap(_get_chatgpt_code, arguments)
    

class BOWR(BOW):
    """
    Ranking oriented BOW.
    """
    def __init__(self, config):
        super().__init__(config)
        # freeze all parameters
        self.plm.requires_grad_(False)
        # unfreeze embeddings (the lm_head will also be unfreezed)
        self.plm.shared.requires_grad_(True)

        # register hook to eliminish gradients for indexes other than eos
        template = torch.ones(self.plm.shared.weight.shape[0], dtype=torch.bool, device=config.device)
        template[1] = False
        def hook(grad):
            grad[template] = 0
            return grad
        self.plm.shared.weight.register_hook(hook)

        self.scorer = torch.nn.Sequential(
            nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.plm.config.hidden_size // 2, 1)
        )


    def forward(self, x):
        x = self._move_to_device(x)
        query = x["query"]
        query_attn_mask = query["attention_mask"]
        encoder_outputs = self.plm.encoder(**query)

        text_token_id = x["text_code"]
        text_attn_mask = (text_token_id != -1).float()
        # remove -1 because it can not be recognized by the model
        text_token_id[text_token_id == -1] = 0

        B, M = text_token_id.shape[:2]
        text_token_id = text_token_id.flatten(0, 1) # B*M, L
        text_attn_mask = text_attn_mask.flatten(0, 1)

        # repeat query encode outputs to batchify
        for k, v in encoder_outputs.items():
            encoder_outputs[k] = v.repeat_interleave(M, 0)
        query_attn_mask = query_attn_mask.repeat_interleave(M, 0)
        
        outputs = self.plm(decoder_input_ids=text_token_id, encoder_outputs=encoder_outputs, attention_mask=query_attn_mask, output_hidden_states=True) # *, L, V
        token_embedding = outputs.decoder_hidden_states[-1]    # *, L, D
        
        valid_token_length = text_attn_mask.sum(dim=-1).long() - 1
        eos_embedding = token_embedding[range(valid_token_length.shape[0]), valid_token_length]
        score = self.scorer(eos_embedding).squeeze(-1)  # B*M

        label = torch.zeros(B, device=self.config.device, dtype=torch.long)
        score = score.view(B, M)
        loss = nn.functional.cross_entropy(score, label)
        return loss

