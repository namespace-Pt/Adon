import torch
import torch.nn as nn
from .DSI import DSI
from transformers import AutoTokenizer


class BOW(DSI):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        x = self._move_to_device(x)
        # squeeze the auxillary dimension
        if "query_code" in x:
            text_code = x["query_code"]
        else:
            text_code = x["text_code"]
    
        # in case there are multiple codes for one text (config.permute_code > 0)
        encoder_outputs = self.plm.encoder(**x["query"])
        query_attn_mask = x["query"]["attention_mask"]
        B, M = text_code.shape[:2]
        for k, v in encoder_outputs.items():
            encoder_outputs[k] = v.repeat_interleave(M, 0)
        query_attn_mask = query_attn_mask.repeat_interleave(M, 0)   # B*M, L

        text_code = text_code.flatten(0,1)  # B*M, CL
        # the code has a leading 0, shift left one position so t5 can shift it back
        # default to -1
        labels = torch.zeros_like(text_code) - 1
        labels[:, :-1] = text_code[:, 1:]
        # replace the padding code with the -100 (ignored when computing loss)
        labels = labels.masked_fill(labels == -1, -100)
        logits = self.plm(attention_mask=query_attn_mask, encoder_outputs=encoder_outputs, labels=labels).logits    # B*M, CL, V

        # ignore_index defaults to -100
        loss = nn.functional.cross_entropy(logits.flatten(0,1), labels.view(-1), reduction="none").view(B, M, -1).mean(-1) # B, M
        # sum
        if self.config.reduce_code == "mean":
            loss = loss.mean()
        elif self.config.reduce_code == "min":
            min_loss, min_index = loss.min(-1)
            loss = min_loss.mean()
        else:
            raise NotImplementedError(f"Reduction type {self.config.reduce_code} is not implemented!")
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
            from utils.index import BeamDecoder, GreedyCodeSorter
            from utils.data import prepare_train_data
            index = self.index(loaders).index

            text_dataset = loaders["text"].dataset
            # set necessary attributes to enable loader_train
            self.config.loader_train = "neg"
            self.config.train_set = [self.config.eval_set]
            self.config.neg_type = "none"
            self.config.batch_size = self.config.eval_batch_size

            loader_train = prepare_train_data(self.config, text_dataset, return_dataloader=True)
            query_dataset = loader_train.dataset.query_datasets[0]

            code_path = os.path.join(self.code_dir, self.config.code_src, query_dataset.query_set, "codes.mmp")
            makedirs(code_path)

            self.logger.info(f"sorting keywords from {self.config.code_type} and saving at {code_path}...")

            if self.config.get("nbeam", 1) > 1:
                sorter = BeamDecoder()
                nseq = self.config.nbeam
            else:
                sorter = GreedyCodeSorter()
                nseq = self.config.get("decode_nseq", 1)

            if self.config.is_main_proc:
                query_codes = np.memmap(
                    code_path,
                    dtype=np.int32,
                    shape=(len(loader_train.dataset), nseq, self.config.code_length),
                    mode="w+"
                )
                # default to -1 to be used as padding
                query_codes[:] = -1
            synchronize()
            query_codes = np.memmap(
                code_path,
                dtype=np.int32,
                mode="r+"
            ).reshape(len(loader_train.dataset), nseq, self.config.code_length)

            
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))

            start_idx = 0
            for i, x in enumerate(tqdm(loader_train, leave=False, ncols=100)):
                # if i < 49:
                #     continue
                qrel_idx = x["qrel_idx"]
                query = self._move_to_device(x["query"])

                encoder_outputs = self.plm.encoder(**query)
                B = query["input_ids"].shape[0]
                end_idx = start_idx + B

                sorter.search(
                    model=self.plm, 
                    query={**query, "encoder_outputs": encoder_outputs},
                    nbeam=self.config.nbeam, 
                    max_new_tokens=self.config.code_length - 1, 
                    constrain_index=index,
                    text_indices=x["text_idx"].squeeze(1).numpy(),
                    # forbid early stop as we must generate the entire sequence
                    do_early_stop=False,
                    do_sample=self.config.get("decode_do_sample", False),
                    num_return_sequences=self.config.get("decode_nseq", 1),
                    temperature=self.config.get("decode_tau", 1),
                )
                # print(tokenizer.decode(x["query"]["input_ids"][0], skip_special_tokens=True), tokenizer.batch_decode(sorter.beams[0]))
                # input()

                res = sorter.beams  # batch_size, nseq, code_length

                # assign query_codes
                for j, y in enumerate(res):
                    length = len(y[0])
                    query_codes[qrel_idx[j], :, :length] = y

                if self.config.get("keep_text_code"):
                    query_codes[qrel_idx, 0] = x["text_code"].view(B, self.config.code_length).numpy()

                start_idx = end_idx
                if self.config.debug:
                    if i > 2:
                        break
            
            if self.config.is_main_proc:
                same_count = 0
                text_codes = text_dataset.text_codes
                for qrel in loader_train.dataset.qrels:
                    qrel_idx, query_set_idx, query_idx, text_idx = qrel
                    query_code = query_codes[qrel_idx]
                    text_code = text_codes[text_idx]
                    for qc in query_code:
                        if (qc == text_code).all():
                            same_count += 1
            
                self.logger.info(f"{same_count}/{query_codes.shape[0] * query_codes.shape[1]} query codes are identical to the text codes!")

        else:
            if self.config.code_type == "chat":
                import multiprocessing as mp
                from utils.util import _get_chatgpt_code

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
    
