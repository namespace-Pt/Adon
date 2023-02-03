import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils.util import synchronize, Config
from utils.data import prepare_data

import hydra
from pathlib import Path
from omegaconf import OmegaConf
@hydra.main(version_base=None, config_path="../data/config/", config_name=f"script/{Path(__file__).stem}")
def get_config(hydra_config: OmegaConf):
    config._from_hydra(hydra_config)


def main(config):
    loaders = prepare_data(config)
    loader_text = loaders["text"]

    max_length = config.query_length
    query_per_doc = config.query_per_doc
    mmp_path = os.path.join(config.cache_root, "dataset", "text", ",".join([str(x) for x in config.text_col]), "t5", "doct5.mmp")
    doct5_path = os.path.join(config.data_root, config.dataset, "doct5.tsv")

    model = T5ForConditionalGeneration.from_pretrained(config.plm_dir).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)

    # generate psudo queries
    if not config.load_encode:
        query_token_ids = np.zeros((len(loader_text.sampler), query_per_doc, max_length), dtype=np.int32)

        with torch.no_grad():
            start_idx = end_idx = 0
            for i, x in enumerate(tqdm(loader_text, ncols=100, desc="Generating Queries")):
                text = x["text"]
                for k, v in text.items():
                    text[k] = v.to(config.device, non_blocking=True)

                B = text["input_ids"].shape[0]

                sequences = model.generate(
                    **text,
                    max_length=max_length,
                    do_sample=True,
                    num_return_sequences=query_per_doc
                ).view(B, query_per_doc, -1).cpu().numpy()   # B, N, L

                end_idx += B
                query_token_ids[start_idx: end_idx, :, :sequences.shape[-1]] = sequences
                start_idx = end_idx

        # use memmap to temperarily save the generated token ids
        if config.is_main_proc:
            query_token_ids_mmp = np.memmap(
                mmp_path,
                shape=(len(loader_text.dataset), query_per_doc, max_length),
                dtype=np.int32,
                mode="w+"
            )
        synchronize()

        query_token_ids_mmp = np.memmap(
            mmp_path,
            dtype=np.int32,
            mode="r+"
        ).reshape(len(loader_text.dataset), query_per_doc, max_length)
        query_token_ids_mmp[loader_text.sampler.start: loader_text.sampler.end] = query_token_ids

        if config.is_main_proc:
            with open(doct5_path, "w") as f:
                for sequences in tqdm(query_token_ids, ncols=100, desc="Decoding"):
                    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)    # N
                    f.write("\t".join(texts) + "\n")

    # tokenize psudo queries by preprocess_plm and save it in the dataset/text/preprocess_plm/doct5.mmp
    # only need to do so when we want to re-tokenize the generated query by another plm
    if config.is_main_proc and config.plm_tokenizer != "t5":
        # load all saved token ids
        query_token_ids = np.memmap(
            mmp_path,
            dtype=np.int32,
            mode="r+"
        ).reshape(len(loader_text.dataset), query_per_doc, max_length)

        cache_dir = os.path.join(config.cache_root, "dataset", "text", ",".join([str(x) for x in config.text_col]), config.plm_tokenizer)
        os.makedirs(cache_dir, exist_ok=True)
        tokenize_thread = config.tokenize_thread
        all_line_count = len(loader_text.dataset)

        config._set_plm(already_on_main_proc=True)
        tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)
        config.logger.info("tokenizing {} in {} threads, output file will be saved at {}".format(doct5_path, tokenize_thread, cache_dir))

        arguments = []
        # create memmap first
        token_ids = np.memmap(
            os.path.join(cache_dir, "doct5.mmp"),
            shape=(all_line_count, query_per_doc, max_length),
            mode="w+",
            dtype=np.int32
        )

        for i in range(tokenize_thread):
            start_idx = round(all_line_count * i / tokenize_thread)
            end_idx = round(all_line_count * (i+1) / tokenize_thread)
            arguments.append((doct5_path, cache_dir, all_line_count, start_idx, end_idx, query_per_doc, tokenizer, max_length))

        with Pool(tokenize_thread) as p:
            id2indexs = p.starmap(_tokenize_text, arguments)


def _tokenize_text(input_path, output_dir, all_line_count, start_idx, end_idx, query_per_doc, tokenizer, max_length):
    """
    tokenize the input text, do padding and truncation, then save the token ids, token_lengths, text ids

    Args:
        input_path: input text file path
        output_dir: directory of output numpy arrays
        start_idx: the begining index to read
        end_idx: the ending index
        tokenizer: transformer tokenizer
        max_length: max length of tokens
        text_type: corpus class
    """
    pad_token_id = -1

    token_ids = np.memmap(
        os.path.join(output_dir, "doct5.mmp"),
        shape=(all_line_count, query_per_doc, max_length),
        mode="r+",
        dtype=np.int32
    )

    with open(input_path, 'r') as f:
        pbar = tqdm(total=end_idx-start_idx, desc="Tokenizing", ncols=100, leave=False)
        for idx, line in enumerate(f):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break

            pseudo_queries = line.split('\t')
            for j, query in enumerate(pseudo_queries):
                token_id = tokenizer.encode(query, add_special_tokens=False, truncation=True, max_length=max_length)
                token_length = len(token_id)

                token_ids[idx, j] = token_id + [pad_token_id] * (max_length - token_length)
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    # manually action="store_true" because hydra doesn't support it
    for i, arg in enumerate(sys.argv):
        if "=" not in arg:
            sys.argv[i] += "=true"

    config = Config()
    get_config()
    config._set_plm("doct5")
    main(config)
