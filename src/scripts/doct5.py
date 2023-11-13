# on zhiyuan machine, must import utils first to load faiss before torch
from utils.util import synchronize, save_pickle, makedirs, Config
from utils.data import prepare_data

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from transformers import T5ForConditionalGeneration, AutoTokenizer

import hydra
from pathlib import Path
from omegaconf import OmegaConf
@hydra.main(version_base=None, config_path="../data/config/", config_name=f"script/{Path(__file__).stem}")
def get_config(hydra_config: OmegaConf):
    config._from_hydra(hydra_config)


def main(config:Config):
    loaders = prepare_data(config)
    loader_text = loaders["text"]

    max_length = config.query_length
    query_per_doc = config.query_per_doc

    doct5_path = os.path.join(config.data_root, config.dataset, "queries.doct5.tsv")
    doct5_qrel_path = os.path.join(config.data_root, config.dataset, "qrels.doct5.tsv")
    cache_dir = os.path.join(config.cache_root, "dataset", "query", "doct5")
    mmp_path = os.path.join(cache_dir, config.plm_tokenizer, "token_ids.mmp")
    makedirs(mmp_path)

    model = T5ForConditionalGeneration.from_pretrained(config.plm_dir).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)

    # generate psudo queries
    if not config.load_encode:
        # -1 is the pad_token_id
        query_token_ids = np.zeros((len(loader_text.sampler), query_per_doc, max_length), dtype=np.int32) - 1
        start_idx = end_idx = 0
        with torch.no_grad():
            for i, x in enumerate(tqdm(loader_text, ncols=100, desc="Generating Queries")):
                text = x["text"]
                for k, v in text.items():
                    text[k] = v.to(config.device, non_blocking=True)

                B = text["input_ids"].shape[0]

                sequences = model.generate(
                    **text,
                    do_sample=True,
                    max_length=max_length,
                    temperature=3.0,
                    top_k=10,
                    num_return_sequences=query_per_doc
                ).view(B, query_per_doc, -1).cpu().numpy()   # B, N, L

                end_idx += B
                query_token_ids[start_idx: end_idx, :, :sequences.shape[-1]] = sequences
                start_idx = end_idx

        # mask eos tokens
        query_token_ids[query_token_ids == config.special_token_ids["eos"][1]] = -1

        # use memmap to temperarily save the generated token ids
        if config.is_main_proc:
            query_token_ids_mmp = np.memmap(
                mmp_path,
                shape=(len(loader_text.dataset) * query_per_doc, max_length),
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
        synchronize()

    if config.is_main_proc:
        # load all saved token ids
        query_token_ids = np.memmap(
            mmp_path,
            dtype=np.int32,
            mode="r+"
        ).reshape(len(loader_text.dataset) * query_per_doc, max_length)

        # decode to strings and write to the query file
        idx = 0
        with open(doct5_path, "w") as f, open(doct5_qrel_path, "w") as g:
            for i, queries in enumerate(tqdm(query_token_ids.reshape(len(loader_text.dataset), query_per_doc, max_length), ncols=100, desc="Decoding")):
                for j, query in enumerate(queries):
                    seq = tokenizer.decode(query[query != -1], skip_special_tokens=True)    # N
                    f.write("\t".join([str(idx), seq]) + "\n")
                    g.write("\t".join([str(idx), "0", str(i), "1"]) + "\n")
                    idx += 1


if __name__ == "__main__":
    # manually action="store_true" because hydra doesn't support it
    for i, arg in enumerate(sys.argv):
        if "=" not in arg:
            sys.argv[i] += "=true"

    config = Config()
    get_config()
    main(config)
