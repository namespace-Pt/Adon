"""
Generate negatives from the ``retrieval_result`` returned by :func:`models.BaseModel.BaseModel.retrieve` over ``train`` set.
"""
import re
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.util import load_pickle, save_pickle, Config

import hydra
from pathlib import Path
from omegaconf import OmegaConf
@hydra.main(version_base=None, config_path="../data/config/", config_name=f"script/{Path(__file__).stem}")
def get_config(hydra_config: OmegaConf):
    config._from_hydra(hydra_config)


if __name__ == "__main__":
    # manually action="store_true" because hydra doesn't support it
    for i, arg in enumerate(sys.argv):
        if "=" not in arg:
            sys.argv[i] += "=true"

    config = Config()
    get_config()

    positives = load_pickle(f"{config.cache_root}/dataset/train/positives.pkl")

    if config.hard_neg_type == "official":
        save_name = "official"
        qid2index = load_pickle(os.path.join(config.cache_root, "dataset", "train", "id2index.pkl"))
        tid2index = load_pickle(os.path.join(config.cache_root, "dataset", "text", "id2index.pkl"))
        if config.dataset == "MSMARCO-passage":
            triple_path = os.path.join(config.data_root, config.dataset, "qidpidtriples.train.full.2.tsv")
            hard_negatives = defaultdict(list)
            with open(triple_path, "r") as f:
                for line in tqdm(f, ncols=100, desc="Collecting Negatives"):
                    qid, pid, nid = line.strip().split("\t")
                    qid = qid2index[qid]
                    nid = tid2index[nid]
                    if nid in positives[qid]:
                        continue
                    hard_negatives[qid].append(nid)
        elif config.dataset == "MSMARCO-Document":
            top100_path = os.path.join(config.data_root, config.dataset, "msmarco-doctrain-top100")
            hard_negatives = defaultdict(list)
            with open(top100_path, "r") as f:
                for line in tqdm(f, ncols=100, desc="Collecting Negatives"):
                    qid, _, nid, _, _, _ = line.strip().split()
                    qid = qid2index[qid]
                    nid = tid2index[nid]
                    if nid in positives[qid]:
                        continue
                    hard_negatives[qid].append(nid)
        else:
            raise NotImplementedError

    elif config.hard_neg_type[:3] == "hg-":
        assert config.dataset == "MSMARCO-passage", "hg-xxx only available on MSMARCO-passage"
        save_name = config.hard_neg_type
        negative_path = os.path.join(config.data_root, config.dataset, "msmarco-hard-negatives.jsonl")
        qid2index = load_pickle(os.path.join(config.cache_root, "dataset", "train", "id2index.pkl"))
        tid2index = load_pickle(os.path.join(config.cache_root, "dataset", "text", "id2index.pkl"))
        hard_negatives = defaultdict(list)

        hg_neg_type = config.hard_neg_type.split("-")[1]

        with open(negative_path, "r") as f:
            for line in tqdm(f, ncols=100, desc="Collecting Negatives"):
                fields = json.loads(line.strip())
                qid = str(fields["qid"])
                pos = str(fields["pos"])
                negs = fields["neg"]

                # confirm the query exists in qrels
                if qid in qid2index:
                    qidx = qid2index[qid]

                    neg_set = set()
                    if hg_neg_type == "all":
                        for neg in negs.values():
                            neg_set.update(neg)
                        hard_negatives[qidx] = list(neg_set)
                    else:
                        # some negatives are not included in fields
                        try:
                            nidxs = [tid2index[str(x)] for x in negs[hg_neg_type]]
                            hard_negatives[qidx] = nidxs
                        except:
                            pass

    elif os.path.exists(f"{config.cache_root}/retrieve/{config.hard_neg_type}/train/retrieval_result.pkl"):
        save_name = config.hard_neg_type
        retrieval_result = load_pickle(f"{config.cache_root}/retrieve/{config.hard_neg_type}/train/retrieval_result.pkl")
        hard_negatives = defaultdict(list)
        for k,v in tqdm(retrieval_result.items(), desc="Collecting Negatives", ncols=100):
            for i, x in enumerate(v):
                if x in positives[k]:
                    continue
                hard_negatives[k].append(x)

    elif os.path.exists(config.hard_neg_type):
        save_name = re.search("retrieve/(.*)?/train", config.hard_neg_type).group(1)
        retrieval_result = load_pickle(config.hard_neg_type)
        hard_negatives = defaultdict(list)
        for k,v in tqdm(retrieval_result.items(), desc="Collecting Negatives", ncols=100):
            for i, x in enumerate(v):
                if x in positives[k]:
                    continue
                hard_negatives[k].append(x)

    else:
        raise NotImplementedError(f"Not Implemented Model {config.hard_neg_type}!")

    neg_nums = np.array([len(x) for x in hard_negatives.values()])
    print(f"the collected query number is {len(hard_negatives)}, whose negative number is MEAN: {np.round(neg_nums.mean(), 1)}, MAX: {neg_nums.max()}, MIN: {neg_nums.min()}")

    if config.name != "default":
        save_name = config.name
    save_path = f"{config.cache_root}/dataset/train/negatives_{save_name}.pkl"
    save_pickle(dict(hard_negatives), save_path)
    print(f"saved negatives at {save_path}")