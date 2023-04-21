"""
Generate negatives from the ``retrieval_result`` returned by :func:`models.BaseModel.BaseModel.retrieve` over ``train`` set.
"""
import sys
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

    for query_set in config.query_set:
        positives = load_pickle(f"{config.cache_root}/dataset/query/{query_set}/positives.pkl")

        retrieval_result = load_pickle(f"{config.cache_root}/retrieve/{config.neg_type}/{query_set}/retrieval_result.pkl")
        hard_negatives = defaultdict(list)
        for k,v in tqdm(retrieval_result.items(), desc="Collecting Negatives", ncols=100):
            for i, x in enumerate(v[:config.hits]):
                if x in positives[k]:
                    continue
                hard_negatives[k].append(x)

        neg_nums = np.array([len(x) for x in hard_negatives.values()])
        print(f"the collected query number is {len(hard_negatives)}, whose negative number is MEAN: {np.round(neg_nums.mean(), 1)}, MAX: {neg_nums.max()}, MIN: {neg_nums.min()}")

        if config.save_name != "default":
            save_name = config.save_name
        else:
            save_name = config.neg_type
        save_path = f"{config.cache_root}/dataset/query/{query_set}/negatives_{save_name}.pkl"
        save_pickle(dict(hard_negatives), save_path)
        print(f"saved negatives at {save_path}")