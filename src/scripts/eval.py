import os
import sys
from utils.util import compute_metrics, load_pickle, Config

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

    if os.path.exists(config.src):
        path = config.src
    elif os.path.exists(os.path.join(config.cache_root, config.eval_mode, config.src, config.eval_set, "retrieval_result.pkl")):
        path = os.path.join(config.cache_root, config.eval_mode, config.src, config.eval_set, "retrieval_result.pkl")
    else:
        raise FileNotFoundError

    retrieval_result = load_pickle(path)

    ground_truth = load_pickle(os.path.join(config.cache_root, "dataset", "query", config.eval_set, "positives.pkl"))
    metrics = compute_metrics(retrieval_result, ground_truth, metrics=config.eval_metric, cutoffs=config.eval_metric_cutoff)
    print()
    print(metrics)