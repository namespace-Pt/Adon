import os
import sys
import scipy.stats as stats
from utils.util import load_pickle, compute_metrics, Config

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

    if os.path.exists(config.x_model):
        x_path = config.x_model
    elif os.path.exists(os.path.join(config.cache_root, config.eval_mode, config.x_model, config.eval_set, "retrieval_result.pkl")):
        x_path = os.path.join(config.cache_root, config.eval_mode, config.x_model, config.eval_set, "retrieval_result.pkl")
    else:
        raise FileNotFoundError

    if os.path.exists(config.y_model):
        y_path = config.y_model
    elif os.path.exists(os.path.join(config.cache_root, config.eval_mode, config.y_model, config.eval_set, "retrieval_result.pkl")):
        y_path = os.path.join(config.cache_root, config.eval_mode, config.y_model, config.eval_set, "retrieval_result.pkl")
    else:
        raise FileNotFoundError

    print(x_path, y_path)

    x_retrieval_result = load_pickle(x_path)
    y_retrieval_result = load_pickle(y_path)

    ground_truth = load_pickle(os.path.join(config.cache_root, "dataset", config.eval_set, "positives.pkl"))

    all_metrics = set()
    cutoffs = set()
    for metric in config.ttest_metric:
        if "@" in metric:
            metric_body, cutoff = metric.split("@")
            all_metrics.add(metric_body.lower())
            cutoffs.add(int(cutoff))
        else:
            all_metrics.add(metric.lower())

    all_metrics = list(all_metrics)
    cutoffs = list(cutoffs)

    x_metrics_per_query = compute_metrics(x_retrieval_result, ground_truth, metrics=all_metrics, cutoffs=cutoffs, return_each_query=True)
    y_metrics_per_query = compute_metrics(y_retrieval_result, ground_truth, metrics=all_metrics, cutoffs=cutoffs, return_each_query=True)

    print("*" * 10 + f"    {config.x_model} (X) v.s. {config.y_model} (Y)    " + "*" * 10)
    for metric in config.ttest_metric:
        print(f"the p of {metric}: {' '*(20 - len(metric))}{stats.ttest_rel(x_metrics_per_query[metric], y_metrics_per_query[metric]).pvalue}")
