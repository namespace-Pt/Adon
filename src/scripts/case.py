import os
import sys
from collections import defaultdict
from utils.util import save_pickle, load_pickle, makedirs, Config

import hydra
from omegaconf import OmegaConf
from pathlib import Path
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

    x_retrieval_result = load_pickle(x_path)
    y_retrieval_result = load_pickle(y_path)

    if config.x_hits > 0:
        for k, v in x_retrieval_result.items():
            x_retrieval_result[k] = v[:config.x_hits]
    if config.y_hits > 0:
        for k, v in y_retrieval_result.items():
            y_retrieval_result[k] = v[:config.y_hits]

    x_dir = os.path.join(*os.path.split(x_path)[:-1])
    y_dir = os.path.join(*os.path.split(y_path)[:-1])

    ground_truth = load_pickle(f"{config.cache_root}/dataset/{config.eval_set}/positives.pkl")

    unified_retrieval_result = defaultdict(lambda: {"x":[], "y":[], "both":[]})
    for k in ground_truth.keys():
        x_res = set(x_retrieval_result[k])
        y_res = set(y_retrieval_result[k])
        unified_retrieval_result[k]["x"] = x_res - y_res
        unified_retrieval_result[k]["y"] = y_res - x_res
        unified_retrieval_result[k]["both"] = x_res.intersection(y_res)

    unified_retrieval_result_gt = defaultdict(lambda: {"x":[], "y":[], "both":[], "neither":[]})
    x_unique_count = 0
    y_unique_count = 0
    both_count = 0
    gt_in_x = 0
    gt_in_y = 0
    gt_in_both = 0
    gt_in_neither = 0

    for k,v in unified_retrieval_result.items():
        x_unique_count += len(v["x"])
        y_unique_count += len(v["y"])
        both_count += len(v["both"])

        gt = set(ground_truth[k])
        x_unique_gt = gt.intersection(v["x"])
        y_unique_gt = gt.intersection(v["y"])
        both_gt = gt.intersection(v["both"])
        neither_gt = gt - v["x"].union(v["y"]).union(v["both"])

        unified_retrieval_result_gt[k]["x"] = x_unique_gt
        unified_retrieval_result_gt[k]["y"] = y_unique_gt
        unified_retrieval_result_gt[k]["both"] = both_gt
        unified_retrieval_result_gt[k]["neither"] = neither_gt

        if len(gt) > 0:
            gt_in_x += len(x_unique_gt) / len(gt)
            gt_in_y += len(y_unique_gt) / len(gt)
            gt_in_both += len(both_gt) / len(gt)
            gt_in_neither += len(neither_gt) / len(gt)
        else:
            gt_in_neither += 1

    print("*" * 10 + f"    {config.x_model} (X) v.s. {config.y_model} (Y)    " + "*" * 10)
    print(f"X Unique Count:         {round(x_unique_count / len(ground_truth))}")
    print(f"Y Unique Count:         {round(y_unique_count / len(ground_truth))}")
    print(f"Overlapping Count:      {round(both_count / len(ground_truth))}")
    print(f"GT in X Ratio:          {round(gt_in_x / len(ground_truth), 2)}")
    print(f"GT in Y Ratio:          {round(gt_in_y / len(ground_truth), 2)}")
    print(f"GT in Both Ratio:       {round(gt_in_both / len(ground_truth), 2)}")
    print(f"GT in Neither Ratio:    {round(gt_in_neither / len(ground_truth), 2)}")

    if config.save_case:
        save_path = os.path.join(config.cache_root, "case", f"unified.pkl")
        print(f"saving merged retrieval result at {save_path}")
        makedirs(save_path)
        save_pickle(dict(unified_retrieval_result_gt), save_path)
