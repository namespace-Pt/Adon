import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.manager import Manager
from utils.util import load_pickle, save_pickle

import hydra
from pathlib import Path
from omegaconf import OmegaConf
@hydra.main(version_base=None, config_path="../data/config/", config_name=f"script/{Path(__file__).stem}")
def get_config(config: OmegaConf):
    manager.setup(config)


def compute_overlap(query_token_ids, text_token_ids, ground_truth, tokenizer, stop_token_ids=None):
    overlap = {}

    for qidx, v in tqdm(ground_truth.items(), ncols=100, desc="Computing Overlaps"):
        gt = v[0]
        gt_token_id = text_token_ids[gt].copy()
        query_token_id = query_token_ids[qidx].copy()

        gt_token_id_not_in_stop_token_ids = gt_token_id[np.isin(gt_token_id, stop_token_ids, invert=True)]

        gt_meaningful_tokens = set()
        for token in tokenizer.convert_ids_to_tokens(gt_token_id_not_in_stop_token_ids):
            if token.startswith("##") and len(token) == 3:
                continue
            gt_meaningful_tokens.add(token)
        gt_token_id_not_in_stop_token_ids = np.asarray(tokenizer.convert_tokens_to_ids(list(gt_meaningful_tokens)), dtype=np.int32)

        gt_overlap = np.isin(gt_token_id_not_in_stop_token_ids, query_token_id).sum()
        overlap[qidx] = gt_overlap

    return overlap


if __name__ == "__main__":
    # manually action="store_true" because hydra doesn't support it
    for i, arg in enumerate(sys.argv):
        if "=" not in arg:
            sys.argv[i] += "=true"

    manager = Manager()
    get_config()
    config = manager.config
    loaders = manager.prepare()
    dataset_text = loaders["text"].dataset
    dataset_query = loaders["query"].dataset

    if config.ground_truth_path == "default":
        ground_truth = load_pickle(os.path.join(config.cache_root, "dataset", config.eval_set, "positives.pkl"))
    else:
        ground_truth = load_pickle(config.ground_truth_path)

    # special_token_ids and punctuations and common words
    stop_tokens = set([x[0] for x in config.special_token_ids.values()] + [x for x in ";:'\\\"`~[]<>()\{\}/|?!@$#%^&*…-_=+,."] + ["what", "how", "where", "when", "why", "which", "a", "about", "also", "am", "to", "an", "and", "another", "any", "anyone", "are", "aren't", "as", "at", "be", "been", "being", "but", "by", "despite", "did", "didn't", "do", "does", "doesn't", "doing", "done", "don't", "each", "etc", "every", "everyone", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's", "him", "himself", "his", "however", "i", "i'd", "if", "i'll", "i'm", "in", "into", "is", "isn't", "it", "its", "it's", "itself", "i've", "just", "let's", "like", "lot", "may", "me", "might", "mightn't", "my", "myself", "no", "nor", "not", "of", "on", "onto", "or", "other", "ought", "oughtn't", "our", "ours", "ourselves", "out", "over", "shall", "shan't", "she", "she'd", "she'll", "she's", "since", "so", "some", "something", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "tht", "to", "too", "usually", "very", "via", "was", "wasn't", "we", "we'd", "well", "we'll", "were", "we're", "weren't", "we've", "will", "with", "without", "won't", "would", "wouldn't", "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've"])
    print(stop_tokens)
    t = AutoTokenizer.from_pretrained(config.plm_dir)
    stop_token_ids = np.asarray(t.convert_tokens_to_ids(list(stop_tokens)), dtype=np.int32)

    overlap = compute_overlap(
        query_token_ids=dataset_query.query_token_ids,
        text_token_ids=dataset_text.text_token_ids,
        tokenizer=t,
        ground_truth=ground_truth,
        stop_token_ids=stop_token_ids,
    )
    save_pickle(overlap, os.path.join(config.cache_root, "case", "overlap.pkl"))

    overall_overlap = round(sum([1 if x > 0 else 0 for x in overlap.values()]) / len(overlap), 2)
    print(f"The ratio of the ground truth that shares common token (except stop tokens) with the query is: {overall_overlap}!")

