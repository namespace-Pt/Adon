import regex
import unicodedata
import argparse
import pickle
import sys
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils.util import save_pickle



def load_test_data(query_andwer_path, collection_path):
    answers = []
    for line in open(query_andwer_path, encoding='utf-8'):
        line = line.strip().split('\t')
        answers.append(eval(line[1]))

    collection = []
    for line in tqdm(open(collection_path, encoding='utf-8'), ncols=100, desc="Collecting Passages", leave=False):
        line = line.strip().split('\t')
        collection.append(line[2])
    return answers, collection


class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncase=False):
        tokens = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()
            # Format data
            if uncase:
                tokens.append(token.lower())
            else:
                tokens.append(token)
        return tokens


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string.
    """
    text = _normalize(text)

    # Answer is a list of possible strings
    text = tokenizer.tokenize(text, uncase=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncase=True)

        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def collate_fn(batch_hits):
    return batch_hits


class EvalDataset(Dataset):
    def __init__(self, retrieval_result, answers, collection):
        self.collection = collection
        self.answers = answers
        self.retrieval_result = retrieval_result
        self.tokenizer = SimpleTokenizer()

    def __getitem__(self, qidx):
        res = self.retrieval_result[qidx]
        hits = []
        for i, tidx in enumerate(res):
            if tidx == -1:
                hits.append(False)
            else:
                hits.append(has_answer(self.answers[qidx], self.collection[tidx], self.tokenizer))
        return hits

    def __len__(self):
        return len(self.retrieval_result)


def validate(retrieval_result, answers, collection, num_workers=16, batch_size=16):
    dataset = EvalDataset(retrieval_result, answers, collection)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    final_scores = []
    for scores in tqdm(dataloader, total=len(dataloader), ncols=100, desc="Computing Metrics"):
        final_scores.extend(scores)

    relaxed_hits = np.zeros(max([len(x) for x in retrieval_result.values()]))
    for question_hits in final_scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            relaxed_hits[best_hit:] += 1

    relaxed_recall = relaxed_hits / len(retrieval_result)

    return {
        "Recall@1": round(relaxed_recall[0], 4),
        "Recall@5": round(relaxed_recall[4], 4),
        "Recall@10": round(relaxed_recall[9], 4),
        "Recall@20": round(relaxed_recall[19], 4),
        "Recall@100": round(relaxed_recall[99], 4)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_result_path")
    parser.add_argument("--query_answer_path", default="../../../Data/NQ/nq-test.qa.csv")
    parser.add_argument("--collection_path", default="../../../Data/NQ/collection.tsv")
    args = parser.parse_args()

    with open(args.retrieval_result_path, "rb") as f:
        retrieval_result = pickle.load(f)

    metric = validate(retrieval_result, *load_test_data(args.query_answer_path, args.collection_path))
    sys.stdout.write(str(metric))