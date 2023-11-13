import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from accelerate import Accelerator


if __name__ == "__main__":
    accelerator = Accelerator(cpu=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="/share/LMs/", device="cpu")
    model = accelerator.prepare_model(model, device_placement=True, evaluation_mode=True).eval()

    kw_model = KeyBERT(model)

    ndoc = 109739
    ndoc_per_node = ndoc / accelerator.num_processes
    start_idx = round(ndoc_per_node * accelerator.process_index)
    end_idx = round(ndoc_per_node * (accelerator.process_index + 1))

    with open("/share/peitian/share/Datasets/Adon/NQ320k/collection.tsv") as f, open(f"/share/peitian/share/Datasets/Adon/NQ320k/phrases/2grams.{accelerator.process_index}.json", "w") as g:
        for i, line in enumerate(tqdm(f, total=109739)):
            if i < start_idx:
                continue
            if i >= end_idx:
                break

            text = " ".join(line.split("\t")[1:]).strip()
            text = " ".join(text.split()[:1024])

            with torch.no_grad():
                phrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 2), stop_words="english", top_n=200, use_mmr=True, diversity=0.5)
            # phrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 2), stop_words="english", top_n=200)
            phrases = [x[0] for x in phrases]
            g.write(json.dumps(phrases, ensure_ascii=False) + "\n")
