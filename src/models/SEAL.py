import os
import torch
import subprocess
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.util import BaseOutput, synchronize, makedirs
from .BaseModel import BaseGenerativeModel


class SEAL(BaseGenerativeModel):
    def __init__(self, config):
        super().__init__(config)
        assert "bart" in self.config.plm
        self.plm = AutoModelForSeq2SeqLM.from_pretrained(self.config.plm_dir)


    def index(self, loaders):
        """
        Build FM index.
        """
        import seal
        assert self.config.index_type == "fm", "Must use fm index!"

        index_dir = os.path.join(self.index_dir, "index")
        index_path = os.path.join(index_dir, "fm_index")
        if self.config.is_main_proc:
            collection_dir = os.path.join(self.index_dir, "collection")
            collection_path = os.path.join(collection_dir, "collection.tsv")

            makedirs(collection_path)
            makedirs(index_path)

            tokenizer = AutoTokenizer.from_pretrained(self.config.plm_dir)

            if (self.config.load_index and os.path.exists(index_path + ".oth")) or (self.config.load_collection and os.path.exists(collection_path)):
                pass
            else:
                assert self.config.get("title_col") is not None, "Must specify title column index!"
                loader_text = loaders["text"]
                with open(f"{self.config.data_root}/{self.config.dataset}/collection.tsv") as f, \
                    open(collection_path, "w") as g:
                    for line in tqdm(f, total=loader_text.dataset.text_num, ncols=100, desc="Building Collection"):
                        fields = line.split("\t")
                        fields = [field.strip() for field in fields]

                        tid = fields[0]
                        title = fields[self.config.title_col]
                        text = " ".join(fields[self.config.title_col + 1:])

                        # for fair comparison
                        text = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:self.config.text_length], skip_special_tokens=True)

                        g.write("\t".join([tid, title, text]) + "\n")

            if self.config.load_index and os.path.exists(index_path + ".oth"):
                pass
            else:
                subprocess.run(
                    f"python -m seal.build_fm_index {collection_path} {index_path} --hf_model {self.config.plm_dir} --jobs {self.config.index_thread} --include_title --lowercase", shell=True)

        synchronize()
        # fm_index = seal.SEALSearcher.load_fm_index(index_path)
        return BaseOutput()
    

    @synchronize
    @torch.no_grad()
    def retrieve(self, loaders):
        import seal
        index = self.index(loaders).index
        loader_query = loaders["query"]

        self.logger.info("searching...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.plm_dir)

        searcher = seal.SEALSearcher.load(os.path.join(self.index_dir, "index", "fm_index"), bart_model_path=f"../../SEAL/{self.config.dataset}/checkpoints/checkpoint_best.pt", backbone=self.config.plm_dir, device=self.config.device)
        # searcher = seal.SEALSearcher.load(os.path.join(self.index_dir, "index", "fm_index"), None, backbone=self.config.plm_dir, device=self.config.device)
        searcher.include_keys = True

        retrieval_result = {}
        for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
            query = x["query"]["input_ids"]
            query_idx = x["query_idx"].tolist()

            query = tokenizer.batch_decode(query, skip_special_tokens=True)
            for j, q in zip(query_idx, query):
                res = searcher.search(q, k=self.config.hits)
                retrieval_result[j] = [doc.id() for doc in res]

            if self.config.get("debug") and i > 1:
                break
        
        return retrieval_result


    def generate_code(self, loaders):
        pass


