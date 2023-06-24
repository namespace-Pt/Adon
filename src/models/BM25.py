import os
import math
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .BaseModel import BaseSparseModel
from transformers import AutoTokenizer
from utils.static import *
from utils.util import BaseOutput, synchronize, save_pickle



class BM25(BaseSparseModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_dir)
    
    def encode_text_step(self, x):
        if self.config.get("save_weight"):
            text_idx = x["text_idx"].tolist()
            text_token_id = x["text"]["input_ids"].numpy()
            text_token_embedding = np.zeros((*text_token_id.shape, self._output_dim), dtype=np.float32)

            for i, (tidx, batch_token_id) in enumerate(zip(text_idx, text_token_id)):
                tokens = self.tokenizer.convert_ids_to_tokens(batch_token_id)
                for j, token in enumerate(tokens):
                    if token not in self.stop_words["special_tokens"]:
                        text_token_embedding[i, j] = self.pretokenize_index.compute_bm25_term_weight(str(tidx), token, analyzer=None)

            if "text_first_mask" in x:
                # mask the duplicated tokens' weight
                text_first_mask = x["text_first_mask"].numpy()
                text_token_embedding[~text_first_mask] = 0
            else:
                text_token_embedding[~x["text"]["attention_mask"].bool().numpy()] = 0

            return text_token_id, text_token_embedding

        else:
            return super().encode_text_step(x)

    @synchronize
    @torch.no_grad()
    def encode_text(self, *args, **kwargs):
        """
        One step in encode_text.

        Args:
            x: a data record.

        Returns:
            the text token id for indexing, array of [B, L]
            the text token embedding for indexing, array of [B, L, D]
        """
        if self.config.pretokenize:
            if self.config.get("save_weight"):
                from pyserini.index.lucene import IndexReader
                self.pretokenize_index = IndexReader(os.path.join(self.index_dir, "index"))
                assert self.config.save_encode
                assert not self.config.load_encode and not self.config.load_text_encode
            return BaseSparseModel.encode_text(self, *args, **kwargs)
        else:
            return BaseOutput()
    
    @synchronize
    @torch.no_grad()
    def encode_query(self, *args, **kwargs):
        """
        One step in encode_text.

        Args:
            x: a data record.

        Returns:
            the query token id for indexing, array of [B, L]
            the query token embedding for indexing, array of [B, L, D]
        """
        if self.config.pretokenize:
            return BaseSparseModel.encode_query(self, *args, **kwargs)
        else:
            return BaseOutput()

    
    @synchronize
    def generate_code(self, loaders: LOADERS):
        """
        Generate code by BM25 term weights.
        """
        import json
        import shutil
        from transformers import AutoModel
        from pyserini.index.lucene import IndexReader
        from utils.util import _get_token_code, makedirs, isempty

        assert self.config.pretokenize, f"Enable pretokenize!"

        # the code is bind to the code_tokenizer
        code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "codes.mmp")
        self.logger.info(f"generating codes from {self.config.code_type} with code_length: {self.config.code_length}, saving at {code_path}...")
        makedirs(code_path)

        tokenizer = AutoTokenizer.from_pretrained(self.config.plm_dir)
        code_tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))

        loader_text = loaders["text"]
        text_num = len(loader_text.dataset)
        start_idx = loader_text.sampler.start
        end_idx = loader_text.sampler.end

        if self.config.is_main_proc:
            # load all saved token ids
            # all codes are led by 0 and padded by -1
            text_codes = np.memmap(
                code_path,
                dtype=np.int32,
                mode="w+",
                shape=(text_num, self.config.code_length)
            )
            model = AutoModel.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))
            try:
                start_token_id = model._get_decoder_start_token_id()
            except ValueError:
                start_token_id = model.config.pad_token_id
                self.logger.warning(f"Decoder start token id not found, use pad token id ({start_token_id}) instead!")
            # the codes are always led by start_token_id and padded by -1
            text_codes[:, 0] = start_token_id
            text_codes[:, 1:] = -1
            
        synchronize()

        stop_words = set()
        punctuations = set([x for x in ";:'\\\"`~[]<>()\{\}/|?!@$#%^&*…-_=+,."])
        nltk_stop_words = set(["a", "about", "also", "am", "to", "an", "and", "another", "any", "anyone", "are", "aren't", "as", "at", "be", "been", "being", "but", "by", "despite", "did", "didn't", "do", "does", "doesn't", "doing", "done", "don't", "each", "etc", "every", "everyone", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's", "him", "himself", "his", "however", "i", "i'd", "if", "i'll", "i'm", "in", "into", "is", "isn't", "it", "its", "it's", "itself", "i've", "just", "let's", "like", "lot", "may", "me", "might", "mightn't", "my", "myself", "no", "nor", "not", "of", "on", "onto", "or", "other", "ought", "oughtn't", "our", "ours", "ourselves", "out", "over", "shall", "shan't", "she", "she'd", "she'll", "she's", "since", "so", "some", "something", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "tht", "to", "too", "usually", "very", "via", "was", "wasn't", "we", "we'd", "well", "we'll", "were", "we're", "weren't", "we've", "will", "with", "without", "won't", "would", "wouldn't", "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've"])
        # include punctuations
        stop_words = stop_words | punctuations
        # include nltk stop words
        stop_words = stop_words | nltk_stop_words
        # include numbers in stopwords
        stop_words.add(r"\d")

        collection_dir = os.path.join(os.path.join(self.index_dir, "collection"), "weighted")

        input_path = f"{collection_dir}/{self.config.rank:02d}.jsonl"

        if self.config.get("load_collection"):
            pass
        else:
            if self.config.is_main_proc:
                if not isempty(collection_dir):
                    shutil.rmtree(collection_dir)
                makedirs(input_path)
            synchronize()
            
            bm25_index = IndexReader(os.path.join(self.index_dir, "index"))
            if self.config.get("use_tfidf"):
                doc_count = bm25_index.stats()["documents"]

            with open(input_path, "w") as f:
                for i in tqdm(range(start_idx, end_idx), leave=False, ncols=100, desc="Collecting DFs"):
                    x = loader_text.dataset[i]
                    text_idx = str(x["text_idx"])

                    doc = bm25_index.get_document_vector(text_idx)
                    if self.config.get("use_tfidf"):
                        doc_len = sum(doc.values())

                    word_weight_pairs = {}
                    for word in doc:
                        if word[-1] in punctuations:
                            continue
                        if word not in word_weight_pairs:
                            # NOTE: set analyzer to None because this is a pretokenized index
                            if self.config.get("use_tfidf"):
                                df = bm25_index.get_term_counts(word, analyzer=None)[0]
                                word_weight_pairs[word] = round(doc[word] / doc_len * math.log(doc_count / df), 3)
                            else:
                                word_weight_pairs[word] = round(bm25_index.compute_bm25_term_weight(text_idx, word, analyzer=None), 3)

                    doc_vec = {"id": int(text_idx), "vector": word_weight_pairs}
                    f.write(json.dumps(doc_vec) + "\n")
        
        code_fields = self.config.code_type.split("-")
        defaults = ["weight", None]
        code_fields.extend(defaults[-(3 - len(code_fields)):])
        code_name, code_init_order, code_post_order = code_fields[:3]
        
        # force to stem
        _get_token_code(
            input_path, 
            code_path, 
            text_num, 
            start_idx, 
            end_idx, 
            code_tokenizer, 
            self.config.code_length, 
            code_init_order, 
            code_post_order, 
            stop_words, 
            self.config.get("code_sep", " "), 
            self.config.get("stem_code"),
            self.config.get("filter_num"),
            self.config.get("filter_unit"),
        )


    # FIXME: refactor
    def rerank(self, loaders: dict):
        from pyserini.index.lucene import IndexReader
        from utils.util import load_pickle

        index_reader = IndexReader(self.index_dir)

        tid2index = load_pickle(os.path.join(os.path.join(self.config.cache_root, "dataset", "text", "id2index.pkl")))
        tindex2id = {v: k for k, v in tid2index.items()}

        loader_rerank = loaders["rerank"]
        retrieval_result = defaultdict(list)

        self.logger.info("reranking...")
        for i, x in enumerate(tqdm(loader_rerank.dataset, ncols=100, leave=False)):
            query_idx = x["query_idx"]
            seq_idx = x["seq_idx"]

            seq_id = tindex2id[seq_idx]
            query = self.tokenizer.decode(x["query_token_id"], skip_special_tokens=True)
            score = index_reader.compute_query_document_score(seq_id, query)

            retrieval_result[query_idx].append((seq_idx, score))

        retrieval_result = self._gather_retrieval_result(retrieval_result)
        return retrieval_result


    @torch.no_grad()
    def compute_flops(self, loaders, log=True):
        """ compute flops as stated in SPLADE
        """
        from pyserini.index.lucene import IndexReader

        # document side
        loader_text = loaders["text"]
        doc_num = len(loader_text.dataset)
        query_num = len(loaders["query"].dataset)

        index = IndexReader(self.index_dir)
        terms = {}
        for k in tqdm(index.terms(), ncols=100, desc="Collecting Vocabulary"):
            terms[k.term] = 0

        D = terms
        Q = terms.copy()

        for i in tqdm(range(doc_num), ncols=100, desc="Collecting Text Terms"):
            tid = str(i)
            base_doc = index.get_document_vector(tid)
            base_key = base_doc.keys()
            for k in base_key:
                D[k] += 1

        with open(f"{self.config.data_root}/{self.config.dataset}/queries.dev.tsv") as f:
            for line in tqdm(f, ncols=100, desc="Collecting Query Terms"):
                qid, text = line.strip().split("\t")
                analysed = index.analyze(text)
                for k in analysed:
                    try:
                        Q[k] += 1
                    except:
                        pass

        flops = 0
        for k in D.keys():
            flops += (D[k] / doc_num * Q[k] / query_num)

        flops = round(flops, 2)
        self.metrics.update({"FLOPs": flops})
        if log:
            self.log_result()
            self.logger.info(f"FLOPs: {flops}")

