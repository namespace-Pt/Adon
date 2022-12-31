import os
import torch
import faiss
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from utils.index import TrieIndex, permute_code
from utils.util import BaseOutput, load_pickle
from .BaseModel import BaseModel



class CrossEncoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.plm = AutoModelForSequenceClassification.from_pretrained(config.plm_dir, num_labels=1)
        # self.plm.pooler = None
        if config.code_size > 0:
            self.plm.resize_token_embeddings(config.vocab_size + config.code_size)


    def _compute_score(self, **kwargs):
        """ concate the query and the input text;
        Args:
            query_token_id: B, LQ
            text_token_id: B, LS
        Returns:
            tensor of [B]
        """
        for k, v in kwargs.items():
            # B, 1+N, L -> B * (1+N), L
            if v.dim() == 3:
                kwargs[k] = v.view(-1, v.shape[-1])

        score = self.plm(**kwargs).logits.squeeze(-1)
        return score


    def compute_score(self, x):
        x = self._move_to_device(x)
        if "text_code" in x:
            # concate query and text code as inputs
            query_token_id = x["query"]["input_ids"]    # B, L
            query_attn_mask = x["query"]["attention_mask"]

            text_code = x["text_code"]  # B, 1+N, LC or B, LC
            if text_code.dim() == 3:
                text_code = text_code.flatten(0, 1) # B*(1+N) or B, LC

            M, L = text_code.shape[0] // query_token_id.shape[0], query_token_id.shape[-1]

            pair_token_id = torch.zeros((text_code.shape[0], text_code.shape[-1] + query_token_id.shape[-1] - 1), device=text_code.device)
            pair_token_id[:, :L] = query_token_id.repeat_interleave(M, 0)
            # remove the leading 0
            pair_token_id[:, L:] = text_code[:, 1:]

            pair_attn_mask = torch.zeros_like(pair_token_id)
            pair_attn_mask[:, :L] = query_attn_mask.repeat_interleave(M, 0)
            pair_attn_mask[:, L:] = (text_code != -1).float()

            pair = {
                "input_ids": pair_token_id,
                "attention_mask": pair_attn_mask
            }
            if "token_type_ids" in x["query"]:
                pair_type_id = torch.zeros_like(pair_attn_mask)
                pair_type_id[:, L:] = 1
                pair["token_type_ids"] = pair_type_id
        else:
            pair = x["pair"]

        score = self._compute_score(**pair) # B or B*(1+N)
        return score


    def forward(self, x):
        pair = x["pair"]
        score = self.compute_score(x) # B*(1+N)

        if pair["input_ids"].dim() == 3:
            # use cross entropy loss
            score = score.view(x["pair"]["input_ids"].shape[0], -1)
            label = torch.zeros(score.shape[0], dtype=torch.long, device=self.config.device)
            loss = F.cross_entropy(score, label)

        elif pair["input_ids"].dim() == 2:
            label = x["label"]
            loss = F.binary_cross_entropy(torch.sigmoid(score), label)

        return loss



class Seq2SeqRanker(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        if config.score_type == "discrim":
            plm = T5Model.from_pretrained(config.plm_dir)
        elif config.score_type == "decode":
            plm = T5ForConditionalGeneration.from_pretrained(config.plm_dir)
        else:
            raise NotImplementedError(f"Ranking type {config.score_type} not implemented!")

        if config.get("code_size", 0) > 0:
            plm.resize_token_embeddings(config.vocab_size + config.code_size)

        if config.code_type == "UniCOIL-weight-align":
            self.logger.info("initializing new tokens embeddings...")
            new_token_embeds = np.load(os.path.join(config.cache_root, "codes", "UniCOIL-weight-align", config.code_tokenizer, str(config.code_length), "new_token_embeds.npy"))
            plm.encoder.embed_tokens.weight.data[-config.code_size:] = torch.tensor(new_token_embeds)

        self.plm = plm

        if config.score_type != "decode":
            self.scoreProject = nn.Linear(self.plm.config.hidden_size, 1)


    def _compute_score(self, text_token_id, **kwargs):
        """
        Wrapped method to compute each token's relevance score.

        Args:
            text_token_id: for score_type=discrim, it is a tensor of [*, L]; for score_type=decode, it is a tensor of [*, L+1];

        Returns:
            token_score: tensor of [*, L]
        """
        if self.config.score_type == "discrim":
            # important to add attention mask
            embedding = self.plm(decoder_input_ids=text_token_id, **kwargs).last_hidden_state # *, L, D
            token_score = self.scoreProject(embedding).squeeze(-1)    # *, L

        elif self.config.score_type == "decode":
            embedding = self.plm(decoder_input_ids=text_token_id, **kwargs).logits # *, L, V
            # remove the leading 0 <-> shift_left
            text_token_id = text_token_id[:, 1:]

            token_score = embedding.gather(index=text_token_id.unsqueeze(-1), dim=-1).squeeze(-1) # *, L

        else:
            raise NotImplementedError(f"Score type {self.config.score_type} not implemented!")

        return token_score


    def forward(self, x):
        x = self._move_to_device(x)
        encoder_outputs = self.plm.encoder(**x["query"])

        if "text_code" in x:
            text_token_id = x["text_code"]  # B, N, L

            text_attn_mask = (text_token_id != -1).float()
            # remove -1 because it can not be recognized by the model
            text_token_id = text_token_id.masked_fill(text_token_id == -1, 0)

            if text_token_id.dim() == 3:
                text_token_id = text_token_id.flatten(0, 1) # B*N, L
                text_attn_mask = text_attn_mask.flatten(0, 1)

            # no shift right for decode ranking because the code is already started by 0
            if self.config.score_type == "discrim":
                # strip off the leading 0
                text_token_id = text_token_id[:, 1:]
                text_attn_mask = text_attn_mask[:, 1:]

        else:
            text_token_id = x["text"].input_ids
            text_attn_mask = x["text"].attention_mask

            if text_token_id.dim() == 3:
                text_token_id = text_token_id.flatten(0, 1) # B*N, L
                text_attn_mask = text_attn_mask.flatten(0, 1)

        B = x["query"]["input_ids"].shape[0]
        M = text_token_id.shape[0] // B

        if M > 1:
            # 1+N case
            # repeat query encode outputs to batchify
            for k, v in encoder_outputs.items():
                encoder_outputs[k] = v.repeat_interleave(M, 0)
            attention_mask = x["query"]["attention_mask"].repeat_interleave(M, 0)

        token_score = self._compute_score(text_token_id, encoder_outputs=encoder_outputs, attention_mask=attention_mask)

        if self.config.score_type == "decode":
            # the gathered score is 1 token shorter than the text_token_id
            text_attn_mask = text_attn_mask[:, 1:]

        L = text_attn_mask.shape[-1]

        if self.config.ranking_token == "last":
            # only use the second last token to generate score
            # the last one is <eos>
            valid_token_length = text_attn_mask.sum(dim=-1).long() - 2  # B
            score = token_score[range(valid_token_length.shape[0]), valid_token_length] # B
            if M > 1:
                # cross entropy
                label = torch.zeros(B, device=self.config.device, dtype=torch.long)
                score = score.view(B, M)
                loss = F.cross_entropy(score, label)
            else:
                # binary
                label = x["label"]
                loss = F.binary_cross_entropy(torch.sigmoid(score), label)

        elif self.config.ranking_token == "all":
            # all tokens are optimized to generate precise intermidiate scores
            if M > 1:
                score = score.view(B, M, L)
                # neg_text_codes which shares same prefix with the positive one is masked, so that they will
                # be 0 after log_softmax
                text_code_prefix_mask = x["text_code_prefix_mask"][:, :, 1:]  # B, M, L
                score = score + (1 - text_code_prefix_mask) * -1e9
                score = score.transpose(1, 2)   # B, L, M

                label = torch.zeros((B, L), dtype=torch.long, device=self.config.device)
                # slice the attention_mask of positive text
                label_mask = text_attn_mask.view(B, M, L)[:, 0] # B, L
                # mask the padded pos_text_code
                label[~label_mask.bool()] = -100

                loss = F.cross_entropy(score.flatten(0, 1), label.view(-1), ignore_index=-100)  # B * (LS)

            else:
                # actually, degrades to DSI
                raise ValueError(f"To use ranking_token=all, must pass more than 1 text per query!")

        else:
            raise NotImplementedError(f"Ranking token {self.config.ranking_token} not implemented!")

        return loss


    def compute_score(self, x):
        if "text_code" in x:
            x = self._move_to_device(x, exclude_keys=["text_idx", "query_idx", "text"])

            text_token_id = x["text_code"]  # B, N, L
            text_attn_mask = (text_token_id != -1).float()
            # remove -1 because it can not be recognized by the model
            text_token_id = text_token_id.masked_fill(text_token_id == -1, 0)

            # no shift right for decode ranking because the code is already started by 0
            if self.config.score_type == "discrim":
                # strip off the leading 0
                text_token_id = text_token_id[:, 1:]
                text_attn_mask = text_attn_mask[:, 1:]

        else:
            x = self._move_to_device(x)
            text_token_id = x["text"].input_ids
            text_attn_mask = x["text"].attention_mask

        token_score = self._compute_score(text_token_id, **x["query"])

        if self.config.score_type == "decode":
            # the gathered score is 1 token shorter than the text_token_id
            text_attn_mask = text_attn_mask[:, 1:]

        if self.config.get("beam_ranking", "none") == "none":
            # ordinal ranking
            valid_token_length = text_attn_mask.sum(dim=-1).long() - 2  # B
            score = token_score[range(valid_token_length.shape[0]), valid_token_length] # B
        else:
            # beam ranking
            score = token_score

        return score


    @torch.no_grad()
    def rerank(self, loaders: dict):
        if self.config.get("beam_ranking", "none") == "none":
            return super().rerank(loaders)

        from torch_scatter import scatter_max

        self.logger.info("beam reranking...")

        if self.config.load_beam:
            self.logger.info("loading intermidiate beam scores...")
            retrieval_result = torch.load(os.path.join(self.retrieve_dir, str(self.config.world_size), f"scores_{self.config.rank}.pt"))

        else:
            loader_rerank = loaders["rerank"]
            retrieval_result = defaultdict(list)
            for i, x in enumerate(tqdm(loader_rerank, ncols=100, leave=False)):
                query_idx = x["query_idx"].tolist()	# B
                text_idx = x["text_idx"]	# B
                score = self.compute_score(x)

                if self.config.return_code:
                    text_token_id = x["text_code"].to(self.config.device, non_blocking=True)[:, 1:]   # B, LC - 1
                else:
                    text_token_id = x["text"]["input_ids"].to(self.config.device, non_blocking=True)

                for j, qidx in enumerate(query_idx):
                    retrieval_result[qidx].append((text_idx[j], text_token_id[j], score[j]))

                if self.config.get("debug"):
                    if i > 5:
                        break

        if self.config.save_beam:
            self.logger.info("saving intermidiate beam scores...")
            os.makedirs(self.retrieve_dir, exist_ok=True)
            torch.save(retrieval_result, os.path.join(self.retrieve_dir, str(self.config.world_size), f"scores_{self.config.rank}.pt"))

        if self.config.get("restore") == "lsh":
            text_codes = loaders["rerank"].dataset.text_codes.copy()
            one_hot_codes = np.zeros((text_codes.shape[0], text_codes.max() + 1), dtype=np.float32)
            # ignore the leading 0 and the tailing 1
            np.put_along_axis(one_hot_codes, text_codes[:, 1: -1], 1., axis=-1)
        elif self.config.get("restore") == "bm25":
            from utils.index import BM25Index
            text_codes = loaders["rerank"].dataset.text_codes.copy()
            bm25 = BM25Index()

        nbeam = self.config.nbeam

        for qidx, res in tqdm(retrieval_result.items(), ncols=100, desc="Beam Searching", leave=False):
            text_indices = torch.stack([x[0] for x in res], dim=0)   # 1000
            text_token_ids = torch.stack([x[1] for x in res], dim=0)  # 1000, 33
            token_scores = torch.stack([x[2] for x in res], dim=0)   # 1000, 33

            num_step = text_token_ids.shape[-1]

            # at first, we inspect all possibilities
            beam_hypothesis = torch.ones(text_token_ids.shape[0], dtype=torch.bool, device=self.config.device)
            # the candidate number will decrease from 1000 to nbeam, using bool index is inappropriate
            candidate_idx = torch.arange(text_token_ids.shape[0], device=self.config.device)

            for i in range(num_step):
                if i == num_step - 1:
                    break

                if isinstance(nbeam, list):
                    # if i exceeds its length, just access the last element
                    current_beam_size = nbeam[min(i, len(nbeam) - 1)]
                else:
                    current_beam_size = nbeam

                token_id_til_this_step = text_token_ids[candidate_idx, :(i + 1)]     # num_hypo, i
                unique_token_id, original_idx_to_unique_idx = token_id_til_this_step.unique(dim=0, return_inverse=True)  # num_unique

                # if there are no more than current_beam_size unique token id, we shall keep them all
                if len(unique_token_id) <= current_beam_size:
                    continue

                score = token_scores[candidate_idx, i]  # num_hypo
                # map the score to the unique score according to the inverse unique index
                unique_score, _ = scatter_max(score, index=original_idx_to_unique_idx)

                # select the topk branch
                topk_unique_score, topk_unique_idx = unique_score.topk(k=current_beam_size)  # current_beam_size
                # whether the i-th row in the unique topk
                beam_hypothesis = (original_idx_to_unique_idx.unsqueeze(-1) == topk_unique_idx).sum(-1).bool()  # beam hypothesis for next step
                candidate_idx = candidate_idx[beam_hypothesis]

            candidate_text_indices = text_indices[candidate_idx]    # num_hypo

            if self.config.get("restore") == "lsh":
                candidate_text_indices = candidate_text_indices.tolist()
                # text ids not in the beams
                except_beams = list(set(text_indices.tolist()) - set(candidate_text_indices))

                except_beams_onehot_codes = one_hot_codes[except_beams]
                lsh = faiss.IndexLSH(one_hot_codes.shape[-1], 32)
                lsh.train(except_beams_onehot_codes)
                lsh.add(except_beams_onehot_codes)

                beam_onehot_codes = one_hot_codes[candidate_text_indices]   # 100, 32100
                _, lsh_candidates = lsh.search(beam_onehot_codes, 2)   # 100, 5
                lsh_candidates = lsh_candidates.reshape(-1).tolist()

                idx_to_text_id = {k: v for k, v in enumerate(except_beams)}
                candidate_set = set(candidate_text_indices)
                # convert offset to text id
                lsh_candidate_set = set([idx_to_text_id[y] for y in lsh_candidates])
                candidate_text_indices = list(candidate_set.union(lsh_candidate_set))

                text_id_to_offset = {v: k for k, v in enumerate(text_indices.tolist())}
                # map back to the original 1000 offset
                candidate_idx = [text_id_to_offset[k] for k in candidate_text_indices]

                beam_hypothesis_score = token_scores[candidate_idx, -2]
                # print(f"old: {old}; new: {len(candidate_text_indices)}")

                sorted_beam_hypothesis = beam_hypothesis_score.argsort(descending=True)    # num_hypo
                res = []
                for idx in sorted_beam_hypothesis:
                    res.append((candidate_text_indices[idx], beam_hypothesis_score[idx]))
                # finally rank the hypothesis
                retrieval_result[qidx] = res

            elif self.config.get("restore") == "bm25":
                candidate_text_indices = candidate_text_indices.tolist()
                # text ids not in the beams
                except_beams = list(set(text_indices.tolist()) - set(candidate_text_indices))
                corpus = text_codes[except_beams]   # num_candidates - nbeams

                bm25.fit(corpus, ids=except_beams)

                beam_code = text_codes[candidate_text_indices]
                bm25_candidates = bm25.search(beam_code, 2)

                candidate_set = set(candidate_text_indices)
                # convert offset to text id
                candidate_text_indices = list(candidate_set.union(set(bm25_candidates)))

                text_id_to_offset = {v: k for k, v in enumerate(text_indices.tolist())}
                # map back to the original 1000 offset
                candidate_idx = [text_id_to_offset[k] for k in candidate_text_indices]

                beam_hypothesis_score = token_scores[candidate_idx, -2]
                # print(f"old: {old}; new: {len(candidate_text_indices)}")

                sorted_beam_hypothesis = beam_hypothesis_score.argsort(descending=True)    # num_hypo
                res = []
                for idx in sorted_beam_hypothesis:
                    res.append((candidate_text_indices[idx], beam_hypothesis_score[idx]))
                # finally rank the hypothesis
                retrieval_result[qidx] = res

            else:
                # we are using the second last token embedding's score for ranking in training
                if self.config.beam_ranking == "last":
                    beam_hypothesis_score = token_scores[candidate_idx, -2]   # num_hypo
                elif self.config.beam_ranking == "fewest":
                    beam_hypothesis_score = score[beam_hypothesis]   # num_hypo
                else:
                    raise NotImplementedError
                sorted_beam_hypothesis = beam_hypothesis_score.argsort(descending=True)    # num_hypo
                # finally rank the hypothesis
                retrieval_result[qidx] = list(zip(candidate_text_indices[sorted_beam_hypothesis.cpu()].tolist(), beam_hypothesis_score[sorted_beam_hypothesis].tolist()))

        return retrieval_result


    def index(self, loaders):
        """ creates a trie (prefix tree) index, where leaf nodes are docuements, and intermidiate nodes are codes
        """
        loader_text = loaders["text"]
        code_dir = os.path.join(self.config.cache_root, "codes", self.name if self.config.code_type == "self" else self.config.code_type, self.config.code_tokenizer)
        text_codes = loader_text.dataset.text_codes[loader_text.sampler.start: loader_text.sampler.end].copy()

        index = TrieIndex(
            save_dir=os.path.join(code_dir, "tries"),
            pad_token_id=self.config.special_token_ids["pad"][1]
        )

        index.fit(
            text_codes=text_codes,
            rebuild_index=self.config.code_type == "self",
            load_index=self.config.load_index,
            # only save at rank==0 because tries across processes are always the same
            save_index=self.config.is_main_proc and self.config.save_index
        )

        return BaseOutput(index=index, codes=text_codes)


    @torch.no_grad()
    def retrieve(self, loaders):
        loader_query = loaders["query"]

        assert self.config.batch_size_eval == 1, "Seq2SeqRanker only supports retrieve with batch_size==1!"

        self.logger.info("searching...")

        # query parallel
        query_start_idx = loader_query.sampler.start
        retrieval_result = defaultdict(list)
        nbeam = self.config.nbeam
        batch_size = 100

        if self.config.get("retrieve_candidate"):
            loader_text = loaders["text"]
            text_codes = loader_text.dataset.text_codes[loader_text.sampler.start: loader_text.sampler.end].copy()
            if os.path.exists(self.config.candidate_type):
                candidate_path = self.config.candidate_type
            elif os.path.exists(os.path.join(self.config.cache_root, "retrieve", self.config.candidate_type, self.config.eval_set, "retrieval_result.pkl")):
                candidate_path = os.path.join(self.config.cache_root, "retrieve", self.config.candidate_type, self.config.eval_set, "retrieval_result.pkl")
            else:
                raise FileNotFoundError(f"{self.config.candidate_type} Not Found!")
            # pad retrieval result
            candidates = load_pickle(candidate_path)
        else:
            output = self.index(loaders)
            trie = output.index

        for qidx, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
            qidx = qidx + query_start_idx

            query = self._move_to_device(x["query"])
            # compute query encode outputs only once
            query_encode_outputs = self.plm.encoder(**query)
            query_attn_mask = query["attention_mask"]

            if self.config.get("retrieve_candidate"):
                candidate = candidates[qidx]
                trie = TrieIndex(
                    save_dir=".",
                    pad_token_id=self.config.special_token_ids["pad"][1]
                )
                candidate_code = text_codes[candidate]
                if self.config.restore == "dup-rotate":
                    rotate_codes = permute_code(candidate_code)
                    for rotate_code in rotate_codes:
                        trie.add(rotate_code, ids=candidate, verbose=False)
                else:
                    # use the candidate idxs as ids
                    trie.add(candidate_code, ids=candidate, verbose=False)
                    batch_size = len(candidate)

            beam_hypotheses = [[0]]
            step = 0
            while True:
                # 0. get current beam size
                if isinstance(nbeam, list):
                    # if i exceeds its length, just access the last element
                    current_beam_size = nbeam[min(step, len(nbeam) - 1)]
                else:
                    current_beam_size = nbeam

                # 1. get next keys of all beams
                new_hypotheses = []
                for hypothesis in beam_hypotheses:
                    next_token_ids = trie.get_next_keys(hypothesis)
                    for next_token_id in next_token_ids:
                        new_hypotheses.append(hypothesis + [next_token_id])

                # 2. quit if there are less new_hypotheses, meaning next_token_ids == []
                if len(new_hypotheses) < len(beam_hypotheses):
                    # optional: rerank all beams according to the last token before quit
                    if self.config.beam_ranking == "last":
                        if isinstance(beam_hypotheses, list):
                            beam_hypotheses = np.array(beam_hypotheses, dtype=np.int64)
                        beam_scores = np.zeros(len(beam_hypotheses))
                        for start_idx in range(0, len(beam_hypotheses), batch_size):
                            end_idx = min(start_idx + batch_size, len(beam_hypotheses))
                            beam_scores[start_idx: end_idx] = self._compute_beam_score(query_attn_mask, query_encode_outputs, beam_hypotheses, start_idx, end_idx, return_score_idx=-2)
                        topk_beam_indices = np.arange(len(beam_scores))
                    break

                # 3. continue to expand hypotheses if not exceed current_beam_size
                if len(new_hypotheses) <= current_beam_size:
                    beam_hypotheses = new_hypotheses
                    continue

                beam_hypotheses = np.array(new_hypotheses, dtype=np.int64)
                beam_scores = np.zeros(len(beam_hypotheses))

                # 4. partition hypotheses by batch size
                for start_idx in range(0, len(beam_hypotheses), batch_size):
                    end_idx = min(start_idx + batch_size, len(beam_hypotheses))
                    beam_scores[start_idx: end_idx] = self._compute_beam_score(query_attn_mask, query_encode_outputs, beam_hypotheses, start_idx, end_idx)

                # 5. slice out the highest current_beam_size hypotheses
                topk_beam_indices = np.argpartition(-beam_scores, current_beam_size)[:current_beam_size]
                # using list facitates further expansion
                beam_hypotheses = beam_hypotheses[topk_beam_indices].tolist()

            text_scores = beam_scores[topk_beam_indices].tolist()   # current_beam_size
            res = defaultdict(list)
            for i, hypothesis in enumerate(beam_hypotheses):
                ids = trie[hypothesis]
                for id in ids:
                    res[id].append(text_scores[i])
            retrieval_result[qidx] = [(k, max(v)) for k, v in res.items()]

        return retrieval_result


    def _compute_beam_score(self, query_attn_mask, query_encode_outputs, beam_hypotheses, start_idx, end_idx, return_score_idx=-1):
        """ compute score for each beam
        """
        encoder_outputs = BaseModelOutput()
        for k, v in query_encode_outputs.items():
            encoder_outputs[k] = v.repeat_interleave(end_idx - start_idx, 0)
        query_attn_mask = query_attn_mask.repeat_interleave(end_idx - start_idx, 0)

        # remove the leading 0
        text_token_id = torch.as_tensor(beam_hypotheses[start_idx: end_idx, 1:], device=self.config.device)   # B, step - 1

        # very essential to pass in the query_attn_mask, otherwise the result differs with encoding from scratch
        token_score = self._compute_score(text_token_id=text_token_id, attention_mask=query_attn_mask, encoder_outputs=encoder_outputs)   # B, step - 1
        score = token_score[:, return_score_idx].cpu().numpy()  # B
        return score
