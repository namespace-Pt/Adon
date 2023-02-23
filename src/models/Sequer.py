import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from .BaseModel import BaseGenerativeModel
from utils.util import synchronize



class Sequer(BaseGenerativeModel):
    def __init__(self, config):
        super().__init__(config)

        plm = T5ForConditionalGeneration.from_pretrained(config.plm_dir)

        if config.get("code_size", 0) > 0:
            plm.resize_token_embeddings(config.vocab_size + config.code_size)

        if config.code_type == "UniCOIL-weight-align":
            self.logger.info("initializing new tokens embeddings...")
            new_token_embeds = np.load(os.path.join(config.cache_root, "codes", "UniCOIL-weight-align", config.code_tokenizer, str(config.code_length), "new_token_embeds.npy"))
            plm.encoder.embed_tokens.weight.data[-config.code_size:] = torch.tensor(new_token_embeds)

        self.plm = plm
        self.scorer = torch.nn.Linear(plm.config.d_model, 1)


    def _prepare_decoder_inputs(self, text=None, text_code=None):
        """
        Prepare for _compute_logits. For regular text, shift right by 1 position; for code, keep it as is and generate attention mask.

        Returns:
            text_token_id: tensor of [B, L]; starting with 0
            text_attn_mask: tensor of [B, L]
        """
        if text_code is not None:
            text_token_id = text_code
            text_attn_mask = (text_token_id != -1).float()
            # remove -1 because it can not be recognized by the model
            text_token_id = text_token_id.masked_fill(text_token_id == -1, 0)

        elif text is not None:
            # the original text_token_id does not start with 0, we append it
            text_token_id = text.input_ids
            text_attn_mask = text.attention_mask
            pad_token_id = torch.zeros((*text_token_id.shape[:-1], 1), dtype=text_token_id.dtype, device=text_token_id.device)
            text_token_id = torch.cat([pad_token_id, text_token_id], dim=-1)
            text_attn_mask = torch.cat([torch.ones_like(pad_token_id), text_attn_mask], dim=-1)

        else:
            raise ValueError(f"Must provide either text or text_codes!")

        return text_token_id, text_attn_mask


    def _compute_logits(self, text_token_id, **kwargs):
        """
        Wrapped method to compute each token's relevance score.

        Returns:
            token_score: tensor of [B, L]
            logits: tensor of [B, L, V]
        """
        outputs = self.plm(decoder_input_ids=text_token_id, output_hidden_states=True, **kwargs) # *, L, V
        token_embedding = outputs.decoder_hidden_states[-1]    # *, L, D
        logits = outputs.logits
        # target_token_id = text_token_id[:, 1:]
        # token_score = logits.gather(index=target_token_id.unsqueeze(-1), dim=-1).squeeze(-1) # *, L - 1 
        return logits, token_embedding


    def forward(self, x):
        x = self._move_to_device(x)
        encoder_outputs = self.plm.encoder(**x["query"])

        # start with 0
        text_token_id, text_attn_mask = self._prepare_decoder_inputs(x["text"], x.get("text_code"))

        if text_token_id.dim() == 3:
            text_token_id = text_token_id.flatten(0, 1) # B*N, L
            text_attn_mask = text_attn_mask.flatten(0, 1)

        B = x["query"]["input_ids"].shape[0]
        M = text_token_id.shape[0] // B

        query_attn_mask = x["query"]["attention_mask"]
        if M > 1:
            # repeat query encode outputs to batchify
            for k, v in encoder_outputs.items():
                encoder_outputs[k] = v.repeat_interleave(M, 0)
            query_attn_mask = query_attn_mask.repeat_interleave(M, 0)

        # important to add attention mask to properly read from encoder_outputs
        logits, token_embedding = self._compute_logits(text_token_id, encoder_outputs=encoder_outputs, attention_mask=query_attn_mask)

        loss = 0

        if "gen" in self.config.train_scheme:
            logits = logits.unflatten(0, (B, M))[:, 0]    # B, L, V
            pos_text_token_id = text_token_id.unflatten(0, (B, M))[:, 0]  # B, L
            pos_text_attn_mask = text_attn_mask.unflatten(0, (B, M))[:, 0]    # B, L

            labels = torch.zeros_like(pos_text_token_id)    # B, L
            labels_mask = torch.zeros_like(pos_text_attn_mask)
            # shift left
            labels[:, :-1] = pos_text_token_id[:, 1:]    # B, L
            labels_mask[:, :-1] = pos_text_attn_mask[:, 1:]     # B, L
            # the pad token will be ignored in computing loss
            labels = labels.masked_fill(~labels_mask.bool(), -100)
            loss += F.cross_entropy(logits.flatten(0, 1), labels.view(-1), ignore_index=-100)

        if "contra" in self.config.train_scheme:
            valid_token_length = text_attn_mask.sum(dim=-1).long() - 1
            eos_embedding = token_embedding[range(valid_token_length.shape[0]), valid_token_length]
            score = self.scorer(eos_embedding).squeeze(-1)
            # cross entropy
            label = torch.zeros(B, device=self.config.device, dtype=torch.long)
            score = score.view(B, M)
            loss += F.cross_entropy(score, label)

        return loss


    def rerank_step(self, x):
        x = self._move_to_device(x)

        text_token_id, text_attn_mask = self._prepare_decoder_inputs(x.get("text"), x.get("text_code"))

        _, token_embedding = self._compute_logits(text_token_id, **x["query"], encoder_outputs=x.get("encoder_outputs"))

        # always use eos token to rank
        valid_token_length = text_attn_mask.sum(dim=-1).long() - 1  # B
        eos_embedding = token_embedding[range(valid_token_length.shape[0]), valid_token_length]
        score = self.scorer(eos_embedding).squeeze(-1)
        return score


    @synchronize
    @torch.no_grad()
    def retrieve(self, loaders):
        from utils.index import BeamManagerConstantLength
        # Use the <eos> token's score as the sequences_score
        index = self.index(loaders).index
        loader_query = loaders["query"]

        retrieval_result = {}
        self.logger.info("searching...")

        start_idx = end_idx = 0
        # in case the query is parallel
        query_start_idx = loader_query.sampler.start

        if self.config.save_encode:
            if self.config.beam_trsd == 0:
                N = min(self.config.hits, self.config.nbeam)
            else:
                N = self.config.beam_trsd
            query_codes = np.full((len(loader_query.sampler), N, self.config.code_length), -1, dtype=np.int32)

        beam_manager = BeamManagerConstantLength()

        for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
            query = self._move_to_device(x["query"])
            encoder_outputs = self.plm.encoder(**query)
            B = query["input_ids"].shape[0]

            beam_manager.search(
                model=self.plm, 
                nbeam=self.config.nbeam, 
                threshold=self.config.beam_trsd, 
                trsd_start_len=self.config.trsd_start_len, 
                max_new_tokens=self.config.code_length - 1, 
                constrain_index=index, 
                **query, 
                encoder_outputs=encoder_outputs
            )
            beams = beam_manager.beams
            eos_hidden_states = beam_manager.eos_hidden_states

            # ranking by score
            if self.config.rank_type == "eos":
                if isinstance(eos_hidden_states, list):
                    eos_hidden_states = torch.cat(eos_hidden_states, dim=0)
                scores = self.scorer(eos_hidden_states).squeeze(-1).tolist() # B * trsd
            elif self.config.rank_type == "prob":            
                # ranking by generation prob
                scores = sum(beam_manager.seq_scores, [])
            else:
                raise NotImplementedError(f"Ranking type {self.config.ranking_type} is not implemented yet!")

            if self.config.save_encode:
                for batch_beam in beams:
                    query_codes[start_idx: end_idx, :len(batch_beam)] = batch_beam

            offset = 0
            for j, batch in enumerate(beams):
                res = defaultdict(list)
                for k, c in enumerate(batch):
                    if beam_manager.constrain_index_type == "trie":
                        ids = index[c]
                    elif beam_manager.constrain_index_type == "intersect":
                        # need to provide prev_text_indices
                        ids = index[c, beam_manager.prev_text_indices[j][k]]
                    for id in ids:
                        res[id].append(scores[offset + k])
                offset += len(batch)
                retrieval_result[j + start_idx + query_start_idx] = [(k, max(v)) for k, v in res.items()]

            end_idx = start_idx + B
            start_idx = end_idx
            if self.config.debug:
                if i > 2:
                    break
        
        if self.config.save_encode:
            self.save_to_mmp(
                os.path.join(self.retrieve_dir, "query_codes.mmp"),
                shape=(len(loader_query.dataset), *query_codes.shape[1:]),
                dtype=query_codes.dtype,
                loader=loader_query,
                obj=query_codes
            )


        return retrieval_result
