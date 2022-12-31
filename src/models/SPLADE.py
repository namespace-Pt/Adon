import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from .BaseModel import BaseSparseModel



class SPLADEv2(BaseSparseModel):
    def __init__(self, config):
        super().__init__(config)

        self.plm = AutoModelForMaskedLM.from_pretrained(config.plm_dir)

        if self.config.mode == "train":
            self._step = 0
            self._lambda_warmup_step = config.lambda_warmup_step

        self._rebuild_index = True
        self._skip_special_tokens = False
        self._text_length = self.config.text_decode_k
        self._query_length = self.config.query_decode_k


    def _encode(self, **kwargs):
        for k, v in kwargs.items():
            # B, 1+N, L -> B * (1+N), L
            if v.dim() == 3:
                kwargs[k] = v.view(-1, v.shape[-1])

        token_all_embedding = self.plm(**kwargs, return_dict=True).logits
        token_all_embedding = torch.log(F.relu(token_all_embedding) + 1) * kwargs["attention_mask"].unsqueeze(-1)    # B, L, V
        token_embedding = token_all_embedding.max(dim=1)[0]  # B, V
        return token_embedding


    def _compute_flops(self, embedding):
        return torch.sum(torch.mean(torch.abs(embedding), dim=0) ** 2)


    def _refresh_lambda(self):
        if self._step <= self._lambda_warmup_step:
            self._text_lambda = self.config.text_lambda * (self._step / self._lambda_warmup_step) ** 2
            self._query_lambda = self.config.query_lambda * (self._step / self._lambda_warmup_step) ** 2
        else:
            self._text_lambda = self.config.text_lambda
            self._query_lambda = self.config.query_lambda
        self._step += 1


    def forward(self, x):
        if self.training:
            self._refresh_lambda()

        x = self._move_to_device(x)

        query_embedding = self._encode(**x["query"])	# B, V
        text_embedding = self._encode(**x["text"])	# B*(1+N), V

        if self.config.is_distributed and self.config.enable_all_gather:
            query_embedding = self._gather_tensors(query_embedding)
            text_embedding = self._gather_tensors(text_embedding)

        score = query_embedding.matmul(text_embedding.transpose(-1,-2))	# B, B*(1+N)

        B = query_embedding.size(0)
        # in batch negative
        if self.config.enable_inbatch_negative:
            label = torch.arange(B, device=self.config.device)
            label = label * (text_embedding.shape[0] // query_embedding.shape[0])
        else:
            label = torch.zeros(B, dtype=torch.long, device=self.config.device)
            score = score.view(B, B, -1)[range(B), range(B)]    # B, 1+N

        query_flops_loss = self._compute_flops(query_embedding) * self._query_lambda
        text_flops_loss = self._compute_flops(text_embedding) * self._text_lambda
        flops_loss = query_flops_loss + text_flops_loss

        loss = self._compute_loss(score, label, self._compute_teacher_score(x)) + flops_loss

        return loss


    def encode_text_step(self, x):
        text = self._move_to_device(x["text"])
        text_embedding = self._encode(**text)

        text_token_weight, text_token_id = text_embedding.topk(k=self._text_length, dim=1)  # B, K

        # unsqueeze to map it to the _output_dim (1)
        return text_token_id.cpu().numpy(), text_token_weight.unsqueeze(-1).cpu().numpy()


    def encode_query_step(self, x):
        query = self._move_to_device(x["query"])
        query_embedding = self._encode(**query)

        query_token_weight, query_token_id = query_embedding.topk(k=self._query_length, dim=1)  # B, K

        # unsqueeze to map it to the _output_dim (1)
        return query_token_id.cpu().numpy(), query_token_weight.unsqueeze(-1).cpu().numpy()
