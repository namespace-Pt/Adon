import torch
import torch.nn as nn
from .COIL import COIL



class UniCOIL(COIL):
    def __init__(self, config):
        config.token_dim = 1
        super().__init__(config)

        plm_dim = self.textEncoder.config.hidden_size
        self.tokenProject = nn.Sequential(
            nn.Linear(plm_dim, plm_dim),
            nn.ReLU(),
            nn.Linear(plm_dim, self._output_dim),
            nn.ReLU()
        )

        self.special_token_ids = [x[1] for x in config.special_token_ids.values() if x[0] is not None]


    def _to_bow(self, token_ids, token_weights):
        """
        Convert the token sequence (maybe repetitive tokens) into BOW (no repetitive tokens except pad token)

        Args:
            token_ids: tensor of B, L
            token_weights: tensor of B, L, 1

        Returns:
            bow representation of B, V
        """
        # create the src
        dest = torch.zeros((*token_ids.shape, self.config.vocab_size), device=token_ids.device) - 1   # B, L, V
        bow = torch.scatter(dest, dim=-1, index=token_ids.unsqueeze(-1), src=token_weights)
        bow = bow.max(dim=1)[0]    # B, V
        # only pad token and the tokens with positive weights are valid
        bow[:, self.special_token_ids] = 0
        return bow


    def encode_text_step(self, x):
        text = self._move_to_device(x["text"])
        text_token_id = text["input_ids"]

        text_token_embedding = self._encode_text(**text)
        text_bow = self._to_bow(text_token_id, text_token_embedding)

        text_token_weight = text_bow.gather(index=text_token_id, dim=-1)

        if "text_first_mask" in x:
            # mask the duplicated tokens' weight
            text_first_mask = self._move_to_device(x["text_first_mask"])
            # mask duplicated tokens' id
            text_token_id = text_token_id.masked_fill(~text_first_mask, 0)
            text_token_weight = text_token_weight.masked_fill(~text_first_mask, 0)
        # unsqueeze to map it to the _output_dim (1)
        return text_token_id.cpu().numpy(), text_token_weight.unsqueeze(-1).cpu().numpy()


    def encode_query_step(self, x):
        query = self._move_to_device(x["query"])
        query_token_id = query["input_ids"]

        query_token_weight = self._encode_query(**query)
        query_token_weight *= query["attention_mask"].unsqueeze(-1)

        # unsqueeze to map it to the _output_dim (1)
        return query_token_id.cpu().numpy(), query_token_weight.cpu().numpy()
