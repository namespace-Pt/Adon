import torch
import torch.nn as nn
from .COIL import COIL



class UniCOIL(COIL):
    def __init__(self, config):
        config.token_dim = 1
        super().__init__(config)

        plm_dim = self.plm.config.hidden_size
        self.tokenProject = nn.Sequential(
            nn.Linear(plm_dim, plm_dim),
            nn.ReLU(),
            nn.Linear(plm_dim, self._output_dim),
            nn.ReLU()
        )

        self.special_token_ids = [x[1] for x in config.special_token_ids.values() if x[0] is not None]
        # critical since COIL is not BOW
        self._is_bow = True


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
        text_token_embedding = self._encode_text(**text)
        text_bow = self._to_bow(text["input_ids"], text_token_embedding)

        text_first_mask = self._move_to_device(x["text_first_mask"])
        # mask the duplicated tokens and special tokens
        valid_token_ids = text["input_ids"].masked_fill(~text_first_mask, 0)
        valid_token_weights = text_bow.gather(index=valid_token_ids, dim=-1).unsqueeze(-1)
        # unsqueeze to map it to the _output_dim (1)
        return valid_token_ids.cpu().numpy(), valid_token_weights.cpu().numpy()


    def encode_query_step(self, x):
        query = self._move_to_device(x["query"])
        query_token_embedding = self._encode_text(**query)
        query_bow = self._to_bow(query["input_ids"], query_token_embedding)

        query_first_mask = self._move_to_device(x["query_first_mask"])
        # mask the duplicated tokens and special tokens
        valid_token_ids = query["input_ids"].masked_fill(~query_first_mask, 0)
        valid_token_weights = query_bow.gather(index=valid_token_ids, dim=-1).unsqueeze(-1)
        # unsqueeze to map it to the _output_dim (1)
        return valid_token_ids.cpu().numpy(), valid_token_weights.cpu().numpy()
