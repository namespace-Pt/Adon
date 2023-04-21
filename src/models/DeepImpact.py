import torch.nn as nn
from .UniCOIL import UniCOIL
from .BaseModel import BaseSparseModel



class DeepImpact(UniCOIL):
    def __init__(self, config):
        """
        `DeepImpact model <https://arxiv.org/abs/2104.12016>`_
        """
        super().__init__(config)


    def encode_query_step(self, x):
        """
        not contextualized
        """
        return BaseSparseModel.encode_query_step(self, x)
