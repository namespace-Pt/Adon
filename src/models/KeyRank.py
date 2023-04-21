import torch
from .BaseModel import BaseModel

class KeyRank(BaseModel):
    """
    Select keywords from the document for ranking, using REINFORCE policy gradient.
    """
    def __init__(self, config):
        super().__init__(config)

    
    def forward(self, x):
        pass
    

    def rerank_step(self, x):
        pass
