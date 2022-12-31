from .DPR import DPR
from utils.util import load_pickle


class AR2(DPR):
    def __init__(self, config):
        super().__init__(config)


    def forward(self, x):
        raise NotImplementedError("AR2 training not implemented!")
