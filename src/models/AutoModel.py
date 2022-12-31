import os
import torch
from .BaseModel import BaseModel

from .AR2 import AR2
from .BM25 import BM25
from .COIL import COIL, UniCOIL
from .ColBERT import ColBERT
from .DSI import DSI
from .DeepImpact import DeepImpact
from .DPR import DPR
from .IVF import IVF, TopIVF, TokIVF
from .SPARTA import SPARTA
from .SPLADE import SPLADEv2
from .Ranker import CrossEncoder, Seq2SeqRanker
from .UniRetriever import UniRetriever
from .VQ import DistillVQ

MODEL_MAP = {
    "bm25": BM25,
    "coil": COIL,
    "colbert": ColBERT,
    "crossenc": CrossEncoder,
    "deepimpact": DeepImpact,
    "distillvq": DistillVQ,
    "dpr": DPR,
    "dsi": DSI,
    "ivf": IVF,
    "sparta": SPARTA,
    "spladev2": SPLADEv2,
    "topivf": TopIVF,
    "tokivf": TokIVF,
    "unicoil": UniCOIL,
    "uniretriever": UniRetriever
}


class AutoModel(BaseModel):
    @classmethod
    def from_pretrained(cls, ckpt_path, device="cpu"):
        state_dict = torch.load(ckpt_path, map_location=torch.device(device))
        config = state_dict["config"]

        model_name_current = os.path.abspath(ckpt_path).split(os.sep)[-2]
        model_name_ckpt = config.name

        # override model name
        config.name = model_name_current

        model_type = model_name_current.split("_")[0].lower()
        config.device = device

        try:
            model = MODEL_MAP[model_type](config).to(config.device)
        except KeyError:
            raise NotImplementedError(f"Model {model_type} not implemented!")
        if model_name_ckpt != model_name_current:
            model.logger.warning(f"model name in the checkpoint is {model_name_ckpt}, while it's {model_name_current} now!")

        model.logger.info(f"loading model from {ckpt_path} with checkpoint config...")
        model.load_state_dict(state_dict["model"])
        model.metrics = state_dict["metrics"]

        model.eval()
        return model
