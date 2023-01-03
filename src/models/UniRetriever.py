import os
from collections import defaultdict
from .BaseModel import BaseModel
from utils.util import load_pickle


class UniRetriever(BaseModel):
    def __init__(self, config):
        from .AutoModel import AutoModel as AM
        super().__init__(config)

        if config.x_model != "none":
            XModel = AM.from_pretrained(os.path.join(config.cache_root, "ckpts", config.x_model, config.x_load_ckpt), device=config.get("x_device", config.device))
            # set load_encode, load_posting etc.
            for k,v in config.items():
                if k.startswith("x_") and k != "x_model":
                    setattr(XModel.config, k[2:], v)
            XModel.config.verifier_type = config.verifier_type
            XModel.config.verifier_src = config.verifier_src
            XModel.config.verifier_index = config.verifier_index
        else:
            XModel = None

        if config.y_model != "none":
            YModel = AM.from_pretrained(os.path.join(config.cache_root, "ckpts", config.y_model, config.y_load_ckpt), device=config.get("y_device", config.device))
            for k,v in config.items():
                if k.startswith("y_") and k != "y_model":
                    setattr(YModel.config, k[2:], v)
            YModel.config.verifier_type = config.verifier_type
            YModel.config.verifier_src = config.verifier_src
            YModel.config.verifier_index = config.verifier_index
        else:
            YModel = None

        self.XModel = XModel
        self.YModel = YModel


    def retrieve(self, loaders):
        """ retrieve by index

        Args:
            encode_query: if true, compute query embedding before retrieving
        """
        if self.XModel is not None:
            x_retrieval_result = self.XModel.retrieve(loaders)
            self.metrics.update({f"X {k}": v for k, v in self.XModel.metrics.items() if k in ["Posting_List_Length"]})
        else:
            x_retrieval_result = {}

        if self.YModel is not None:
            y_retrieval_result = self.YModel.retrieve(loaders)
            self.metrics.update({f"Y {k}": v for k, v in self.YModel.metrics.items() if k in ["Posting_List_Length"]})
        else:
            y_retrieval_result = {}

        if self.config.get("save_intm_result"):
            self.XModel._gather_retrieval_result(
                x_retrieval_result,
                retrieval_result_path=os.path.join(self.retrieve_dir, "x_retrieval_result.pkl")
            )
            self.YModel._gather_retrieval_result(
                y_retrieval_result,
                retrieval_result_path=os.path.join(self.retrieve_dir, "y_retrieval_result.pkl")
            )

        loader_query = loaders["query"]
        retrieval_result = {}
        for qidx in range(loader_query.sampler.start, loader_query.sampler.end):
            res = dict(x_retrieval_result.get(qidx, []))
            res.update(dict(y_retrieval_result.get(qidx, [])))
            sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)[:self.config.hits]
            retrieval_result[qidx] = sorted_res

        return retrieval_result
