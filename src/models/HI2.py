import os
from collections import defaultdict
from .BaseModel import BaseModel
from utils.util import load_pickle


class HI2(BaseModel):
    def __init__(self, config):
        from .AutoModel import AutoModel as AM
        super().__init__(config)

        if config.x_model != "none":
            additional_kwargs = {
                "data_root": config.data_root,
                "plm_root": config.plm_root,
                "text_col": config.text_col,
                "text_type": config.text_type,
                "device": config.get("x_device", config.device),
                "verifier_type": config.verifier_type,
                "verifier_src": config.verifier_src,
                "verifier_index": config.verifier_index,
                "verifier_hits": config.verifier_hits,
                "save_res": config.save_res,
            }
            for k,v in config.items():
                if k.startswith("x_") and k != "x_model":
                    additional_kwargs[k[2:]] = v

            XModel = AM.from_pretrained(os.path.join(config.cache_root, "ckpts", config.x_model, config.x_load_ckpt), **additional_kwargs)
        else:
            XModel = None

        if config.y_model != "none":
            additional_kwargs = {
                "data_root": config.data_root,
                "plm_root": config.plm_root,
                "text_col": config.text_col,
                "text_type": config.text_type,
                "device": config.get("y_device", config.device),
                "verifier_type": config.verifier_type,
                "verifier_src": config.verifier_src,
                "verifier_index": config.verifier_index,
                "verifier_hits": config.verifier_hits,
                "save_res": config.save_res,
            }
            for k,v in config.items():
                if k.startswith("y_") and k != "y_model":
                    additional_kwargs[k[2:]] = v

            YModel = AM.from_pretrained(os.path.join(config.cache_root, "ckpts", config.y_model, config.y_load_ckpt), **additional_kwargs)

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

        posting_length = 0
        if "X Posting_List_Length" in self.metrics:
            posting_length += self.metrics["X Posting_List_Length"]
        if "Y Posting_List_Length" in self.metrics:
            posting_length += self.metrics["Y Posting_List_Length"]            
        self.metrics.update({"Posting_List_Length": posting_length})

        if self.config.get("save_intm_res"):
            self.XModel.gather_retrieval_result(
                x_retrieval_result,
                retrieval_result_path=os.path.join(self.retrieve_dir, "x_retrieval_result.pkl")
            )
            self.YModel.gather_retrieval_result(
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
