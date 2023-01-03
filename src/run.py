import sys
import hydra
import subprocess
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from utils.manager import Manager
from models.AutoModel import MODEL_MAP

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

name = None

@hydra.main(version_base=None, config_path="data/config/")
def get_config(config: OmegaConf):
    manager.setup(config)
    manager.config.name = name
    manager.logger.info(f"using model {name}...")


def main(manager:Manager):
    """ train/dev/test the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    model = MODEL_MAP[manager.config.model_type](manager.config).to(manager.config.device)

    loaders = manager.prepare()

    if manager.config.mode == "train":
        model.load()
        manager.train(model, loaders)

    elif manager.config.mode == "eval":
        model.load()
        model.evaluate(loaders)

    elif manager.config.mode == "encode":
        model.load()
        model.encode(loaders)

    elif manager.config.mode == "encode-text":
        model.load()
        model.encode_text(loaders["text"])

    elif manager.config.mode == "encode-query":
        model.load()
        model.encode_query(loaders["query"])

    elif manager.config.mode == "flops":
        model.load()
        text_output = model.encode_text(loaders["text"])
        query_output = model.encode_query(loaders["query"])
        model.compute_flops(loaders, text_output.token_ids, text_output.embeddings, query_output.token_ids, query_output.embeddings, log=True)

    elif manager.config.mode == "cluster":
        model.load()
        model.cluster(loaders)

    elif manager.config.mode == "code":
        model.load()
        model.generate_code(loaders)

    elif manager.config.mode == "migrate":
        from utils.util import load_from_previous
        if manager.config.is_main_proc:
            path = f"{manager.config.cache_root}/ckpts/{model.name}/{manager.config.load_ckpt}"
            load_from_previous(model, path)
            model.save()

    elif manager.config.mode == "deploy":
        model.load()
        model.deploy()

    elif manager.config.mode == "index":
        model.load()
        model.index(loaders)

    else:
        raise ValueError(f"Invalid mode {manager.config.mode}!")

    if manager.config.save_model:
        model.save()

    manager.cleanup()


if __name__ == "__main__":
    # get the model full name
    name = sys.argv.pop(1)
    # parse the config_name, which is the first part in the list split by _
    config_name = name.split("_")[0].lower()
    # add the parsed config_name back to the sys.argv so that hydra can use it
    sys.argv.insert(1, config_name)
    sys.argv.insert(1, "--config-name")

    # manually action="store_true" because hydra doesn't support it
    for i, arg in enumerate(sys.argv):
        if i > 2 and "=" not in arg:
            sys.argv[i] += "=true"

    manager = Manager()
    get_config()

    main(manager)
