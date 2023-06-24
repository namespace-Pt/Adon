import sys
import hydra
from omegaconf import OmegaConf
from utils.util import Config
from utils.data import prepare_data
from models.AutoModel import MODEL_MAP
name = None

@hydra.main(version_base=None, config_path="data/config/")
def get_config(hydra_config: OmegaConf):
    config._from_hydra(hydra_config)
    config.name = name


def main(config:Config):
    """ train/dev/test the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    loaders = prepare_data(config)
    model = MODEL_MAP[config.model_type](config).to(config.device)

    if config.mode == "train":
        from utils.trainer import train
        model.load()
        train(model, loaders)

    elif config.mode == "eval":
        model.load()
        model.evaluate(loaders)

    elif config.mode == "encode":
        model.load()
        model.encode(loaders)

    elif config.mode == "cluster":
        model.load()
        model.cluster(loaders)

    elif config.mode == "code":
        model.load()
        model.generate_code(loaders)

    elif config.mode == "migrate":
        from utils.util import load_from_previous
        if config.is_main_proc:
            path = f"{config.cache_root}/ckpts/{model.name}/{config.load_ckpt}"
            load_from_previous(model, path)
            model.save()

    elif config.mode == "deploy":
        model.load()
        model.deploy()

    elif config.mode == "index":
        model.load()
        model.index(loaders)

    else:
        raise ValueError(f"Invalid mode {config.mode}!")

    if config.save_model:
        model.save()



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

    config = Config()
    get_config()

    main(config)
