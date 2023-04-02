# very essential that we import faiss before torch on zhiyuan machine
import faiss
import torch

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s (%(name)s) %(message)s")
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)

import transformers
# prevent warning of transformers
transformers.logging.set_verbosity_error()

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
