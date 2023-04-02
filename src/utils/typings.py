import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Any, Callable, Union, Literal, Optional, Mapping, Iterable

DEVICE = Union[int,Literal["cpu"]]

RETRIEVAL_MAPPING = Union[dict[int, list[int]],dict[int, list[tuple[int,float]]]]
ID_MAPPING = dict[str, int]

DENSE_METRIC = Literal["ip", "cos", "l2"]
DATA_FORMAT = Literal["memmap", "raw"]

TENSOR = torch.Tensor
LOADERS = dict[str,DataLoader]
NN_MODULE = torch.nn.Module
INDICES = Union[np.ndarray,list,torch.Tensor]
