
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import getpass
import logging
import os
import pprint
import random
import socket
import time
import warnings
from typing import Tuple

import torch
import torch.optim
import torch.utils.data
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)
logger.setLevel(level=logging.DEBUG)
logging.Formatter.converter = time.gmtime


def setup(cfg: DictConfig) -> Tuple[torch.device, torch.dtype]:
    logger.info(f"getpass.getuser() = {getpass.getuser()}")
    logger.info(f"socket.gethostname() = {socket.gethostname()}")
    logger.info(f"os.getcwd() = {os.getcwd()}")
    logger.info(f"cfg.prng.seed = {cfg.prng.seed}")
    logger.info(pprint.pformat(dict(cfg)))

    torch.manual_seed(cfg.prng.seed)
    torch.cuda.manual_seed_all(cfg.prng.seed)
    random.seed(cfg.prng.seed)

    cuda_wanted = cfg.hardware.cuda_wanted
    cuda_available = torch.cuda.is_available()
    cuda_used = cuda_wanted and cuda_available
    device = torch.device("cuda") if cuda_used else torch.device("cpu")
    if cfg.hardware.benchmark:
        torch.backends.cudnn.benchmark = True

    if "float32" == cfg.torch.dtype:
        dtype = torch.float32
    elif "float16" == cfg.torch.dtype:
        dtype = torch.float16
    elif "bfloat16" == cfg.torch.dtype:
        dtype = torch.bfloat16
    else:
        raise ValueError(f"dtype = {cfg.torch.dtype} is not configured")

    if cfg.torch.deterministic:
        torch.use_deterministic_algorithms(True)
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if cfg.torch.anomaly_detection:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.autograd.detect_anomaly()

    if cuda_used:
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        logger.info(f"Running on {current_device_properties}")
        logger.info(f"torch.version.cuda = {torch.version.cuda}")

    for path in cfg.paths.items():
        # print(path)
        if path[1]:
            os.makedirs(path[1], exist_ok=True)
    return device, dtype
