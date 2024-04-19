import torch
import numpy as np
import random
import os

import omegaconf
from omegaconf import OmegaConf
import logging

def load_config() -> omegaconf.DictConfig:
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load(cli_cfg.default_config)
    if 'MODEL_path' in cli_cfg:
        cfg['MODEL'] = OmegaConf.load(cli_cfg.MODEL_path)
    
    if 'DATA_path' in cli_cfg:
        cfg['DATA'] = OmegaConf.load(cli_cfg.DATA_path)

    for key, value in cli_cfg.items():
        if key in cfg:
            cfg[key] = value

    return cfg

def torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_logger(cfg : omegaconf.DictConfig) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    logging.basicConfig(
        filename = os.path.join(cfg.log_dir, 'train.log'),
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    return logger

def init_logger(logger, cfg):
    logger.info(">> Configurations")
    for key, value in cfg.items():
        if type(value) == omegaconf.dictconfig.DictConfig:
            logger.info(f"## {key} ##")
            for k, v in value.items():
                logger.info(f"{k} : {v}")
        else :
            logger.info(f"{key} : {value}")
    logger.info("=====================================")
    logger.info(">> Start training")