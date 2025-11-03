import random
import numpy as np
import json
import copy
import re
import torch

'''
miscellaneous module
'''

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def format_cfg(cfg):
    """Format experiment config for friendly display"""

    def list2str(cfg):
        for key, value in cfg.items():
            if isinstance(value, dict):
                cfg[key] = list2str(value)
            elif isinstance(value, list):
                if len(value) == 0 or isinstance(value[0], (int, float)):
                    cfg[key] = str(value)
                else:
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            value[i] = list2str(item)
                    cfg[key] = value
        return cfg

    cfg = list2str(copy.deepcopy(cfg))
    json_str = json.dumps(cfg, indent=2, ensure_ascii=False).split(r"\n")
    json_str = [re.sub(r"(\"|,$|\{|\}|\[$)", "", line) for line in json_str if line.strip() not in "{}[]"]
    cfg_str = r"\n".join([line.rstrip() for line in json_str if line.strip()])
    return cfg_str

def build_cls_instance(module, cfg):
    """Build instance for given cls"""
    cls = getattr(module, cfg['type'])
    # list indicates different lr for pretrained optical flow net
    if isinstance(cfg["kwargs"], list):
        return cls(cfg["kwargs"])
    return cls(**cfg['kwargs'])
