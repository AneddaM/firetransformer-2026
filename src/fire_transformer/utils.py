import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def device_from_arg(name: str):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name
