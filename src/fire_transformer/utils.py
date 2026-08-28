from pathlib import Path
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p


def device_from_arg(arg):
    if arg == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    return arg
