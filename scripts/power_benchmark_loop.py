#!/usr/bin/env python3
"""Run repeated batch-1 CPU inference while an external meter records power."""
import argparse,time
import torch
from fire_transformer.config import load_config
from fire_transformer.model import build_model
ap=argparse.ArgumentParser(); ap.add_argument('--model',default='ft64',choices=['ft32','ft64','ft128']); ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--seconds',type=float,default=60.0); ap.add_argument('--threads',type=int,default=4); args=ap.parse_args()
torch.set_num_threads(args.threads); cfg=load_config(args.config); model=build_model(cfg['models'][args.model],20,3).eval(); x=torch.zeros(1,60,20); n=0; start=time.perf_counter()
with torch.inference_mode():
    for _ in range(500): model(x)
    while time.perf_counter()-start<args.seconds: model(x); n+=1
print(f'inferences={n} elapsed_s={time.perf_counter()-start:.6f}')
