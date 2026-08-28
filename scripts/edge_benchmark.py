#!/usr/bin/env python3
import argparse,time,json
import numpy as np, torch
from fire_transformer.config import load_config
from fire_transformer.model import build_model

ap=argparse.ArgumentParser(); ap.add_argument('--model',default='ft64',choices=['ft32','ft64','ft128']); ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--checkpoint'); ap.add_argument('--warmup',type=int,default=1000); ap.add_argument('--runs',type=int,default=5000); ap.add_argument('--threads',type=int,default=4); args=ap.parse_args()
torch.set_num_threads(args.threads); cfg=load_config(args.config); model=build_model(cfg['models'][args.model],n_features=20,gas_feature_index=3).eval()
if args.checkpoint:
    obj=torch.load(args.checkpoint,map_location='cpu'); model.load_state_dict(obj['state_dict'] if isinstance(obj,dict) and 'state_dict' in obj else obj)
x=torch.zeros(1,60,20)
with torch.inference_mode():
    for _ in range(args.warmup): model(x)
    times=[]
    for _ in range(args.runs):
        t0=time.perf_counter_ns(); model(x); t1=time.perf_counter_ns(); times.append((t1-t0)/1e6)
print(json.dumps({'model':args.model,'runs':args.runs,'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'threads':args.threads},indent=2))
