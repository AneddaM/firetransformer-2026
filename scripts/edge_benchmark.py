#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from fire_transformer.model import build_model
from fire_transformer.evaluation import benchmark_latency
from fire_transformer.utils import device_from_arg


def main():
    ap=argparse.ArgumentParser(description="Batch-1 edge latency benchmark")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--output", default=None)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    ckpt=torch.load(args.checkpoint,map_location="cpu")
    model=build_model(ckpt["model_cfg"], ckpt["n_features"], gas_feature_index=3)
    model.load_state_dict(ckpt["state_dict"])
    device=device_from_arg(args.device)
    sample=torch.zeros(1, int(ckpt.get("window",60)), int(ckpt["n_features"]), dtype=torch.float32)
    stats=benchmark_latency(model,sample,device,args.warmup,args.iterations)
    stats.update({"checkpoint":str(args.checkpoint),"model":ckpt.get("model_name"),"device":device,"threads":args.threads,"batch_size":1,"warmup":args.warmup,"iterations":args.iterations})
    print(json.dumps(stats,indent=2))
    if args.output: Path(args.output).write_text(json.dumps(stats,indent=2),encoding="utf-8")

if __name__=="__main__": main()
