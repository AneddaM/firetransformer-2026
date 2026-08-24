#!/usr/bin/env python3
"""Run continuous batch-1 inference while an external power meter logs system power.

The JSON output contains wall-clock start/end timestamps so the matching interval can
be selected from the meter log without estimating energy from PSU ratings.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from fire_transformer.model import build_model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--seconds",type=float,default=60.0)
    ap.add_argument("--threads",type=int,default=1)
    ap.add_argument("--output",default="power_benchmark_interval.json")
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    ckpt=torch.load(args.checkpoint,map_location="cpu")
    model=build_model(ckpt["model_cfg"],ckpt["n_features"],gas_feature_index=3).eval()
    model.load_state_dict(ckpt["state_dict"])
    x=torch.zeros(1,int(ckpt.get("window",60)),int(ckpt["n_features"]))
    with torch.inference_mode():
        for _ in range(500): model(x)
        start_epoch=time.time(); start_perf=time.perf_counter(); lat=[]; n=0
        while time.perf_counter()-start_perf < args.seconds:
            t=time.perf_counter_ns(); model(x); lat.append((time.perf_counter_ns()-t)/1e6); n+=1
        end_perf=time.perf_counter(); end_epoch=time.time()
    a=np.asarray(lat)
    result={"start_epoch_s":start_epoch,"end_epoch_s":end_epoch,"duration_s":end_perf-start_perf,"inferences":n,"latency_mean_ms":float(a.mean()),"latency_median_ms":float(np.median(a)),"latency_p95_ms":float(np.percentile(a,95)),"threads":args.threads,"batch_size":1}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
