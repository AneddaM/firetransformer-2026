#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description="Compute measured energy per inference from a power log")
    ap.add_argument("--csv",required=True,help="CSV containing timestamp and power columns")
    ap.add_argument("--interval-json",required=True,help="Output of power_benchmark_loop.py")
    ap.add_argument("--timestamp-col",default="timestamp_s")
    ap.add_argument("--power-col",default="power_w")
    ap.add_argument("--idle-w",type=float,default=None,help="Measured idle power; omit to report system energy only")
    ap.add_argument("--output",default=None)
    args=ap.parse_args()
    df=pd.read_csv(args.csv)
    interval=json.loads(Path(args.interval_json).read_text(encoding="utf-8"))
    t=pd.to_numeric(df[args.timestamp_col],errors="coerce").to_numpy(float)
    p=pd.to_numeric(df[args.power_col],errors="coerce").to_numpy(float)
    mask=np.isfinite(t)&np.isfinite(p)&(t>=interval["start_epoch_s"])&(t<=interval["end_epoch_s"])
    if mask.sum()<2: raise ValueError("Power log has fewer than two samples inside inference interval")
    tt,pp=t[mask],p[mask]
    order=np.argsort(tt); tt,pp=tt[order],pp[order]
    joules=float(np.trapz(pp,tt)); n=int(interval["inferences"])
    result={"samples":int(mask.sum()),"mean_active_power_w":float(np.mean(pp)),"total_system_energy_j":joules,"system_energy_per_inference_mj":1000*joules/n}
    if args.idle_w is not None:
        dynamic=max(0.0,joules-float(args.idle_w)*(tt[-1]-tt[0]))
        result.update({"measured_idle_power_w":float(args.idle_w),"total_dynamic_energy_j":dynamic,"dynamic_energy_per_inference_mj":1000*dynamic/n})
    print(json.dumps(result,indent=2))
    if args.output: Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")

if __name__=="__main__": main()
