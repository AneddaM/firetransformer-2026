#!/usr/bin/env python3
"""Create a tiny five-node dataset for pipeline smoke testing only."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="data/synthetic"); ap.add_argument("--samples",type=int,default=180); args=ap.parse_args()
    root=Path(args.output); rng=np.random.default_rng(7)
    for node in range(1,6):
        d=root/f"NODO{node}"; d.mkdir(parents=True,exist_ok=True)
        for f in range(4):
            n=args.samples; t=np.arange(n,dtype=float)
            onset=110 + (node+f)%20
            fire=(t>=onset).astype(int)
            temp=24+0.01*t+1.5*fire+rng.normal(0,.08,n)
            hum=55-0.015*t-2.0*fire+rng.normal(0,.15,n)
            press=1012+rng.normal(0,.2,n)
            gas=120000-30*t-15000*fire+rng.normal(0,800,n)
            pd.DataFrame({"timestamp":t,"temperature":temp,"humidity":hum,"pressure":press,"gas_resistance":gas,"fire":fire}).to_csv(d/f"acq_{f+1}.csv",index=False)
    print(root)
if __name__=="__main__": main()
