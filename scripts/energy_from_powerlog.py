#!/usr/bin/env python3
import argparse, pandas as pd
ap=argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="CSV with a power_w column")
ap.add_argument("--latency-ms", type=float, required=True)
ap.add_argument("--idle-w", type=float, required=True)
args=ap.parse_args()
df=pd.read_csv(args.csv)
p=float(df["power_w"].mean())
t=args.latency_ms/1000.0
print(f"active_power_W={p:.6f}")
print(f"system_energy_mJ={p*t*1000:.6f}")
print(f"dynamic_energy_mJ={(p-args.idle_w)*t*1000:.6f}")
