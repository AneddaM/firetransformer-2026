#!/usr/bin/env python3
import argparse
import pandas as pd
from fire_transformer.config import load_config
from fire_transformer.model import build_model
from fire_transformer.complexity import transformer_macs, parameter_bytes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", default=None)
    args=ap.parse_args()
    cfg=load_config(args.config)
    rows=[]
    for name in ["ft32","ft64","ft128"]:
        mcfg=cfg["models"][name]
        model=build_model(mcfg, n_features=20, gas_feature_index=3)
        params=sum(p.numel() for p in model.parameters())
        macs=transformer_macs(cfg["window"]+1,20,mcfg["d_model"],mcfg["d_ff"],mcfg["n_layers"])
        rows.append({
            "model":name,"parameters":params,"fp32_weight_kib":parameter_bytes(model,4)/1024,
            "theoretical_int8_weight_kib":parameter_bytes(model,1)/1024,
            "mac_per_inference":macs,"mmac_per_inference":macs/1e6,"approx_mflop":2*macs/1e6,
        })
    df=pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x:f"{x:.3f}"))
    if args.output:
        df.to_csv(args.output,index=False)

if __name__=="__main__": main()
