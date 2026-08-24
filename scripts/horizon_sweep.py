#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from fire_transformer.config import load_config
from fire_transformer.data import DatasetCatalog, build_fold, to_loader
from fire_transformer.model import build_model
from fire_transformer.training import train_model
from fire_transformer.evaluation import collect_probabilities, choose_threshold, compute_metrics
from fire_transformer.utils import seed_everything, ensure_dir, device_from_arg


def main():
    ap = argparse.ArgumentParser(description="Sensitivity analysis over future-warning horizon H")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="ft64", choices=["ft32","ft64","ft128"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[5,15,30,60])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1,2,3])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="runs/horizon")
    ap.add_argument("--heldout", nargs="*", default=None, help="Default: all five nodes")
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    device = device_from_arg(args.device)
    catalog = DatasetCatalog(args.data, cfg["schema"])
    heldouts = args.heldout or [f"NODO{i}" for i in range(1,6)]
    outdir = ensure_dir(args.output)
    rows = []

    for horizon in args.horizons:
        for outer_index, heldout in enumerate(heldouts):
            train_b, val_b, test_b, _, feature_names, train_a, val_a, test_a = build_fold(
                catalog, heldout, cfg["rolling_window"], cfg["window"], horizon,
                cfg["val_fraction_files"], split_seed=1000 + outer_index,
            )
            for seed in args.seeds:
                seed_everything(seed)
                model_cfg = cfg["models"][args.model]
                model = build_model(model_cfg, train_b.X.shape[-1], gas_feature_index=3)
                tr = to_loader(train_b, cfg["training"]["batch_size"], True, cfg["training"]["num_workers"])
                va = to_loader(val_b, cfg["training"]["batch_size"], False, cfg["training"]["num_workers"])
                te = to_loader(test_b, cfg["training"]["batch_size"], False, cfg["training"]["num_workers"])
                model, _, best_val_auc = train_model(model, tr, va, model_cfg, cfg["training"], cfg["loss"], device=device)
                yv, pv = collect_probabilities(model, va, device)
                threshold, _ = choose_threshold(yv, pv, cfg["threshold_objective"])
                yt, pt = collect_probabilities(model, te, device)
                m = compute_metrics(yt, pt, threshold, test_b.window_end_ts, test_b.onset_ts)
                m.update({"model": args.model, "horizon": horizon, "outer_test_node": heldout, "seed": seed, "best_inner_val_auc": best_val_auc})
                rows.append(m)
                print(f"H={horizon} {heldout} seed={seed} AUC={m['auc_roc']:.4f} F1={m['f1']:.4f} lead={m.get('lead_mean_s', np.nan):.3f}s")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "horizon_runs.csv", index=False)
    metrics = ["auc_roc","f1","precision","recall","coverage","lead_mean_s","false_alarm_rate","missed_detection_rate"]
    df.groupby(["model","horizon"])[metrics].agg(["mean","std"]).to_csv(outdir / "horizon_summary.csv")
    print(f"Saved results to {outdir}")


if __name__ == "__main__":
    main()
