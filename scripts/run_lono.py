#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from fire_transformer.config import load_config
from fire_transformer.data import DatasetCatalog, build_fold, to_loader
from fire_transformer.model import build_model
from fire_transformer.training import train_model
from fire_transformer.evaluation import collect_probabilities, choose_threshold, compute_metrics
from fire_transformer.utils import seed_everything, ensure_dir, device_from_arg


def main():
    ap = argparse.ArgumentParser(description="Full five-fold leakage-controlled leave-one-node-out evaluation")
    ap.add_argument("--data", required=True, help="Root containing CSV files under NODO1...NODO5 paths")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="+", default=["ft64"], choices=["ft32","ft64","ft128","bilstm_bce","bilstm_fl"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1,2,3])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="runs/lono")
    ap.add_argument("--heldout", nargs="*", default=None, help="Optional subset, e.g. NODO5")
    ap.add_argument("--max-epochs", type=int, default=None, help="Override config for quick tests")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    device = device_from_arg(args.device)
    outdir = ensure_dir(args.output)
    catalog = DatasetCatalog(args.data, cfg["schema"])
    expected = [f"NODO{i}" for i in range(1,6)]
    missing = sorted(set(expected) - set(catalog.nodes))
    if missing:
        raise ValueError(f"Full LONO requires NODO1...NODO5. Missing: {missing}; found: {catalog.nodes}")
    heldouts = args.heldout or expected
    rows = []

    for model_name in args.models:
        model_cfg = cfg["models"][model_name]
        for outer_index, heldout in enumerate(heldouts):
            train_b, val_b, test_b, scaler, feature_names, train_a, val_a, test_a = build_fold(
                catalog, heldout,
                rolling_window=cfg["rolling_window"], window=cfg["window"], horizon=cfg["horizon"],
                val_fraction_files=cfg["val_fraction_files"], split_seed=1000 + outer_index,
            )
            n_features = train_b.X.shape[-1]
            # Gas-resistance raw feature is canonical index 3 before rolling features.
            gas_idx = 3
            positive = float((train_b.y == 1).sum())
            negative = float((train_b.y == 0).sum())
            pos_weight = negative / max(1.0, positive)

            for seed in args.seeds:
                seed_everything(seed)
                train_loader = to_loader(train_b, cfg["training"]["batch_size"], True, cfg["training"]["num_workers"])
                val_loader = to_loader(val_b, cfg["training"]["batch_size"], False, cfg["training"]["num_workers"])
                test_loader = to_loader(test_b, cfg["training"]["batch_size"], False, cfg["training"]["num_workers"])
                model = build_model(model_cfg, n_features=n_features, gas_feature_index=gas_idx)
                model, history, best_val_auc = train_model(
                    model, train_loader, val_loader, model_cfg, cfg["training"], cfg["loss"],
                    device=device, positive_weight=pos_weight if model_name == "bilstm_bce" else None,
                    use_augmentation=True,
                )
                yv, pv = collect_probabilities(model, val_loader, device)
                threshold, threshold_score = choose_threshold(yv, pv, cfg["threshold_objective"])
                yt, pt = collect_probabilities(model, test_loader, device)
                metrics = compute_metrics(yt, pt, threshold, test_b.window_end_ts, test_b.onset_ts)
                metrics.update({
                    "model": model_name, "outer_test_node": heldout, "seed": seed,
                    "best_inner_val_auc": best_val_auc, "threshold_selection_score": threshold_score,
                    "n_train": len(train_b.y), "n_val": len(val_b.y), "n_test": len(test_b.y),
                    "train_files": len(train_a), "val_files": len(val_a), "test_files": len(test_a),
                    "horizon": cfg["horizon"], "window": cfg["window"],
                })
                rows.append(metrics)
                stem = f"{model_name}_{heldout}_seed{seed}"
                torch.save({
                    "state_dict": model.state_dict(), "model_name": model_name, "model_cfg": model_cfg,
                    "n_features": n_features, "feature_names": feature_names, "threshold": threshold,
                    "heldout_node": heldout, "seed": seed, "window": cfg["window"], "horizon": cfg["horizon"]
                }, outdir / f"{stem}.pth")
                pd.DataFrame(history).to_csv(outdir / f"{stem}_history.csv", index=False)
                print(json.dumps({k: metrics[k] for k in ["model","outer_test_node","seed","auc_roc","f1","recall","coverage","lead_mean_s"]}, indent=2))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "lono_runs.csv", index=False)
    metric_cols = [c for c in ["precision","recall","f1","auc_roc","auc_pr","coverage","lead_mean_s","false_alarm_rate","missed_detection_rate"] if c in df]
    summary = df.groupby("model")[metric_cols].agg(["mean","std"])
    summary.to_csv(outdir / "lono_macro_summary.csv")
    node_summary = df.groupby(["model","outer_test_node"])[metric_cols].agg(["mean","std"])
    node_summary.to_csv(outdir / "lono_per_node_summary.csv")
    print(f"\nSaved: {outdir / 'lono_runs.csv'}")
    print(f"Saved: {outdir / 'lono_macro_summary.csv'}")
    print(f"Saved: {outdir / 'lono_per_node_summary.csv'}")


if __name__ == "__main__":
    main()
