#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fire_transformer.config import load_config
from fire_transformer.data import DatasetCatalog, build_fold, to_loader
from fire_transformer.evaluation import (
    choose_threshold,
    collect_probabilities,
    compute_metrics,
)
from fire_transformer.model import build_model
from fire_transformer.training import train_model
from fire_transformer.utils import device_from_arg, ensure_dir, seed_everything


def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity analysis over future-warning horizon H"
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--model",
        default="ft64",
        choices=["ft64", "ft128"],
    )
    ap.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[5, 15, 30, 60],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="runs/horizon")
    ap.add_argument(
        "--heldout",
        nargs="*",
        default=None,
        help="Default: all five nodes",
    )
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs

    device = device_from_arg(args.device)
    catalog = DatasetCatalog(args.data, cfg["schema"])

    expected_nodes = [f"NODO{i}" for i in range(1, 6)]
    missing = sorted(set(expected_nodes) - set(catalog.nodes))
    if missing:
        raise ValueError(
            "Full horizon LONO requires NODO1...NODO5. "
            f"Missing: {missing}; found: {catalog.nodes}"
        )

    heldouts = args.heldout or expected_nodes
    invalid = sorted(set(heldouts) - set(expected_nodes))
    if invalid:
        raise ValueError(f"Invalid held-out node(s): {invalid}")

    outdir = ensure_dir(args.output)
    rows = []

    for horizon_value in args.horizons:
        for outer_index, heldout in enumerate(heldouts):
            (
                train_b,
                val_b,
                test_b,
                _,
                _,
                _,
                _,
                _,
            ) = build_fold(
                catalog,
                heldout,
                cfg["rolling_window"],
                cfg["window"],
                horizon_value,
                cfg["val_fraction_files"],
                split_seed=1000 + outer_index,
            )

            for seed in args.seeds:
                seed_everything(seed)

                model_cfg = cfg["models"][args.model]
                model = build_model(
                    model_cfg,
                    train_b.X.shape[-1],
                    gas_feature_index=3,
                )

                tr = to_loader(
                    train_b,
                    cfg["training"]["batch_size"],
                    True,
                    cfg["training"]["num_workers"],
                )
                va = to_loader(
                    val_b,
                    cfg["training"]["batch_size"],
                    False,
                    cfg["training"]["num_workers"],
                )
                te = to_loader(
                    test_b,
                    cfg["training"]["batch_size"],
                    False,
                    cfg["training"]["num_workers"],
                )

                model, _, best_val_auc = train_model(
                    model,
                    tr,
                    va,
                    model_cfg,
                    cfg["training"],
                    cfg["loss"],
                    device=device,
                )

                y_val, p_val = collect_probabilities(model, va, device)
                threshold, _ = choose_threshold(
                    y_val,
                    p_val,
                    cfg["threshold_objective"],
                )

                y_test, p_test = collect_probabilities(model, te, device)
                metrics = compute_metrics(
                    y_test,
                    p_test,
                    threshold,
                    test_b.window_end_ts,
                    test_b.onset_ts,
                    file_ids=test_b.file_ids,
                )

                metrics.update(
                    {
                        "model": args.model,
                        "horizon": horizon_value,
                        "outer_test_node": heldout,
                        "seed": seed,
                        "best_inner_val_auc": best_val_auc,
                    }
                )
                rows.append(metrics)

                print(
                    f"H={horizon_value} {heldout} seed={seed} "
                    f"AUC={metrics['auc_roc']:.4f} "
                    f"F1={metrics['f1']:.4f} "
                    f"coverage={metrics.get('coverage', np.nan):.4f} "
                    f"lead={metrics.get('lead_mean_s', np.nan):.3f}s"
                )

    if not rows:
        raise RuntimeError("No horizon runs were executed.")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "horizon_runs.csv", index=False)

    candidate_metrics = [
        "auc_roc",
        "f1",
        "precision",
        "recall",
        "coverage",
        "lead_mean_s",
        "false_alarm_rate",
        "missed_detection_rate",
    ]
    metrics = [c for c in candidate_metrics if c in df.columns]

    horizon_node_means = (
        df.groupby(
            ["model", "horizon", "outer_test_node"],
            as_index=False,
        )[metrics]
        .mean()
    )
    horizon_node_means.to_csv(
        outdir / "horizon_per_node_means.csv",
        index=False,
    )

    horizon_summary = (
        horizon_node_means.groupby(["model", "horizon"])[metrics]
        .agg(["mean", "std"])
    )
    horizon_summary.to_csv(outdir / "horizon_summary.csv")

    horizon_seed_std = (
        df.groupby(["model", "horizon", "outer_test_node"])[metrics]
        .std()
        .reset_index()
    )
    horizon_seed_std.to_csv(
        outdir / "horizon_seed_std_by_node.csv",
        index=False,
    )

    print(f"Saved results to {outdir}")


if __name__ == "__main__":
    main()
