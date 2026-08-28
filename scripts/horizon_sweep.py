#!/usr/bin/env python3
"""Prediction-horizon sensitivity under grouped acquisition-level CV."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fire_transformer.config import load_config
from fire_transformer.data import DatasetCatalog, build_fold, outer_fold_manifest, to_loader
from fire_transformer.evaluation import choose_threshold, collect_probabilities, compute_metrics
from fire_transformer.model import build_model
from fire_transformer.training import train_model
from fire_transformer.utils import device_from_arg, ensure_dir, seed_everything


def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity analysis over sample-based future-warning horizon H"
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="ft64", choices=["ft64", "ft128"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[5, 15, 30, 60])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="runs/horizon")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="Default: all outer folds")
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs

    cv = cfg["cv"]
    n_splits = int(cv["n_splits"])
    outer_seed = int(cv["outer_split_seed"])
    inner_seed_base = int(cv["inner_split_seed_base"])
    val_fraction = float(cv["val_fraction_files"])

    folds = args.folds or list(range(1, n_splits + 1))
    invalid = sorted(f for f in folds if f < 1 or f > n_splits)
    if invalid:
        raise ValueError(f"Invalid fold(s): {invalid}; valid range is 1..{n_splits}")

    device = device_from_arg(args.device)
    catalog = DatasetCatalog(args.data, cfg["schema"])
    outdir = ensure_dir(args.output)

    outer_fold_manifest(
        catalog,
        n_splits=n_splits,
        split_seed=outer_seed,
        window=cfg["window"],
    ).to_csv(outdir / "cv_fold_manifest.csv", index=False)

    rows = []
    model_cfg = cfg["models"][args.model]

    for horizon_value in args.horizons:
        for outer_fold in folds:
            (
                train_b,
                val_b,
                test_b,
                _,
                _,
                train_a,
                val_a,
                test_a,
            ) = build_fold(
                catalog,
                outer_fold=outer_fold,
                rolling_window=cfg["rolling_window"],
                window=cfg["window"],
                horizon=horizon_value,
                val_fraction_files=val_fraction,
                n_splits=n_splits,
                outer_split_seed=outer_seed,
                inner_split_seed=inner_seed_base + outer_fold - 1,
            )

            positive = float((train_b.y == 1).sum())
            negative = float((train_b.y == 0).sum())
            pos_weight = negative / max(1.0, positive)

            for seed in args.seeds:
                seed_everything(seed)
                model = build_model(model_cfg, train_b.X.shape[-1], gas_feature_index=3)

                tr = to_loader(train_b, cfg["training"]["batch_size"], True, cfg["training"]["num_workers"])
                va = to_loader(val_b, cfg["training"]["batch_size"], False, cfg["training"]["num_workers"])
                te = to_loader(test_b, cfg["training"]["batch_size"], False, cfg["training"]["num_workers"])

                model, _, best_val_auc = train_model(
                    model,
                    tr,
                    va,
                    model_cfg,
                    cfg["training"],
                    cfg["loss"],
                    device=device,
                    positive_weight=(pos_weight if args.model == "bilstm_bce" else None),
                    use_augmentation=True,
                )

                y_val, p_val = collect_probabilities(model, va, device)
                threshold, threshold_score = choose_threshold(
                    y_val, p_val, cfg["threshold_objective"]
                )

                y_test, p_test = collect_probabilities(model, te, device)
                metrics = compute_metrics(
                    y_test,
                    p_test,
                    threshold,
                    test_b.window_end_ts,
                    test_b.onset_ts,
                    file_ids=test_b.file_ids,
                    physical_events=test_b.physical_events,
                    evaluable_events=test_b.evaluable_events,
                )

                metrics.update(
                    {
                        "model": args.model,
                        "horizon": horizon_value,
                        "outer_fold": outer_fold,
                        "seed": seed,
                        "best_inner_val_auc": best_val_auc,
                        "threshold_selection_score": threshold_score,
                        "train_files": len(train_a),
                        "val_files": len(val_a),
                        "test_files": len(test_a),
                        "test_file_ids": ";".join(a.file_id for a in test_a),
                    }
                )
                rows.append(metrics)

                print(
                    f"H={horizon_value} fold={outer_fold} seed={seed} "
                    f"AUC={metrics['auc_roc']:.4f} F1={metrics['f1']:.4f} "
                    f"coverage={metrics.get('coverage', np.nan):.4f} "
                    f"lead={metrics.get('lead_mean_s', np.nan):.3f}s "
                    f"events={metrics.get('events_covered', 0)}/"
                    f"{metrics.get('events_evaluable_total', 0)}"
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

    fold_means = (
        df.groupby(["model", "horizon", "outer_fold"], as_index=False)[metrics]
        .mean()
        .sort_values(["model", "horizon", "outer_fold"])
    )
    fold_means.to_csv(outdir / "horizon_per_fold_means.csv", index=False)

    horizon_summary = (
        fold_means.groupby(["model", "horizon"])[metrics]
        .agg(["mean", "std"])
    )
    horizon_summary.to_csv(outdir / "horizon_summary.csv")

    horizon_seed_std = (
        df.groupby(["model", "horizon", "outer_fold"])[metrics]
        .std()
        .reset_index()
    )
    horizon_seed_std.to_csv(outdir / "horizon_seed_std_by_fold.csv", index=False)

    print(f"Saved results to {outdir}")


if __name__ == "__main__":
    main()
