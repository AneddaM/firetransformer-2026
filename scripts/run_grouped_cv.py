#!/usr/bin/env python3
"""Grouped five-fold cross-validation over complete acquisition CSV files."""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from fire_transformer.config import load_config
from fire_transformer.data import (
    DatasetCatalog,
    build_fold,
    outer_fold_manifest,
    to_loader,
)
from fire_transformer.evaluation import (
    choose_threshold,
    collect_probabilities,
    compute_metrics,
)
from fire_transformer.model import build_model
from fire_transformer.training import train_model
from fire_transformer.utils import device_from_arg, ensure_dir, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Leakage-controlled grouped cross-validation. The unit of splitting is "
            "the complete acquisition CSV; NODO1...NODO5 folders are metadata only."
        )
    )
    ap.add_argument("--data", required=True, help="Root containing the 20 acquisition CSV files")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["ft64"],
        choices=["ft32", "ft64", "ft128", "bilstm_bce", "bilstm_fl"],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="runs/grouped_cv")
    ap.add_argument(
        "--folds",
        nargs="*",
        type=int,
        default=None,
        help="Optional outer-fold subset for debugging, e.g. --folds 1",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override config for quick tests only",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs

    cv_cfg = cfg["cv"]
    n_splits = int(cv_cfg["n_splits"])
    outer_split_seed = int(cv_cfg["outer_split_seed"])
    inner_seed_base = int(cv_cfg["inner_split_seed_base"])
    val_fraction_files = float(cv_cfg["val_fraction_files"])

    folds = args.folds or list(range(1, n_splits + 1))
    invalid = sorted(f for f in folds if f < 1 or f > n_splits)
    if invalid:
        raise ValueError(f"Invalid outer fold(s): {invalid}; valid range is 1..{n_splits}")

    device = device_from_arg(args.device)
    outdir = ensure_dir(args.output)
    catalog = DatasetCatalog(args.data, cfg["schema"])

    manifest = outer_fold_manifest(
        catalog,
        n_splits=n_splits,
        split_seed=outer_split_seed,
        window=cfg["window"],
    )
    manifest.to_csv(outdir / "cv_fold_manifest.csv", index=False)

    rows = []

    for model_name in args.models:
        model_cfg = cfg["models"][model_name]

        for outer_fold in folds:
            (
                train_b,
                val_b,
                test_b,
                scaler,
                feature_names,
                train_a,
                val_a,
                test_a,
            ) = build_fold(
                catalog,
                outer_fold=outer_fold,
                rolling_window=cfg["rolling_window"],
                window=cfg["window"],
                horizon=cfg["horizon"],
                val_fraction_files=val_fraction_files,
                n_splits=n_splits,
                outer_split_seed=outer_split_seed,
                inner_split_seed=inner_seed_base + outer_fold - 1,
            )

            n_features = train_b.X.shape[-1]
            gas_idx = 3
            positive = float((train_b.y == 1).sum())
            negative = float((train_b.y == 0).sum())
            pos_weight = negative / max(1.0, positive)

            for seed in args.seeds:
                seed_everything(seed)

                train_loader = to_loader(
                    train_b,
                    cfg["training"]["batch_size"],
                    True,
                    cfg["training"]["num_workers"],
                )
                val_loader = to_loader(
                    val_b,
                    cfg["training"]["batch_size"],
                    False,
                    cfg["training"]["num_workers"],
                )
                test_loader = to_loader(
                    test_b,
                    cfg["training"]["batch_size"],
                    False,
                    cfg["training"]["num_workers"],
                )

                model = build_model(
                    model_cfg,
                    n_features=n_features,
                    gas_feature_index=gas_idx,
                )

                model, history, best_val_auc = train_model(
                    model,
                    train_loader,
                    val_loader,
                    model_cfg,
                    cfg["training"],
                    cfg["loss"],
                    device=device,
                    positive_weight=(pos_weight if model_name == "bilstm_bce" else None),
                    use_augmentation=True,
                )

                y_val, p_val = collect_probabilities(model, val_loader, device)
                threshold, threshold_score = choose_threshold(
                    y_val,
                    p_val,
                    cfg["threshold_objective"],
                )

                y_test, p_test = collect_probabilities(model, test_loader, device)
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
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "seed": seed,
                        "best_inner_val_auc": best_val_auc,
                        "threshold_selection_score": threshold_score,
                        "n_train": len(train_b.y),
                        "n_val": len(val_b.y),
                        "n_test": len(test_b.y),
                        "train_files": len(train_a),
                        "val_files": len(val_a),
                        "test_files": len(test_a),
                        "train_file_ids": ";".join(a.file_id for a in train_a),
                        "val_file_ids": ";".join(a.file_id for a in val_a),
                        "test_file_ids": ";".join(a.file_id for a in test_a),
                        "horizon": cfg["horizon"],
                        "window": cfg["window"],
                        "outer_split_seed": outer_split_seed,
                        "inner_split_seed": inner_seed_base + outer_fold - 1,
                    }
                )
                rows.append(metrics)

                stem = f"{model_name}_fold{outer_fold}_seed{seed}"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_name": model_name,
                        "model_cfg": model_cfg,
                        "n_features": n_features,
                        "feature_names": feature_names,
                        "threshold": threshold,
                        "outer_fold": outer_fold,
                        "seed": seed,
                        "window": cfg["window"],
                        "horizon": cfg["horizon"],
                        "train_file_ids": [a.file_id for a in train_a],
                        "val_file_ids": [a.file_id for a in val_a],
                        "test_file_ids": [a.file_id for a in test_a],
                        "physical_events": list(test_b.physical_events),
                        "evaluable_events": list(test_b.evaluable_events),
                        "split_protocol": "five-fold grouped CV by complete acquisition file",
                        "storage_groups_used_for_splitting": False,
                        "documented_heater_profiles": 17,
                        "state_semantics": {
                            "warmup": [0],
                            "pre_fire": [1001],
                            "fire": [1002, 1003, 1004, 1005, 1006, 1007],
                            "post_fire": [1008, 1009],
                            "physical_onset": [1001, 1002],
                        },
                    },
                    outdir / f"{stem}.pth",
                )

                pd.DataFrame(history).to_csv(
                    outdir / f"{stem}_history.csv",
                    index=False,
                )

                printable = {
                    key: metrics.get(key)
                    for key in [
                        "model",
                        "outer_fold",
                        "seed",
                        "auc_roc",
                        "f1",
                        "recall",
                        "events_physical_total",
                        "events_evaluable_total",
                        "events_covered",
                        "coverage",
                        "lead_mean_s",
                    ]
                }
                print(json.dumps(printable, indent=2, default=float))

    if not rows:
        raise RuntimeError("No grouped-CV runs were executed.")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "cv_runs.csv", index=False)

    candidate_metrics = [
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "auc_pr",
        "coverage",
        "lead_mean_s",
        "false_alarm_rate",
        "missed_detection_rate",
    ]
    metric_cols = [c for c in candidate_metrics if c in df.columns]
    if not metric_cols:
        raise RuntimeError("No expected metric columns were produced.")

    # Hierarchical aggregation:
    # 1) average seeds within each outer acquisition fold;
    # 2) macro mean across the five fold-wise means;
    # 3) sigma_fold = sample std across fold-wise means;
    # 4) keep seed variability separate.
    fold_summary = (
        df.groupby(["model", "outer_fold"])[metric_cols]
        .agg(["mean", "std"])
        .sort_index()
    )
    fold_summary.to_csv(outdir / "cv_per_fold_summary.csv")

    fold_means = (
        df.groupby(["model", "outer_fold"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["model", "outer_fold"])
    )
    fold_means.to_csv(outdir / "cv_per_fold_means.csv", index=False)

    macro_summary = (
        fold_means.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .sort_index()
    )
    macro_summary.to_csv(outdir / "cv_macro_summary.csv")

    seed_std_by_fold = (
        df.groupby(["model", "outer_fold"])[metric_cols]
        .std()
        .reset_index()
        .sort_values(["model", "outer_fold"])
    )
    seed_std_by_fold.to_csv(outdir / "cv_seed_std_by_fold.csv", index=False)

    seed_variability = (
        seed_std_by_fold.groupby("model")[metric_cols]
        .mean()
        .sort_index()
    )
    seed_variability.to_csv(outdir / "cv_seed_variability.csv")

    for filename in [
        "cv_fold_manifest.csv",
        "cv_runs.csv",
        "cv_per_fold_summary.csv",
        "cv_per_fold_means.csv",
        "cv_macro_summary.csv",
        "cv_seed_std_by_fold.csv",
        "cv_seed_variability.csv",
    ]:
        print(f"Saved: {outdir / filename}")


if __name__ == "__main__":
    main()
