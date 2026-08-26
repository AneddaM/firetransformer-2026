#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full five-fold leakage-controlled leave-one-node-out evaluation"
    )
    ap.add_argument(
        "--data",
        required=True,
        help="Root containing CSV files under NODO1...NODO5 paths",
    )
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["ft64"],
        choices=["ft32", "ft64", "ft128", "bilstm_bce", "bilstm_fl"],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="runs/lono")
    ap.add_argument(
        "--heldout",
        nargs="*",
        default=None,
        help="Optional subset for debugging, e.g. --heldout NODO5",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override config for quick tests",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs

    device = device_from_arg(args.device)
    outdir = ensure_dir(args.output)

    catalog = DatasetCatalog(args.data, cfg["schema"])
    expected_nodes = [f"NODO{i}" for i in range(1, 6)]

    missing = sorted(set(expected_nodes) - set(catalog.nodes))
    if missing:
        raise ValueError(
            "Full LONO requires NODO1...NODO5. "
            f"Missing: {missing}; found: {catalog.nodes}"
        )

    heldouts = args.heldout or expected_nodes
    invalid = sorted(set(heldouts) - set(expected_nodes))
    if invalid:
        raise ValueError(f"Invalid held-out node(s): {invalid}")

    rows = []

    for model_name in args.models:
        model_cfg = cfg["models"][model_name]

        for outer_index, heldout in enumerate(heldouts):
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
                heldout,
                rolling_window=cfg["rolling_window"],
                window=cfg["window"],
                horizon=cfg["horizon"],
                val_fraction_files=cfg["val_fraction_files"],
                split_seed=1000 + outer_index,
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
                    positive_weight=(
                        pos_weight if model_name == "bilstm_bce" else None
                    ),
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
                        "outer_test_node": heldout,
                        "seed": seed,
                        "best_inner_val_auc": best_val_auc,
                        "threshold_selection_score": threshold_score,
                        "n_train": len(train_b.y),
                        "n_val": len(val_b.y),
                        "n_test": len(test_b.y),
                        "train_files": len(train_a),
                        "val_files": len(val_a),
                        "test_files": len(test_a),
                        "horizon": cfg["horizon"],
                        "window": cfg["window"],
                    }
                )
                rows.append(metrics)

                stem = f"{model_name}_{heldout}_seed{seed}"

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_name": model_name,
                        "model_cfg": model_cfg,
                        "n_features": n_features,
                        "feature_names": feature_names,
                        "threshold": threshold,
                        "heldout_node": heldout,
                        "seed": seed,
                        "window": cfg["window"],
                        "horizon": cfg["horizon"],
                        "physical_events": list(test_b.physical_events),
                        "evaluable_events": list(test_b.evaluable_events),
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
                        "outer_test_node",
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
        raise RuntimeError("No LONO runs were executed.")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "lono_runs.csv", index=False)

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
    # 1) average seeds within each held-out node;
    # 2) macro mean across node-wise means;
    # 3) sigma_node = sample std across node-wise means;
    # 4) keep seed variability separate.
    node_summary = (
        df.groupby(["model", "outer_test_node"])[metric_cols]
        .agg(["mean", "std"])
        .sort_index()
    )
    node_summary.to_csv(outdir / "lono_per_node_summary.csv")

    node_means = (
        df.groupby(["model", "outer_test_node"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["model", "outer_test_node"])
    )
    node_means.to_csv(outdir / "lono_per_node_means.csv", index=False)

    macro_summary = (
        node_means.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .sort_index()
    )
    macro_summary.to_csv(outdir / "lono_macro_summary.csv")

    seed_std_by_node = (
        df.groupby(["model", "outer_test_node"])[metric_cols]
        .std()
        .reset_index()
        .sort_values(["model", "outer_test_node"])
    )
    seed_std_by_node.to_csv(
        outdir / "lono_seed_std_by_node.csv",
        index=False,
    )

    seed_variability = (
        seed_std_by_node.groupby("model")[metric_cols]
        .mean()
        .sort_index()
    )
    seed_variability.to_csv(outdir / "lono_seed_variability.csv")

    for filename in [
        "lono_runs.csv",
        "lono_per_node_summary.csv",
        "lono_per_node_means.csv",
        "lono_macro_summary.csv",
        "lono_seed_std_by_node.csv",
        "lono_seed_variability.csv",
    ]:
        print(f"Saved: {outdir / filename}")


if __name__ == "__main__":
    main()
