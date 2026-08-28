#!/usr/bin/env python3
"""Preflight the acquisition-level grouped cross-validation before training."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fire_transformer.config import load_config
from fire_transformer.data import DatasetCatalog, build_fold, outer_fold_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Grouped-CV split/window preflight")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", default="runs/preflight")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cv = cfg["cv"]
    catalog = DatasetCatalog(args.data, cfg["schema"])

    n_splits = int(cv["n_splits"])
    outer_seed = int(cv["outer_split_seed"])
    inner_seed_base = int(cv["inner_split_seed_base"])
    val_fraction = float(cv["val_fraction_files"])

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = outer_fold_manifest(
        catalog,
        n_splits=n_splits,
        split_seed=outer_seed,
        window=cfg["window"],
    )
    manifest.to_csv(outdir / "cv_fold_manifest.csv", index=False)

    rows = []
    test_files_seen = []

    for fold in range(1, n_splits + 1):
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
            outer_fold=fold,
            rolling_window=cfg["rolling_window"],
            window=cfg["window"],
            horizon=cfg["horizon"],
            val_fraction_files=val_fraction,
            n_splits=n_splits,
            outer_split_seed=outer_seed,
            inner_split_seed=inner_seed_base + fold - 1,
        )

        train_ids = {a.file_id for a in train_a}
        val_ids = {a.file_id for a in val_a}
        test_ids = {a.file_id for a in test_a}
        assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
        assert train_ids | val_ids | test_ids == {a.file_id for a in catalog.acquisitions}
        test_files_seen.extend(sorted(test_ids))

        for split_name, bundle, acqs in [
            ("train", train_b, train_a),
            ("val", val_b, val_a),
            ("test", test_b, test_a),
        ]:
            n_pos = int((bundle.y == 1).sum())
            n_neg = int((bundle.y == 0).sum())
            rows.append(
                {
                    "outer_fold": fold,
                    "split": split_name,
                    "files": len(acqs),
                    "windows": len(bundle.y),
                    "positive": n_pos,
                    "negative": n_neg,
                    "positive_pct": 100.0 * n_pos / len(bundle.y) if len(bundle.y) else np.nan,
                    "physical_events": len(bundle.physical_events),
                    "evaluable_events": len(bundle.evaluable_events),
                }
            )

    preflight = pd.DataFrame(rows)
    preflight.to_csv(outdir / "cv_preflight.csv", index=False)

    all_ids = sorted(a.file_id for a in catalog.acquisitions)
    assert sorted(test_files_seen) == all_ids, "Each acquisition must appear once in outer test"
    assert len(test_files_seen) == len(set(test_files_seen)), "Duplicate outer-test acquisition"
    assert int(manifest["physical_events"].sum()) == len(catalog.acquisitions)

    expected_evaluable = sum(a.pre_onset_samples >= cfg["window"] for a in catalog.acquisitions)
    assert int(manifest["evaluable_events"].sum()) == expected_evaluable

    if (preflight["windows"] == 0).any():
        raise AssertionError("At least one train/validation/test split has zero windows")
    if (preflight["positive"] == 0).any():
        raise AssertionError("At least one split has no positive early-warning windows")
    if (preflight["negative"] == 0).any():
        raise AssertionError("At least one split has no negative early-warning windows")

    print("=" * 72)
    print("ACQUISITION-LEVEL GROUPED CV PREFLIGHT")
    print("=" * 72)
    print(preflight.to_string(index=False))
    print("\nOuter test acquisitions are a disjoint partition of all CSV files: PASSED")
    print(f"Total outer-test acquisition files : {len(test_files_seen)}")
    print(f"Total physical onsets             : {int(manifest['physical_events'].sum())}")
    print(f"Total evaluable events W={cfg['window']}      : {int(manifest['evaluable_events'].sum())}")
    print("Storage/development-kit folders are NOT used as CV groups.")
    print("GROUPED CV PREFLIGHT: PASSED")
    print(f"Saved: {outdir / 'cv_fold_manifest.csv'}")
    print(f"Saved: {outdir / 'cv_preflight.csv'}")


if __name__ == "__main__":
    main()
