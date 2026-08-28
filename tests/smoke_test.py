#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_transformer.config import load_config
from fire_transformer.data import (
    DatasetCatalog,
    STATE_FIRE,
    STATE_POSTFIRE,
    STATE_PREFIRE,
    STATE_WARMUP,
    build_fold,
    decode_label_tags,
    fire_onset_indices,
    fit_feature_scaler,
    make_outer_folds,
    outer_fold_manifest,
    physical_onset_indices,
)
from fire_transformer.evaluation import event_coverage, lead_time_stats
from fire_transformer.model import build_model


def main():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cv = cfg["cv"]

    import pandas as pd

    raw_labels = pd.Series([0, 0, 1001, 1001, 1002, 1003, 1007, 1008, 1009])
    labels, states = decode_label_tags(raw_labels, cfg["schema"])
    assert np.array_equal(
        states,
        np.asarray(
            [
                STATE_WARMUP,
                STATE_WARMUP,
                STATE_PREFIRE,
                STATE_PREFIRE,
                STATE_FIRE,
                STATE_FIRE,
                STATE_FIRE,
                STATE_POSTFIRE,
                STATE_POSTFIRE,
            ],
            dtype=np.int8,
        ),
    )
    assert np.array_equal(physical_onset_indices(labels), np.asarray([4], dtype=np.int64))

    fire = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    assert np.array_equal(fire_onset_indices(fire), np.asarray([2, 6], dtype=np.int64))

    pred = np.asarray([0, 1, 1, 1, 0], dtype=np.int64)
    y = np.ones(5, dtype=np.int64)
    end_ts = np.asarray([5, 7, 9, 6, 8], dtype=np.float64)
    onset_ts = np.asarray([10, 10, 10, 10, 10], dtype=np.float64)
    file_ids = np.asarray(["a", "a", "a", "b", "b"], dtype=object)
    expected_events = (("a", 10.0), ("b", 10.0))

    assert np.isclose(
        event_coverage(
            pred,
            onset_ts,
            file_ids=file_ids,
            evaluable_events=expected_events,
        ),
        1.0,
    )
    lead = lead_time_stats(pred, y, end_ts, onset_ts, file_ids=file_ids)
    assert np.isclose(lead["lead_mean_s"], 3.5)

    with tempfile.TemporaryDirectory() as td:
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "make_synthetic_dataset.py"),
                "--output",
                td,
                "--samples",
                "220",
            ]
        )

        catalog = DatasetCatalog(td, cfg["schema"])
        assert len(catalog.acquisitions) == 20
        assert catalog.storage_groups == [f"NODO{i}" for i in range(1, 6)]

        first = catalog.acquisitions[0]
        dt = np.diff(first.timestamps[:5])
        assert np.all(dt > 0)
        assert np.all(dt < 10.0)

        for acq in catalog.acquisitions:
            assert len(physical_onset_indices(acq.labels)) == 1
            assert acq.labels[acq.onset_index - 1] == 1001
            assert acq.labels[acq.onset_index] == 1002

        folds = make_outer_folds(
            catalog,
            n_splits=cv["n_splits"],
            split_seed=cv["outer_split_seed"],
        )
        assert len(folds) == 5
        assert all(len(fold) == 4 for fold in folds)
        outer_ids = [a.file_id for fold in folds for a in fold]
        assert len(outer_ids) == 20
        assert len(set(outer_ids)) == 20
        assert set(outer_ids) == {a.file_id for a in catalog.acquisitions}

        manifest = outer_fold_manifest(
            catalog,
            n_splits=cv["n_splits"],
            split_seed=cv["outer_split_seed"],
            window=cfg["window"],
        )
        assert manifest["physical_events"].sum() == 20
        assert manifest["evaluable_events"].sum() == 19

        all_test_ids = []
        total_test_physical = 0
        total_test_evaluable = 0
        first_fold_objects = None

        for fold in range(1, 6):
            result = build_fold(
                catalog,
                outer_fold=fold,
                rolling_window=cfg["rolling_window"],
                window=cfg["window"],
                horizon=cfg["horizon"],
                val_fraction_files=cv["val_fraction_files"],
                n_splits=cv["n_splits"],
                outer_split_seed=cv["outer_split_seed"],
                inner_split_seed=cv["inner_split_seed_base"] + fold - 1,
            )
            train, val, test, _, _, train_a, val_a, test_a = result
            if first_fold_objects is None:
                first_fold_objects = result

            train_ids = {a.file_id for a in train_a}
            val_ids = {a.file_id for a in val_a}
            test_ids = {a.file_id for a in test_a}
            assert len(train_a) == 12
            assert len(val_a) == 4
            assert len(test_a) == 4
            assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
            assert train_ids | val_ids | test_ids == {a.file_id for a in catalog.acquisitions}

            all_test_ids.extend(test_ids)
            total_test_physical += len(test.physical_events)
            total_test_evaluable += len(test.evaluable_events)

            for acq in test_a:
                if acq.pre_onset_samples < cfg["window"]:
                    continue
                mask = test.file_ids == acq.file_id
                assert mask.any()
                assert np.all(test.window_end_ts[mask] < acq.onset_timestamp)

        assert len(all_test_ids) == 20
        assert len(set(all_test_ids)) == 20
        assert total_test_physical == 20
        assert total_test_evaluable == 19

        train, val, test, _, _, train_a, _, _ = first_fold_objects
        assert train.X.shape[-1] == 20

        scaler_a, _ = fit_feature_scaler(train_a, cfg["rolling_window"], cfg["window"])
        contaminated = copy.deepcopy(train_a)
        for acq in contaminated:
            acq.raw.iloc[acq.onset_index :, :] = 1e12
        scaler_b, _ = fit_feature_scaler(contaminated, cfg["rolling_window"], cfg["window"])
        assert np.allclose(scaler_a.data_min_, scaler_b.data_min_)
        assert np.allclose(scaler_a.data_max_, scaler_b.data_max_)

        model = build_model(cfg["models"]["ft64"], n_features=20, gas_feature_index=3)
        with torch.inference_mode():
            output = model(torch.from_numpy(train.X[:2]))
        assert output.shape == (2,)
        assert sum(p.numel() for p in model.parameters()) == 110338

        ft32 = build_model(cfg["models"]["ft32"], n_features=20, gas_feature_index=3)
        assert sum(p.numel() for p in ft32.parameters()) == 20002

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
