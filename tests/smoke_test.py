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
    physical_onset_indices,
)
from fire_transformer.evaluation import event_coverage, lead_time_stats
from fire_transformer.model import build_model


def main():
    cfg = load_config(ROOT / "configs" / "default.yaml")

    # ------------------------------------------------------------------
    # Raw label semantics and exact physical onset.
    # ------------------------------------------------------------------
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
    assert np.array_equal(
        physical_onset_indices(labels),
        np.asarray([4], dtype=np.int64),
    )

    # Generic binary helper remains strict 0->1.
    fire = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    assert np.array_equal(
        fire_onset_indices(fire),
        np.asarray([2, 6], dtype=np.int64),
    )

    # ------------------------------------------------------------------
    # Event identity and earliest correct warning.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # End-to-end state-aware synthetic dataset.
    # ------------------------------------------------------------------
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
        assert catalog.nodes == [f"NODO{i}" for i in range(1, 6)]
        assert len(catalog.acquisitions) == 20

        # timestamp_since_poweron must have been converted from ms to seconds.
        first = catalog.acquisitions[0]
        dt = np.diff(first.timestamps[:5])
        assert np.all(dt > 0)
        assert np.all(dt < 10.0)

        # Every synthetic acquisition has exactly one exact 1001->1002 onset.
        for acq in catalog.acquisitions:
            assert len(physical_onset_indices(acq.labels)) == 1
            assert acq.labels[acq.onset_index - 1] == 1001
            assert acq.labels[acq.onset_index] == 1002

        (
            train,
            val,
            test,
            _,
            _,
            train_a,
            val_a,
            test_a,
        ) = build_fold(
            catalog,
            "NODO5",
            cfg["rolling_window"],
            cfg["window"],
            cfg["horizon"],
            cfg["val_fraction_files"],
            1337,
        )

        assert train.X.shape[-1] == 20
        assert set(a.node for a in test_a) == {"NODO5"}
        assert all(a.node != "NODO5" for a in train_a + val_a)

        # NODO5 has four physical events but one intentionally has only 20
        # pre-onset observations, so W=60 leaves exactly three evaluable events.
        assert len(test.physical_events) == 4
        assert len(test.evaluable_events) == 3
        assert sum(a.pre_onset_samples < 60 for a in test_a) == 1

        # Active-fire/post-fire samples never enter model windows: every generated
        # window for each file ends before that file's physical onset timestamp.
        onset_by_file = {
            a.file_id: a.onset_timestamp
            for a in test_a
            if a.pre_onset_samples >= cfg["window"]
        }
        for file_id, onset in onset_by_file.items():
            mask = test.file_ids == file_id
            assert mask.any()
            assert np.all(test.window_end_ts[mask] < onset)

        finite = np.isfinite(test.onset_ts)
        assert finite.any()
        assert np.all(test.window_end_ts[finite] < test.onset_ts[finite])

        # Scaler must be invariant to arbitrarily large post-onset values.
        scaler_a, _ = fit_feature_scaler(
            train_a,
            cfg["rolling_window"],
            cfg["window"],
        )
        contaminated = copy.deepcopy(train_a)
        for acq in contaminated:
            acq.raw.iloc[acq.onset_index :, :] = 1e12
        scaler_b, _ = fit_feature_scaler(
            contaminated,
            cfg["rolling_window"],
            cfg["window"],
        )
        assert np.allclose(scaler_a.data_min_, scaler_b.data_min_)
        assert np.allclose(scaler_a.data_max_, scaler_b.data_max_)

        model = build_model(
            cfg["models"]["ft64"],
            n_features=20,
            gas_feature_index=3,
        )
        with torch.inference_mode():
            output = model(torch.from_numpy(train.X[:2]))
        assert output.shape == (2,)
        assert sum(p.numel() for p in model.parameters()) == 110338

        ft32 = build_model(
            cfg["models"]["ft32"],
            n_features=20,
            gas_feature_index=3,
        )
        assert sum(p.numel() for p in ft32.parameters()) == 20002

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
