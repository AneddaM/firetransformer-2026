#!/usr/bin/env python3
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
    build_fold,
    fire_onset_indices,
)
from fire_transformer.evaluation import event_coverage, lead_time_stats
from fire_transformer.model import build_model


def main():
    # A physical onset is strictly a 0->1 transition.
    fire = np.asarray(
        [0, 0, 1, 1, 0, 0, 1, 1],
        dtype=np.int64,
    )
    assert np.array_equal(
        fire_onset_indices(fire),
        np.asarray([2, 6], dtype=np.int64),
    )

    # Equal timestamps in separate files must remain distinct events.
    # Lead time is based on the earliest correct warning per event.
    pred = np.asarray([0, 1, 1, 1, 0], dtype=np.int64)
    y = np.ones(5, dtype=np.int64)
    end_ts = np.asarray([5, 7, 9, 6, 8], dtype=np.float64)
    onset_ts = np.asarray([10, 10, 10, 10, 10], dtype=np.float64)
    file_ids = np.asarray(["a", "a", "a", "b", "b"], dtype=object)

    assert np.isclose(
        event_coverage(
            pred,
            onset_ts,
            file_ids=file_ids,
        ),
        1.0,
    )

    lead = lead_time_stats(
        pred,
        y,
        end_ts,
        onset_ts,
        file_ids=file_ids,
    )
    assert np.isclose(lead["lead_mean_s"], 3.5)

    with tempfile.TemporaryDirectory() as td:
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "make_synthetic_dataset.py"),
                "--output",
                td,
                "--samples",
                "180",
            ]
        )

        cfg = load_config(ROOT / "configs" / "default.yaml")
        catalog = DatasetCatalog(td, cfg["schema"])

        assert catalog.nodes == [f"NODO{i}" for i in range(1, 6)]

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

        finite = np.isfinite(test.onset_ts)
        assert finite.any()

        # Every positive warning window must end before its physical onset.
        assert np.all(
            test.window_end_ts[finite] <
            test.onset_ts[finite]
        )

        # The synthetic held-out node has one 0->1 event per acquisition file.
        event_keys = {
            (
                str(file_id),
                round(float(onset), 6),
            )
            for file_id, onset in zip(
                test.file_ids[finite],
                test.onset_ts[finite],
            )
        }
        assert len(event_keys) == len(test_a) == 4

        model = build_model(
            cfg["models"]["ft64"],
            n_features=20,
            gas_feature_index=3,
        )

        with torch.inference_mode():
            output = model(torch.from_numpy(train.X[:2]))

        assert output.shape == (2,)
        assert sum(p.numel() for p in model.parameters()) == 110338

        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
