#!/usr/bin/env python3
"""Audit the public dataset before any training run.

The script validates the state machine through DatasetCatalog and reports raw/sample
counts, physical/evaluable onset counts, node distribution, heater-profile metadata,
and pre-fire duration. Use --strict-public to enforce the counts verified for Mendeley
Data version 1 (DOI 10.17632/48j7mm8k56.1).
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from fire_transformer.config import load_config
from fire_transformer.data import (
    DatasetCatalog,
    STATE_FIRE,
    STATE_POSTFIRE,
    STATE_PREFIRE,
)


PUBLIC_EXPECTED = {
    "files": 20,
    "nodes": 5,
    "raw_rows": 131370,
    "warmup_rows": 23877,
    "annotated_rows": 107493,
    "prefire_rows": 11762,
    "fire_rows": 92395,
    "postfire_rows": 3336,
    "physical_events": 20,
    "evaluable_w60": 19,
}


def main():
    ap = argparse.ArgumentParser(description="Audit FireTransformer public dataset")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--strict-public", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    window = int(args.window if args.window is not None else cfg["window"])
    catalog = DatasetCatalog(args.data, cfg["schema"])

    acquisitions = catalog.acquisitions
    labels = np.concatenate([a.labels for a in acquisitions])
    states = np.concatenate([a.states for a in acquisitions])
    label_counts = Counter(labels.tolist())

    raw_rows = int(sum(a.raw_rows for a in acquisitions))
    warmup_rows = int(sum(a.warmup_rows for a in acquisitions))
    annotated_rows = int(sum(len(a.labels) for a in acquisitions))
    prefire_rows = int(np.sum(states == STATE_PREFIRE))
    fire_rows = int(np.sum(states == STATE_FIRE))
    postfire_rows = int(np.sum(states == STATE_POSTFIRE))
    physical_events = len(acquisitions)
    evaluable = [a for a in acquisitions if a.pre_onset_samples >= window]
    non_evaluable = [a for a in acquisitions if a.pre_onset_samples < window]

    filename_profile_indices = sorted(
    {
        a.filename_profile_index
        for a in acquisitions
        if a.filename_profile_index is not None
    }
    )
    recovery_files = int(sum(np.any(a.states == STATE_POSTFIRE) for a in acquisitions))

    print("=" * 68)
    print("DATASET AUDIT — PHYSICAL-ONSET EARLY WARNING")
    print("=" * 68)
    print(f"CSV files                         : {len(acquisitions)}")
    print(f"Nodes                             : {len(catalog.nodes)} {catalog.nodes}")
    print(
        "Acquisition profile indices "
        f"from filenames : {len(filename_profile_indices)} "
        f"{filename_profile_indices}"
    )
    print(
        "Documented BME688 heater profiles           : 17"
    )
    print(f"Raw rows                          : {raw_rows}")
    print(f"Warm-up rows (label 0)            : {warmup_rows}")
    print(f"Annotated rows after warm-up      : {annotated_rows}")
    print(f"Pre-fire rows (1001)              : {prefire_rows}")
    print(f"Active-fire rows (1002-1007)      : {fire_rows}")
    print(f"Post-fire rows (1008-1009)        : {postfire_rows}")
    print(f"Files with observed recovery      : {recovery_files}")
    print(f"Physical 1001->1002 onsets        : {physical_events}")
    print(f"Evaluable events for W={window:<3}         : {len(evaluable)}")
    print(f"Non-evaluable events for W={window:<3}     : {len(non_evaluable)}")

    print("\nRaw label_tag distribution:")
    for label in sorted(label_counts):
        print(f"  {label:>4}: {label_counts[label]}")

    rows = []
    for a in acquisitions:
        rows.append(
            {
                "file": a.file_id,
                "node": a.node,
                "profile": a.heater_profile,
                "pre_onset_samples": a.pre_onset_samples,
                "prefire_duration_s": a.prefire_duration_s,
                "onset_time_s": a.onset_timestamp,
                "evaluable": a.pre_onset_samples >= window,
                "postfire_observed": bool(np.any(a.states == STATE_POSTFIRE)),
            }
        )

    detail = pd.DataFrame(rows).sort_values(["node", "file"])
    print("\nPer-acquisition summary:")
    print(detail.to_string(index=False))

    if non_evaluable:
        print("\nNon-evaluable physical events:")
        for a in non_evaluable:
            print(
                f"  {a.file_id}: pre_onset_samples={a.pre_onset_samples}, "
                f"prefire_duration_s={a.prefire_duration_s:.3f}, "
                f"onset_time_s={a.onset_timestamp:.3f}"
            )

    if args.strict_public:
        observed = {
            "files": len(acquisitions),
            "nodes": len(catalog.nodes),
            "raw_rows": raw_rows,
            "warmup_rows": warmup_rows,
            "annotated_rows": annotated_rows,
            "prefire_rows": prefire_rows,
            "fire_rows": fire_rows,
            "postfire_rows": postfire_rows,
            "physical_events": physical_events,
            "evaluable_w60": len(
                [a for a in acquisitions if a.pre_onset_samples >= 60]
            ),
        }
        mismatches = {
            key: (PUBLIC_EXPECTED[key], observed[key])
            for key in PUBLIC_EXPECTED
            if PUBLIC_EXPECTED[key] != observed[key]
        }
        if mismatches:
            raise AssertionError(f"Public-dataset count mismatch: {mismatches}")
        print("\nSTRICT PUBLIC-DATASET CHECK: PASSED")


if __name__ == "__main__":
    main()
