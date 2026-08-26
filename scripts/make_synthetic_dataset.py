#!/usr/bin/env python3
"""Create a tiny five-node state-aware dataset for pipeline smoke testing only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/synthetic")
    ap.add_argument("--samples", type=int, default=220)
    args = ap.parse_args()

    root = Path(args.output)
    rng = np.random.default_rng(7)

    for node in range(1, 6):
        d = root / f"NODO{node}"
        d.mkdir(parents=True, exist_ok=True)

        for f in range(4):
            n = args.samples
            warmup = 15

            # One deliberately non-evaluable physical event: 20 pre-onset samples.
            if node == 5 and f == 0:
                prefire = 20
            else:
                prefire = 95 + ((node + f) % 25)

            onset = warmup + prefire
            if onset + 30 >= n:
                raise ValueError("Increase --samples for the requested synthetic layout")

            # Different synthetic profile cadences test ms->s conversion without
            # resampling. Profile ID itself is metadata, not an ML feature.
            profile_id = ((node - 1) * 4 + f) % 17 + 1
            dt_s = 0.5 + 0.15 * (profile_id % 5)
            t_s = np.arange(n, dtype=float) * dt_s

            labels = np.full(n, 1001, dtype=np.int64)
            labels[:warmup] = 0

            # Exact physical onset 1001 -> 1002.
            fire_end = min(n, onset + 55)
            labels[onset:fire_end] = 1002
            if fire_end - onset > 20:
                labels[onset + 20 : fire_end] = 1003
            if fire_end - onset > 40:
                labels[onset + 40 : fire_end] = 1007

            # Most acquisitions include post-fire/recovery, mirroring the real state
            # progression without attempting to reproduce exact public counts.
            if not (node == 1 and f == 0):
                labels[fire_end:] = 1008
                if n - fire_end >= 3 and (node + f) % 3 == 0:
                    labels[-2:] = 1009
            else:
                labels[fire_end:] = 1007

            fire = np.isin(labels, [1002, 1003, 1004, 1005, 1006, 1007]).astype(float)
            temp = 24 + 0.01 * np.arange(n) + 1.5 * fire + rng.normal(0, 0.08, n)
            hum = 55 - 0.015 * np.arange(n) - 2.0 * fire + rng.normal(0, 0.15, n)
            press = 1012 + rng.normal(0, 0.2, n)
            gas = 120000 - 30 * np.arange(n) - 15000 * fire + rng.normal(0, 800, n)

            pd.DataFrame(
                {
                    "sensor_index": np.arange(n),
                    "sensor_id": node,
                    "timestamp_since_poweron": t_s * 1000.0,
                    "real_time_clock": 1_700_000_000 + np.floor(t_s).astype(int),
                    "temperature": temp,
                    "pressure": press,
                    "relative_humidity": hum,
                    "resistance_gassensor": gas,
                    "heater_profile_step_index": np.arange(n) % 10,
                    "scanning_enabled": 1,
                    "scanning_cycle_index": np.arange(n),
                    "label_tag": labels,
                    "error_code": 0,
                }
            ).to_csv(
                d / f"Nodo_{node}_profilo_{profile_id}.csv",
                index=False,
            )

    print(root)


if __name__ == "__main__":
    main()
