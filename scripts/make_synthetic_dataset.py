#!/usr/bin/env python3
"""Create a tiny 20-acquisition state-aware dataset for smoke testing only.

Five directories mimic the public archive's Bosch development-kit/storage layout.
They are intentionally *not* statistical groups. Across 20 acquisitions, the synthetic
metadata cycles through 17 heater-profile IDs, with three profiles repeated once.
"""
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
    acquisition_counter = 0

    for storage_index in range(1, 6):
        d = root / f"NODO{storage_index}"
        d.mkdir(parents=True, exist_ok=True)

        for local_slot in range(1, 5):
            n = args.samples
            warmup = 15

            # One deliberately non-evaluable physical event: 20 pre-onset samples.
            if storage_index == 5 and local_slot == 1:
                prefire = 20
            else:
                prefire = 95 + ((storage_index + local_slot) % 25)

            onset = warmup + prefire
            if onset + 30 >= n:
                raise ValueError("Increase --samples for the requested synthetic layout")

            # 17 distinct profiles across 20 acquisition files; profiles 1,2,3 repeat.
            heater_profile_id = (acquisition_counter % 17) + 1
            acquisition_counter += 1
            dt_s = 0.5 + 0.15 * (heater_profile_id % 5)
            t_s = np.arange(n, dtype=float) * dt_s

            labels = np.full(n, 1001, dtype=np.int64)
            labels[:warmup] = 0

            fire_end = min(n, onset + 55)
            labels[onset:fire_end] = 1002
            if fire_end - onset > 20:
                labels[onset + 20 : fire_end] = 1003
            if fire_end - onset > 40:
                labels[onset + 40 : fire_end] = 1007

            if not (storage_index == 1 and local_slot == 1):
                labels[fire_end:] = 1008
                if n - fire_end >= 3 and (storage_index + local_slot) % 3 == 0:
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
                    "sensor_id": storage_index,
                    "synthetic_heater_profile_id": heater_profile_id,
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
                d / f"Acquisition_{acquisition_counter:02d}_profilo_{local_slot}.csv",
                index=False,
            )

    print(root)


if __name__ == "__main__":
    main()
