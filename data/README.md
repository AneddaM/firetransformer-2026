# Dataset

The raw dataset is intentionally not redistributed in this repository.

Use:

**M. Anedda, “Dataset Smoke Detection,” Mendeley Data, Version 1, 2026.**  
DOI: `10.17632/48j7mm8k56.1`

The original archive layout can be preserved:

```text
data/raw/
├── NODO1/
├── NODO2/
├── NODO3/
├── NODO4/
└── NODO5/
```

**Important:** `NODO1...NODO5` are Bosch development-kit/storage groups used to
organize acquisition files. They are not independent statistical nodes or evaluation
domains and are never used as cross-validation groups.

## Acquisition and heater-profile policy

The dataset is analyzed as **20 complete acquisition CSV files** from one multi-profile
campaign. The campaign documents **17 distinct BME688 heater profiles** across those 20
acquisitions; some profiles are repeated. All acquisitions/profiles are retained.

The repository does not select a heater profile, compare heater profiles, or infer a
global heater-profile identity from the small `profilo_X` filename token. Heater-profile
identity is not an ML feature.

## Public annotation state machine

```text
0              -> initial warm-up/stabilization; excluded
1001           -> valid pre-fire state
1001 -> 1002   -> physical fire onset
1002...1007    -> active-fire state
1008, 1009     -> post-fire/recovery state
```

The physical onset is the first and unique `1001 -> 1002` transition inside each
acquisition. Active-fire and post-fire samples are never used as early-warning model
inputs.

The public Version 1 archive contains 131,370 raw rows. Removing 23,877 initial warm-up
rows leaves 107,493 annotated rows: 11,762 pre-fire, 92,395 active-fire, and 3,336
post-fire/recovery samples.

## Time and heater cadence

No temporal resampling is applied. `W` and `H` are observation counts. Physical lead
time is computed from `timestamp_since_poweron`, converted from milliseconds to seconds.

## Leakage-control rule

Every CSV is one indivisible statistical group. Five deterministic outer folds partition
all 20 files, four test acquisitions per fold. Inner train/validation splitting is also
performed by complete file before sliding-window generation.

The MinMax scaler is fitted only on eligible pre-onset samples from inner-training files.
Neither validation/test acquisitions nor active/post-fire observations influence scaling.

## Event denominator

There are 20 physical onset events. With `W=60`, 19 are evaluable. One acquisition has
only 20 pre-onset observations and cannot generate a complete input window. Coverage is
computed over evaluable physical events while the physical event remains tracked.

Run before training:

```bash
python scripts/dataset_audit.py --data data/raw --strict-public
python scripts/cv_preflight.py --data data/raw
```
