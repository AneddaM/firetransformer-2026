# Dataset

The raw dataset is intentionally not redistributed in this repository.

Use:

**M. Anedda, “Dataset Smoke Detection,” Mendeley Data, Version 1, 2026.**  
DOI: `10.17632/48j7mm8k56.1`

Expected layout:

```text
data/raw/
├── NODO1/
├── NODO2/
├── NODO3/
├── NODO4/
└── NODO5/
```

The loader searches recursively, so original filenames may be preserved.

## Public annotation state machine

The revised pipeline uses the raw `label_tag` values directly:

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

The public Version 1 archive contains 131,370 raw rows. Removing the 23,877 initial
warm-up rows leaves 107,493 annotated rows: 11,762 pre-fire, 92,395 active-fire, and
3,336 post-fire/recovery samples.

## Heater profiles and time

All 17 BME688 heater profiles are retained. Their different acquisition-cycle timing is
not resampled. Therefore `W` and `H` are sample-based quantities. Physical lead time is
computed from `timestamp_since_poweron`, which is converted from milliseconds to
seconds using a factor of `0.001`.

## Leakage-control rule

Every CSV is treated as one indivisible acquisition file. Inner validation is selected
at file level before sliding windows are generated. The outer held-out node is never
used for scaler fitting, training, early stopping, or threshold selection.

The MinMax scaler is fitted only on eligible **pre-onset** samples from inner-training
files. Neither active-fire/post-fire values nor the held-out node influence scaling.

## Event denominator

There are 20 physical onset events. With `W=60`, 19 are evaluable. One NODO5
acquisition contains only 20 pre-onset observations and therefore cannot generate a
complete input window. Coverage is computed over evaluable physical events, while the
physical event remains explicitly tracked.

Run before training:

```bash
python scripts/dataset_audit.py --data data/raw --strict-public
```
