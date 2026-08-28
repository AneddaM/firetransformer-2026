# FireTransformer — WF-IoT 2026

Code and experimental protocol for the revision of
**“Edge-Aware FireTransformer for Horizon-Based Early Wildfire Warning in IoT Sensor Networks.”**

The repository implements a physical-onset early-warning task over the **20 public
acquisition CSV files as one pooled multi-profile campaign**. The five `NODO1...NODO5`
directories reflect Bosch development-kit/storage organization used during acquisition;
they are **not independent statistical nodes, domains, or cross-validation groups**.

## Experimental interpretation

- 20 acquisition CSV files are retained;
- 17 distinct BME688 heater profiles are represented across those 20 acquisitions;
- some heater profiles are repeated, but no profile is selected as “best”;
- heater-profile identity is not a model feature and is not an evaluation target;
- no temporal resampling is applied;
- cross-validation is grouped by complete acquisition file;
- all reported predictive results must come from the new acquisition-level grouped-CV pipeline.

The repository deliberately does **not** perform comparisons among `NODO1...NODO5`.

## Dataset state semantics

The public dataset contains **131,370 raw samples**. `label_tag` is interpreted as:

```text
0              initial warm-up / stabilization -> excluded
1001           valid pre-fire state
1001 -> 1002   physical fire onset
1002...1007    active-fire state
1008, 1009     post-fire / recovery state
```

After removing the 23,877 warm-up samples, **107,493 annotated samples** remain:

- 11,762 pre-fire samples (`1001`);
- 92,395 active-fire samples (`1002`–`1007`);
- 3,336 post-fire/recovery samples (`1008`–`1009`).

All 20 acquisition files contain exactly one physical `1001 -> 1002` onset. With the
reference `W=60`, 19 physical onset events are evaluable; one acquisition contains only
20 valid pre-onset observations and cannot produce a complete input window.

These acquisition-state counts are not the early-warning class counts. Model windows
are generated **only before the physical onset**.

## Early-warning target

For acquisition onset index `o`, a `W`-observation window ending at `t` is eligible only
when:

```text
W - 1 <= t < o
```

The target is:

```text
y_t = 1  if  0 < o - t <= H
y_t = 0  if      o - t > H
```

Active-fire and post-fire/recovery samples never enter the input tensor.

The 4 raw BME688 variables (temperature, humidity, pressure, gas resistance) are enriched
with rolling mean/std/min/max, producing 20 features. The MinMax scaler is fitted only
on eligible pre-onset samples from inner-training acquisition files.

## Heater profiles and physical time

All **17 documented BME688 heater profiles** are retained. The public filenames contain
small `profilo_X` tokens inside the storage layout; the repository does not treat those
tokens as global heater-profile identifiers.

Because heater profiles have different acquisition-cycle timing, `W` and `H` are numbers
of **observations**, not fixed seconds. Realized lead time is calculated from:

```text
timestamp_since_poweron / 1000.0
```

and is therefore reported in seconds. Lead time is conditional on events that are
actually covered by at least one correct warning.

## Leakage-controlled five-fold grouped cross-validation

The unit of splitting is the **complete acquisition CSV**.

For the public 20-file dataset:

```text
Outer grouped CV: 5 folds x 4 test acquisitions

For each outer fold:
    4 CSV  -> TEST
   16 CSV  -> development
             |-- 12 TRAIN
             `--  4 VALIDATION
```

The outer fold assignment is deterministic (`outer_split_seed: 2026`). Inner
train/validation assignment is also fixed per outer fold and does not change across
models or training seeds.

No windows from one acquisition can cross train/validation/test boundaries. The
`NODO1...NODO5` storage directories are ignored by the split algorithm.

For each outer fold:

- scaler: inner-training pre-onset samples only;
- training: inner-training acquisitions only;
- early stopping: inner-validation only;
- decision threshold: selected on inner-validation only;
- outer test: untouched until final evaluation.

## Statistical reporting

For metric `m`, repeated training seeds are first averaged within each outer acquisition
fold:

```text
m_bar_fold(k) = mean over seeds for fold k
```

The paper reports:

```text
mu_CV       = mean of the five fold means
sigma_fold  = sample std of the five fold means
```

Thus `mu_CV ± sigma_fold` describes variability across grouped acquisition folds, not
across Bosch kits. Run-to-run seed variability is stored and reported separately.

The paper should compare **models**, not storage directories. Per-fold outputs are
retained for reproducibility/diagnostics but are not interpreted as physical-node
comparisons.

## Dataset

Use **M. Anedda, “Dataset Smoke Detection,” Mendeley Data, Version 1, 2026**, DOI
`10.17632/48j7mm8k56.1`.

The original archive layout can be kept unchanged:

```text
data/raw/
├── NODO1/
├── NODO2/
├── NODO3/
├── NODO4/
└── NODO5/
```

These folder names are storage/development-kit metadata only.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
python tests/smoke_test.py
```

## 0. Audit the real dataset

```bash
python scripts/dataset_audit.py \
  --data data/raw \
  --strict-public
```

Strict Version-1 checks include:

```text
20 acquisition CSV files
5 storage/development-kit groups (metadata only)
17 documented heater profiles
131370 raw rows
23877 warm-up rows
107493 annotated rows
11762 pre-fire rows
92395 active-fire rows
3336 post-fire/recovery rows
20 physical onsets
19 evaluable events for W=60
```

## 1. Preflight the grouped CV

Run this before any publication training:

```bash
python scripts/cv_preflight.py \
  --data data/raw \
  --output runs/preflight
```

The preflight verifies that:

- each of the 20 acquisitions appears exactly once as outer test;
- five outer folds contain four acquisition files each;
- train/validation/test file sets are disjoint;
- no split is empty or one-class;
- outer-test physical/evaluable totals sum to 20/19 for `W=60`.

It also saves `cv_fold_manifest.csv`.

## 2. Grouped-CV predictive evaluation

Reference FT-64:

```bash
python scripts/run_grouped_cv.py \
  --data data/raw \
  --models ft64 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --device cuda \
  --output runs/grouped_cv_ft64
```

Main model comparison:

```bash
python scripts/run_grouped_cv.py \
  --data data/raw \
  --models ft64 ft128 bilstm_bce bilstm_fl \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --device cuda \
  --output runs/grouped_cv_main
```

FT-32 remains the compact prospective node-edge configuration and may be evaluated
separately if desired.

Main generated files:

```text
cv_fold_manifest.csv
cv_runs.csv
cv_per_fold_summary.csv
cv_per_fold_means.csv
cv_macro_summary.csv
cv_seed_std_by_fold.csv
cv_seed_variability.csv
```

## 3. Prediction-horizon sensitivity

```bash
python scripts/horizon_sweep.py \
  --data data/raw \
  --model ft64 \
  --horizons 5 15 30 60 \
  --seeds 1 2 3 \
  --device cuda
```

`H` is a number of future observations; use timestamp-based lead time for physical-time
interpretation.

## 4. Paper table rows

After the full main comparison:

```bash
python scripts/make_paper_tables.py \
  --cv runs/grouped_cv_main/cv_runs.csv \
  --output runs/grouped_cv_main/model_table_rows.tex
```

The generated rows report `mean ± sigma_fold` for each model.

## 5. Computational complexity

```bash
python scripts/complexity.py --output runs/complexity.csv
```

| Model | Parameters | FP32 weights | MMAC/inference |
|---|---:|---:|---:|
| FT-32 | 20,002 | 78.1 KiB | 1.518 |
| FT-64 | 110,338 | 431.0 KiB | 7.514 |
| FT-128 | 433,666 | 1.65 MiB | 27.038 |

These are analytical quantities, not hardware measurements.

## Results policy

The repository contains no estimated or provisional predictive results. Previous LONO
runs based on the five storage directories are **invalid for the intended experimental
interpretation and must not be reported**. Publication values must be regenerated using
`run_grouped_cv.py`.
