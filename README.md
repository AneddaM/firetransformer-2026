# FireTransformer — WF-IoT 2026

Code and experimental protocol for the camera-ready revision of
**“Edge-Aware FireTransformer for Horizon-Based Early Wildfire Warning in IoT Sensor Networks.”**

This repository addresses the WF-IoT review requests on cross-node generalization,
prediction-horizon sensitivity, computational complexity, edge latency, and energy consumption.

## What is implemented

- FireTransformer FT-32, FT-64, and FT-128;
- FT-64 as the reference configuration for full LONO and prediction-horizon sensitivity;
- FT-32 retained as the compact prospective node-edge configuration;
- BiLSTM-BCE and BiLSTM-FL baselines;
- full five-fold leave-one-node-out (LONO) outer evaluation;
- outer test node completely excluded from scaler fitting, training, early stopping, and threshold selection;
- inner train/validation split by complete acquisition file before sliding-window generation;
- physical fire onset defined strictly by a `0 -> 1` transition of the fire annotation;
- early-warning samples generated only while the current observed fire state is `0`;
- 20-dimensional feature representation: 4 raw variables plus rolling mean/std/min/max;
- horizon labels with `W=60` and configurable `H`;
- repeated random seeds and per-node / macro summaries;
- event-aware coverage and lead time, with events disambiguated by acquisition-file identity;
- analytical parameter count, weight footprint and MAC count;
- Raspberry Pi / CPU batch-1 latency benchmark;
- externally measured power-log integration for system and dynamic energy per inference;
- synthetic smoke test and GitHub Actions CI.

## Scientific validation rule

For outer fold `NODOk`:

```text
Development nodes (4)                   Outer test node (1)
      |                                        |
      +-- file-grouped train/validation        +-- final test only
          |                                    |
          +-- fit scaler                       +-- never fit scaler
          +-- training                         +-- never train
          +-- early stopping                   +-- never early-stop
          +-- choose threshold                 +-- never choose threshold
```

Because windows overlap heavily, no acquisition file is allowed to contribute windows
to both inner training and inner validation.

## Physical onset and early-warning target

A physical onset is defined only by a transition

```text
fire[t-1] = 0  and  fire[t] = 1
```

within the same acquisition file.

A window ending at sample `t` is used for early-warning classification only if
`fire[t] == 0`. Its target is positive when at least one physical onset occurs in

```text
(t, t + H]
```

and negative otherwise.

Coverage and lead time are computed per physical event. Event identity is the pair
`(acquisition_file, onset_timestamp)`.

For a covered event:

```text
lead_time = onset_timestamp - earliest_warning_window_end
```

## Dataset

Use the public **Dataset Smoke Detection**, Mendeley Data, Version 1 (2026), DOI
`10.17632/48j7mm8k56.1`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
python tests/smoke_test.py
```

## 1. Full LONO evaluation

Reference FT-64 run:

```bash
python scripts/run_lono.py \
  --data data/raw \
  --models ft64 \
  --seeds 1 2 3 4 5 6 7 8 9 10
```

Main paper comparison:

```bash
python scripts/run_lono.py \
  --data data/raw \
  --models ft64 ft128 bilstm_bce bilstm_fl \
  --seeds 1 2 3 4 5 6 7 8 9 10
```

FT-32 can also be evaluated under the same LONO implementation when desired:

```bash
python scripts/run_lono.py \
  --data data/raw \
  --models ft32 \
  --seeds 1 2 3
```

FT-32 is retained primarily as the compact prospective node-edge configuration; FT-64
remains the reference model used for the full LONO and horizon-sensitivity analyses
reported in the WF-IoT revision.

### Statistical aggregation convention

For each held-out node, metrics are first averaged across independent training seeds.
The macro mean is then computed across the five node-wise means, while
`σ_node` is their sample standard deviation. Seed-level variability is reported separately.

## 2. Horizon sensitivity

```bash
python scripts/horizon_sweep.py \
  --data data/raw \
  --model ft64 \
  --horizons 5 15 30 60 \
  --seeds 1 2 3
```

## 3. Computational complexity

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

This repository intentionally contains no estimated or provisional experimental
results. Only analytical architecture quantities are pre-populated. LONO, horizon,
latency, power, and energy values must come from actual executions or measurements.
