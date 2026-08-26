# FireTransformer — WF-IoT 2026

Code and experimental protocol for the camera-ready revision of
**“Edge-Aware FireTransformer for Horizon-Based Early Wildfire Warning in IoT Sensor Networks.”**

This repository implements the physical-onset formulation used in the revised paper.
It retains all 17 BME688 heater profiles, performs no temporal resampling, and uses
sample-based windows/horizons while reporting realized warning anticipation from the
recorded timestamps in seconds.

## What is implemented

- FireTransformer FT-32, FT-64, and FT-128;
- FT-64 as the reference configuration for full LONO and prediction-horizon sensitivity;
- FT-32 retained as the compact prospective node-edge configuration;
- BiLSTM-BCE and BiLSTM-FL baselines;
- full five-fold leave-one-node-out (LONO) outer evaluation;
- outer test node completely excluded from scaler fitting, training, early stopping,
  threshold selection, and hyperparameter selection;
- inner train/validation split by complete acquisition file before window generation;
- exact public-dataset state decoding from `label_tag`;
- physical fire onset defined by the first and unique `1001 -> 1002` transition;
- model input windows generated exclusively from pre-onset observations;
- active-fire and post-fire/recovery observations excluded from model inputs;
- MinMaxScaler fitted only on eligible pre-onset samples from inner-training files;
- explicit physical-event and evaluable-event bookkeeping;
- 20-dimensional feature representation: 4 raw variables plus rolling mean/std/min/max;
- sample-based horizon labels with `W=60` and configurable `H`;
- timestamp conversion from `timestamp_since_poweron` milliseconds to seconds;
- repeated random seeds and hierarchical per-node / macro summaries;
- event-aware coverage and lead time, with events disambiguated by acquisition-file identity;
- analytical parameter count, weight footprint, and MAC count;
- Raspberry Pi / CPU batch-1 latency benchmark;
- externally measured power-log integration for system and dynamic energy per inference;
- state-aware synthetic smoke test and GitHub Actions CI;
- public-dataset audit script that must be run before training.

## Dataset state semantics

The public dataset contains **131,370 raw samples**. The revised pipeline interprets
`label_tag` as follows:

```text
label_tag = 0
    sensor warm-up / stabilization
    EXCLUDED

label_tag = 1001
    valid pre-fire state

1001 -> 1002
    physical fire onset

label_tag = 1002...1007
    active-fire acquisition

label_tag = 1008, 1009
    post-fire / recovery acquisition
```

After removal of the 23,877 warm-up samples, **107,493 annotated samples** remain:

- 11,762 pre-fire samples (`1001`);
- 92,395 active-fire samples (`1002`–`1007`);
- 3,336 post-fire/recovery samples (`1008`–`1009`).

These counts describe acquisition states. They are **not** the early-warning window
class counts. Early-warning windows are generated only from the pre-onset phase.

Every one of the 20 acquisition files contains exactly one physical `1001 -> 1002`
onset. With `W=60`, 19 physical onset events are evaluable. One acquisition contains
only 20 valid pre-onset observations and therefore cannot produce a complete input
window; the physical event remains tracked but is excluded from the event-coverage
denominator for `W=60`.

See `docs/state_semantics.md` for the full state machine and rationale.

## Heater-profile policy

All **17 BME688 heater profiles** are retained. No heater-profile selection and no
temporal resampling are applied in this revision. Because the heater profiles have
different acquisition-cycle timing, `W` and `H` are defined in **observations**, not in
fixed physical seconds. Heater-profile identity is metadata only and is not used as an
input feature.

Realized event lead time is calculated from:

```text
timestamp_since_poweron / 1000.0
```

so that lead-time results are reported in seconds despite heterogeneous acquisition
cadences.

## Scientific validation rule

For outer fold `NODOk`:

```text
Development nodes (4)                   Outer test node (1)
      |                                        |
      +-- file-grouped train/validation        +-- final test only
          |                                    |
          +-- pre-onset-only scaler            +-- never fit scaler
          +-- training                         +-- never train
          +-- early stopping                   +-- never early-stop
          +-- choose threshold                 +-- never choose threshold
```

Because windows overlap heavily, no acquisition file is allowed to contribute windows
to both inner training and inner validation.

## Physical-onset early-warning target

For each acquisition, let `o` denote the physical onset index after warm-up removal.
A `W`-sample input window ending at sample `t` is eligible only when:

```text
W - 1 <= t < o
```

Its target is:

```text
y_t = 1  if  0 < o - t <= H
y_t = 0  if      o - t > H
```

No active-fire or post-fire sample is ever included in an input window.

Coverage is computed over **evaluable physical onset events**, where event identity is:

```text
(acquisition_file, onset_timestamp)
```

For a covered event:

```text
lead_time = onset_timestamp - earliest_correct_warning_window_end
```

## Dataset

Use the public **Dataset Smoke Detection**, Mendeley Data, Version 1 (2026), DOI
`10.17632/48j7mm8k56.1`.

Expected layout:

```text
data/raw/
├── NODO1/
├── NODO2/
├── NODO3/
├── NODO4/
└── NODO5/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
python tests/smoke_test.py
```

## 0. Audit the real dataset before training

Run this first:

```bash
python scripts/dataset_audit.py \
  --data data/raw \
  --strict-public
```

For Mendeley Data Version 1, the strict check must confirm:

```text
20 CSV files
5 nodes
131370 raw rows
23877 warm-up rows
107493 annotated rows
11762 pre-fire rows
92395 active-fire rows
3336 post-fire/recovery rows
20 physical onsets
19 evaluable events for W=60
```

Do not start publication runs if the strict audit fails.

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

FT-32 can also be evaluated under the same LONO implementation:

```bash
python scripts/run_lono.py \
  --data data/raw \
  --models ft32 \
  --seeds 1 2 3
```

FT-32 is retained primarily as the compact prospective node-edge configuration; FT-64
remains the reference model used for full LONO and horizon-sensitivity analyses.

### Statistical aggregation convention

For each held-out node, metrics are first averaged across independent training seeds.
The macro mean is then computed across the five node-wise means, while
`sigma_node` is their sample standard deviation. Seed-level variability is reported
separately.

The event denominator is not silently inferred from existing positive windows. The
pipeline explicitly stores both physical and evaluable events. With `W=60`, NODO1–4
contain 4 physical / 4 evaluable events each, while NODO5 contains 4 physical / 3
evaluable events.

## 2. Horizon sensitivity

```bash
python scripts/horizon_sweep.py \
  --data data/raw \
  --model ft64 \
  --horizons 5 15 30 60 \
  --seeds 1 2 3
```

`H` is a number of future observations. Use `lead_mean_s` for a common physical-time
interpretation across heater profiles.

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
