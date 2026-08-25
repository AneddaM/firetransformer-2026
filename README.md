# FireTransformer — WF-IoT 2026

Code and experimental protocol for the camera-ready revision of
**“Edge-Aware FireTransformer for Horizon-Based Early Wildfire Warning in IoT Sensor Networks.”**

This repository addresses the WF-IoT review requests on cross-node generalization,
prediction-horizon sensitivity, computational complexity, edge latency, and energy
consumption.

## What is implemented

- FireTransformer FT-64 and FT-128, i.e. the Transformer configurations considered in the WF-IoT evaluation;
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
- AUC-ROC, AUC-PR, precision, recall, F1, specificity, FAR and MDR;
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
`(acquisition_file, onset_timestamp)`, so acquisitions whose timestamps restart from
zero cannot be accidentally merged.

For a covered event, lead time is computed from the earliest correctly warning window:

```text
lead_time = onset_timestamp - earliest_warning_window_end
```

## Dataset

Use the public **Dataset Smoke Detection**, Mendeley Data, Version 1 (2026), DOI
`10.17632/48j7mm8k56.1`.

The dataset itself is not redistributed here. See `data/README.md` for the expected
directory structure and schema mapping.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

Run the included smoke test:

```bash
python tests/smoke_test.py
```

## 1. Full LONO evaluation

Quick FT-64 run with three seeds:

```bash
python scripts/run_lono.py \
  --data data/raw \
  --models ft64 \
  --seeds 1 2 3
```

Complete paper comparison:

```bash
python scripts/run_lono.py \
  --data data/raw \
  --models ft64 ft128 bilstm_bce bilstm_fl \
  --seeds 1 2 3 4 5 6 7 8 9 10
```

Outputs include raw runs, per-node summaries, macro summaries, checkpoints and
training histories under `runs/lono/`.

### Statistical aggregation convention

For each held-out node, metrics are first averaged across the independent training
seeds.

Let \(m_{k,r}\) denote a metric obtained for held-out node \(k\) and random seed \(r\).
The node-wise estimate is

\[
\bar m_k = \frac{1}{R}\sum_{r=1}^{R}m_{k,r}.
\]

The reported macro mean is then computed across the five node-wise means,

\[
\mu_{\mathrm{macro}}
=
\frac{1}{5}\sum_{k=1}^{5}\bar m_k,
\]

while \(\sigma_{\mathrm{node}}\) is the sample standard deviation of the five
node-wise means.

Therefore, the paper reports

\[
\mu_{\mathrm{macro}} \pm \sigma_{\mathrm{node}},
\]

which quantifies cross-node variability. Run-to-run variability across random seeds
is computed separately and is not pooled with between-node variability.

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
| FT-64 | 110,338 | 431.0 KiB | 7.514 |
| FT-128 | 433,666 | 1.65 MiB | 27.038 |

These are analytical quantities, not hardware measurements.

## 4. Raspberry Pi / edge latency

```bash
python scripts/edge_benchmark.py \
  --checkpoint runs/lono/ft64_NODO5_seed1.pth \
  --device cpu \
  --threads 1 \
  --warmup 500 \
  --iterations 5000 \
  --output runs/pi5_latency.json
```

## 5. Measured energy per inference

```bash
python scripts/power_benchmark_loop.py \
  --checkpoint runs/lono/ft64_NODO5_seed1.pth \
  --seconds 60 \
  --output runs/power_interval.json
```

Then:

```bash
python scripts/energy_from_powerlog.py \
  --csv power_logs/pi5.csv \
  --interval-json runs/power_interval.json \
  --timestamp-col timestamp_s \
  --power-col power_w \
  --idle-w <MEASURED_IDLE_POWER> \
  --output runs/pi5_energy.json
```

## 6. Paper table rows

```bash
python scripts/make_paper_tables.py
```

## Results policy

This repository intentionally contains no estimated or provisional experimental
results. Only analytical architecture quantities are pre-populated. LONO, horizon,
latency, power and energy values must come from actual executions or measurements.


