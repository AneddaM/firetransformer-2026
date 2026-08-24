# FireTransformer — WF-IoT 2026 reproducibility revision

Code and experimental protocol for the camera-ready revision of **“Edge-Aware
FireTransformer for Horizon-Based Early Wildfire Warning in IoT Sensor Networks.”**

This repository extends the earlier FireTransformer implementation to address the
WF-IoT review requests on cross-node generalization, prediction-horizon sensitivity,
computational complexity, edge latency, and energy consumption.

## What is implemented

- FireTransformer FT-32, FT-64 and FT-128 with the parameter counts reported in the paper;
- BiLSTM-BCE and BiLSTM-FL baselines;
- full five-fold leave-one-node-out (LONO) outer evaluation;
- outer test node completely excluded from scaler fitting, early stopping and threshold selection;
- inner train/validation split by complete acquisition file before sliding-window generation;
- 20-dimensional feature representation: 4 raw variables plus rolling mean/std/min/max;
- horizon labels with `W=60` and configurable `H`;
- repeated random seeds and per-node / macro summaries;
- AUC-ROC, AUC-PR, precision, recall, F1, specificity, FAR, MDR, event coverage and lead time;
- analytical parameters, weight footprint and MAC count;
- Raspberry Pi / CPU batch-1 latency benchmark;
- externally measured power-log integration for system and dynamic energy per inference;
- synthetic smoke test and GitHub Actions CI;
- BibTeX additions and reviewer-revision map.

## Scientific validation rule

For outer fold `NODOk`:

```text
Development nodes (4)                   Outer test node (1)
      |                                        |
      +-- file-grouped train/validation        +-- final test only
          |                                    |
          +-- fit scaler                       +-- never fit scaler
          +-- early stopping                   +-- never early-stop
          +-- choose threshold                 +-- never choose threshold
```

Because windows overlap heavily, no acquisition file is allowed to contribute windows
to both inner training and inner validation.

## Dataset

Use the public **Dataset Smoke Detection**, Mendeley Data, Version 1 (2026), DOI
`10.17632/48j7mm8k56.1`. The dataset itself is not redistributed here. See
`data/README.md` for the expected directory structure and schema mapping.

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

FT-64 only, three seeds:

```bash
python scripts/run_lono.py \
  --data data/raw \
  --models ft64 \
  --seeds 1 2 3
```

For the complete paper comparison:

```bash
python scripts/run_lono.py \
  --data data/raw \
  --models ft64 ft128 bilstm_bce bilstm_fl \
  --seeds 1 2 3 4 5 6 7 8 9 10
```

Outputs include raw runs, per-node summaries, macro summaries, checkpoints and training histories under `runs/lono/`.

## 2. Horizon sensitivity

```bash
python scripts/horizon_sweep.py \
  --data data/raw \
  --model ft64 \
  --horizons 5 15 30 60 \
  --seeds 1 2 3
```

This directly supports the reviewer-requested analysis of the trade-off between longer
prediction horizons and precursor separability.

## 3. Computational complexity

```bash
python scripts/complexity.py --output runs/complexity.csv
```

With the default architecture, expected **analytical** parameter counts are:

| Model | Parameters | FP32 weights |
|---|---:|---:|
| FT-32 | 20,002 | 78.1 KiB |
| FT-64 | 110,338 | 431.0 KiB |
| FT-128 | 433,666 | 1.65 MiB |

The MAC values are analytical estimates under the counting convention documented in
`src/fire_transformer/complexity.py`; they are not hardware measurements.

## 4. Raspberry Pi / edge latency

After producing a checkpoint:

```bash
python scripts/edge_benchmark.py \
  --checkpoint runs/lono/ft64_NODO5_seed1.pth \
  --device cpu \
  --threads 1 \
  --warmup 500 \
  --iterations 5000 \
  --output runs/pi5_latency.json
```

Report at least median and p95 latency, batch size 1, thread count and hardware/software environment.

## 5. Measured energy per inference

While an external USB-C power meter or laboratory supply records timestamped input
power, run:

```bash
python scripts/power_benchmark_loop.py \
  --checkpoint runs/lono/ft64_NODO5_seed1.pth \
  --seconds 60 \
  --output runs/power_interval.json
```

Then integrate the measured log:

```bash
python scripts/energy_from_powerlog.py \
  --csv power_logs/pi5.csv \
  --interval-json runs/power_interval.json \
  --timestamp-col timestamp_s \
  --power-col power_w \
  --idle-w <MEASURED_IDLE_POWER> \
  --output runs/pi5_energy.json
```

This reports both system energy and dynamic energy per inference. Never infer model
energy from the Raspberry Pi power-supply rating.

## 6. Paper table rows

After the measured full LONO run:

```bash
python scripts/make_paper_tables.py
```

The generated LaTeX rows can be copied into the camera-ready results table after manual verification.


## Repository structure

```text
.
├── .github/workflows/smoke-test.yml
├── configs/default.yaml
├── data/README.md
├── docs/
│   ├── energy_protocol.md
│   ├── results_policy.md
│   └── reviewer_revision_map.md
├── paper/
│   ├── README.md
│   └── references_wfiot_revision.bib
├── results/.gitkeep
├── scripts/
│   ├── complexity.py
│   ├── edge_benchmark.py
│   ├── energy_from_powerlog.py
│   ├── horizon_sweep.py
│   ├── make_paper_tables.py
│   ├── make_synthetic_dataset.py
│   ├── power_benchmark_loop.py
│   └── run_lono.py
├── src/fire_transformer/
│   ├── augmentation.py
│   ├── complexity.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── lead_time.py
│   ├── losses.py
│   ├── model.py
│   ├── training.py
│   └── utils.py
├── tests/smoke_test.py
├── CITATION.cff
├── GITHUB_UPLOAD.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Results policy

This repository intentionally contains no estimated or provisional experimental
results. Only analytical architecture quantities are pre-populated. LONO, horizon,
latency, power and energy values must come from actual executions / measurements.

## Citation

Use `CITATION.cff` for the software citation and `paper/references_wfiot_revision.bib`
for paper-related references.
