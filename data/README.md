# Dataset

The raw dataset is intentionally not redistributed in this repository.

Use:

**M. Anedda, “Dataset Smoke Detection,” Mendeley Data, Version 1, 2026.**  
DOI: `10.17632/48j7mm8k56.1`

Expected layout after download/extraction:

```text
data/raw/
├── NODO1/
│   ├── acquisition_1.csv
│   └── ...
├── NODO2/
├── NODO3/
├── NODO4/
└── NODO5/
```

The loader searches recursively, so filenames may be preserved. The path must contain
`NODO1` ... `NODO5` (or `NODE1` ... `NODE5`) so the physical node can be inferred.

## Column mapping

The default configuration recognizes common aliases for:

- temperature;
- relative humidity;
- atmospheric pressure;
- gas resistance;
- binary fire/smoke annotation;
- timestamp (optional, but required for physical lead-time reporting).

If the released CSV headers differ, edit only the `schema:` section of
`configs/default.yaml`; do not modify the splitting logic.

## Leakage-control rule

Every CSV is treated as one indivisible acquisition file. Inner validation is selected
at file level before rolling features and sliding windows are used. The outer held-out
node is never used for scaler fitting, early stopping, threshold selection, or any
hyperparameter decision.
