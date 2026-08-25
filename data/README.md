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

## Leakage-control rule

Every CSV is treated as one indivisible acquisition file. Inner validation is selected
at file level before rolling features and sliding windows are generated. The outer
held-out node is never used for scaler fitting, training, early stopping, or threshold
selection.

## Physical onset rule

A physical fire onset is defined strictly as a `0 -> 1` transition of the binary fire
annotation inside one acquisition file.

A warning window ending at sample `t` is included only when the current observed state
is `fire[t] == 0`. It is labeled positive if a physical onset occurs within the next
`H` samples.

Coverage and lead time are computed per physical onset. Events are identified using
both acquisition-file identity and onset timestamp, so independent files with equal or
restarted timestamps remain separate events.
