# Results policy

This public repository intentionally contains **no fabricated, expected, or provisional
experimental results**.

Safe to report before new experiments:

- model parameter counts;
- FP32 / theoretical INT8 weight footprint;
- analytical MAC estimate under the stated counting convention.

Must be measured before publication:

- five-fold LONO AUC, precision, recall, F1, coverage and lead time;
- horizon-sensitivity metrics;
- Raspberry Pi latency;
- active and idle power;
- system and dynamic energy per inference.

The scripts write measured outputs under `runs/` (ignored by Git until intentionally
reviewed and committed).
