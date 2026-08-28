# Results policy

This repository intentionally contains **no fabricated, expected, or provisional
experimental results**.

Safe to report before new experiments:

- model parameter counts;
- FP32 / theoretical INT8 weight footprint;
- analytical MAC estimate under the stated counting convention;
- public-dataset state counts verified directly from Mendeley Data Version 1.

Must be measured with the acquisition-level physical-onset pipeline before publication:

- grouped five-fold CV AUC, precision, recall, F1, coverage and lead time;
- random-seed stability;
- horizon-sensitivity metrics;
- Raspberry Pi latency;
- active and idle power;
- system and dynamic energy per inference.

Previous leave-one-`NODO` results are invalid for the intended experimental
interpretation because the five directory groups are acquisition-storage/development-kit
partitions rather than independent evaluation nodes. They must not be reported.

## Event definition

A physical event is the first and unique `1001 -> 1002` transition within one acquisition
file. Event identity is `(acquisition_file, onset_timestamp)`. Only pre-onset observations
are used as model inputs.

With `W=60`, the public dataset contains 20 physical onsets and 19 evaluable events.
Coverage is the fraction of evaluable physical onset events receiving at least one
correct positive-target warning. Lead time uses the earliest correctly warning
positive-target window and is conditional on covered events.

## Statistical reporting

Grouped-CV statistics are computed hierarchically:

1. metrics are averaged across random seeds separately within each outer acquisition fold;
2. the macro CV mean is computed from the five fold-wise means;
3. the accompanying sample standard deviation is `sigma_fold` across those five fold means;
4. random-seed variability is reported separately and is not pooled with `sigma_fold`.

The paper should compare models using these aggregate grouped-CV statistics. Individual
fold values are retained for reproducibility but are not interpreted as physical-node or
heater-profile comparisons.
