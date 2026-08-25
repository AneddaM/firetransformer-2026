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

The scripts write measured outputs under `runs/`, which is ignored by Git until results
are intentionally reviewed and selected for release.

## Event definition

A physical event is defined only by a `0 -> 1` transition of the fire annotation within
one acquisition file.

Early-warning windows are generated only while the current fire state is `0`. For a
window ending at `t`, the target is positive when a physical onset occurs in `(t, t+H]`.

Coverage and lead time are event-level metrics. Events are identified by
`(acquisition_file, onset_timestamp)` to prevent unrelated acquisitions with the same
timestamp values from being merged.

For each covered event, lead time uses the **earliest correctly warning window** before
the physical onset.

## Statistical reporting

LONO macro statistics are computed hierarchically.

1. Metrics are first averaged across random seeds separately for each held-out node.
2. The macro mean is computed from the five node-wise means.
3. The standard deviation reported together with the macro mean is the between-node
   sample standard deviation (`sigma_node`).
4. Random-seed variability is reported separately and is not pooled with between-node
   variability.

The repository must therefore not compute the paper `mean ± std` directly over all
node-seed runs.
