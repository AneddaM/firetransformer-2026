# Results policy

This public repository intentionally contains **no fabricated, expected, or provisional
experimental results**.

Safe to report before new experiments:

- model parameter counts;
- FP32 / theoretical INT8 weight footprint;
- analytical MAC estimate under the stated counting convention;
- public-dataset state counts verified directly from Mendeley Data Version 1.

Must be re-measured with the physical-onset pipeline before publication:

- five-fold LONO AUC, precision, recall, F1, coverage and lead time;
- random-seed stability;
- horizon-sensitivity metrics;
- Raspberry Pi latency;
- active and idle power;
- system and dynamic energy per inference.

Old predictive values generated using an earlier target/preprocessing implementation
must not be mixed with the new results.

## Event definition

A physical event is the first and unique `1001 -> 1002` annotation transition within
one acquisition file. Event identity is:

```text
(acquisition_file, onset_timestamp)
```

Only pre-onset observations are used as model inputs. Active-fire and post-fire states
are excluded from window generation.

A physical event is evaluable for a given `W` only when at least one complete pre-onset
input window exists. With `W=60`, the public dataset contains 20 physical onsets and 19
evaluable events.

Coverage is the fraction of **evaluable** physical onset events for which at least one
positive prediction is produced on a positive-target warning window. Lead time uses the
earliest correctly warning positive-target window for each covered event.

## Statistical reporting

LONO macro statistics are computed hierarchically.

1. Metrics are first averaged across random seeds separately for each held-out node.
2. The macro mean is computed from the five node-wise means.
3. The standard deviation reported together with the macro mean is the between-node
   sample standard deviation (`sigma_node`).
4. Random-seed variability is reported separately and is not pooled with between-node
   variability.

The paper `mean ± std` must therefore not be computed directly over all node-seed runs.
