# WF-IoT reviewer-revision implementation map

This file documents the reproducibility changes implemented for the WF-IoT revision.

## Models

- FT-32: compact prospective node-edge configuration;
- FT-64: reference configuration for grouped-CV and horizon-sensitivity analyses;
- FT-128: higher-capacity Transformer comparison;
- BiLSTM-BCE and BiLSTM-FL: recurrent baselines.

Old predictive values from earlier preprocessing/evaluation implementations must not be
mixed with the new physical-onset grouped-CV statistics.

## Acquisition-level evaluation

The 20 CSV files are treated as one pooled multi-profile acquisition campaign.
`NODO1...NODO5` directory labels are Bosch development-kit/storage metadata only.

Implemented safeguards:

- five deterministic outer folds over complete acquisition files;
- every acquisition appears exactly once as outer test;
- four test acquisitions per fold for the 20-file public release;
- inner train/validation split by complete acquisition file;
- no acquisition can contribute windows to more than one split in a fold;
- outer test excluded from scaler fitting, training, early stopping and threshold selection;
- repeated random seeds;
- hierarchical reporting with `sigma_fold`, not `sigma_node`.

## Physical onset and acquisition phases

```text
0            -> warm-up/stabilization
1001         -> pre-fire
1002-1007    -> active fire
1008-1009    -> post-fire/recovery
```

The physical onset is the first and unique `1001 -> 1002` transition. Only pre-onset
samples generate input windows. Active-fire and post-fire samples are excluded.

The public dataset contains 20 physical onset events and 19 evaluable events at `W=60`.

## Heater-profile handling

The campaign documents 17 distinct BME688 heater profiles across 20 acquisitions, with
some profiles repeated. All are retained. No heater-profile ranking, selection, or
per-profile performance comparison is performed. No temporal resampling is applied.
`W` and `H` remain observation counts; timestamp-based lead time is reported in seconds.

## Prediction-horizon sensitivity

FT-64 is the reference model for `H in {5, 15, 30, 60}` under the same grouped
acquisition-level protocol.

## Computational complexity and energy

Analytical complexity is reported for FT-32, FT-64 and FT-128. Raspberry Pi
latency/energy may be reported only after actual hardware measurements.

## Scientific integrity rule

No estimated/provisional grouped-CV, horizon, latency, power, or energy value may be
presented as measured evidence.
