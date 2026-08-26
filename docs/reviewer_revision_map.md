# WF-IoT reviewer-revision implementation map

This file documents the reproducibility changes implemented for the WF-IoT camera-ready revision.

## Models

The repository contains three FireTransformer configurations:

- FT-32: compact prospective node-edge configuration;
- FT-64: reference configuration for full LONO and horizon-sensitivity analyses;
- FT-128: higher-capacity Transformer comparison.

The recurrent baselines are BiLSTM-BCE and BiLSTM-FL.

FT-32 remains part of the architecture and computational-complexity characterization.
Its earlier predictive evaluation used a reduced training budget and a single random
seed, so those earlier values must not be mixed with the new full-LONO statistics.

## Cross-node evaluation

- full five-fold leave-one-node-out outer evaluation;
- outer-node exclusion from scaler fitting, training, early stopping, threshold calibration, and hyperparameter selection;
- inner train/validation split by complete acquisition file;
- scaler fitted only on eligible pre-onset inner-training samples;
- repeated random seeds;
- hierarchical macro reporting.

## Physical onset, acquisition phases, coverage, and lead time

Raw annotation states are interpreted as:

```text
0            -> warm-up/stabilization
1001         -> pre-fire
1002-1007    -> active fire
1008-1009    -> post-fire/recovery
```

The physical fire onset is the first and unique `1001 -> 1002` transition in each
acquisition. Only pre-onset samples are used to generate early-warning input windows.
Active-fire and post-fire/recovery samples are excluded from model inputs.

The public dataset contains 20 physical onset events. With `W=60`, 19 are evaluable;
one NODO5 acquisition contains only 20 pre-onset observations.

Event identity is `(acquisition_file, onset_timestamp)`. Coverage is computed over the
explicit set of evaluable physical events, and lead time uses the earliest correctly
warning positive-target window.

## Heater-profile handling

All 17 BME688 heater profiles are retained. No heater-profile selection and no temporal
resampling are performed. `W` and `H` are sample-based quantities. Realized lead time
is computed from `timestamp_since_poweron / 1000.0` and reported in seconds.

## Prediction-horizon sensitivity

FT-64 is the reference model for `H in {5, 15, 30, 60}` under the same leakage-controlled protocol.

## Computational complexity and energy

Analytical complexity is reported for FT-32, FT-64, and FT-128.
Measured Raspberry Pi latency/energy must be reported only after actual hardware measurements.

## Scientific integrity rule

No estimated or provisional LONO, horizon, latency, power, or energy result may be
presented as measured evidence. Old predictive values generated with the previous
state/target definition must not be reused after the physical-onset preprocessing change.
