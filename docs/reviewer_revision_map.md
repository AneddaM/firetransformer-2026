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
- outer-node exclusion from scaler fitting, training, early stopping, and threshold calibration;
- inner train/validation split by complete acquisition file;
- repeated random seeds;
- hierarchical macro reporting.

## Physical onset, coverage, and lead time

A physical onset is defined strictly by a `0 -> 1` transition within one acquisition file.
Windows ending while fire is already active are excluded.
Event identity is `(acquisition_file, onset_timestamp)`.
Coverage is event-level and lead time uses the earliest correctly warning window.

## Prediction-horizon sensitivity

FT-64 is the reference model for `H in {5, 15, 30, 60}` under the same leakage-controlled protocol.

## Computational complexity and energy

Analytical complexity is reported for FT-32, FT-64, and FT-128.
Measured Raspberry Pi latency/energy must be reported only after actual hardware measurements.

## Scientific integrity rule

No estimated or provisional LONO, horizon, latency, power, or energy result may be
presented as measured evidence.
