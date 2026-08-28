# Public-dataset state semantics and physical-onset task

This document fixes the acquisition-state interpretation used by the WF-IoT revision.
The same definitions are implemented in `configs/default.yaml`,
`src/fire_transformer/data.py`, and the paper methodology.

## 1. Experimental unit

The statistical unit is the **complete acquisition CSV file**. The public archive stores
20 acquisitions under five `NODO1...NODO5` directories associated with Bosch development
kits used during acquisition. Those directory labels are storage/traceability metadata
only and are not interpreted as five independent sensing nodes or experimental domains.

The campaign documents 17 distinct BME688 heater profiles across the 20 acquisitions,
with some profiles reused. All acquisitions are retained; no heater-profile selection or
heater-profile comparison is performed.

## 2. Raw state labels

| `label_tag` | Interpretation | Used as model input? |
|---:|---|---|
| 0 | initial BME688 sensor warm-up / stabilization | No |
| 1001 | valid pre-fire acquisition | Yes, before onset only |
| 1002–1007 | active-fire acquisition | No |
| 1008–1009 | post-fire / recovery acquisition | No |

The physical fire onset is the first and unique transition:

```text
1001 -> 1002
```

All 20 public acquisition files contain exactly one such transition.

## 3. Acquisition state progression

```text
WARM-UP
   |
   v
PRE-FIRE
   |
   | 1001 -> 1002
   v
ACTIVE FIRE
   |
   v
OPTIONAL POST-FIRE / RECOVERY
```

Warm-up is an initial contiguous phase and is removed before feature generation.
Pre-fire state may not reappear after physical onset. If post-fire state appears, active
fire may not reappear afterwards. Violations raise an error.

## 4. Counts verified for Mendeley Data Version 1

```text
Raw samples                         131,370
Warm-up label 0                      23,877
Annotated samples                   107,493

Pre-fire label 1001                  11,762
Active-fire labels 1002-1007         92,395
Post-fire labels 1008-1009            3,336
```

The 15,098 non-fire annotations in the raw acquisition labels consist of 11,762 pre-fire
and 3,336 post-fire/recovery records. For the early-warning task, post-fire observations
are not ordinary negative samples because they occur after the event to be anticipated.
They are therefore excluded from model inputs.

## 5. Early-warning window definition

Let `o` be the physical onset index after warm-up removal. A window of length `W` ending
at `t` is eligible only if:

```text
W - 1 <= t < o
```

Its target is:

```text
y_t = 1   if   0 < o - t <= H
y_t = 0   if       o - t > H
```

No active-fire or post-fire observation is used in an input tensor.

## 6. Physical versus evaluable events

A physical event is identified by:

```text
(acquisition_file, physical_onset_timestamp)
```

A physical event is evaluable for a given `W` only if at least one complete pre-onset
window exists. With `W=60`:

```text
Physical onset events       20
Evaluable onset events      19
Non-evaluable events         1
```

The non-evaluable acquisition has 20 valid pre-onset observations. Its physical event is
tracked but excluded from the event-coverage denominator at `W=60`.

## 7. Heater profiles and physical time

This revision:

- retains all 17 documented heater profiles;
- uses all 20 acquisition files together;
- performs no heater-profile ranking or selection;
- does not use heater-profile identity as an ML feature;
- does not temporally resample acquisitions;
- interprets `W` and `H` as numbers of observations;
- converts `timestamp_since_poweron` from milliseconds to seconds;
- reports timestamp-based lead time for physical-time interpretation.

## 8. Acquisition-level grouped CV

Five deterministic outer folds partition the 20 acquisition files. Each acquisition
appears exactly once as outer test. For each outer fold, the remaining 16 files are split
at file level into 12 inner-training and 4 inner-validation acquisitions.

The `NODO1...NODO5` storage directories do not enter fold construction.

## 9. Scaling rule

The MinMax scaler is fitted only on eligible pre-onset samples from inner-training
acquisition files. It does not use:

- outer-test acquisitions;
- inner-validation acquisitions;
- active-fire observations;
- post-fire/recovery observations;
- non-evaluable training acquisitions that cannot generate a `W`-sample input window.
