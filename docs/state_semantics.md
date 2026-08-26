# Public-dataset state semantics and physical-onset task

This document fixes the acquisition-state interpretation used by the WF-IoT revision.
The same definitions are implemented in `configs/default.yaml`,
`src/fire_transformer/data.py`, and the paper methodology.

## 1. Raw state labels

| `label_tag` | Interpretation in this revision | Used as model input? |
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

## 2. Acquisition state progression

The validated state progression is:

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
Pre-fire state is not allowed to reappear after the physical onset. If post-fire state
appears, active fire is not allowed to reappear afterwards. Violations raise an error
rather than being silently coerced.

## 3. Sample counts verified for Mendeley Data Version 1

```text
Raw samples                         131,370
Warm-up label 0                      23,877
Annotated samples                   107,493

Pre-fire label 1001                  11,762
Active-fire labels 1002-1007         92,395
Post-fire labels 1008-1009            3,336
```

The previously reported 15,098 non-fire annotations equal:

```text
11,762 pre-fire + 3,323 label-1008 + 13 label-1009 = 15,098
```

For the early-warning task, post-fire observations are **not** ordinary negative
samples. They occur after the event that the model is intended to anticipate and are
therefore excluded from model inputs.

## 4. Early-warning window definition

Let `o` be the physical onset index after warm-up removal. A window of length `W`
ending at `t` is eligible only if:

```text
W - 1 <= t < o
```

Its target is:

```text
y_t = 1   if   0 < o - t <= H
y_t = 0   if       o - t > H
```

No active-fire or post-fire sample is used in the input tensor.

## 5. Physical versus evaluable events

A physical event is identified by:

```text
(acquisition_file, physical_onset_timestamp)
```

A physical event is evaluable for a given `W` only if at least one complete pre-onset
window exists. With the reference `W=60`:

```text
Physical onset events       20
Evaluable onset events      19
Non-evaluable events         1
```

The non-evaluable event belongs to `NODO5/Nodo_5_profilo_1.csv`, which contains 20
valid pre-onset observations. Event-level coverage uses 19 as the global set of
evaluable events for `W=60`; the physical count of 20 is still stored and reported.

## 6. Heater profiles and physical time

The public archive uses 17 BME688 heater profiles. Each profile has different
acquisition-cycle timing, so the number of samples observed during a similar physical
pre-fire duration can vary substantially.

This revision therefore:

- retains all 17 heater profiles;
- does not select one profile;
- does not temporally resample the acquisitions;
- does not use heater-profile identity as an ML feature;
- interprets `W` and `H` as numbers of observations;
- converts `timestamp_since_poweron` from milliseconds to seconds;
- reports timestamp-based lead time for a common physical-time interpretation.

## 7. Scaling rule

The MinMax scaler is fitted only on pre-onset samples from **eligible inner-training
acquisition files**. It does not use:

- the held-out outer node;
- inner-validation files;
- active-fire samples;
- post-fire/recovery samples;
- non-evaluable training acquisitions that cannot generate a `W`-sample input window.

This aligns preprocessing with the population actually used by the anticipatory task.
