from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y, p, objective="f1"):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_s = 0.5, -np.inf

    for t in thresholds:
        pred = (p >= t).astype(int)

        if objective == "f1":
            score = f1_score(y, pred, zero_division=0)
        elif objective == "recall":
            score = recall_score(y, pred, zero_division=0)
        else:
            raise ValueError("threshold objective must be 'f1' or 'recall'")

        if score > best_s:
            best_s, best_t = score, float(t)

    return best_t, best_s


def _normalize_event_key(event):
    file_id, onset = event
    return (str(file_id), round(float(onset), 6))


def _normalize_events(events):
    if events is None:
        return None
    return tuple(dict.fromkeys(_normalize_event_key(event) for event in events))


def _event_groups(onset_ts, file_ids=None):
    """Group positive-target warning windows by physical event.

    Event identity is `(acquisition_file, onset_timestamp)`, so independent
    acquisitions with equal/restarted timestamps cannot be merged.
    """
    onset_ts = np.asarray(onset_ts, dtype=np.float64)
    n = len(onset_ts)

    if file_ids is None:
        file_ids = np.full(n, "__single_acquisition__", dtype=object)
    else:
        file_ids = np.asarray(file_ids, dtype=object)
        if len(file_ids) != n:
            raise ValueError("file_ids and onset_ts must have the same length")

    groups = {}
    for index in np.flatnonzero(np.isfinite(onset_ts)):
        key = (
            str(file_ids[index]),
            round(float(onset_ts[index]), 6),
        )
        groups.setdefault(key, []).append(int(index))

    return groups


def event_coverage(
    pred,
    onset_ts,
    file_ids=None,
    evaluable_events=None,
):
    """Fraction of evaluable physical onsets receiving a correct warning."""
    groups = _event_groups(onset_ts, file_ids)
    expected = _normalize_events(evaluable_events)
    if expected is None:
        expected = tuple(groups.keys())

    if not expected:
        return np.nan

    pred = np.asarray(pred)
    covered = 0

    for event in expected:
        indices = groups.get(event, [])
        if not indices:
            continue
        idx = np.asarray(indices, dtype=int)
        if np.any(pred[idx] == 1):
            covered += 1

    return covered / len(expected)


def lead_time_stats(
    pred,
    y,
    window_end_ts,
    onset_ts,
    file_ids=None,
):
    """Compute one timestamp-based lead-time value per covered physical onset.

    For each event, lead time uses the earliest correctly warning *positive-target*
    window before the physical 1001->1002 onset.
    """
    pred = np.asarray(pred)
    y = np.asarray(y)
    window_end_ts = np.asarray(window_end_ts, dtype=np.float64)
    onset_ts = np.asarray(onset_ts, dtype=np.float64)

    groups = _event_groups(onset_ts, file_ids)
    event_leads = []

    for indices in groups.values():
        idx = np.asarray(indices, dtype=int)
        correct_warning = idx[(pred[idx] == 1) & (y[idx] == 1)]

        if correct_warning.size == 0:
            continue

        event_onset = float(onset_ts[correct_warning[0]])
        earliest_warning = float(np.min(window_end_ts[correct_warning]))
        event_leads.append(event_onset - earliest_warning)

    if not event_leads:
        return {
            "lead_mean_s": np.nan,
            "lead_median_s": np.nan,
            "lead_p25_s": np.nan,
            "lead_p75_s": np.nan,
        }

    lead = np.asarray(event_leads, dtype=np.float64)
    return {
        "lead_mean_s": float(np.mean(lead)),
        "lead_median_s": float(np.median(lead)),
        "lead_p25_s": float(np.percentile(lead, 25)),
        "lead_p75_s": float(np.percentile(lead, 75)),
    }


def compute_metrics(
    y,
    p,
    threshold,
    window_end_ts=None,
    onset_ts=None,
    file_ids=None,
    physical_events=None,
    evaluable_events=None,
):
    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc_roc": (
            float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        ),
        "auc_pr": (
            float(average_precision_score(y, p))
            if len(np.unique(y)) > 1
            else np.nan
        ),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "missed_detection_rate": float(fn / (fn + tp)) if (fn + tp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if onset_ts is not None:
        groups = _event_groups(onset_ts, file_ids)

        expected = _normalize_events(evaluable_events)
        if expected is None:
            expected = tuple(groups.keys())

        physical = _normalize_events(physical_events)
        if physical is None:
            physical = expected

        covered = 0
        for event in expected:
            indices = groups.get(event, [])
            if not indices:
                continue
            idx = np.asarray(indices, dtype=int)
            if np.any(pred[idx] == 1):
                covered += 1

        out["events_physical_total"] = int(len(physical))
        out["events_evaluable_total"] = int(len(expected))
        # Backward-compatible name used by existing table scripts.
        out["events_total"] = int(len(expected))
        out["events_covered"] = int(covered)
        out["coverage"] = (
            float(covered / len(expected)) if expected else np.nan
        )

    if window_end_ts is not None and onset_ts is not None:
        out.update(
            lead_time_stats(
                pred,
                y,
                window_end_ts,
                onset_ts,
                file_ids=file_ids,
            )
        )

    return out


@torch.inference_mode()
def collect_probabilities(model, loader, device="cpu"):
    model.eval()
    ys, ps = [], []

    for xb, yb in loader:
        logits = model(xb.to(device))
        ys.append(yb.cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(ys), np.concatenate(ps)


@torch.inference_mode()
def benchmark_latency(
    model,
    sample,
    device="cpu",
    warmup=200,
    iterations=2000,
):
    model.eval().to(device)
    x = sample.to(device)

    for _ in range(warmup):
        _ = model(x)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        _ = model(x)

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

        times.append((time.perf_counter_ns() - t0) / 1e6)

    arr = np.asarray(times)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_median_ms": float(np.median(arr)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_std_ms": float(arr.std()),
    }
