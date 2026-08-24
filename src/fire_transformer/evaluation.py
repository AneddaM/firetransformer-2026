from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, average_precision_score
)


def choose_threshold(y, p, objective="f1"):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_s = 0.5, -np.inf
    for t in thresholds:
        pred = (p >= t).astype(int)
        if objective == "f1":
            s = f1_score(y, pred, zero_division=0)
        elif objective == "recall":
            s = recall_score(y, pred, zero_division=0)
        else:
            raise ValueError("threshold objective must be 'f1' or 'recall'")
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s


def event_coverage(pred, onset_ts):
    valid = np.isfinite(onset_ts)
    if not valid.any():
        return np.nan
    # Multiple windows may point to the same physical onset. Round to microseconds before unique.
    events = np.unique(np.round(onset_ts[valid], 6))
    covered = 0
    for e in events:
        mask = valid & (np.round(onset_ts, 6) == e)
        covered += int(np.any(pred[mask] == 1))
    return covered / len(events) if len(events) else np.nan


def lead_time_stats(pred, y, window_end_ts, onset_ts):
    mask = (pred == 1) & (y == 1) & np.isfinite(onset_ts)
    if not mask.any():
        return {"lead_mean_s": np.nan, "lead_median_s": np.nan, "lead_p25_s": np.nan, "lead_p75_s": np.nan}
    lead = onset_ts[mask] - window_end_ts[mask]
    return {
        "lead_mean_s": float(np.mean(lead)),
        "lead_median_s": float(np.median(lead)),
        "lead_p25_s": float(np.percentile(lead, 25)),
        "lead_p75_s": float(np.percentile(lead, 75)),
    }


def compute_metrics(y, p, threshold, window_end_ts=None, onset_ts=None):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "auc_pr": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "missed_detection_rate": float(fn / (fn + tp)) if (fn + tp) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if onset_ts is not None:
        out["coverage"] = float(event_coverage(pred, onset_ts))
    if window_end_ts is not None and onset_ts is not None:
        out.update(lead_time_stats(pred, y, window_end_ts, onset_ts))
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
def benchmark_latency(model, sample, device="cpu", warmup=200, iterations=2000):
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
    a = np.asarray(times)
    return {
        "latency_mean_ms": float(a.mean()),
        "latency_median_ms": float(np.median(a)),
        "latency_p95_ms": float(np.percentile(a, 95)),
        "latency_std_ms": float(a.std()),
    }
