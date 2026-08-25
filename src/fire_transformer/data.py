"""Leakage-controlled raw CSV preparation for the WF-IoT revision.

Principles:
1) outer split is by physical node;
2) inner validation split is by complete acquisition file;
3) rolling features are computed independently per acquisition file;
4) MinMaxScaler is fitted on training files only;
5) sliding windows are created independently per file after the split;
6) a physical fire onset is defined only by a 0->1 transition;
7) early-warning windows are generated only while the current fire state is 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def infer_node(path: str | Path) -> str:
    s = str(path).upper()
    for i in range(1, 6):
        if f"NODO{i}" in s or f"NODE{i}" in s:
            return f"NODO{i}"
    raise ValueError(
        f"Cannot infer node from '{path}'. Put files inside NODO1...NODO5 folders "
        "or rename paths so the node identifier is present."
    )


def list_csvs(root: str | Path):
    return sorted(Path(root).rglob("*.csv"))


def _resolve_column(columns, aliases, required=True):
    normalized = {_norm(c): c for c in columns}
    for a in aliases:
        na = _norm(a)
        if na in normalized:
            return normalized[na]

    for a in aliases:
        na = _norm(a)
        if len(na) < 4:
            continue
        for nc, original in normalized.items():
            if na in nc or nc in na:
                return original

    if required:
        raise KeyError(
            f"Could not resolve any of aliases {aliases}. "
            f"Available columns: {list(columns)}"
        )
    return None


def resolve_schema(df: pd.DataFrame, schema_cfg: dict):
    mapping = {}
    for canonical, aliases in schema_cfg["feature_aliases"].items():
        mapping[canonical] = _resolve_column(
            df.columns, [canonical] + list(aliases)
        )

    mapping["fire"] = _resolve_column(
        df.columns,
        schema_cfg.get("fire_aliases", ["fire", "label"]),
    )
    mapping["timestamp"] = _resolve_column(
        df.columns,
        schema_cfg.get("timestamp_aliases", ["timestamp", "time"]),
        required=False,
    )
    return mapping


def normalize_fire_label(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        arr = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
        return (arr > 0).astype(np.int64)

    vals = series.astype(str).str.strip().str.lower()
    pos = {"1", "true", "fire", "smoke", "yes", "positive", "burning"}
    return vals.isin(pos).astype(np.int64).to_numpy()


def parse_timestamps(series: pd.Series | None, n: int):
    if series is None:
        return np.arange(n, dtype=np.float64)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        return numeric.interpolate(limit_direction="both").to_numpy(dtype=np.float64)

    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().mean() > 0.95:
        return (dt.astype("int64") / 1e9).to_numpy(dtype=np.float64)

    return np.arange(n, dtype=np.float64)


def fire_onset_indices(fire: np.ndarray) -> np.ndarray:
    """Return physical onset indices defined strictly by 0->1 transitions.

    A sequence beginning in the fire state is not assigned an onset at index 0,
    because the preceding 0 state is not observed.
    """
    fire = np.asarray(fire, dtype=np.int64)
    if fire.size < 2:
        return np.empty((0,), dtype=np.int64)

    return (
        np.flatnonzero((fire[:-1] == 0) & (fire[1:] == 1)) + 1
    ).astype(np.int64)


@dataclass
class Acquisition:
    path: Path
    node: str
    raw: pd.DataFrame
    fire: np.ndarray
    timestamps: np.ndarray


@dataclass
class WindowBundle:
    X: np.ndarray
    y: np.ndarray
    window_end_ts: np.ndarray
    onset_ts: np.ndarray
    file_ids: np.ndarray


class DatasetCatalog:
    def __init__(self, root, schema_cfg):
        self.root = Path(root)
        self.schema_cfg = schema_cfg
        self.acquisitions = []

        for p in list_csvs(self.root):
            df = pd.read_csv(p)
            mapping = resolve_schema(df, schema_cfg)

            canonical = pd.DataFrame({
                "temperature": pd.to_numeric(df[mapping["temperature"]], errors="coerce"),
                "humidity": pd.to_numeric(df[mapping["humidity"]], errors="coerce"),
                "pressure": pd.to_numeric(df[mapping["pressure"]], errors="coerce"),
                "gas_resistance": pd.to_numeric(df[mapping["gas_resistance"]], errors="coerce"),
            })

            valid = canonical.notna().all(axis=1)
            canonical = canonical.loc[valid].reset_index(drop=True)

            fire = normalize_fire_label(
                df.loc[valid, mapping["fire"]].reset_index(drop=True)
            )

            ts_series = (
                None
                if mapping["timestamp"] is None
                else df.loc[valid, mapping["timestamp"]].reset_index(drop=True)
            )
            ts = parse_timestamps(ts_series, len(canonical))

            if len(canonical) == 0:
                continue

            self.acquisitions.append(
                Acquisition(p, infer_node(p), canonical, fire, ts)
            )

        if not self.acquisitions:
            raise FileNotFoundError(f"No usable CSV files found under {self.root}")

    @property
    def nodes(self):
        return sorted({a.node for a in self.acquisitions})

    def by_node(self, node):
        return [a for a in self.acquisitions if a.node == node]


def enrich(raw: pd.DataFrame, rolling_window=5) -> pd.DataFrame:
    feats = [raw.reset_index(drop=True)]
    roll = raw.rolling(rolling_window, min_periods=1)

    for name, obj in [
        ("mean", roll.mean()),
        ("std", roll.std().fillna(0)),
        ("min", roll.min()),
        ("max", roll.max()),
    ]:
        obj = obj.copy()
        obj.columns = [f"{c}_r_{name}" for c in raw.columns]
        feats.append(obj.reset_index(drop=True))

    return pd.concat(feats, axis=1)


def split_outer_inner(
    catalog: DatasetCatalog,
    heldout_node: str,
    val_fraction_files=0.25,
    split_seed=1337,
):
    test = [a for a in catalog.acquisitions if a.node == heldout_node]
    dev = [a for a in catalog.acquisitions if a.node != heldout_node]

    if not test:
        raise ValueError(f"No acquisitions found for held-out node {heldout_node}")

    train, val = [], []
    rng = np.random.default_rng(split_seed)

    for node in sorted({a.node for a in dev}):
        node_files = [a for a in dev if a.node == node]
        idx = np.arange(len(node_files))
        rng.shuffle(idx)

        n_val = (
            max(1, int(round(len(node_files) * val_fraction_files)))
            if len(node_files) > 1
            else 0
        )

        val_idx = set(idx[:n_val].tolist())

        for i, acq in enumerate(node_files):
            (val if i in val_idx else train).append(acq)

    if not train or not val:
        raise ValueError(
            "Need at least two acquisition files among development nodes "
            "for grouped train/validation split"
        )

    return train, val, test


def fit_feature_scaler(train_acqs, rolling_window=5):
    frames = [enrich(a.raw, rolling_window) for a in train_acqs]
    feature_names = list(frames[0].columns)

    scaler = MinMaxScaler().fit(
        pd.concat(frames, ignore_index=True).to_numpy(dtype=np.float64)
    )

    return scaler, feature_names


def make_windows_one(
    acq: Acquisition,
    scaler: MinMaxScaler,
    rolling_window=5,
    window=60,
    horizon=15,
):
    """Generate pre-onset early-warning windows for one acquisition.

    A window ending at sample t is eligible only when the current observed
    fire state is 0. Its target is positive if a physical 0->1 onset occurs
    in samples (t, t+H].
    """
    features = enrich(acq.raw, rolling_window).to_numpy(dtype=np.float64)
    x = scaler.transform(features).astype(np.float32)

    yraw = np.asarray(acq.fire, dtype=np.int64)
    ts = np.asarray(acq.timestamps, dtype=np.float64)
    physical_onsets = fire_onset_indices(yraw)

    X, y, end_ts, onset_ts, file_ids = [], [], [], [], []

    last_t = len(x) - horizon - 1

    for t in range(window - 1, last_t + 1):
        # Early-warning classification is defined only before an active fire.
        if yraw[t] != 0:
            continue

        candidates = physical_onsets[
            (physical_onsets > t) & (physical_onsets <= t + horizon)
        ]

        label = int(candidates.size > 0)

        X.append(x[t - window + 1:t + 1])
        y.append(label)
        end_ts.append(ts[t])

        if label:
            onset_ts.append(ts[int(candidates[0])])
        else:
            onset_ts.append(np.nan)

        file_ids.append(str(acq.path))

    if not X:
        nfeat = x.shape[1]
        return WindowBundle(
            np.empty((0, window, nfeat), np.float32),
            np.empty((0,), np.int64),
            np.empty((0,), np.float64),
            np.empty((0,), np.float64),
            np.empty((0,), object),
        )

    return WindowBundle(
        np.stack(X),
        np.asarray(y, np.int64),
        np.asarray(end_ts, np.float64),
        np.asarray(onset_ts, np.float64),
        np.asarray(file_ids, object),
    )


def concat_bundles(bundles):
    good = [b for b in bundles if len(b.y)]

    if not good:
        raise ValueError(
            "No windows generated. Check window/horizon length and input files."
        )

    return WindowBundle(
        np.concatenate([b.X for b in good]),
        np.concatenate([b.y for b in good]),
        np.concatenate([b.window_end_ts for b in good]),
        np.concatenate([b.onset_ts for b in good]),
        np.concatenate([b.file_ids for b in good]),
    )


def build_fold(
    catalog,
    heldout_node,
    rolling_window=5,
    window=60,
    horizon=15,
    val_fraction_files=0.25,
    split_seed=1337,
):
    train_a, val_a, test_a = split_outer_inner(
        catalog,
        heldout_node,
        val_fraction_files,
        split_seed,
    )

    scaler, feature_names = fit_feature_scaler(train_a, rolling_window)

    def make(items):
        return concat_bundles([
            make_windows_one(
                a,
                scaler,
                rolling_window,
                window,
                horizon,
            )
            for a in items
        ])

    train = make(train_a)
    val = make(val_a)
    test = make(test_a)

    return (
        train,
        val,
        test,
        scaler,
        feature_names,
        train_a,
        val_a,
        test_a,
    )


def to_loader(
    bundle: WindowBundle,
    batch_size=256,
    shuffle=False,
    num_workers=0,
):
    ds = TensorDataset(
        torch.from_numpy(bundle.X),
        torch.from_numpy(bundle.y),
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
