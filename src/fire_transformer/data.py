"""Leakage-controlled data preparation for physical-onset early warning.

The public BME688/BME690 dataset is interpreted as a state sequence:

    label 0             -> sensor warm-up/stabilization (excluded)
    label 1001          -> valid pre-fire state
    labels 1002..1007   -> active-fire acquisition
    labels 1008, 1009   -> post-fire/recovery acquisition

A physical fire onset is the first and unique 1001 -> 1002 transition inside an
acquisition. Early-warning model inputs are generated exclusively from the pre-onset
portion of an acquisition. Active-fire and post-fire observations are retained only as
metadata for state validation and are never used as model input windows.

Leakage-control principles:
1) outer split is by physical node;
2) inner validation split is by complete acquisition file;
3) rolling features are computed independently per acquisition file;
4) MinMaxScaler is fitted only on eligible inner-training pre-onset samples;
5) sliding windows are created independently per acquisition after the split;
6) event coverage uses an explicit set of evaluable physical onset events.
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


STATE_WARMUP = -1
STATE_PREFIRE = 0
STATE_FIRE = 1
STATE_POSTFIRE = 2


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


def infer_heater_profile(path: str | Path) -> int | None:
    """Infer a profile identifier from filenames such as `...profilo_12.csv`.

    Heater-profile identity is metadata only and is not used as a model feature.
    """
    m = re.search(r"profilo[\s_\-]*(\d+)", Path(path).stem, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def list_csvs(root: str | Path):
    return sorted(Path(root).rglob("*.csv"))


def _resolve_column(columns, aliases, required=True):
    normalized = {_norm(c): c for c in columns}

    for alias in aliases:
        na = _norm(alias)
        if na in normalized:
            return normalized[na]

    for alias in aliases:
        na = _norm(alias)
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
            df.columns,
            [canonical] + list(aliases),
        )

    mapping["label"] = _resolve_column(
        df.columns,
        schema_cfg.get("label_aliases", ["label_tag"]),
    )
    mapping["timestamp"] = _resolve_column(
        df.columns,
        schema_cfg.get("timestamp_aliases", ["timestamp_since_poweron"]),
    )
    return mapping


def decode_label_tags(series: pd.Series, schema_cfg: dict):
    """Decode raw label_tag values into acquisition-state codes.

    Unknown label values raise rather than being silently mapped to fire/non-fire.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("NaN or non-numeric values found in label_tag")

    labels = numeric.astype(np.int64).to_numpy()
    state_cfg = schema_cfg["label_states"]

    warmup = set(map(int, state_cfg["warmup"]))
    pre_fire = set(map(int, state_cfg["pre_fire"]))
    fire = set(map(int, state_cfg["fire"]))
    post_fire = set(map(int, state_cfg["post_fire"]))
    known = warmup | pre_fire | fire | post_fire

    unknown = sorted(set(np.unique(labels).tolist()) - known)
    if unknown:
        raise ValueError(f"Unknown label_tag values: {unknown}")

    states = np.empty(len(labels), dtype=np.int8)
    states[np.isin(labels, list(warmup))] = STATE_WARMUP
    states[np.isin(labels, list(pre_fire))] = STATE_PREFIRE
    states[np.isin(labels, list(fire))] = STATE_FIRE
    states[np.isin(labels, list(post_fire))] = STATE_POSTFIRE
    return labels, states


def parse_timestamps(
    series: pd.Series | None,
    n: int,
    scale: float = 1.0,
):
    """Parse timestamps and express numeric values in seconds using `scale`.

    For the public dataset `timestamp_since_poweron` is in milliseconds and the
    configuration therefore uses `timestamp_scale: 0.001`.
    """
    if series is None:
        return np.arange(n, dtype=np.float64)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        arr = (
            numeric.interpolate(limit_direction="both")
            .to_numpy(dtype=np.float64)
        )
        return arr * float(scale)

    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().mean() > 0.95:
        return (dt.astype("int64") / 1e9).to_numpy(dtype=np.float64)

    raise ValueError("Timestamp column cannot be parsed reliably")


def physical_onset_indices(
    labels: np.ndarray,
    from_label: int = 1001,
    to_label: int = 1002,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size < 2:
        return np.empty((0,), dtype=np.int64)
    return (
        np.flatnonzero(
            (labels[:-1] == int(from_label))
            & (labels[1:] == int(to_label))
        )
        + 1
    ).astype(np.int64)


def fire_onset_indices(fire: np.ndarray) -> np.ndarray:
    """Generic binary 0->1 helper retained for tests/backward-compatible utilities."""
    fire = np.asarray(fire, dtype=np.int64)
    if fire.size < 2:
        return np.empty((0,), dtype=np.int64)
    return (
        np.flatnonzero((fire[:-1] == 0) & (fire[1:] == 1)) + 1
    ).astype(np.int64)


def validate_state_sequence(
    labels: np.ndarray,
    states: np.ndarray,
    path: str | Path,
    schema_cfg: dict,
):
    """Validate the experimental state machine and return warm-up/onset indices.

    Returns
    -------
    first_valid_raw_index:
        Index of the first non-warm-up sample in the raw acquisition.
    onset_index_after_warmup:
        Physical onset index after removal of the warm-up block.
    """
    labels = np.asarray(labels, dtype=np.int64)
    states = np.asarray(states, dtype=np.int8)

    non_warmup = np.flatnonzero(states != STATE_WARMUP)
    if len(non_warmup) == 0:
        raise ValueError(f"{path}: acquisition contains only warm-up samples")

    first_valid = int(non_warmup[0])
    if np.any(states[first_valid:] == STATE_WARMUP):
        raise ValueError(
            f"{path}: warm-up label reappears after the valid acquisition begins"
        )

    labels_valid = labels[first_valid:]
    states_valid = states[first_valid:]

    onset_cfg = schema_cfg.get(
        "physical_onset",
        {"from_label": 1001, "to_label": 1002},
    )
    onset = physical_onset_indices(
        labels_valid,
        int(onset_cfg.get("from_label", 1001)),
        int(onset_cfg.get("to_label", 1002)),
    )

    if len(onset) != 1:
        raise ValueError(
            f"{path}: expected exactly one physical "
            f"{onset_cfg.get('from_label', 1001)}->"
            f"{onset_cfg.get('to_label', 1002)} onset, found {len(onset)}"
        )

    onset_idx = int(onset[0])

    if onset_idx <= 0:
        raise ValueError(f"{path}: no observed pre-fire sample before onset")

    if np.any(states_valid[:onset_idx] != STATE_PREFIRE):
        raise ValueError(f"{path}: unexpected acquisition state before physical onset")

    if states_valid[onset_idx] != STATE_FIRE:
        raise ValueError(f"{path}: physical onset does not enter active-fire state")

    if np.any(states_valid[onset_idx:] == STATE_PREFIRE):
        raise ValueError(f"{path}: pre-fire state reappears after physical onset")

    post = np.flatnonzero(states_valid == STATE_POSTFIRE)
    if len(post):
        first_post = int(post[0])
        if np.any(states_valid[first_post:] == STATE_FIRE):
            raise ValueError(f"{path}: active-fire state reappears after recovery")

    return first_valid, onset_idx


@dataclass
class Acquisition:
    path: Path
    file_id: str
    node: str
    heater_profile: int | None
    raw: pd.DataFrame
    labels: np.ndarray
    states: np.ndarray
    fire: np.ndarray
    timestamps: np.ndarray
    onset_index: int
    onset_timestamp: float
    raw_rows: int
    warmup_rows: int

    @property
    def pre_onset_samples(self) -> int:
        return int(self.onset_index)

    @property
    def physical_event(self):
        return (
            str(self.file_id),
            round(float(self.onset_timestamp), 6),
        )

    @property
    def prefire_duration_s(self) -> float:
        if self.onset_index <= 0:
            return float("nan")
        return float(self.onset_timestamp - self.timestamps[0])


@dataclass
class WindowBundle:
    X: np.ndarray
    y: np.ndarray
    window_end_ts: np.ndarray
    onset_ts: np.ndarray
    file_ids: np.ndarray
    physical_events: tuple
    evaluable_events: tuple


class DatasetCatalog:
    def __init__(self, root, schema_cfg):
        self.root = Path(root)
        self.schema_cfg = schema_cfg
        self.acquisitions = []

        for p in list_csvs(self.root):
            df = pd.read_csv(p)
            mapping = resolve_schema(df, schema_cfg)

            canonical = pd.DataFrame(
                {
                    "temperature": pd.to_numeric(
                        df[mapping["temperature"]], errors="coerce"
                    ),
                    "humidity": pd.to_numeric(
                        df[mapping["humidity"]], errors="coerce"
                    ),
                    "pressure": pd.to_numeric(
                        df[mapping["pressure"]], errors="coerce"
                    ),
                    "gas_resistance": pd.to_numeric(
                        df[mapping["gas_resistance"]], errors="coerce"
                    ),
                }
            )

            # The public release has valid values for all four canonical channels.
            # Fail loudly instead of introducing an undocumented sample-removal rule.
            invalid_feature_rows = ~canonical.notna().all(axis=1)
            if invalid_feature_rows.any():
                raise ValueError(
                    f"{p}: found {int(invalid_feature_rows.sum())} rows with invalid "
                    "canonical sensor values"
                )

            labels_all, states_all = decode_label_tags(
                df[mapping["label"]],
                schema_cfg,
            )
            first_valid, onset_idx = validate_state_sequence(
                labels_all,
                states_all,
                p,
                schema_cfg,
            )

            ts_all = parse_timestamps(
                df[mapping["timestamp"]],
                len(df),
                scale=float(schema_cfg.get("timestamp_scale", 1.0)),
            )
            if len(ts_all) > 1 and np.any(np.diff(ts_all) <= 0):
                raise ValueError(f"{p}: timestamps are not strictly increasing")

            # Remove only the initial warm-up/stabilization block.
            canonical = canonical.iloc[first_valid:].reset_index(drop=True)
            labels = labels_all[first_valid:]
            states = states_all[first_valid:]
            timestamps = ts_all[first_valid:]

            if onset_idx >= len(canonical):
                raise ValueError(f"{p}: onset index is outside valid acquisition")

            onset_timestamp = float(timestamps[onset_idx])
            fire = (states == STATE_FIRE).astype(np.int64)

            self.acquisitions.append(
                Acquisition(
                    path=p,
                    file_id=p.relative_to(self.root).as_posix(),
                    node=infer_node(p),
                    heater_profile=infer_heater_profile(p),
                    raw=canonical,
                    labels=labels,
                    states=states,
                    fire=fire,
                    timestamps=timestamps,
                    onset_index=onset_idx,
                    onset_timestamp=onset_timestamp,
                    raw_rows=len(df),
                    warmup_rows=first_valid,
                )
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


def fit_feature_scaler(
    train_acqs,
    rolling_window=5,
    window=60,
):
    """Fit scaler only on pre-onset samples that can contribute to model windows."""
    eligible = [a for a in train_acqs if a.pre_onset_samples >= window]
    if not eligible:
        raise ValueError(f"No evaluable training acquisitions for W={window}")

    frames = [
        enrich(
            a.raw.iloc[: a.onset_index].reset_index(drop=True),
            rolling_window,
        )
        for a in eligible
    ]
    feature_names = list(frames[0].columns)

    scaler = MinMaxScaler().fit(
        pd.concat(frames, ignore_index=True).to_numpy(dtype=np.float64)
    )
    return scaler, feature_names


def _empty_bundle(acq: Acquisition, window: int, n_features: int):
    return WindowBundle(
        X=np.empty((0, window, n_features), dtype=np.float32),
        y=np.empty((0,), dtype=np.int64),
        window_end_ts=np.empty((0,), dtype=np.float64),
        onset_ts=np.empty((0,), dtype=np.float64),
        file_ids=np.empty((0,), dtype=object),
        physical_events=(acq.physical_event,),
        evaluable_events=(),
    )


def make_windows_one(
    acq: Acquisition,
    scaler: MinMaxScaler,
    rolling_window=5,
    window=60,
    horizon=15,
):
    """Generate physical-onset warning windows from one acquisition.

    Every input window ends strictly before the first physical 1001->1002 onset.
    A target is positive iff that onset occurs within the next H observations.
    Active-fire and post-fire/recovery observations never enter model inputs.
    """
    if window <= 0 or horizon <= 0:
        raise ValueError("window and horizon must be positive")

    onset = int(acq.onset_index)
    raw_pre = acq.raw.iloc[:onset].reset_index(drop=True)
    features = enrich(raw_pre, rolling_window).to_numpy(dtype=np.float64)
    n_features = features.shape[1] if features.ndim == 2 else 20

    if onset < window:
        return _empty_bundle(acq, window, n_features)

    x = scaler.transform(features).astype(np.float32)
    ts = np.asarray(acq.timestamps[:onset], dtype=np.float64)

    X, y, end_ts, onset_ts, file_ids = [], [], [], [], []

    for t in range(window - 1, onset):
        distance_to_onset = onset - t
        label = int(0 < distance_to_onset <= horizon)

        X.append(x[t - window + 1 : t + 1])
        y.append(label)
        end_ts.append(ts[t])
        onset_ts.append(acq.onset_timestamp if label else np.nan)
        file_ids.append(acq.file_id)

    return WindowBundle(
        X=np.stack(X),
        y=np.asarray(y, dtype=np.int64),
        window_end_ts=np.asarray(end_ts, dtype=np.float64),
        onset_ts=np.asarray(onset_ts, dtype=np.float64),
        file_ids=np.asarray(file_ids, dtype=object),
        physical_events=(acq.physical_event,),
        evaluable_events=(acq.physical_event,),
    )


def _dedupe_events(events):
    return tuple(dict.fromkeys(events))


def concat_bundles(bundles):
    if not bundles:
        raise ValueError("No acquisition bundles were provided")

    physical_events = _dedupe_events(
        event for bundle in bundles for event in bundle.physical_events
    )
    evaluable_events = _dedupe_events(
        event for bundle in bundles for event in bundle.evaluable_events
    )

    good = [b for b in bundles if len(b.y)]
    if not good:
        raise ValueError(
            "No windows generated. Check W, acquisition pre-onset length, and split."
        )

    return WindowBundle(
        X=np.concatenate([b.X for b in good]),
        y=np.concatenate([b.y for b in good]),
        window_end_ts=np.concatenate([b.window_end_ts for b in good]),
        onset_ts=np.concatenate([b.onset_ts for b in good]),
        file_ids=np.concatenate([b.file_ids for b in good]),
        physical_events=physical_events,
        evaluable_events=evaluable_events,
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

    scaler, feature_names = fit_feature_scaler(
        train_a,
        rolling_window,
        window,
    )

    def make(items):
        return concat_bundles(
            [
                make_windows_one(
                    a,
                    scaler,
                    rolling_window,
                    window,
                    horizon,
                )
                for a in items
            ]
        )

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
