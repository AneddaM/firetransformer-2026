#!/usr/bin/env python3
"""Convert grouped-CV outputs into compact LaTeX rows for the paper."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PAPER_METRICS = ["precision", "recall", "f1", "auc_roc", "coverage"]
MODEL_LABELS = {
    "ft32": "FT-32",
    "ft64": "FT-64",
    "ft128": "FT-128",
    "bilstm_bce": "BiLSTM-BCE",
    "bilstm_fl": "BiLSTM-FL",
}


def fmt_mean_std(mean_value: float, std_value: float, decimals: int = 4) -> str:
    return f"{mean_value:.{decimals}f} $\\pm$ {std_value:.{decimals}f}"


def validate_event_denominators(df: pd.DataFrame) -> None:
    required = {"events_physical_total", "events_evaluable_total"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Physical-onset runs must contain events_physical_total and "
            "events_evaluable_total. Re-run the updated grouped-CV pipeline."
        )

    for (model, fold), group in df.groupby(["model", "outer_fold"]):
        if group["events_physical_total"].nunique() != 1:
            raise ValueError(
                f"Physical-event denominator changes across seeds: {model}/fold{fold}"
            )
        if group["events_evaluable_total"].nunique() != 1:
            raise ValueError(
                f"Evaluable-event denominator changes across seeds: {model}/fold{fold}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate model-comparison LaTeX rows from acquisition-level grouped CV. "
            "Seeds are averaged within each fold, then mean ± sigma_fold is computed "
            "across outer folds."
        )
    )
    ap.add_argument("--cv", default="runs/grouped_cv/cv_runs.csv")
    ap.add_argument("--output", default="runs/grouped_cv/model_table_rows.tex")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.cv)
    required = {
        "model",
        "outer_fold",
        "seed",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "coverage",
        "events_physical_total",
        "events_evaluable_total",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Input CV CSV is missing required columns: " + ", ".join(missing))

    validate_event_denominators(df)

    if args.models:
        wanted = set(args.models)
        missing_models = sorted(wanted - set(df["model"].unique()))
        if missing_models:
            raise ValueError(f"Requested models not found in {args.cv}: {missing_models}")
        df = df[df["model"].isin(wanted)].copy()

    fold_means = (
        df.groupby(["model", "outer_fold"], as_index=False)[PAPER_METRICS]
        .mean()
        .sort_values(["model", "outer_fold"])
    )

    model_summary = (
        fold_means.groupby("model")[PAPER_METRICS]
        .agg(["mean", "std"])
        .sort_index()
    )

    lines = [
        "% Five-fold grouped CV by complete acquisition file.",
        "% Values are mean ± sample std across the five fold-wise seed means (sigma_fold).",
        "% NODO1...NODO5 storage directories are not evaluation groups.",
    ]

    preferred_order = ["ft64", "ft128", "bilstm_bce", "bilstm_fl", "ft32"]
    available = list(model_summary.index)
    order = [m for m in preferred_order if m in available] + [
        m for m in available if m not in preferred_order
    ]

    for model in order:
        row = model_summary.loc[model]
        cells = [
            MODEL_LABELS.get(model, model),
            fmt_mean_std(row[("precision", "mean")], row[("precision", "std")]),
            fmt_mean_std(row[("recall", "mean")], row[("recall", "std")]),
            fmt_mean_std(row[("f1", "mean")], row[("f1", "std")]),
            fmt_mean_std(row[("auc_roc", "mean")], row[("auc_roc", "std")]),
            (
                f"{100.0 * row[('coverage', 'mean')]:.1f}"
                r" $\pm$ "
                f"{100.0 * row[('coverage', 'std')]:.1f}\\%"
            ),
        ]
        lines.append(" & ".join(cells) + r" \\")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
