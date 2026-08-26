#!/usr/bin/env python3
"""Convert measured LONO CSV outputs into compact LaTeX table rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PAPER_METRICS = [
    "precision",
    "recall",
    "f1",
    "auc_roc",
    "coverage",
]


def fmt_mean_std(mean_value: float, std_value: float, decimals: int = 4) -> str:
    return f"{mean_value:.{decimals}f} $\\pm$ {std_value:.{decimals}f}"


def validate_event_denominators(df: pd.DataFrame) -> None:
    required = {"events_physical_total", "events_evaluable_total"}
    if not required.issubset(df.columns):
        raise ValueError(
            "New physical-onset runs must contain events_physical_total and "
            "events_evaluable_total. Re-run the updated pipeline."
        )

    for (model, node), group in df.groupby(["model", "outer_test_node"]):
        if group["events_physical_total"].nunique() != 1:
            raise ValueError(f"Physical-event denominator changes across seeds: {model}/{node}")
        if group["events_evaluable_total"].nunique() != 1:
            raise ValueError(f"Evaluable-event denominator changes across seeds: {model}/{node}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate LaTeX rows using hierarchical LONO aggregation: seed means "
            "within node, then macro mean ± between-node std."
        )
    )
    ap.add_argument("--lono", default="runs/lono/lono_runs.csv")
    ap.add_argument("--output", default="runs/lono/lono_table_rows.tex")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.lono)
    required = {
        "model",
        "outer_test_node",
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
        raise ValueError("Input LONO CSV is missing required columns: " + ", ".join(missing))

    validate_event_denominators(df)

    if args.model is not None:
        df = df[df["model"] == args.model].copy()
        if df.empty:
            raise ValueError(f"Model '{args.model}' not found in {args.lono}")

    lines = []

    for model, group in df.groupby("model", sort=True):
        node_means = (
            group.groupby("outer_test_node")[PAPER_METRICS]
            .mean()
            .sort_index()
        )
        if node_means.empty:
            continue

        event_counts = (
            group.groupby("outer_test_node")[["events_physical_total", "events_evaluable_total"]]
            .first()
            .sort_index()
        )

        lines.append(
            f"% model={model}; held-out nodes={len(node_means)}; "
            "per-node values are seed-averaged"
        )
        lines.append(
            "% physical/evaluable events by node: "
            + ", ".join(
                f"{node}={int(row.events_physical_total)}/{int(row.events_evaluable_total)}"
                for node, row in event_counts.iterrows()
            )
        )

        for node, row in node_means.iterrows():
            cells = [
                str(node),
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1']:.4f}",
                f"{row['auc_roc']:.4f}",
                f"{100.0 * row['coverage']:.1f}\\%",
            ]
            lines.append(" & ".join(cells) + r" \\")

        macro_mean = node_means.mean()
        macro_std = node_means.std(ddof=1)
        macro_cells = [
            r"Macro $\mu\pm\sigma_{\mathrm{node}}$",
            fmt_mean_std(macro_mean["precision"], macro_std["precision"]),
            fmt_mean_std(macro_mean["recall"], macro_std["recall"]),
            fmt_mean_std(macro_mean["f1"], macro_std["f1"]),
            fmt_mean_std(macro_mean["auc_roc"], macro_std["auc_roc"]),
            (
                f"{100.0 * macro_mean['coverage']:.1f}"
                r" $\pm$ "
                f"{100.0 * macro_std['coverage']:.1f}\\%"
            ),
        ]
        lines.append(" & ".join(macro_cells) + r" \\")
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
