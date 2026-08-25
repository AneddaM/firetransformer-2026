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


def fmt_mean_std(
    mean_value: float,
    std_value: float,
    decimals: int = 4,
) -> str:
    """Format mean ± sample standard deviation for LaTeX."""
    return (
        f"{mean_value:.{decimals}f} "
        f"$\\pm$ "
        f"{std_value:.{decimals}f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate LaTeX rows using hierarchical "
            "LONO aggregation: seed means within node, "
            "then macro mean ± between-node std."
        )
    )
    ap.add_argument(
        "--lono",
        default="runs/lono/lono_runs.csv",
        help="CSV containing individual node/seed LONO runs",
    )
    ap.add_argument(
        "--output",
        default="runs/lono/lono_table_rows.tex",
        help="Output LaTeX file",
    )
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Optional single model to export, e.g. ft64. "
            "If omitted, all models are emitted in "
            "separate commented blocks."
        ),
    )
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
    }
    missing = sorted(
        required - set(df.columns)
    )
    if missing:
        raise ValueError(
            "Input LONO CSV is missing required "
            "column(s): "
            + ", ".join(missing)
        )

    if args.model is not None:
        df = df[
            df["model"] == args.model
        ].copy()

        if df.empty:
            raise ValueError(
                f"Model '{args.model}' not found "
                f"in {args.lono}"
            )

    lines = []

    for model, group in df.groupby(
        "model",
        sort=True,
    ):
        # ------------------------------------------------------------
        # Statistical convention:
        #
        # 1. Average seeds within each held-out node.
        # 2. Compute macro statistics across node-wise means.
        # 3. sigma_node is therefore BETWEEN-NODE variability.
        # ------------------------------------------------------------

        node_means = (
            group.groupby(
                "outer_test_node"
            )[PAPER_METRICS]
            .mean()
            .sort_index()
        )

        if node_means.empty:
            continue

        lines.append(
            f"% model={model}; "
            f"held-out nodes={len(node_means)}; "
            "per-node values are seed-averaged"
        )

        # Per-node rows.
        for node, row in node_means.iterrows():
            cells = [
                str(node),
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1']:.4f}",
                f"{row['auc_roc']:.4f}",
                (
                    f"{100.0 * row['coverage']:.1f}"
                    "\\%"
                ),
            ]

            lines.append(
                " & ".join(cells) + r" \\"
            )

        # Macro mean and between-node sample standard deviation.
        macro_mean = node_means.mean()
        macro_std = node_means.std(
            ddof=1
        )

        macro_cells = [
            (
                r"Macro "
                r"$\mu\pm\sigma_{\mathrm{node}}$"
            ),
            fmt_mean_std(
                macro_mean["precision"],
                macro_std["precision"],
            ),
            fmt_mean_std(
                macro_mean["recall"],
                macro_std["recall"],
            ),
            fmt_mean_std(
                macro_mean["f1"],
                macro_std["f1"],
            ),
            fmt_mean_std(
                macro_mean["auc_roc"],
                macro_std["auc_roc"],
            ),
            (
                f"{100.0 * macro_mean['coverage']:.1f}"
                r" $\pm$ "
                f"{100.0 * macro_std['coverage']:.1f}"
                "\\%"
            ),
        ]

        lines.append(
            " & ".join(macro_cells) + r" \\"
        )
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )

    print(output)


if __name__ == "__main__":
    main()
