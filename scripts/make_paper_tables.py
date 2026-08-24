#!/usr/bin/env python3
"""Convert measured CSV outputs into compact LaTeX table rows."""
import argparse
from pathlib import Path
import pandas as pd


def fmt(mu,sd,d=4):
    return f"{mu:.{d}f} $\\pm$ {sd:.{d}f}"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--lono",default="runs/lono/lono_runs.csv"); ap.add_argument("--output",default="runs/lono/lono_table_rows.tex"); args=ap.parse_args()
    df=pd.read_csv(args.lono)
    cols=["precision","recall","f1","auc_roc"]
    lines=[]
    for model,g in df.groupby("model"):
        cells=[model]
        for c in cols:
            cells.append(fmt(g[c].mean(),g[c].std()))
        cov=g["coverage"].mean()*100 if "coverage" in g else float("nan")
        lead=g["lead_mean_s"].mean() if "lead_mean_s" in g else float("nan")
        cells += [f"{cov:.1f}\\%", f"{lead:+.2f} s"]
        lines.append(" & ".join(cells)+r" \\")
    Path(args.output).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(Path(args.output))
if __name__=="__main__": main()
