#!/usr/bin/env python3
"""
Plot true-easy stability statistics from before/after evaluation JSON files.

Expected JSON format: a list of dicts like:
{
  "example_id": 2,
  "true_label": "entailment",
  "predicted_label": "entailment",
  "all_probs": {...},
  "true_prob": 0.58,
  "nll": 0.53,
  "correct": true
}

This script produces:
- histogram_after_true_prob.png
- histogram_conf_drop.png
- histogram_loss_increase.png
- scatter_before_after_true_prob.png
- delta_table.csv (joined table for further analysis)

It also prints percentiles to help pick thresholds.
"""

import argparse
import json
import os
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd


def load_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def to_df(records: List[Dict[str, Any]], prefix: str) -> pd.DataFrame:
    """
    Keep only the columns we need; prefix them to avoid collisions.
    """
    rows = []
    for r in records:
        if "example_id" not in r:
            continue
        rows.append({
            "example_id": r["example_id"],
            f"{prefix}true_prob": r.get("true_prob", None),
            f"{prefix}nll": r.get("nll", None),
            f"{prefix}correct": r.get("correct", None),
            f"{prefix}true_label": r.get("true_label", None),
            f"{prefix}predicted_label": r.get("predicted_label", None),
        })
    return pd.DataFrame(rows)


def percentile_report(series: pd.Series, name: str, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    clean = series.dropna().astype(float)
    if len(clean) == 0:
        print(f"[WARN] No values for {name}")
        return
    print(f"\n{name} (n={len(clean)}):")
    for p in ps:
        print(f"  p{p:02d}: {clean.quantile(p/100):.6f}")
    print(f"  min: {clean.min():.6f}")
    print(f"  max: {clean.max():.6f}")
    print(f"  mean: {clean.mean():.6f}")
    print(f"  std: {clean.std(ddof=1):.6f}")


def main():
    ap = argparse.ArgumentParser(description="Plot before/after true-easy stability stats")
    ap.add_argument("--before_json", type=str, required=True, help="Path to easy_eval_before_phase2.json")
    ap.add_argument("--after_json", type=str, required=True, help="Path to easy_eval_after_phase2.json")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for plots and CSV")
    ap.add_argument("--inner_join", action="store_true",
                    help="If set, only keep example_ids present in both files (recommended).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    before = load_list(args.before_json)
    after = load_list(args.after_json)

    df_b = to_df(before, prefix="before_")
    df_a = to_df(after, prefix="after_")

    how = "inner" if args.inner_join else "outer"
    df = df_b.merge(df_a, on="example_id", how=how)

    # Derived metrics
    df["conf_drop"] = df["before_true_prob"] - df["after_true_prob"]
    df["loss_increase"] = df["after_nll"] - df["before_nll"]

    # Save table
    csv_path = os.path.join(args.out_dir, "delta_table.csv")
    df.sort_values("example_id").to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved joined table: {csv_path}")
    print(f"Rows: {len(df)} (before: {len(df_b)}, after: {len(df_a)}, join: {how})")

    # Reports to guide threshold selection
    percentile_report(df["after_true_prob"], "after_true_prob")
    percentile_report(df["conf_drop"], "conf_drop = before_true_prob - after_true_prob")
    percentile_report(df["loss_increase"], "loss_increase = after_nll - before_nll")

    # Plot 1: histogram of after_true_prob
    plt.figure()
    df["after_true_prob"].dropna().astype(float).hist(bins=50)
    plt.xlabel("after_true_prob")
    plt.ylabel("count")
    plt.title("Histogram: after_true_prob")
    p1 = os.path.join(args.out_dir, "histogram_after_true_prob.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p1}")

    # Plot 2: histogram of conf_drop
    plt.figure()
    df["conf_drop"].dropna().astype(float).hist(bins=50)
    plt.xlabel("conf_drop (before - after)")
    plt.ylabel("count")
    plt.title("Histogram: conf_drop")
    p2 = os.path.join(args.out_dir, "histogram_conf_drop.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p2}")

    # Plot 3: histogram of loss_increase
    plt.figure()
    df["loss_increase"].dropna().astype(float).hist(bins=50)
    plt.xlabel("loss_increase (after_nll - before_nll)")
    plt.ylabel("count")
    plt.title("Histogram: loss_increase")
    p3 = os.path.join(args.out_dir, "histogram_loss_increase.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p3}")

    # Plot 4: scatter before vs after
    sub = df[["before_true_prob", "after_true_prob"]].dropna().astype(float)
    plt.figure()
    plt.scatter(sub["before_true_prob"], sub["after_true_prob"], s=6, alpha=0.5)
    plt.xlabel("before_true_prob")
    plt.ylabel("after_true_prob")
    plt.title("Scatter: before_true_prob vs after_true_prob")
    p4 = os.path.join(args.out_dir, "scatter_before_after_true_prob.png")
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p4}")

    print("\nTip for thresholds:")
    print("  - true_easy_min_conf_after: pick a cutoff on after_true_prob histogram (e.g., keep top 20-40%).")
    print("  - true_easy_max_conf_drop: pick a small tail cutoff on conf_drop (e.g., p80-p95).")
    print("  - true_easy_max_loss_increase: similar, use loss_increase tail (e.g., p80-p95).")


if __name__ == "__main__":
    main()
