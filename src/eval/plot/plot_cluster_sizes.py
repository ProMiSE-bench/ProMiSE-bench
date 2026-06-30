#!/usr/bin/env python3
"""Histogram and CDF of sequence cluster sizes from `data/clusters.json`.

Style matches other eval plots (plot_msa_pref_distribution, plot_hits_eff_distribution):
Arial/Helvetica sans-serif, bold axis labels, blue bars, PNG + PDF.

Example:
  PYTHONPATH=src python -m eval.plot.plot_cluster_sizes \\
    --input-json data/clusters.json --output-dir data_eval/plots/cluster_sizes
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

_STAT_LINE_COLOR = "#dc2626"
_STAT_LINE_KW = dict(color=_STAT_LINE_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)


def _v2_style_rc() -> dict:
    """Match plot_correlation_dynamic_v2 / plot_msa_pref_distribution."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    }


def _style_axes(ax) -> None:
    ax.tick_params(axis="both", which="major", labelsize=11, length=5, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def load_cluster_sizes(path: Path) -> list[int]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Expected JSON array of cluster objects.")
    sizes: list[int] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        members = entry.get("members")
        if not isinstance(members, list):
            continue
        sizes.append(len(members))
    if not sizes:
        raise SystemExit("No clusters with 'members' lists found.")
    return sizes


def write_stats_csv(
    out_csv: Path,
    sizes: list[int],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = len(sizes)
    total_members = sum(sizes)
    row = {
        "n_clusters": n,
        "n_members_total": total_members,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": statistics.mean(sizes),
        "median_size": statistics.median(sizes),
        "stdev_size": statistics.stdev(sizes) if n >= 2 else 0.0,
    }
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def plot_cluster_sizes(sizes: list[int], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(sizes, dtype=float)
    n = len(arr)
    mean_v = float(np.mean(arr))
    med_v = float(np.median(arr))
    min_s = max(float(np.min(arr)), 1.0)
    max_s = float(np.max(arr))

    with plt.rc_context(_v2_style_rc()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Log-spaced bins on raw sizes (heavy tail)
        n_bins = min(60, max(20, int(np.sqrt(n))))
        bin_edges = np.logspace(np.log10(min_s), np.log10(max_s), n_bins + 1)
        ax1.hist(arr, bins=bin_edges, color="#3b82f6", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax1.set_xscale("log")
        ax1.set_xlabel(r"Cluster size ($|\mathrm{members}|$)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax1.grid(alpha=0.25)
        _style_axes(ax1)
        ax1.axvline(med_v, **_STAT_LINE_KW)
        ax1.axvline(mean_v, linestyle=":", color=_STAT_LINE_COLOR, linewidth=1.0, alpha=0.75)
        ax1.text(
            0.02,
            0.98,
            f"N clusters = {n:,}\n"
            f"N members = {int(np.sum(arr)):,}\n"
            f"mean = {mean_v:.2f}\n"
            f"median = {med_v:.1f}\n"
            f"max = {max_s:.0f}",
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="sans-serif",
        )
        # CDF
        x_sorted = np.sort(arr)
        y = np.linspace(0.0, 1.0, len(x_sorted), endpoint=True)
        ax2.plot(x_sorted, y, color="black", linewidth=1.5, alpha=0.85)
        ax2.set_xscale("log")
        ax2.set_xlabel(r"Cluster size ($|\mathrm{members}|$)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("CDF", fontsize=12, fontweight="bold")
        ax2.grid(alpha=0.25)
        _style_axes(ax2)
        ax2.axvline(med_v, **_STAT_LINE_KW)
        ax2.axvline(mean_v, linestyle=":", color=_STAT_LINE_COLOR, linewidth=1.0, alpha=0.75)

        fig.tight_layout()
        out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)


@click.command()
@click.option(
    "--input-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="clusters.json (array of {center, members}).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for PNG, PDF, and cluster_size_stats.csv.",
)
@click.option(
    "--basename",
    type=str,
    default="sequence_cluster_sizes",
    show_default=True,
    help="Output filename stem (writes <basename>.png/.pdf).",
)
def main(input_json: Path, output_dir: Path, basename: str) -> None:
    sizes = load_cluster_sizes(input_json)
    out_png = output_dir / f"{basename}.png"
    plot_cluster_sizes(sizes, out_png)
    write_stats_csv(output_dir / "cluster_size_stats.csv", sizes)
    print(f"Saved cluster size plot: {out_png}")
    print(f"Saved stats CSV: {output_dir / 'cluster_size_stats.csv'}")
    print(f"N clusters: {len(sizes):,}  total members: {sum(sizes):,}")


if __name__ == "__main__":
    main()
