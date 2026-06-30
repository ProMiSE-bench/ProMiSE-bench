#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory


DEFAULT_QUANTILES = [20, 25, 50, 75, 95, 99]

# Vertical quantile guides (histogram + CDF)
_QUANTILE_LINE_COLOR = "#dc2626"
_QUANTILE_LINE_KW = dict(color=_QUANTILE_LINE_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)
# CDF: labels stack on the left; only annotate from this percentile upward
_CDF_LABEL_MIN_QUANTILE = 25.0

# PYTHONPATH=src python -m src.eval.plot.plot_hits_eff_distribution   --input-json data_eval/merged_valid_pairs_data_with_bias_ratio_diff.json   --output-dir data_eval/plots/hits_eff_distribution


def _v2_style_rc() -> dict:
    """Match matplotlib defaults to plot_correlation_dynamic_v2 (axis label fonts, spine/tick widths)."""
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


def collect_hits_eff(merged_data: dict, model_filter: str | None, category_filter: str | None) -> list[float]:
    values: list[float] = []
    for model_name, model_data in merged_data.items():
        if model_filter and model_name != model_filter:
            continue
        if not isinstance(model_data, dict):
            continue
        for category, category_data in model_data.items():
            if category_filter and category != category_filter:
                continue
            if not isinstance(category_data, dict):
                continue
            for cluster_data in category_data.values():
                if not isinstance(cluster_data, dict):
                    continue
                for pair_info in cluster_data.values():
                    if not isinstance(pair_info, dict):
                        continue
                    v = pair_info.get("hits_eff")
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(fv):
                        values.append(max(0.0, fv))
    return values


def write_quantile_table(values: np.ndarray, quantiles: list[float], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantile_percent", "hits_eff_value"])
        for q in quantiles:
            writer.writerow([q, float(np.percentile(values, q))])


def _neg_log10_hits_eff(arr: np.ndarray) -> np.ndarray:
    """Requires hits_eff > 0."""
    return -np.log10(np.asarray(arr, dtype=float))


def plot_distribution(values: np.ndarray, quantiles: list[float], out_png: Path) -> None:
    """`values` must be strictly positive (exclude hits_eff == 0)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_v2_style_rc()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Top: histogram on -log10(hits_eff)
        x_nlog = _neg_log10_hits_eff(values)
        ax1.hist(x_nlog, bins=60, color="#3b82f6", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax1.set_xlabel(r"$-\log_{10}(\mathrm{hits\_eff})$", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax1.grid(alpha=0.25)
        _style_axes(ax1)

        trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)

        # Mark quantile cuts + q-labels (same style as CDF)
        for q in quantiles:
            qv = float(np.percentile(values, q))
            qx = float(-np.log10(qv))
            ax1.axvline(qx, **_QUANTILE_LINE_KW)
            ax1.text(
                qx,
                0.98,
                f"q{q:g}",
                transform=trans1,
                rotation=90,
                va="top",
                ha="right",
                fontsize=9,
                color=_QUANTILE_LINE_COLOR,
            )

        # Bottom: CDF on raw scale
        x_sorted = np.sort(values)
        y = np.linspace(0, 1, len(x_sorted), endpoint=True)
        ax2.plot(x_sorted, y, color="black", linewidth=1.5, alpha=0.85)
        ax2.set_xlabel(r"$\mathrm{hits\_eff}$", fontsize=12, fontweight="bold")
        ax2.set_ylabel("CDF", fontsize=12, fontweight="bold")
        ax2.grid(alpha=0.25)
        _style_axes(ax2)

        for q in quantiles:
            qv = float(np.percentile(values, q))
            ax2.axvline(qv, **_QUANTILE_LINE_KW)
            if q >= _CDF_LABEL_MIN_QUANTILE:
                ax2.text(qv, 0.02, f"q{q:g}", rotation=90, va="bottom", ha="right", fontsize=9, color=_QUANTILE_LINE_COLOR)

        fig.tight_layout()
        out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)


@click.command()
@click.option("--input-json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Merged JSON containing hits_eff.")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True, help="Directory to write plots and quantile CSV.")
@click.option("--model", type=str, default=None, help="Optional model filter, e.g. af3.")
@click.option("--category", type=str, default=None, help="Optional category filter, e.g. intrinsic.")
@click.option("--quantiles", type=str, default="25,50,75,95", show_default=True, help="Comma-separated quantiles in percent.")
def main(input_json: Path, output_dir: Path, model: str | None, category: str | None, quantiles: str) -> None:
    with open(input_json, "r") as f:
        merged_data = json.load(f)

    q_list = [float(x.strip()) for x in quantiles.split(",") if x.strip()]
    if not q_list:
        q_list = DEFAULT_QUANTILES

    values = collect_hits_eff(merged_data, model, category)
    if not values:
        raise SystemExit("No hits_eff values found for the selected scope.")

    arr = np.asarray(values, dtype=float)
    n_zero = int(np.sum(arr <= 0))
    arr_pos = arr[arr > 0]
    if arr_pos.size == 0:
        raise SystemExit(
            "No hits_eff > 0 for the selected scope (all values are zero; excluded from plot/CDF/quantiles)."
        )

    scope_parts = []
    if model:
        scope_parts.append(model)
    if category:
        scope_parts.append(category)
    scope = "__".join(scope_parts) if scope_parts else "all"

    out_png = output_dir / f"hits_eff_distribution_{scope}.png"
    out_csv = output_dir / f"hits_eff_quantiles_{scope}.csv"

    plot_distribution(arr_pos, q_list, out_png)
    write_quantile_table(arr_pos, q_list, out_csv)

    print(f"Saved distribution plot: {out_png}")
    print(f"Saved quantile table: {out_csv}")
    print(f"N hits_eff > 0: {len(arr_pos)} (excluded hits_eff == 0: {n_zero})")


if __name__ == "__main__":
    main()
