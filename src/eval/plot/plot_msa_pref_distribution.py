#!/usr/bin/env python3
"""Histogram of MSA preference (msa_pref_sum) with fixed ±MSA threshold guides.

Use the same merged JSON as the correlation pipeline (not the MSA-filtered subset),
so the distribution is full-data and the vertical lines mark the filter cutoffs.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory


_THRESHOLD_LINE_COLOR = "#dc2626"
_THRESHOLD_LINE_KW = dict(color=_THRESHOLD_LINE_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)


def _v2_style_rc() -> dict:
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


def collect_msa_pref_sum(merged_data: dict, model_filter: str | None, category_filter: str | None) -> list[float]:
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
                    v = pair_info.get("msa_pref_sum")
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(fv):
                        values.append(fv)
    return values


def plot_msa_pref_distribution(values: np.ndarray, msa_threshold: float, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    t = float(msa_threshold)
    with plt.rc_context(_v2_style_rc()):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(values, bins=60, color="#3b82f6", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xlabel(r"$\mathrm{MSA}_{\mathrm{pref}}$", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        _style_axes(ax)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for qx, lab in ((-t, f"-{t:g}"), (t, f"+{t:g}")):
            ax.axvline(qx, **_THRESHOLD_LINE_KW)
            ax.text(
                qx,
                0.98,
                lab,
                transform=trans,
                rotation=90,
                va="top",
                ha="right",
                fontsize=9,
                color=_THRESHOLD_LINE_COLOR,
            )

        fig.tight_layout()
        out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)


@click.command()
@click.option("--input-json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Merged JSON with msa_pref_sum per pair.")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True, help="Directory for PNG/PDF.")
@click.option("--msa-threshold", type=float, required=True, help="Draw vertical lines at ±this value (same as MSA filter threshold).")
@click.option("--model", type=str, default=None, help="Optional model filter, e.g. af3.")
@click.option("--category", type=str, default=None, help="Optional category filter, e.g. intrinsic.")
@click.option("--suffix", type=str, default="", show_default=True, help="Optional filename suffix before .png, e.g. _v2 or _legacy.")
def main(
    input_json: Path,
    output_dir: Path,
    msa_threshold: float,
    model: str | None,
    category: str | None,
    suffix: str,
) -> None:
    with open(input_json, "r") as f:
        merged_data = json.load(f)

    raw = collect_msa_pref_sum(merged_data, model, category)
    if not raw:
        raise SystemExit("No msa_pref_sum values found for the selected scope.")

    arr = np.asarray(raw, dtype=float)
    scope_parts = []
    if model:
        scope_parts.append(model)
    if category:
        scope_parts.append(category)
    scope = "__".join(scope_parts) if scope_parts else "all"
    suf = suffix if suffix.startswith("_") or not suffix else f"_{suffix}"
    out_png = output_dir / f"msa_pref_distribution_{scope}{suf}.png"

    plot_msa_pref_distribution(arr, msa_threshold, out_png)
    print(f"Saved MSA pref distribution: {out_png}")
    print(f"N pairs: {len(arr)}  (±MSA threshold lines at {msa_threshold:g})")


if __name__ == "__main__":
    main()
