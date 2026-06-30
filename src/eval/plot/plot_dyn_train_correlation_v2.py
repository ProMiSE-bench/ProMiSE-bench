#!/usr/bin/env python3
"""
Quick explorer: DynDisto_holo vs Train_holo (bias_ratio_diff).

Uses the same filtered JSON inputs and panel scopes as plot_correlation_dynamic_v2,
but scatter x=DynDisto, y=Train_holo and reports Pearson/Spearman for that pair
(with the same |DynDisto| and bias hit-sum filters as the main v2 plots).
"""

from __future__ import annotations

import csv
from pathlib import Path

import click
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .plot_correlation_dynamic_v2 import (
    AXIS_TICKS,
    DYN_HOLO_LABEL,
    MARKER_SIZE_RANGE,
    TRAIN_HOLO_LABEL,
    _default_filtered_dir,
    _style_colorbar_ticks,
    extract_apo_holo_separated,
    extract_data_from_filtered,
    filter_by_method,
    filter_by_pair_type,
    load_data,
    normalize_marker_sizes,
)

STRUCT_HOLO_LABEL = r"$\mathrm{Struct_{holo}}$"


def compute_dyn_train_correlations(
    dyn_vals,
    train_vals,
    hit_sum_vals,
    dyn_threshold: float,
    hit_sum_min_quantile: float,
):
    x = np.asarray(dyn_vals, dtype=float)
    y = np.asarray(train_vals, dtype=float)
    hit_sum = np.asarray(hit_sum_vals, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite_mask):
        return None
    mask = finite_mask & (np.abs(x) < dyn_threshold)
    hit_sum_mask = mask & np.isfinite(hit_sum)
    if np.sum(hit_sum_mask) >= 3 and np.nanmax(hit_sum[hit_sum_mask]) > np.nanmin(hit_sum[hit_sum_mask]):
        hit_cut = np.percentile(hit_sum[hit_sum_mask], hit_sum_min_quantile)
        filtered_mask = hit_sum_mask & (hit_sum > hit_cut)
        if np.sum(filtered_mask) >= 3:
            mask = filtered_mask
    if np.sum(mask) < 3:
        return None
    pearson_r, _ = pearsonr(x[mask], y[mask])
    spearman_rho, _ = spearmanr(x[mask], y[mask])
    return pearson_r, spearman_rho, int(np.sum(mask))


def _annotate_dyn_train_correlations(ax, dyn, train, hit_sum, dyn_threshold, hit_sum_min_quantile, fs=11):
    stats = compute_dyn_train_correlations(dyn, train, hit_sum, dyn_threshold, hit_sum_min_quantile)
    line = "Pearson r: N/A\nSpearman ρ: N/A"
    if stats is not None:
        pearson_r, spearman_rho, _ = stats
        line = f"Pearson r: {pearson_r:.3f}\nSpearman ρ: {spearman_rho:.3f}"
    ax.text(
        0.98, 0.04, line, transform=ax.transAxes, fontsize=fs,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def _style_dyn_train_axes(ax, *, show_ylabel: bool, label_fontsize: float, tick_fontsize: float):
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(AXIS_TICKS)
    ax.set_yticks(AXIS_TICKS)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, length=5, width=1.2)
    ax.set_xlabel(DYN_HOLO_LABEL, fontsize=label_fontsize, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel(TRAIN_HOLO_LABEL, fontsize=label_fontsize, fontweight="bold")


def _plot_dyn_train_panel(ax, d, title, dyn_threshold, hit_sum_min_quantile, *, show_ylabel: bool, label_fontsize: float, tick_fontsize: float):
    dyn = np.array(d["x"], dtype=float)
    train = np.array(d["bias_ratio_diff"], dtype=float)
    struct = np.array(d["y"], dtype=float)
    hit_sum = np.array(d["bias_hit_sum"], dtype=float)
    _style_dyn_train_axes(ax, show_ylabel=show_ylabel, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize)
    ax.set_title(title, fontsize=16, fontweight="bold")
    if len(dyn) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return None
    sizes = normalize_marker_sizes(hit_sum, *MARKER_SIZE_RANGE)
    scat = ax.scatter(
        dyn, train, c=struct, cmap=cm.coolwarm, vmin=-1, vmax=1,
        alpha=0.75, s=sizes, edgecolors="black", linewidth=0.7,
    )
    ax.axvline(x=dyn_threshold, color="black", linestyle=":", linewidth=1.2, alpha=0.65)
    ax.axvline(x=-dyn_threshold, color="black", linestyle=":", linewidth=1.2, alpha=0.65)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.0, alpha=0.4)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=1.0, alpha=0.4)
    _annotate_dyn_train_correlations(ax, dyn, train, hit_sum, dyn_threshold, hit_sum_min_quantile, fs=11)
    return scat


def create_4panel_dyn_train(data, output_dir: Path, dyn_threshold: float, hit_sum_min_quantile: float, output_suffix: str = ""):
    import matplotlib.gridspec as gridspec

    extracted = extract_data_from_filtered(data)
    apo_holo = extract_apo_holo_separated(data)

    af3_ligand_apo = filter_by_pair_type(filter_by_method(apo_holo["apo"], "af3"), "ligand-induced")
    boltz2_ligand_apo = filter_by_pair_type(filter_by_method(apo_holo["apo"], "boltz2"), "ligand-induced")
    af3_apo = filter_by_pair_type(filter_by_method(extracted, "af3"), "intrinsic")
    boltz2_apo = filter_by_pair_type(filter_by_method(extracted, "boltz2"), "intrinsic")

    panel_data = [
        (af3_ligand_apo, "AF3\nLigand-induced (Apo-conditioned)"),
        (boltz2_ligand_apo, "Boltz-2\nLigand-induced (Apo-conditioned)"),
        (af3_apo, "AF3\nIntrinsic Multi-State"),
        (boltz2_apo, "Boltz-2\nIntrinsic Multi-State"),
    ]

    fig = plt.figure(figsize=(28, 6.5))
    gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.04], wspace=0.20, right=0.96, left=0.04)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cbar_ax = fig.add_subplot(gs[0, 4])
    scat = None
    for idx, (ax, (d, title)) in enumerate(zip(axes, panel_data)):
        scat = _plot_dyn_train_panel(
            ax, d, title, dyn_threshold, hit_sum_min_quantile,
            show_ylabel=idx == 0, label_fontsize=18, tick_fontsize=16,
        ) or scat
    if scat is not None:
        cbar = fig.colorbar(scat, cax=cbar_ax, orientation="vertical")
        cbar.set_label(STRUCT_HOLO_LABEL, fontsize=16)
        _style_colorbar_ticks(cbar, tick_fontsize=15)
    out_png = output_dir / f"dyn_train_correlation_4panel{output_suffix}.png"
    out_pdf = output_dir / f"dyn_train_correlation_4panel{output_suffix}.pdf"
    plt.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    click.echo(f"Saved 4-panel DynDisto vs Train: {out_png}")
    plt.close()


def _collect_dyn_train_rows(data, scope_name, dyn_threshold, hit_sum_min_quantile, rows):
    dyn = np.array(data["x"], dtype=float)
    train = np.array(data["bias_ratio_diff"], dtype=float)
    hit_sum = np.array(data["bias_hit_sum"], dtype=float)
    stats = compute_dyn_train_correlations(dyn, train, hit_sum, dyn_threshold, hit_sum_min_quantile)
    row = {
        "scope": scope_name,
        "dyn_threshold": dyn_threshold,
        "n": "",
        "pearson_r": "",
        "spearman_rho": "",
    }
    if stats is not None:
        pearson_r, spearman_rho, n = stats
        row["n"] = str(n)
        row["pearson_r"] = f"{pearson_r:.6f}"
        row["spearman_rho"] = f"{spearman_rho:.6f}"
    rows.append(row)


def write_dyn_train_table(data, table_path: Path, dyn_threshold: float, hit_sum_min_quantile: float):
    extracted = extract_data_from_filtered(data)
    apo_holo = extract_apo_holo_separated(data)
    rows = []
    _collect_dyn_train_rows(extracted, "all", dyn_threshold, hit_sum_min_quantile, rows)
    for m in sorted(set(extracted["methods"])):
        _collect_dyn_train_rows(filter_by_method(extracted, m), f"{m}::all", dyn_threshold, hit_sum_min_quantile, rows)
        _collect_dyn_train_rows(
            filter_by_pair_type(filter_by_method(extracted, m), "intrinsic"),
            f"{m}::intrinsic", dyn_threshold, hit_sum_min_quantile, rows,
        )
        _collect_dyn_train_rows(
            filter_by_pair_type(filter_by_method(apo_holo["apo"], m), "ligand-induced"),
            f"{m}::ligand-induced::apo", dyn_threshold, hit_sum_min_quantile, rows,
        )
        _collect_dyn_train_rows(
            filter_by_pair_type(filter_by_method(apo_holo["holo"], m), "ligand-induced"),
            f"{m}::ligand-induced::holo", dyn_threshold, hit_sum_min_quantile, rows,
        )
        _collect_dyn_train_rows(
            filter_by_pair_type(filter_by_method(apo_holo["apo"], m), "protein-induced"),
            f"{m}::protein-induced::apo", dyn_threshold, hit_sum_min_quantile, rows,
        )
        _collect_dyn_train_rows(
            filter_by_pair_type(filter_by_method(apo_holo["holo"], m), "protein-induced"),
            f"{m}::protein-induced::holo", dyn_threshold, hit_sum_min_quantile, rows,
        )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "dyn_threshold", "n", "pearson_r", "spearman_rho"])
        writer.writeheader()
        writer.writerows(rows)
    click.echo(f"Saved DynDisto vs Train table: {table_path}")


@click.command()
@click.option(
    "--filtered-json",
    type=click.Path(path_type=Path),
    default=lambda: _default_filtered_dir() / "filtered_pairs_msa0.3_v2.json",
    show_default=True,
    help="Filtered pairs JSON (same as plot_correlation_dynamic_v2).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=lambda: _default_filtered_dir().parent / "plots" / "dyn_train_v2",
    show_default=True,
)
@click.option("--dyn-threshold", type=float, default=0.7, show_default=True)
@click.option("--hit-sum-min-quantile", type=float, default=20.0, show_default=True)
@click.option("--output-suffix", default="_msa0.3_v2", show_default=True)
@click.option("--plot/--no-plot", default=True, show_default=True)
@click.option("--table-csv", type=click.Path(path_type=Path), default=None, help="CSV path; default: <output-dir>/dyn_train_correlation_table.csv")
def main(
    filtered_json: Path,
    output_dir: Path,
    dyn_threshold: float,
    hit_sum_min_quantile: float,
    output_suffix: str,
    plot: bool,
    table_csv: Path | None,
):
    click.echo("DynDisto_holo vs Train_holo correlation (v2)")
    data = load_data(filtered_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = table_csv or (output_dir / "dyn_train_correlation_table.csv")
    write_dyn_train_table(data, table_path, dyn_threshold, hit_sum_min_quantile)

    if plot:
        create_4panel_dyn_train(data, output_dir, dyn_threshold, hit_sum_min_quantile, output_suffix=output_suffix)


if __name__ == "__main__":
    main()
