#!/usr/bin/env python3
"""
Dynamic correlation plotting (legacy):
- color uses bias_ratio_diff
- marker size uses total hits (bias_entry1_hits + bias_entry2_hits)
- sweep table reports Spearman(bias_ratio_diff vs Struct_holo)
  for |DynDisto_holo| < threshold and low-hit points excluded.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import click
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils._config import eval_cfg as E


def load_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def compute_filtered_correlations(x_vals, y_vals, bias_vals, hits_vals, dyn_threshold, hits_min_quantile):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    b = np.asarray(bias_vals, dtype=float)
    h = np.asarray(hits_vals, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(b) & np.isfinite(h)
    if not np.any(finite_mask):
        return None
    hit_cut = np.percentile(h[finite_mask], hits_min_quantile)
    mask = finite_mask & (np.abs(x) < dyn_threshold) & (h > hit_cut)
    if np.sum(mask) < 3:
        return None
    pearson_r, _ = pearsonr(b[mask], y[mask])
    spearman_rho, _ = spearmanr(b[mask], y[mask])
    return pearson_r, spearman_rho, int(np.sum(mask))


def _extract_xy(entry):
    if "confbench_mean" in entry and "distogram_dynamic_confbench" in entry:
        return entry.get("distogram_dynamic_confbench"), entry.get("confbench_mean")
    if "confbench_apo_pred" in entry and "confbench_holo_pred" in entry:
        da = entry.get("distogram_dynamic_confbench_apo")
        dh = entry.get("distogram_dynamic_confbench_holo")
        ca = entry.get("confbench_apo_pred")
        ch = entry.get("confbench_holo_pred")
        if None in (da, dh, ca, ch):
            return None, None
        vals = [da, dh, ca, ch]
        if any(isinstance(v, float) and np.isnan(v) for v in vals):
            return None, None
        return (da + dh) / 2, (ca + ch) / 2
    return None, None


def extract_data_from_filtered(data):
    result = {
        "x": [],
        "y": [],
        "msa_pref_sum": [],
        "bias_ratio_diff": [],
        "total_hits": [],
        "labels": [],
        "methods": [],
        "pair_types": [],
    }
    for method, pair_types_data in data.items():
        if not isinstance(pair_types_data, dict):
            continue
        for pair_type, clusters in pair_types_data.items():
            if not isinstance(clusters, dict):
                continue
            for cluster_id, pairs in clusters.items():
                if not isinstance(pairs, dict):
                    continue
                for pair_key, entry in pairs.items():
                    if not isinstance(entry, dict):
                        continue
                    x_val, y_val = _extract_xy(entry)
                    if x_val is None or y_val is None:
                        continue
                    if any(isinstance(v, float) and np.isnan(v) for v in [x_val, y_val]):
                        continue
                    h1 = entry.get("bias_entry1_hits", 0) or 0
                    h2 = entry.get("bias_entry2_hits", 0) or 0
                    result["x"].append(x_val)
                    result["y"].append(y_val)
                    result["msa_pref_sum"].append(entry.get("msa_pref_sum", 0) or 0)
                    result["bias_ratio_diff"].append(entry.get("bias_ratio_diff", 0) or 0)
                    result["total_hits"].append(h1 + h2)
                    result["labels"].append(f"{method}_{pair_type}_{cluster_id}_{pair_key}")
                    result["methods"].append(method)
                    result["pair_types"].append(pair_type)
    return result


def filter_by_pair_type(data, pair_type):
    idxs = [i for i, pt in enumerate(data["pair_types"]) if pt == pair_type]
    return {k: [v[i] for i in idxs] for k, v in data.items()}


def filter_by_method(data, method):
    idxs = [i for i, m in enumerate(data["methods"]) if m == method]
    return {k: [v[i] for i in idxs] for k, v in data.items()}


def extract_apo_holo_separated(data):
    out = {
        "apo": {"x": [], "y": [], "msa_pref_sum": [], "bias_ratio_diff": [], "total_hits": [], "labels": [], "methods": [], "pair_types": []},
        "holo": {"x": [], "y": [], "msa_pref_sum": [], "bias_ratio_diff": [], "total_hits": [], "labels": [], "methods": [], "pair_types": []},
    }
    for method, pair_types_data in data.items():
        if not isinstance(pair_types_data, dict):
            continue
        for pair_type, clusters in pair_types_data.items():
            if pair_type not in ["ligand-induced", "protein-induced"] or not isinstance(clusters, dict):
                continue
            for cluster_id, pairs in clusters.items():
                if not isinstance(pairs, dict):
                    continue
                for pair_key, entry in pairs.items():
                    if not isinstance(entry, dict):
                        continue
                    da = entry.get("distogram_dynamic_confbench_apo")
                    dh = entry.get("distogram_dynamic_confbench_holo")
                    ca = entry.get("confbench_apo_pred")
                    ch = entry.get("confbench_holo_pred")
                    mp = entry.get("msa_pref_sum", 0) or 0
                    br = entry.get("bias_ratio_diff", 0) or 0
                    h1 = entry.get("bias_entry1_hits", 0) or 0
                    h2 = entry.get("bias_entry2_hits", 0) or 0
                    ht = h1 + h2
                    if da is not None and ca is not None and not any(isinstance(v, float) and np.isnan(v) for v in [da, ca]):
                        out["apo"]["x"].append(da)
                        out["apo"]["y"].append(ca)
                        out["apo"]["msa_pref_sum"].append(mp)
                        out["apo"]["bias_ratio_diff"].append(br)
                        out["apo"]["total_hits"].append(ht)
                        out["apo"]["labels"].append(f"{method}_{pair_type}_{cluster_id}_{pair_key}_apo")
                        out["apo"]["methods"].append(method)
                        out["apo"]["pair_types"].append(pair_type)
                    if dh is not None and ch is not None and not any(isinstance(v, float) and np.isnan(v) for v in [dh, ch]):
                        out["holo"]["x"].append(dh)
                        out["holo"]["y"].append(ch)
                        out["holo"]["msa_pref_sum"].append(mp)
                        out["holo"]["bias_ratio_diff"].append(br)
                        out["holo"]["total_hits"].append(ht)
                        out["holo"]["labels"].append(f"{method}_{pair_type}_{cluster_id}_{pair_key}_holo")
                        out["holo"]["methods"].append(method)
                        out["holo"]["pair_types"].append(pair_type)
    return out


def _annotate_spearman(ax, x, y, b, h, dyn_threshold, hits_min_quantile, fs=11):
    stats = compute_filtered_correlations(x, y, b, h, dyn_threshold, hits_min_quantile)
    line = "Pearson r: N/A\nSpearman ρ: N/A"
    if stats is not None:
        pearson_r, spearman_rho, n = stats
        line = f"Pearson r: {pearson_r:.3f}\nSpearman ρ: {spearman_rho:.3f}"
    ax.text(
        0.98, 0.04, line, transform=ax.transAxes, fontsize=fs,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def create_4panel_subplot(msa_data, output_dir: Path, dyn_threshold: float, hits_min_quantile: float, output_suffix="", use_msa_pref_color: bool = False):
    import matplotlib.gridspec as gridspec

    extracted = extract_data_from_filtered(msa_data)
    apo_holo = extract_apo_holo_separated(msa_data)
    af3_ligand_apo = filter_by_pair_type(filter_by_method(apo_holo["apo"], "af3"), "ligand-induced")
    boltz2_ligand_apo = filter_by_pair_type(filter_by_method(apo_holo["apo"], "boltz2"), "ligand-induced")
    af3_apo = filter_by_pair_type(filter_by_method(extracted, "af3"), "intrinsic")
    boltz2_apo = filter_by_pair_type(filter_by_method(extracted, "boltz2"), "intrinsic")

    panel_data = [
        (af3_ligand_apo, "AF3\nLigand-induced (Apo-conditioned)"),
        (boltz2_ligand_apo, "Boltz-2\nLigand-induced (Apo-conditioned)"),
        (af3_apo, "AF3\nIntrinsic Dynamics"),
        (boltz2_apo, "Boltz-2\nIntrinsic Dynamics"),
    ]

    fig = plt.figure(figsize=(28, 6.5))
    gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.04], wspace=0.20, right=0.96, left=0.04)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cbar_ax = fig.add_subplot(gs[0, 4])
    cmap = cm.coolwarm
    scat = None
    for ax, (d, title) in zip(axes, panel_data):
        x = np.array(d["x"])
        y = np.array(d["y"])
        b = np.array(d["bias_ratio_diff"])
        color_vals = np.array(d["msa_pref_sum"]) if use_msa_pref_color else b
        h = np.array(d["total_hits"])
        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        sizes = np.clip(h * 4, 30, 400)
        scat = ax.scatter(x, y, c=color_vals, cmap=cmap, vmin=-1, vmax=1, alpha=0.75, s=sizes, edgecolors="black", linewidth=0.7)
        ax.plot([-1.05, 1.05], [-1.05, 1.05], "k--", alpha=0.5, linewidth=1.5)
        ax.axvline(x=dyn_threshold, color="black", linestyle=":", linewidth=1.2, alpha=0.65)
        ax.axvline(x=-dyn_threshold, color="black", linestyle=":", linewidth=1.2, alpha=0.65)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        _annotate_spearman(ax, x, y, b, h, dyn_threshold, hits_min_quantile, fs=11)
    if scat is not None:
        cbar = fig.colorbar(scat, cax=cbar_ax, orientation="vertical")
        cbar.set_label(r"$\mathrm{MSA_{holo}}$" if use_msa_pref_color else r"$\mathrm{Train_{holo}}$", fontsize=16)
    out_png = output_dir / f"correlation_4panel_comparison{output_suffix}.png"
    out_pdf = output_dir / f"correlation_4panel_comparison{output_suffix}.pdf"
    plt.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved 4-panel plot: {out_png}")
    plt.close()


def create_supplementary_figure(msa_data, output_dir: Path, dyn_threshold: float, hits_min_quantile: float, output_suffix="", use_msa_pref_color: bool = False):
    import matplotlib.gridspec as gridspec

    extracted = extract_data_from_filtered(msa_data)
    apo_holo = extract_apo_holo_separated(msa_data)
    models = ["af3", "boltz1", "boltz2"]
    model_labels = ["AF3", "Boltz-1", "Boltz-2"]
    conditions = [
        ("intrinsic", None, "Intrinsic Dynamics"),
        ("ligand-induced", "apo", "Ligand-induced (Apo-conditioned)"),
        ("ligand-induced", "holo", "Ligand-induced (Holo-conditioned)"),
        ("protein-induced", "apo", "Protein-induced (Apo-conditioned)"),
        ("protein-induced", "holo", "Protein-induced (Holo-conditioned)"),
    ]

    fig = plt.figure(figsize=(18, 24))
    gs = gridspec.GridSpec(5, 4, figure=fig, width_ratios=[1, 1, 1, 0.04], hspace=0.25, wspace=0.25)
    cmap = cm.coolwarm
    scat = None
    for r, (ptype, ah, row_label) in enumerate(conditions):
        for c, (m, ml) in enumerate(zip(models, model_labels)):
            ax = fig.add_subplot(gs[r, c])
            d0 = filter_by_method(extracted, m) if ptype == "intrinsic" else filter_by_method(apo_holo[ah], m)
            d = filter_by_pair_type(d0, ptype)
            x = np.array(d["x"])
            y = np.array(d["y"])
            b = np.array(d["bias_ratio_diff"])
            color_vals = np.array(d["msa_pref_sum"]) if use_msa_pref_color else b
            h = np.array(d["total_hits"])
            if len(x) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            sizes = np.clip(h * 3, 20, 300)
            scat = ax.scatter(x, y, c=color_vals, cmap=cmap, vmin=-1, vmax=1, alpha=0.7, s=sizes, edgecolors="black", linewidth=0.5)
            ax.plot([-1.05, 1.05], [-1.05, 1.05], "k--", alpha=0.4, linewidth=1.2)
            ax.axvline(x=dyn_threshold, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
            ax.axvline(x=-dyn_threshold, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
            if r == 0:
                ax.set_title(ml, fontsize=16, fontweight="bold")
            if c == 0:
                ax.set_ylabel(row_label + "\n\n" + r"$\mathrm{Struct_{holo}}$", fontsize=12, fontweight="bold")
            if r == 4:
                ax.set_xlabel(r"$\mathrm{DynDisto_{holo}}$", fontsize=12, fontweight="bold")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            _annotate_spearman(ax, x, y, b, h, dyn_threshold, hits_min_quantile, fs=9)
    cbar_ax = fig.add_subplot(gs[1:4, 3])
    if scat is not None:
        cbar = fig.colorbar(scat, cax=cbar_ax, orientation="vertical")
        cbar.set_label(r"$\mathrm{MSA_{holo}}$" if use_msa_pref_color else r"$\mathrm{Train_{holo}}$", fontsize=14)
    out_png = output_dir / f"supplementary_figure_all_models_conditions{output_suffix}.png"
    out_pdf = output_dir / f"supplementary_figure_all_models_conditions{output_suffix}.pdf"
    plt.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved supplementary figure: {out_png}")
    plt.close()


def _collect_scope_rows(data, scope_name, dyn_thresholds, hits_min_quantile, rows):
    x = np.array(data["x"])
    y = np.array(data["y"])
    b = np.array(data["bias_ratio_diff"])
    h = np.array(data["total_hits"])
    for dt in dyn_thresholds:
        stats = compute_filtered_correlations(x, y, b, h, dt, hits_min_quantile)
        row = {
            "scope": scope_name,
            "dyn_threshold": dt,
            "pearson_r": "",
            "spearman_rho": "",
            "n": "",
        }
        if stats is not None:
            pearson_r, spearman_rho, n = stats
            row["pearson_r"] = f"{pearson_r:.6f}"
            row["spearman_rho"] = f"{spearman_rho:.6f}"
            row["n"] = str(n)
        rows.append(row)


def write_sweep_table(msa_data, table_path: Path, dyn_thresholds, hits_min_quantile: float):
    extracted = extract_data_from_filtered(msa_data)
    apo_holo = extract_apo_holo_separated(msa_data)
    rows = []
    _collect_scope_rows(extracted, "all", dyn_thresholds, hits_min_quantile, rows)
    models = sorted(set(extracted["methods"]))
    for m in models:
        _collect_scope_rows(filter_by_method(extracted, m), f"{m}::all", dyn_thresholds, hits_min_quantile, rows)
        _collect_scope_rows(filter_by_pair_type(filter_by_method(extracted, m), "intrinsic"), f"{m}::intrinsic", dyn_thresholds, hits_min_quantile, rows)
        _collect_scope_rows(filter_by_pair_type(filter_by_method(apo_holo["apo"], m), "ligand-induced"), f"{m}::ligand-induced::apo", dyn_thresholds, hits_min_quantile, rows)
        _collect_scope_rows(filter_by_pair_type(filter_by_method(apo_holo["holo"], m), "ligand-induced"), f"{m}::ligand-induced::holo", dyn_thresholds, hits_min_quantile, rows)
        _collect_scope_rows(filter_by_pair_type(filter_by_method(apo_holo["apo"], m), "protein-induced"), f"{m}::protein-induced::apo", dyn_thresholds, hits_min_quantile, rows)
        _collect_scope_rows(filter_by_pair_type(filter_by_method(apo_holo["holo"], m), "protein-induced"), f"{m}::protein-induced::holo", dyn_thresholds, hits_min_quantile, rows)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scope", "dyn_threshold", "pearson_r", "spearman_rho", "n"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved sweep table: {table_path}")


@click.command()
@click.option("--filtered-bias-json", type=click.Path(path_type=Path), default=E.dir("filtered_pairs") / "filtered_pairs_bias0.3_legacy.json", show_default=True)
@click.option("--filtered-msa-json", type=click.Path(path_type=Path), default=E.dir("filtered_pairs") / "filtered_pairs_msa0.3_legacy.json", show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=E.dir("plots") / "dynamic_legacy", show_default=True)
@click.option("--only-4panel/--all-plots", default=True, show_default=True)
@click.option("--dyn-threshold", type=float, default=0.7, show_default=True)
@click.option("--hits-min-quantile", type=float, default=25.0, show_default=True)
@click.option("--dyn-thresholds", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8", show_default=True)
@click.option("--table-output-csv", type=click.Path(path_type=Path), default=None)
def main(filtered_bias_json: Path, filtered_msa_json: Path, output_dir: Path, only_4panel: bool, dyn_threshold: float, hits_min_quantile: float, dyn_thresholds: str, table_output_csv: Path | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    bias_data = load_data(filtered_bias_json)
    msa_data = load_data(filtered_msa_json)
    sweep_thresholds = [float(x.strip()) for x in dyn_thresholds.split(",") if x.strip()]

    if only_4panel:
        create_4panel_subplot(
            bias_data, output_dir, dyn_threshold, hits_min_quantile,
            output_suffix="_no_training_bias_legacy", use_msa_pref_color=True
        )
        create_4panel_subplot(msa_data, output_dir, dyn_threshold, hits_min_quantile, output_suffix="_no_msa_bias_legacy")
        create_supplementary_figure(
            bias_data, output_dir, dyn_threshold, hits_min_quantile,
            output_suffix="_no_training_bias_legacy", use_msa_pref_color=True
        )
        create_supplementary_figure(msa_data, output_dir, dyn_threshold, hits_min_quantile, output_suffix="_no_msa_bias_legacy")
    else:
        create_4panel_subplot(msa_data, output_dir, dyn_threshold, hits_min_quantile, output_suffix="_legacy")
        create_supplementary_figure(
            bias_data, output_dir, dyn_threshold, hits_min_quantile,
            output_suffix="_no_training_bias_legacy", use_msa_pref_color=True
        )
        create_supplementary_figure(msa_data, output_dir, dyn_threshold, hits_min_quantile, output_suffix="_no_msa_bias_legacy")

    if table_output_csv is not None:
        write_sweep_table(msa_data, table_output_csv, sweep_thresholds, hits_min_quantile)


if __name__ == "__main__":
    main()
