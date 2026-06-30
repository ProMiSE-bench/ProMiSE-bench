#!/usr/bin/env python3
"""
Dynamic correlation plotting (v2, DynDisto-correlation variant):
- Keep plotting behavior identical to plot_correlation_dynamic_v2.py
- Change on-plot correlation annotation target from Struct_holo to DynDisto_holo
"""

from __future__ import annotations

from eval.plot import plot_correlation_dynamic_v2 as base


def _annotate_correlations_vs_dyn(ax, x, y, correlate_vals, dyn_threshold, fs=11):
    # Same filter as base script: |DynDisto_holo| < threshold.
    # Correlation target is DynDisto_holo (x), not Struct_holo (y).
    stats = base.compute_dyn_correlations(x, correlate_vals, dyn_threshold)
    line = "Pearson r: N/A\nSpearman ρ: N/A"
    if stats is not None:
        pearson_r, spearman_rho, _ = stats
        line = f"Pearson r: {pearson_r:.3f}\nSpearman ρ: {spearman_rho:.3f}"
    ax.text(
        0.98,
        0.04,
        line,
        transform=ax.transAxes,
        fontsize=fs,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


# Monkey-patch only the annotation function so every plot path stays identical.
base._annotate_correlations = _annotate_correlations_vs_dyn


if __name__ == "__main__":
    base.main()
