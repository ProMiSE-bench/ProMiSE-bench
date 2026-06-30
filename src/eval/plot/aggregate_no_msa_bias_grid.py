#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt


@click.command()
@click.option(
    "--plot-root-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Root plot directory that contains msa_<threshold>/ subdirectories.",
)
@click.option(
    "--msa-thresholds",
    type=str,
    required=True,
    help='Comma-separated MSA thresholds, e.g. "0.2,0.3,0.4".',
)
@click.option(
    "--input-filename",
    type=str,
    default="correlation_4panel_comparison_no_msa_bias_v2.png",
    show_default=True,
    help="Per-threshold panel filename to aggregate.",
)
@click.option(
    "--output-filename",
    type=str,
    default="correlation_12panel_no_msa_bias_aggregate.png",
    show_default=True,
    help="Output filename (saved under plot-root-dir).",
)
def main(plot_root_dir: Path, msa_thresholds: str, input_filename: str, output_filename: str) -> None:
    thresholds = [t.strip() for t in msa_thresholds.split(",") if t.strip()]
    if not thresholds:
        raise SystemExit("Error: --msa-thresholds is empty.")

    rows = []
    for t in thresholds:
        img_path = plot_root_dir / f"msa_{t}" / input_filename
        if not img_path.exists():
            raise SystemExit(f"Error: Missing image for msa threshold {t}: {img_path}")
        rows.append((t, plt.imread(img_path)))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(20, 5.8 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ax, (t, img) in zip(axes, rows):
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.01,
            1.03,
            f"MSA threshold = {t}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            #bbox=dict(boxstyle="round,pad=0.30", facecolor="white", alpha=0.9),
            clip_on=False,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_png = plot_root_dir / output_filename
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved aggregate image: {out_png}")
    print(f"Saved aggregate image (PDF): {out_pdf}")


if __name__ == "__main__":
    main()
