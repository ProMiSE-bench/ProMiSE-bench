#!/usr/bin/env python3
"""Aggregate per-seed msa_bias_results.csv into per_pair_summary.csv."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from utils._config import eval_cfg as E


def summarize_msa_bias(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["set_name", "cluster_id", "conf1_name", "conf2_name"]
    rows = []
    for key, group in df.groupby(group_cols, sort=True):
        prefs = group["msa_pref"].astype(float)
        mean_pref = float(prefs.mean()) if len(prefs) else 0.0
        majority_sign = np.sign(mean_pref) if mean_pref != 0 else 0.0
        if majority_sign == 0:
            same_sign = float((prefs == 0).mean())
        else:
            same_sign = float((np.sign(prefs) == majority_sign).mean())
        rows.append(
            {
                "set_name": key[0],
                "cluster_id": key[1],
                "conf1_name": key[2],
                "conf2_name": key[3],
                "msa_pref_sum": float(prefs.sum()),
                "msa_pref_avg": mean_pref,
                "same_sign_sum_avg": same_sign,
                "over_coverage_0.1": float((prefs.abs() > 0.1).mean()),
                "n_seeds": int(len(prefs)),
            }
        )
    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--input-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=lambda: E.file("msa_bias_csv"),
    show_default=True,
)
@click.option(
    "--output-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=lambda: E.file("msa_pref_csv"),
    show_default=True,
)
def main(input_csv: Path, output_csv: Path) -> None:
    """Write per_pair_summary.csv from msa_bias_results.csv."""
    df = pd.read_csv(input_csv)
    required = {"set_name", "cluster_id", "conf1_name", "conf2_name", "msa_pref"}
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(f"Missing columns in {input_csv}: {sorted(missing)}")

    summary = summarize_msa_bias(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    click.echo(f"Wrote {len(summary)} rows → {output_csv}")


if __name__ == "__main__":
    main()
