#!/usr/bin/env python3
"""
Add bias_ratio_diff and hits_eff in merged_valid_pairs_data.json from training_bias files.

This script never overwrites the input merged JSON. It writes a new output file.
"""

from __future__ import annotations

import json
from pathlib import Path

import click


# merged model -> training bias filename
BIAS_FILE_MAP = {
    "af3": "training_bias_per_pair_af3.json",
    "boltz1": "training_bias_per_pair_af3.json",  # boltz1 shares af3 bias
    "boltz2": "training_bias_per_pair_boltz_2.json",
    "chai": "training_bias_per_pair_chai_1.json",
    "bioemu": "training_bias_per_pair_bioemu.json",
}


def build_training_bias_lookup(
    training_bias_dir: Path,
) -> dict[str, dict[tuple[str, str, str, str], tuple[float, float, float]]]:
    """Build lookup: model -> (set, cluster, conf1, conf2) -> (bias_ratio_diff, hits_eff_a, hits_eff_b)."""
    lookup: dict[str, dict[tuple[str, str, str, str], tuple[float, float, float]]] = {}

    for model_name, bias_filename in BIAS_FILE_MAP.items():
        bias_path = training_bias_dir / bias_filename
        if not bias_path.exists():
            click.echo(f"Warning: bias file not found for {model_name}: {bias_path}")
            continue

        with open(bias_path, "r") as f:
            bias_data = json.load(f)

        lookup[model_name] = {}
        for set_name, entries in bias_data.items():
            for entry in entries:
                cluster_id = entry["cluster_name"]
                entry1 = entry["entry1"]
                entry2 = entry["entry2"]
                bias_ratio_raw = entry.get("bias_ratio_diff")
                if bias_ratio_raw is None:
                    continue
                bias_ratio_diff = float(bias_ratio_raw)
                # hits_eff for each side
                hits_eff_a = entry.get("hits_eff_a")
                if hits_eff_a is None:
                    hits_eff_a = entry.get("entry1_hits", 0.0)
                hits_eff_b = entry.get("hits_eff_b")
                if hits_eff_b is None:
                    hits_eff_b = entry.get("entry2_hits", 0.0)

                # Forward order
                lookup[model_name][(set_name, cluster_id, entry1, entry2)] = (
                    bias_ratio_diff,
                    hits_eff_a,
                    hits_eff_b,
                )
                # Reverse order
                lookup[model_name][(set_name, cluster_id, entry2, entry1)] = (
                    -bias_ratio_diff,
                    hits_eff_b,
                    hits_eff_a,
                )

    return lookup


def add_training_bias_fields(
    merged_data: dict,
    bias_lookup: dict[str, dict[tuple[str, str, str, str], tuple[float, float, float]]],
) -> tuple[dict, int, int]:
    """Return updated merged data and counts of updated/missing pairs."""
    updated_count = 0
    missing_count = 0

    for model_name, model_data in merged_data.items():
        model_lookup = bias_lookup.get(model_name, {})
        if not isinstance(model_data, dict):
            continue

        for set_name, set_data in model_data.items():
            if not isinstance(set_data, dict):
                continue

            for cluster_id, cluster_data in set_data.items():
                if not isinstance(cluster_data, dict):
                    continue

                for pair_key, pair_info in cluster_data.items():
                    # pair_key format: "{conf1}-{conf2}"
                    if "-" not in pair_key:
                        missing_count += 1
                        continue
                    conf1, conf2 = pair_key.split("-", 1)
                    bias_fields = model_lookup.get((set_name, cluster_id, conf1, conf2))
                    if bias_fields is None:
                        missing_count += 1
                        continue

                    bias_ratio_diff, hits_eff_a, hits_eff_b = bias_fields
                    hits_eff = hits_eff_a + hits_eff_b
                    pair_info["bias_ratio_diff"] = bias_ratio_diff
                    pair_info["hits_eff_a"] = hits_eff_a
                    pair_info["hits_eff_b"] = hits_eff_b
                    pair_info["hits_eff"] = hits_eff
                    updated_count += 1

    return merged_data, updated_count, missing_count


@click.command()
@click.option(
    "--merged-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data_eval/merged_valid_pairs_data.json"),
    show_default=True,
    help="Path to merged_valid_pairs_data.json",
)
@click.option(
    "--training-bias-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data_eval/train/training_bias"),
    show_default=True,
    help="Directory containing training_bias_per_pair_*.json files",
)
@click.option(
    "--output-json",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for updated merged JSON (new file)",
)
def main(merged_json: Path, training_bias_dir: Path, output_json: Path | None) -> None:
    if output_json is None:
        output_json = merged_json.with_name(f"{merged_json.stem}_with_bias_ratio_diff.json")

    click.echo(f"Loading merged JSON: {merged_json}")
    with open(merged_json, "r") as f:
        merged_data = json.load(f)

    click.echo(f"Loading training bias from: {training_bias_dir}")
    bias_lookup = build_training_bias_lookup(training_bias_dir)
    updated_data, updated_count, missing_count = add_training_bias_fields(merged_data, bias_lookup)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(updated_data, f, indent=2)

    click.echo("Done.")
    click.echo(f"Pairs with new bias_ratio_diff/hits_eff: {updated_count}")
    click.echo(f"Missing pairs (no training-bias match): {missing_count}")
    click.echo(f"Saved new file: {output_json}")


if __name__ == "__main__":
    main()
