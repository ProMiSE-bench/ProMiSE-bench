from __future__ import annotations

import json
from pathlib import Path

import click

from utils._config import eval_cfg as E

BIAS_RATIO_FIELD = "bias_ratio_diff"


def _count_pairs(data: dict) -> int:
    return sum(
        len(cluster_data)
        for model_data in data.values()
        for category_data in model_data.values()
        for cluster_data in category_data.values()
    )


def _default_input_json() -> Path:
    output_input = E.dir("output") / E.filename("merged_json")
    if output_input.exists():
        return output_input
    return Path(E.file("merged_json"))


def _default_output_dir() -> Path:
    filtered_dir = E.dir("filtered_pairs")
    if filtered_dir.name.endswith("_v2"):
        return filtered_dir
    return filtered_dir.with_name(f"{filtered_dir.name}_v2")


def _numeric_value(entry: dict, field_name: str) -> float | None:
    value = entry.get(field_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@click.command()
@click.option("--threshold", type=float, default=0.3, show_default=True, help="Threshold for both filters.")
@click.option(
    "--input-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_default_input_json,
    show_default="eval output merged JSON when available",
    help="Merged input JSON.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=_default_output_dir,
    show_default="eval filtered_pairs dir with _v2 suffix",
    help="Directory to write filtered JSON files.",
)
@click.option(
    "--suffix",
    type=str,
    default="_v2",
    show_default=True,
    help="Optional suffix appended to output filename before .json (e.g. _v2).",
)
def main(threshold: float, input_json: Path, output_dir: Path, suffix: str) -> None:
    click.echo(f"Using threshold: {threshold}")
    click.echo(f"Loading JSON file: {input_json}")
    with open(input_json, "r") as f:
        data = json.load(f)

    filtered_bias = {}
    filtered_msa = {}
    bias_values_seen = 0
    msa_values_seen = 0

    for model_name, model_data in data.items():
        filtered_bias[model_name] = {}
        filtered_msa[model_name] = {}
        for category, category_data in model_data.items():
            filtered_bias[model_name][category] = {}
            filtered_msa[model_name][category] = {}
            for cluster_id, cluster_data in category_data.items():
                filtered_bias[model_name][category][cluster_id] = {}
                filtered_msa[model_name][category][cluster_id] = {}
                for pair_name, pair_info in cluster_data.items():
                    bias_ratio_diff = _numeric_value(pair_info, BIAS_RATIO_FIELD)
                    if bias_ratio_diff is not None:
                        bias_values_seen += 1
                    if bias_ratio_diff is not None and abs(bias_ratio_diff) < threshold:
                        filtered_bias[model_name][category][cluster_id][pair_name] = pair_info
                    msa_pref_sum = _numeric_value(pair_info, "msa_pref_sum")
                    if msa_pref_sum is not None:
                        msa_values_seen += 1
                    if msa_pref_sum is not None and abs(msa_pref_sum) < threshold:
                        filtered_msa[model_name][category][cluster_id][pair_name] = pair_info

                if not filtered_bias[model_name][category][cluster_id]:
                    del filtered_bias[model_name][category][cluster_id]
                if not filtered_msa[model_name][category][cluster_id]:
                    del filtered_msa[model_name][category][cluster_id]
            if not filtered_bias[model_name][category]:
                del filtered_bias[model_name][category]
            if not filtered_msa[model_name][category]:
                del filtered_msa[model_name][category]
        if not filtered_bias[model_name]:
            del filtered_bias[model_name]
        if not filtered_msa[model_name]:
            del filtered_msa[model_name]

    total_pairs = _count_pairs(data)
    if bias_values_seen == 0:
        click.echo(
            "Warning: no bias_ratio_diff values found; "
            "writing an unfiltered training-bias JSON so plotting can proceed."
        )
        filtered_bias = data
    if msa_values_seen == 0:
        click.echo("Warning: no msa_pref_sum values found; writing an empty MSA-filtered JSON.")

    bias_filtered_pairs = _count_pairs(filtered_bias)
    msa_filtered_pairs = _count_pairs(filtered_msa)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_bias = output_dir / f"filtered_pairs_bias{threshold}{suffix}.json"
    output_file_msa = output_dir / f"filtered_pairs_msa{threshold}{suffix}.json"

    with open(output_file_bias, "w") as f:
        json.dump(filtered_bias, f, indent=2)
    with open(output_file_msa, "w") as f:
        json.dump(filtered_msa, f, indent=2)

    click.echo("Filtering complete.")
    click.echo(f"Total pairs: {total_pairs}")
    click.echo(f"Filter 1 - |bias_ratio_diff| < {threshold}: {bias_filtered_pairs} pairs")
    click.echo(f"Output: {output_file_bias}")
    click.echo(f"Filter 2 - |msa_pref_sum| < {threshold}: {msa_filtered_pairs} pairs")
    click.echo(f"Output: {output_file_msa}")


if __name__ == "__main__":
    main()
