from __future__ import annotations

import json
from pathlib import Path

import click

from utils._config import eval_cfg as E


def _count_pairs(data: dict) -> int:
    return sum(
        len(cluster_data)
        for model_data in data.values()
        for category_data in model_data.values()
        for cluster_data in category_data.values()
    )


@click.command()
@click.option("--threshold", type=float, default=0.2, show_default=True, help="Threshold for both filters.")
@click.option(
    "--input-json",
    type=click.Path(path_type=Path),
    default=E.file("merged_json"),
    show_default=True,
    help="Merged input JSON (legacy: bias_ratio_diff fields expected).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=E.dir("filtered_pairs"),
    show_default=True,
    help="Directory to write filtered JSON files.",
)
@click.option(
    "--suffix",
    type=str,
    default="",
    help="Optional suffix appended to output filename before .json (e.g. _legacy).",
)
def main(threshold: float, input_json: Path, output_dir: Path, suffix: str) -> None:
    print(f"Using threshold: {threshold}")
    print(f"Loading JSON file: {input_json}")
    with open(input_json, "r") as f:
        data = json.load(f)

    filtered_bias = {}
    filtered_msa = {}

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
                    bias_ratio_diff = pair_info.get("bias_ratio_diff")
                    if bias_ratio_diff is not None and abs(bias_ratio_diff) < threshold:
                        filtered_bias[model_name][category][cluster_id][pair_name] = pair_info

                    msa_pref_sum = pair_info.get("msa_pref_sum")
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
    bias_filtered_pairs = _count_pairs(filtered_bias)
    msa_filtered_pairs = _count_pairs(filtered_msa)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_bias = output_dir / f"filtered_pairs_bias{threshold}{suffix}.json"
    output_file_msa = output_dir / f"filtered_pairs_msa{threshold}{suffix}.json"

    with open(output_file_bias, "w") as f:
        json.dump(filtered_bias, f, indent=2)
    with open(output_file_msa, "w") as f:
        json.dump(filtered_msa, f, indent=2)

    print("\nFiltering complete!")
    print(f"Total pairs: {total_pairs}")
    print(f"\nFilter 1 - |bias_ratio_diff| < {threshold}:")
    print(f"  Filtered pairs: {bias_filtered_pairs}")
    print(f"  Output: {output_file_bias}")
    print(f"\nFilter 2 - |msa_pref_sum| < {threshold}:")
    print(f"  Filtered pairs: {msa_filtered_pairs}")
    print(f"  Output: {output_file_msa}")


if __name__ == "__main__":
    main()
