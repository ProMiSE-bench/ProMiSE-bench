#!/usr/bin/env python3
"""
Audit AF3/Boltz chain-mapping JSON coverage for ``data/dataset`` CSV scope only.

Uses the same cluster/tag set as ``make_pairs`` (CSV → clusters.json filter →
on-disk predictions). Reports keys missing from each mapping JSON.

Usage::

    PYTHONPATH=src python -m curation.audit_chain_mappings \\
        --examples-dir /home.galaxy4/bonjae02/projects/promise-samples
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import click

from curation.make_pairs import (
    SET_NAMES,
    _enrich_cfg,
    _load_chain_mapping_json_cached,
    _map_cfg_path,
    add_predictions_to_data,
    build_valid_pairs_from_csv,
    extract_chain_from_yaml_tag,
    filter_data,
    load_csv,
    load_valid_centers,
    process_induced_set,
    process_intrinsic,
    restrict_to_valid_pairs,
)
_METHODS = ("af3", "boltz2", "boltz1")
_CONFIG_KEYS = {
    "af3": "af3_chain_mappings",
    "boltz2": "boltz_chain_mappings",
    "boltz1": "boltz1_chain_mappings",
}


def _load_csv_scope(csv_dir: Path, clusters_json: Path) -> Dict:
    data = {
        "intrinsic": process_intrinsic(load_csv(csv_dir / "intrinsic.csv")),
    }
    for set_name in ("protein-induced", "ligand-induced"):
        data[set_name] = process_induced_set(load_csv(csv_dir / f"{set_name}.csv"))
    return filter_data(data, load_valid_centers(clusters_json))


def _required_mapping_keys(data: Dict) -> Dict[str, List[Tuple[str, str, str]]]:
    """method -> [(json_key, interested_chain, side), ...] unique by json_key."""
    out: Dict[str, dict[str, Tuple[str, str, str]]] = {m: {} for m in _METHODS}
    for set_name, clusters in data.items():
        for cluster_id, info in clusters.items():
            for yaml_tag, preds in (info.get("apo_predictions") or {}).items():
                if not isinstance(preds, dict):
                    continue
                for method in _METHODS:
                    if method in preds:
                        key = f"{set_name}/{cluster_id}/{yaml_tag}"
                        chain = extract_chain_from_yaml_tag(yaml_tag)
                        out[method][key] = (key, chain, "apo")
            for yaml_tag, preds in (info.get("holo_predictions") or {}).items():
                for method in _METHODS:
                    if method in preds:
                        key = f"{set_name}/{cluster_id}/{yaml_tag}"
                        chain = extract_chain_from_yaml_tag(yaml_tag)
                        out[method][key] = (key, chain, "holo")
    return {m: list(d.values()) for m, d in out.items()}


def _mapping_has_chain(
    all_mappings: dict, key: str, interested_chain: str
) -> bool:
    entry = all_mappings.get(key)
    if not isinstance(entry, dict):
        return False
    modeled = entry.get("mapping")
    if not isinstance(modeled, dict) or not modeled:
        return False
    rev = {v: k for k, v in modeled.items()}
    return interested_chain in rev


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--csv-dir",
    type=click.Path(exists=True, file_okay=False),
    default="data/dataset",
    show_default=True,
)
@click.option(
    "--clusters-json",
    type=click.Path(exists=True, dir_okay=False),
    default="data/clusters.json",
    show_default=True,
)
@click.option(
    "--examples-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Prediction root (e.g. promise-samples).",
)
@click.option(
    "--missing-out",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional JSON file listing missing keys per method.",
)
def main(csv_dir, clusters_json, examples_dir, missing_out) -> None:
    """Report chain-mapping JSON gaps for data/dataset CSV scope."""
    csv_dir = Path(csv_dir)
    examples_dir = Path(examples_dir)
    de = _enrich_cfg()

    data = _load_csv_scope(csv_dir, Path(clusters_json))
    valid_pairs = build_valid_pairs_from_csv(csv_dir, data)
    data = restrict_to_valid_pairs(data, valid_pairs, csv_dir)
    data = add_predictions_to_data(data, examples_dir)
    required = _required_mapping_keys(data)

    n_clusters = sum(len(v) for v in data.values())
    click.echo(f"CSV scope: {n_clusters} clusters across {', '.join(SET_NAMES)}")

    missing_by_method: Dict[str, List[dict]] = defaultdict(list)
    for method in _METHODS:
        map_path = _map_cfg_path(de.get(_CONFIG_KEYS[method]))
        entries = required[method]
        click.echo(f"\n{method}: {len(entries)} keys (CSV scope + on-disk prediction)")
        if not map_path or not Path(map_path).exists():
            click.echo(f"  [MISSING FILE] {_CONFIG_KEYS[method]} -> {map_path or '(not set)'}")
            for key, chain, side in entries:
                missing_by_method[method].append(
                    {"key": key, "interested_chain": chain, "side": side, "reason": "no_file"}
                )
            continue

        all_mappings = _load_chain_mapping_json_cached(map_path)
        missing = []
        for key, chain, side in entries:
            if not _mapping_has_chain(all_mappings, key, chain):
                missing.append((key, chain, side))
                missing_by_method[method].append(
                    {"key": key, "interested_chain": chain, "side": side, "reason": "no_entry"}
                )
        click.echo(f"  file: {map_path}")
        click.echo(f"  covered: {len(entries) - len(missing)}/{len(entries)}")
        click.echo(f"  missing: {len(missing)}")
        for key, chain, side in missing[:10]:
            click.echo(f"    {key}  chain={chain!r}  ({side})")
        if len(missing) > 10:
            click.echo(f"    ... and {len(missing) - 10} more")

    if missing_out:
        out_path = Path(missing_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(dict(missing_by_method), fh, indent=2)
        click.echo(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
