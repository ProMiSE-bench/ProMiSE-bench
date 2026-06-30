#!/usr/bin/env python3
"""Update preference_scores.json with fresh ConfBench outputs for one model."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click

SET_NAME_MAP = {
    "intrinsic": "intrinsic-dynamics",
    "ligand-induced": "ligand-induced",
    "protein-induced": "protein-induced",
}


def _strip_state_suffix(yaml_tag: str) -> str:
    if len(yaml_tag) >= 2 and yaml_tag[-2] == "_" and yaml_tag[-1] in "mx":
        return yaml_tag[:-2]
    return yaml_tag


def _preference_pair_key(conf1: str, conf2: str) -> str:
    return f"{_strip_state_suffix(conf1)}-{_strip_state_suffix(conf2)}"


def _find_confbench_entry(
    confbench: dict,
    set_name: str,
    cluster_id: str,
    conf1: str,
    conf2: str,
) -> Optional[dict]:
    for pk in (f"{cluster_id}_{conf1}_{conf2}", f"{cluster_id}_{conf2}_{conf1}"):
        entry = confbench.get(set_name, {}).get(pk)
        if entry is not None:
            return entry
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isnan(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _update_intrinsic(entry: dict, struct_m: dict, disto_m: dict, rmsd: Optional[float]) -> None:
    if rmsd is not None:
        entry["rmsd_conf1_conf2"] = rmsd
    entry["struct_holo"] = _as_float(struct_m.get("mean_confbench_score"))
    entry["disto_holo"] = _as_float(disto_m.get("mean_confbench_score"))
    entry["dyndisto_holo"] = _as_float(disto_m.get("mean_dynamic_confbench_score"))


def _update_induced(entry: dict, struct_m: dict, disto_m: dict, rmsd: Optional[float]) -> None:
    if rmsd is not None:
        entry["rmsd_conf1_conf2"] = rmsd
    struct_apo = struct_m.get("apo_predictions") or {}
    struct_holo = struct_m.get("holo_predictions") or {}
    disto_apo = disto_m.get("apo_predictions") or {}
    disto_holo = disto_m.get("holo_predictions") or {}

    entry["confbench_apo_pred"] = _as_float(struct_apo.get("mean_confbench_score"))
    entry["confbench_holo_pred"] = _as_float(struct_holo.get("mean_confbench_score"))
    entry["distogram_confbench_apo"] = _as_float(disto_apo.get("mean_confbench_score"))
    entry["distogram_confbench_holo"] = _as_float(disto_holo.get("mean_confbench_score"))
    entry["distogram_dynamic_confbench_apo"] = _as_float(
        disto_apo.get("mean_dynamic_confbench_score")
    )
    entry["distogram_dynamic_confbench_holo"] = _as_float(
        disto_holo.get("mean_dynamic_confbench_score")
    )


def update_preference_scores(
    preference_path: Path,
    valid_pairs_path: Path,
    struct_confbench_path: Path,
    distogram_confbench_path: Path,
    model: str,
    *,
    backup: bool = True,
) -> dict[str, int]:
    with open(preference_path) as f:
        preference = json.load(f)
    with open(valid_pairs_path) as f:
        valid_pairs = json.load(f)
    with open(struct_confbench_path) as f:
        struct_confbench = json.load(f)
    with open(distogram_confbench_path) as f:
        distogram_confbench = json.load(f)

    if model not in preference:
        raise click.ClickException(f"Model {model!r} not found in {preference_path}")

    stats = {
        "pairs_total": 0,
        "pairs_updated": 0,
        "missing_struct": 0,
        "missing_distogram": 0,
        "missing_pref_entry": 0,
    }

    for set_name, clusters in valid_pairs.items():
        pref_set_name = SET_NAME_MAP[set_name]
        model_set = preference[model].setdefault(pref_set_name, {})

        for cluster_id, pairs in clusters.items():
            cluster_bucket = model_set.setdefault(cluster_id, {})

            for conf1, conf2 in pairs:
                stats["pairs_total"] += 1
                pair_key = _preference_pair_key(conf1, conf2)
                if pair_key not in cluster_bucket:
                    stats["missing_pref_entry"] += 1
                    continue

                struct_entry = _find_confbench_entry(
                    struct_confbench, set_name, cluster_id, conf1, conf2
                )
                disto_entry = _find_confbench_entry(
                    distogram_confbench, set_name, cluster_id, conf1, conf2
                )
                if struct_entry is None:
                    stats["missing_struct"] += 1
                    continue
                if disto_entry is None:
                    stats["missing_distogram"] += 1
                    continue

                struct_m = struct_entry.get("models", {}).get(model, {})
                disto_m = disto_entry.get("models", {}).get(model, {})
                rmsd = _as_float(
                    struct_entry.get("rmsd_ref1_ref2")
                    or struct_entry.get("rmsd_apo_holo_ref")
                )

                pref_entry = cluster_bucket[pair_key]
                if set_name == "intrinsic":
                    _update_intrinsic(pref_entry, struct_m, disto_m, rmsd)
                else:
                    _update_induced(pref_entry, struct_m, disto_m, rmsd)

                stats["pairs_updated"] += 1

    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = preference_path.with_suffix(f".json.bak.{ts}")
        shutil.copy2(preference_path, backup_path)
        click.echo(f"Backup: {backup_path}")

    with open(preference_path, "w") as f:
        json.dump(preference, f, indent=2)
        f.write("\n")

    return stats


@click.command()
@click.option(
    "--preference-scores",
    type=click.Path(path_type=Path),
    default=Path("data/preference_scores.json"),
    show_default=True,
)
@click.option(
    "--valid-pairs",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/dataset/valid_pairs.json"),
    show_default=True,
)
@click.option(
    "--struct-confbench",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data_eval/confbench_scores_boltz2.json"),
    show_default=True,
)
@click.option(
    "--distogram-confbench",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data_eval/confbench_scores_distogram_boltz2.json"),
    show_default=True,
)
@click.option("--model", "-m", default="boltz2", show_default=True)
@click.option("--no-backup", is_flag=True, default=False)
def main(
    preference_scores: Path,
    valid_pairs: Path,
    struct_confbench: Path,
    distogram_confbench: Path,
    model: str,
    no_backup: bool,
) -> None:
    """Patch ConfBench fields in preference_scores.json for one model."""
    if not preference_scores.exists():
        raise click.ClickException(f"Missing {preference_scores}")

    stats = update_preference_scores(
        preference_scores,
        valid_pairs,
        struct_confbench,
        distogram_confbench,
        model,
        backup=not no_backup,
    )

    click.echo(f"Updated {preference_scores} for model={model}")
    for key, value in stats.items():
        click.echo(f"  {key}: {value}")


if __name__ == "__main__":
    main()
