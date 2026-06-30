"""Classify cluster members as apo vs holo using the same assembly logic as prepare_inputs_gemmi.

Run from the repo root (promise conda env)::

    conda activate promise
    PYTHONPATH=src python -m curation.pipeline.cluster_apo_scan --mmcif-dir /path/to/mmcif_files
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import gettempdir

import click
import pandas as pd
from tqdm.auto import tqdm

from utils._config import pipeline_cfg as C

from .prepare_inputs_gemmi import fetch_assemblies, read_cluster_members
from .types import AssemblyResult
from ..utils._pdb_helpers import check_auth_base


def cluster_csv_to_path(cluster_csv: str, clusters_root: Path) -> Path:
    """Map ``C9/1C9B_3`` -> ``clusters_root/C9/1C9B_3.csv``."""
    s = cluster_csv.strip()
    if "/" not in s:
        raise ValueError(f"cluster_csv must contain '/': {cluster_csv!r}")
    a, b = s.split("/", 1)
    return clusters_root / a / f"{b}.csv"


def classify_member(
    assemblies: list[AssemblyResult] | None, auth: str
) -> tuple[str, str, str, list[AssemblyResult]]:
    """Return (status, detail, ligand_summary, matched_assemblies).

    * **apo** — every assembly that contains this auth chain has no ligands (after exclusions).
    * **holo** — at least one such assembly has a ligand.
    * **unknown** — cannot classify (missing CIF, no assemblies, or chain not in any assembly).
    """
    if assemblies is None:
        return "unknown", "missing_mmcif", "", []
    if not assemblies:
        return "unknown", "no_assemblies_after_filters", "", []

    matched: list[AssemblyResult] = []
    for asm in assemblies:
        for ach in asm.chain_list_author:
            if check_auth_base(auth, ach):
                matched.append(asm)
                break

    if not matched:
        return "unknown", "no_matching_assembly", "", []

    if any(a.ligand_count > 0 for a in matched):
        comps: set[str] = set()
        for a in matched:
            if a.ligand_count > 0:
                comps.update(a.ligand_list)
        return "holo", "", ";".join(sorted(comps)), matched

    return "apo", "", "", matched


def save_assembly(asm: AssemblyResult, pdb: str, save_dir: Path) -> None:
    """Save assembly CIF + auth-to-mmcif map, matching prepare_inputs_gemmi layout."""
    pdb_upper = pdb.upper()
    rep = pdb_upper[1:3]
    cif_file = save_dir / rep / pdb_upper / f"asm_{pdb}_{asm.assembly_id}.cif"
    map_file = save_dir / rep / pdb_upper / f"asm_{pdb}_{asm.assembly_id}_map.json"
    if cif_file.exists() and map_file.exists():
        return
    cif_file.parent.mkdir(parents=True, exist_ok=True)
    asm.assembly.make_mmcif_block().write_file(str(cif_file))
    with map_file.open("w", encoding="utf-8") as f:
        json.dump(asm.assembly_auth_to_mmcif, f, indent=2, ensure_ascii=False)


def main_inner(
    intrinsic_csv: Path,
    clusters_root: Path,
    mmcif_dir: Path,
    out_csv: Path,
    max_resolution: float,
    max_polymer_instances: int,
    max_lig_instances: int,
    exclude_na: bool,
    save_assemblies_dir: Path | None = None,
) -> None:
    df = pd.read_csv(intrinsic_csv)
    if "cluster_csv" not in df.columns:
        raise SystemExit(f"{intrinsic_csv}: missing column cluster_csv")

    cluster_csvs = sorted(df["cluster_csv"].dropna().unique().astype(str))
    dummy_save = Path(gettempdir()) / "promise_cluster_apo_dummy"

    rows_out: list[dict] = []

    for cc in tqdm(cluster_csvs, desc="clusters", unit="cluster"):
        try:
            cluster_path = cluster_csv_to_path(cc, clusters_root)
        except ValueError as e:
            tqdm.write(f"[skip] {cc}: {e}", file=sys.stderr)
            continue

        if not cluster_path.is_file():
            tqdm.write(f"[skip] missing file {cluster_path}", file=sys.stderr)
            continue

        members = read_cluster_members(cluster_path)
        by_pdb: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for pdb, auth, lab in members:
            by_pdb[pdb].append((pdb, auth, lab))

        assem_cache: dict[str, list[AssemblyResult] | None] = {}
        for pdb in by_pdb:
            cif_path = mmcif_dir / f"{pdb.lower()}.cif"
            if not cif_path.is_file():
                assem_cache[pdb] = None
                continue
            assem_cache[pdb] = fetch_assemblies(
                pdb,
                cif_path,
                dummy_save,
                resolution_cutoff=max_resolution,
                max_polymer_instances=max_polymer_instances,
                max_lig_instances=max_lig_instances,
                exclude_na=exclude_na,
            )

        for pdb, auth, lab in members:
            assemblies = assem_cache.get(pdb)
            status, detail, lig_summary, matched = classify_member(assemblies, auth)
            if status == "apo" and save_assemblies_dir is not None:
                for asm in matched:
                    save_assembly(asm, pdb.lower(), save_assemblies_dir)
            if matched:
                for asm in matched:
                    rows_out.append(
                        {
                            "cluster_csv": cc,
                            "cluster_file": str(cluster_path.relative_to(clusters_root)),
                            "chain": f"{pdb}_{auth}",
                            "pdb": pdb.lower(),
                            "auth": auth,
                            "assembly_id": asm.assembly_id,
                            "label": lab,
                            "status": status,
                            "detail": detail,
                            "ligand_comp_ids": ";".join(sorted(asm.ligand_list)),
                        }
                    )
            else:
                rows_out.append(
                    {
                        "cluster_csv": cc,
                        "cluster_file": str(cluster_path.relative_to(clusters_root)),
                        "chain": f"{pdb}_{auth}",
                        "pdb": pdb.lower(),
                        "auth": auth,
                        "assembly_id": "",
                        "label": lab,
                        "status": status,
                        "detail": detail,
                        "ligand_comp_ids": "",
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_csv",
        "cluster_file",
        "chain",
        "pdb",
        "auth",
        "assembly_id",
        "label",
        "status",
        "detail",
        "ligand_comp_ids",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    n_apo = sum(1 for r in rows_out if r["status"] == "apo")
    n_holo = sum(1 for r in rows_out if r["status"] == "holo")
    n_unk = sum(1 for r in rows_out if r["status"] == "unknown")
    click.echo(
        f"[done] wrote {out_csv} ({len(rows_out)} rows: "
        f"apo={n_apo}, holo={n_holo}, unknown={n_unk})"
    )
    if save_assemblies_dir is not None:
        click.echo(f"[done] apo assemblies saved under {save_assemblies_dir}")


@click.command()
@click.option(
    "--intrinsic-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(Path("data/dataset-verbose/intrinsic.csv")),
    show_default=True,
    help="CSV with cluster_csv column (e.g. dataset-verbose/intrinsic.csv).",
)
@click.option(
    "--clusters-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Root directory of cluster CSV trees (default: pipeline data/clusters).",
)
@click.option(
    "--mmcif-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    default="/store/AlphaFold/pdb_mmcif/mmcif_files",
    show_default=True,
    help="Directory containing <pdb>.cif mmCIF files.",
)
@click.option(
    "--out-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=str(Path("data/dataset-verbose/intrinsic_cluster_apo.csv")),
    show_default=True,
    help="Output CSV path.",
)
@click.option("--max-resolution", type=float, default=5.0, show_default=True)
@click.option("--max-polymer-instances", type=int, default=12, show_default=True)
@click.option("--max-lig-instances", type=int, default=12, show_default=True)
@click.option("--exclude-na/--no-exclude-na", default=True, show_default=True)
@click.option(
    "--save-assemblies-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Save apo assembly CIF + map.json here (same layout as prepare_inputs_gemmi). "
         "Default: data/cif-asms from pipeline config.",
)
def main(
    intrinsic_csv: Path,
    clusters_root: Path | None,
    mmcif_dir: Path,
    out_csv: Path,
    max_resolution: float,
    max_polymer_instances: int,
    max_lig_instances: int,
    exclude_na: bool,
    save_assemblies_dir: Path | None,
):
    """Label every cluster member (apo/holo/unknown) for clusters listed in intrinsic.csv."""
    cr = Path(C.dir("clusters")) if clusters_root is None else clusters_root
    if save_assemblies_dir is None:
        save_assemblies_dir = Path(C.dir("cif_asms"))
    save_assemblies_dir.mkdir(parents=True, exist_ok=True)
    main_inner(
        intrinsic_csv=intrinsic_csv,
        clusters_root=cr,
        mmcif_dir=mmcif_dir,
        out_csv=out_csv,
        max_resolution=max_resolution,
        max_polymer_instances=max_polymer_instances,
        max_lig_instances=max_lig_instances,
        exclude_na=exclude_na,
        save_assemblies_dir=save_assemblies_dir,
    )


if __name__ == "__main__":
    main()
