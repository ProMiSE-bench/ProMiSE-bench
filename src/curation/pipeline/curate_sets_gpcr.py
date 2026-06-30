"""GPCR-specific pair curation.

Extracts conformational pairs from GPCR receptor chains using
per-structure chain annotations from GPCRdb.

Side **a** is always a *real apo* receptor (no ligand bound, Inactive).
Side **b** is classified into four categories:

1. **apo-antagonist** — b has an antagonist small-molecule ligand and is
   Inactive.  No conformational change expected, so same ``conf_label``
   is allowed.
2. **apo-agonist** — b has an agonist small-molecule ligand.  Requires
   different ``conf_label``.
3. **apo-antagonist-peptide** — b has an antagonist peptide/protein
   ligand and is Inactive.  Same ``conf_label`` allowed.
4. **apo-agonist-peptide** — b has an agonist peptide/protein ligand.
   Requires different ``conf_label``.

Only receptor chains are considered, identified via GPCRdb's
``preferred_chain`` annotation (loaded from ``--chain-map``).

Output
------
Four CSVs in ``--outdir``: ``gpcr-apo-antagonist.csv``,
``gpcr-apo-agonist.csv``, ``gpcr-apo-antagonist-peptide.csv``, and
``gpcr-apo-agonist-peptide.csv``.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set

import click
import pandas as pd

from utils._config import pipeline_cfg as C
from utils._data_root import DataRootCommand
from .curate_sets import (
    _is_in_filtered_pairs,
    emit_min_row,
    load_filtered_pair_keys,
    normalize_df,
    safe_drop_duplicates,
    write_per_type,
)
from .types import DatasetPair


# ------------------------------------------------------------------ constants

# GPCRdb ligand function values grouped by pharmacological effect.
ANTAGONIST_FUNCTIONS: Set[str] = {
    "Antagonist",
    "Inverse agonist",
    "NAM",
    "Allosteric antagonist",
}
AGONIST_FUNCTIONS: Set[str] = {
    "Agonist",
    "Agonist (partial)",
    "PAM",
    "Ago-PAM",
    "Allosteric agonist",
}
APO_FUNCTIONS: Set[str] = {"Apo (no ligand)"}

# Ligand types that count as peptide/protein.
PEPTIDE_TYPES: Set[str] = {"peptide", "protein"}
SMALL_MOL_TYPES: Set[str] = {"small-molecule", "lipid"}


# ------------------------------------------------------------------ helpers


def _file_checksum(path: Path) -> str:
    """Fast MD5 of a file for cache invalidation."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_filtered_pairs_cached(path: Path) -> set:
    """Load filtered-pair keys with a pickle cache next to the CSV."""
    cache_path = path.with_suffix(".filtered_pairs.pkl")
    checksum = _file_checksum(path)

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                cached = pickle.load(fh)
            if cached.get("checksum") == checksum:
                click.echo(f"[cache] loaded {len(cached['keys'])} pair keys from {cache_path.name}")
                return cached["keys"]
        except Exception:
            pass

    keys = load_filtered_pair_keys(path)
    with open(cache_path, "wb") as fh:
        pickle.dump({"checksum": checksum, "keys": keys}, fh)
    click.echo(f"[cache] built and saved {len(keys)} pair keys to {cache_path.name}")
    return keys


def _load_gpcr_centers_cached(
    root: Path,
    pattern: str,
    chain_map: Dict[str, dict],
    cache_path: Path,
) -> List[Path]:
    """Return only center CSVs that contain GPCR receptor rows.

    A JSON cache (keyed on chain-map PDB set) stores the relative paths
    of matching CSVs so subsequent runs skip the full scan.
    """
    all_files = sorted(root.rglob(pattern))
    map_key = hashlib.md5(";".join(sorted(chain_map.keys())).encode()).hexdigest()

    if cache_path.exists():
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            if cached.get("map_key") == map_key:
                paths = [root / p for p in cached["paths"]]
                paths = [p for p in paths if p.exists()]
                click.echo(f"[cache] {len(paths)} GPCR center CSVs from {cache_path.name}")
                return paths
        except Exception:
            pass

    click.echo(f"[scan] scanning {len(all_files)} CSVs for GPCR receptor rows ...")
    gpcr_pdbs = set(chain_map.keys())
    hits: List[str] = []
    for f in all_files:
        try:
            df = pd.read_csv(f, usecols=["pdb"])
            if df.empty:
                continue
            pdbs = set(df["pdb"].astype(str).str.strip().str.upper())
            if pdbs & gpcr_pdbs:
                hits.append(f.relative_to(root).as_posix())
        except Exception:
            continue

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump({"map_key": map_key, "paths": hits}, fh)
    click.echo(f"[cache] saved {len(hits)} GPCR center paths to {cache_path.name}")
    return [root / p for p in hits]


def load_chain_map(path: Path) -> Dict[str, dict]:
    """Load GPCRdb chain-map JSON (PDB code → receptor/signalling info)."""
    with open(path) as fh:
        return {k.upper(): v for k, v in json.load(fh).items()}


def _classify_ligands(info: dict):
    """Return (has_agonist, has_antagonist, lig_type) for a structure.

    ``lig_type`` is ``"peptide"`` if any non-apo ligand is peptide/protein,
    ``"small-molecule"`` if any non-apo ligand is small-molecule/lipid,
    or ``"none"`` for apo structures.  When both peptide and small-molecule
    ligands are present, ``"peptide"`` takes priority (the peptide partner
    dominates the classification).
    """
    has_agonist = False
    has_antagonist = False
    has_peptide = False
    has_small = False
    for lig in info.get("ligands", []):
        fn = lig.get("function", "")
        tp = lig.get("type", "")
        if fn in APO_FUNCTIONS:
            continue
        if fn in AGONIST_FUNCTIONS:
            has_agonist = True
        elif fn in ANTAGONIST_FUNCTIONS:
            has_antagonist = True
        if tp in PEPTIDE_TYPES:
            has_peptide = True
        elif tp in SMALL_MOL_TYPES:
            has_small = True
    if has_peptide:
        lig_type = "peptide"
    elif has_small:
        lig_type = "small-molecule"
    else:
        lig_type = "none"
    return has_agonist, has_antagonist, lig_type


def _is_real_apo(info: dict) -> bool:
    """A structure is real-apo when all its ligands are Apo (no ligand)."""
    ligs = info.get("ligands", [])
    if not ligs:
        return True
    return all(lig.get("function", "") in APO_FUNCTIONS for lig in ligs)


def _receptor_rows(
    df: pd.DataFrame,
    chain_map: Dict[str, dict],
) -> pd.DataFrame:
    """Keep only GPCR receptor-chain rows and annotate with GPCRdb fields."""
    df = df.copy()
    df["_pdb_upper"] = df["pdb"].astype(str).str.strip().str.upper()
    gpcr = df[df["_pdb_upper"].isin(chain_map)]
    if gpcr.empty:
        return gpcr

    def _is_receptor(row):
        info = chain_map.get(row["_pdb_upper"])
        if not info:
            return False
        return str(row["chain_author"]).strip() == info["receptor_chain"]

    gpcr = gpcr[gpcr.apply(_is_receptor, axis=1)]
    if gpcr.empty:
        return gpcr

    gpcr = gpcr.copy()
    gpcr["_state"] = gpcr["_pdb_upper"].map(
        lambda p: chain_map[p].get("state", "")
    )
    gpcr["_is_apo"] = gpcr["_pdb_upper"].map(
        lambda p: _is_real_apo(chain_map[p])
    )

    def _lig_info(pdb):
        has_ag, has_ant, lt = _classify_ligands(chain_map[pdb])
        return pd.Series({"_has_agonist": has_ag, "_has_antagonist": has_ant, "_lig_type": lt})

    lig_cols = gpcr["_pdb_upper"].apply(_lig_info)
    gpcr = pd.concat([gpcr, lig_cols], axis=1)

    subset_cols = [
        "pdb",
        "chain_auth_asm",
        "conf_label",
        "assembly_id",
        "contact_ligands",
    ]
    gpcr = safe_drop_duplicates(gpcr, subset_cols)
    return gpcr


# ------------------------------------------------------------------ pair finders


def _find_pairs(
    apo: pd.DataFrame,
    holo: pd.DataFrame,
    cluster_csv: str,
    require_diff_conf: bool,
) -> List[DatasetPair]:
    """Emit (apo, holo) pairs, optionally requiring different conf_label."""
    if apo.empty or holo.empty:
        return []
    out: List[DatasetPair] = []
    for _, ra in apo.iterrows():
        for _, rb in holo.iterrows():
            if require_diff_conf and ra["conf_label"] == rb["conf_label"]:
                continue
            out.append(emit_min_row(cluster_csv, ra, rb))
    return out


def classify_pairs(
    gpcr: pd.DataFrame,
    cluster_csv: str,
) -> Dict[str, List[DatasetPair]]:
    """Classify all apo ↔ X pairs into four categories plus all inactive-active."""
    results: Dict[str, List[DatasetPair]] = {}

    # Categories 1-4 need real apo rows
    apo = gpcr[(gpcr["_is_apo"]) & (gpcr["_state"] == "Inactive")]

    # 1. apo ↔ antagonist small-molecule (same conf_label OK)
    antag_sm = gpcr[
        (gpcr["_has_antagonist"])
        & (~gpcr["_has_agonist"])
        & (gpcr["_lig_type"] == "small-molecule")
        & (gpcr["_state"] == "Inactive")
    ]
    results["gpcr-apo-antagonist"] = _find_pairs(
        apo, antag_sm, cluster_csv, require_diff_conf=False
    )

    # 2. apo ↔ agonist small-molecule (require different conf_label)
    agon_sm = gpcr[
        (gpcr["_has_agonist"])
        & (gpcr["_lig_type"] == "small-molecule")
    ]
    results["gpcr-apo-agonist"] = _find_pairs(
        apo, agon_sm, cluster_csv, require_diff_conf=True
    )

    # 3. apo ↔ antagonist peptide/protein (same conf_label OK)
    antag_pep = gpcr[
        (gpcr["_has_antagonist"])
        & (~gpcr["_has_agonist"])
        & (gpcr["_lig_type"] == "peptide")
        & (gpcr["_state"] == "Inactive")
    ]
    results["gpcr-apo-antagonist-peptide"] = _find_pairs(
        apo, antag_pep, cluster_csv, require_diff_conf=False
    )

    # 4. apo ↔ agonist peptide/protein (require different conf_label)
    agon_pep = gpcr[
        (gpcr["_has_agonist"])
        & (gpcr["_lig_type"] == "peptide")
    ]
    results["gpcr-apo-agonist-peptide"] = _find_pairs(
        apo, agon_pep, cluster_csv, require_diff_conf=True
    )

    # 5. all Inactive ↔ Active (regardless of ligand, require diff conf_label)
    inactive = gpcr[gpcr["_state"] == "Inactive"]
    active = gpcr[gpcr["_state"] == "Active"]
    results["gpcr-inactive-active"] = _find_pairs(
        inactive, active, cluster_csv, require_diff_conf=True
    )

    return results


# ------------------------------------------------------------------ CLI


@click.command(
    cls=DataRootCommand, context_settings=dict(help_option_names=["-h", "--help"])
)
@click.option(
    "--filtered-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Root directory (e.g., asms-metal).",
    default=str(C.dir("asms_metal")),
)
@click.option(
    "--pattern",
    default="**/*_asm_subset_filtered.csv",
    show_default=True,
    help="Glob pattern (recursive).",
)
@click.option(
    "--outdir",
    type=click.Path(dir_okay=True, file_okay=False, writable=True),
    default=str(C.dir("combinations")),
    show_default=True,
    help="Directory to write gpcr-*.csv files.",
)
@click.option(
    "--chain-map",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    default="data-95/gpcrdb_chain_map.json",
    show_default=True,
    help="GPCRdb chain-map JSON (from fetch_gpcrdb_chains.py).",
)
@click.option(
    "--filtered-pairs",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    default=str(C.file("filtered_pairs")),
    show_default=True,
    help="Only keep pairs listed in this filtered-pairs CSV.",
)
def cli(filtered_dir, pattern, outdir, chain_map, filtered_pairs):
    cmap = load_chain_map(Path(chain_map))
    click.echo(f"[chain-map] {len(cmap)} GPCR structures loaded from {chain_map}")

    allowed = _load_filtered_pairs_cached(Path(filtered_pairs))

    root = Path(filtered_dir)
    cache_dir = Path(outdir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = _load_gpcr_centers_cached(
        root, pattern, cmap, cache_dir / ".gpcr_centers_cache.json"
    )
    click.echo(f"[files] {len(files)} GPCR center CSVs to process")

    CATEGORIES = [
        "gpcr-apo-antagonist",
        "gpcr-apo-agonist",
        "gpcr-apo-antagonist-peptide",
        "gpcr-apo-agonist-peptide",
        "gpcr-inactive-active",
    ]
    all_pairs: Dict[str, List[DatasetPair]] = {c: [] for c in CATEGORIES}
    processed = 0

    for f in files:
        try:
            df_raw = pd.read_csv(f)
            if df_raw.empty:
                continue
            need = {
                "pdb",
                "chain_author",
                "chain_auth_asm",
                "conf_label",
                "protein_count",
                "chain_list_author",
            }
            miss = [c for c in need if c not in df_raw.columns]
            if miss:
                continue

            df = normalize_df(df_raw)
            cluster_csv = f.relative_to(root).as_posix()
            gpcr = _receptor_rows(df, cmap)
            if len(gpcr) < 2:
                continue

            found_map = classify_pairs(gpcr, cluster_csv)
            for cat in CATEGORIES:
                found = found_map.get(cat, [])
                found = [p for p in found if _is_in_filtered_pairs(p, allowed)]
                all_pairs[cat].extend(found)

            processed += 1
        except Exception as e:
            click.echo(f"[!] Skip {f}: {e}")

    outdir_p = Path(outdir)
    write_per_type(outdir_p, all_pairs)

    for cat in CATEGORIES:
        click.echo(f"[+] {cat}: {len(all_pairs[cat])} pairs")
    click.echo(f"[=] processed {processed} centers")


if __name__ == "__main__":
    cli()
