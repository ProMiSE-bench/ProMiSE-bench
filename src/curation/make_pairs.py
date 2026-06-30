#!/usr/bin/env python3
"""
make_pairs.py - Generate seq_cluster_to_answer_map.json and valid_pairs.json

Usage:
    python -m curation.make_pairs
    python -m curation.make_pairs --csv-dir data/dataset --outdir data/dataset --examples-dir examples

Next: alignment tasks — ``python -m eval.align.generate_alignment_tasks --help``.
Distogram eval consumes this same enriched map (optionally after
``python -m eval.distogram.extract_reference_cb --answer-map …`` for ``reference_cb_json``).
"""

import functools
import json
import csv
import string
import re
from glob import glob
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple, Optional

import click

from utils._config import pipeline_cfg


# ============================================================================
# Configuration
# ============================================================================
SET_NAMES = ["intrinsic", "protein-induced", "ligand-induced"]

MODEL_PATTERNS: Dict[str, str | List[str]] = {
    "af3": [
        # examples/ layout: seed_6/sample_0/model.cif
        "{examples_dir}/af3/{set_name}/{cluster}/{tag}/seed_*/sample_*/model.cif",
        # promise-samples layout: seed_6/{tag}/seed-6_sample-0/model.cif
        "{examples_dir}/af3/{set_name}/{cluster}/{tag}/seed_*/*/seed-*_sample-*/model.cif",
    ],
    "bioemu": "{examples_dir}/bioemu/{set_name}/{cluster}/pdbs/sample_*.pdb",
    "boltz1": [
        # examples/ layout: seed_2853/{tag}_model_0.cif
        "{examples_dir}/boltz1/{set_name}/{cluster}/{tag}/seed_*/{tag}_model_*.cif",
        # promise-samples layout: seed_6/boltz_results_{tag}/predictions/{tag}/{tag}_model_0.cif
        "{examples_dir}/boltz1/{set_name}/{cluster}/{tag}/seed_*/boltz_results_{tag}/predictions/{tag}/{tag}_model_*.cif",
    ],
    "boltz2": [
        # examples/ layout: seed_2853/{tag}_model_0.cif
        "{examples_dir}/boltz2/{set_name}/{cluster}/{tag}/seed_*/{tag}_model_*.cif",
        # promise-samples layout: seed_6/boltz_results_{tag}/predictions/{tag}/{tag}_model_0.cif
        "{examples_dir}/boltz2/{set_name}/{cluster}/{tag}/seed_*/boltz_results_{tag}/predictions/{tag}/{tag}_model_*.cif",
    ],
    "chai": [
        # standard: .../seed_*/pred.model_idx_*.cif
        "{examples_dir}/chai/{set_name}/{cluster}/{tag}/seed_*/pred.model_idx_*.cif",
        # protein-induced multimer: .../{tag}/{entity}/seed_*/pred.model_idx_*.cif
        "{examples_dir}/chai/{set_name}/{cluster}/{tag}/*/seed_*/pred.model_idx_*.cif",
    ],
}


# ============================================================================
# Load Valid Centers from clusters.json
# ============================================================================
def load_valid_centers(clusters_json: Path) -> Set[str]:
    if not clusters_json.exists():
        click.echo(f"  [WARN] clusters.json not found: {clusters_json}")
        return set()
    with open(clusters_json) as f:
        clusters = json.load(f)
    return {c["center"] for c in clusters}


# ============================================================================
# Helper Functions
# ============================================================================
def make_entry_id(pdb: str, asm: str, chain: str, conf: str) -> str:
    return f"{pdb.lower()}_{asm}_{chain}_conf_{conf}"


def make_tag(pdb: str, asm: str, chain: str, suffix: str = "m") -> str:
    return f"{pdb.lower()}_{asm}_{chain}_{suffix}"


def get_conf(entry_id: str) -> Optional[int]:
    match = re.search(r'_conf_(\d+)$', entry_id)
    return int(match.group(1)) if match else None


def tag_to_key(tag: str) -> Tuple[str, str, str]:
    base = tag_base(tag)
    parts = base.split('_')
    return (parts[0].lower(), parts[1], parts[2]) if len(parts) >= 3 else (None, None, None)


def tag_base(tag: str) -> str:
    """Strip apo/holo suffix (_m / _x) for conformation-agnostic matching."""
    if tag.endswith("_m") or tag.endswith("_x"):
        return tag[:-2]
    return tag


def tag_suffix_variants(tag: str) -> List[str]:
    """Return ``tag`` plus the alternate _m/_x form when applicable."""
    variants = [tag]
    if tag.endswith("_m"):
        alt = f"{tag[:-2]}_x"
    elif tag.endswith("_x"):
        alt = f"{tag[:-2]}_m"
    else:
        return variants
    if alt not in variants:
        variants.append(alt)
    return variants


# ============================================================================
# Prediction Paths
# ============================================================================
def _glob_prediction_files(pattern: str) -> List[str]:
    if "**" in pattern:
        return glob(pattern, recursive=True)
    return glob(pattern)


def _disk_set_names(set_name: str) -> List[str]:
    """Eval set names vs on-disk folder names (e.g. intrinsic vs apo-monomers)."""
    if set_name == "intrinsic":
        return ["intrinsic", "apo-monomers"]
    return [set_name]


def _prediction_entry(
    examples_dir: Path,
    set_name: str,
    cluster: str,
    disk_tag: str,
    canonical_tag: str,
    pattern_templates: List[str],
) -> Optional[Dict[str, Any]]:
    for disk_set in _disk_set_names(set_name):
        for pattern_template in pattern_templates:
            pattern = pattern_template.format(
                examples_dir=examples_dir,
                set_name=disk_set,
                cluster=cluster,
                tag=disk_tag,
            )
            files = _glob_prediction_files(pattern)
            if files:
                entry: Dict[str, Any] = {
                    "pattern": pattern,
                    "count": len(files),
                    "yaml_tag": disk_tag,
                }
                if disk_tag != canonical_tag:
                    entry["canonical_yaml_tag"] = canonical_tag
                return entry
    return None


@functools.lru_cache(maxsize=512)
def _intrinsic_disk_tags_for_cluster(examples_dir_str: str, cluster: str) -> Tuple[str, ...]:
    """Tag subdirs under an intrinsic cluster that contain at least one model's predictions."""
    examples_dir = Path(examples_dir_str)
    if not examples_dir.is_dir():
        return ()
    tags: Set[str] = set()
    for model, templates in MODEL_PATTERNS.items():
        if model == "bioemu":
            continue
        tpls = [templates] if isinstance(templates, str) else templates
        for disk_set in _disk_set_names("intrinsic"):
            cluster_dir = examples_dir / model / disk_set / cluster
            if not cluster_dir.is_dir():
                continue
            for sub in cluster_dir.iterdir():
                if not sub.is_dir() or sub.name == "log":
                    continue
                if _prediction_entry(
                    examples_dir, "intrinsic", cluster, sub.name, sub.name, tpls
                ):
                    tags.add(sub.name)
    return tuple(sorted(tags))


def _disk_tag_candidates(set_name: str, cluster: str, tag: str, examples_dir: Path) -> List[str]:
    """Ordered disk-tag probes: exact/suffix variants, then any intrinsic cluster tag."""
    seen: Set[str] = set()
    out: List[str] = []
    for dt in tag_suffix_variants(tag):
        if dt not in seen:
            seen.add(dt)
            out.append(dt)
    if set_name == "intrinsic":
        for dt in _intrinsic_disk_tags_for_cluster(str(examples_dir), cluster):
            if dt not in seen:
                seen.add(dt)
                out.append(dt)
    return out


def get_predictions(examples_dir: Path, set_name: str, cluster: str, tag: str) -> Dict[str, dict]:
    if not examples_dir or not examples_dir.exists():
        return {}

    predictions = {}
    disk_tags = _disk_tag_candidates(set_name, cluster, tag, examples_dir)
    for model, pattern_templates in MODEL_PATTERNS.items():
        if isinstance(pattern_templates, str):
            pattern_templates = [pattern_templates]
        for disk_tag in disk_tags:
            entry = _prediction_entry(
                examples_dir, set_name, cluster, disk_tag, tag, pattern_templates
            )
            if entry:
                predictions[model] = entry
                break
    return predictions


def add_predictions_to_data(data: Dict[str, Dict], examples_dir: Optional[Path]) -> Dict[str, Dict]:
    """Populate apo_predictions / holo_predictions for every cluster.

    Both are keyed by yaml tag -> ``{model: {pattern, count, ...}}``.
    """
    for set_name, clusters in data.items():
        for cluster_name, info in clusters.items():
            apo_by_tag: Dict[str, Dict[str, dict]] = {}
            for tag in info.get("apo_tags") or []:
                preds = get_predictions(examples_dir, set_name, cluster_name, tag)
                if preds:
                    apo_by_tag[tag] = preds
            info["apo_predictions"] = apo_by_tag

            holo_preds: Dict[str, Dict[str, dict]] = {}
            for tag in info.get("holo_tags") or []:
                preds = get_predictions(examples_dir, set_name, cluster_name, tag)
                if preds:
                    holo_preds[tag] = preds
            info["holo_predictions"] = holo_preds
    return data


# ============================================================================
# CSV Parsing (supports both old and new formats)
# ============================================================================
def load_csv(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        click.echo(f"  [WARN] CSV not found: {csv_path}")
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def get_cluster_name(row: dict) -> str:
    """Get cluster name from row (supports both formats)."""
    cluster = row.get('cluster_csv') or row.get('cluster', '')
    # Handle 'AB/8ABP_1' -> '8ABP_1'
    if '/' in cluster:
        return cluster.split('/')[1]
    return cluster


def _row_side(row: dict, side: str) -> Tuple[str, str, str, str]:
    """Return (pdb, asm, chain, conf) for side ``a`` or ``b`` (both CSV formats)."""
    suffix = "_a" if side == "a" else "_b"
    pdb = (row.get(f"{side}_pdb") or row.get(f"pdb{suffix}", "")).lower()
    asm = row.get(f"{side}_assembly_id") or row.get(f"asm{suffix}", "")
    chain = row.get(f"{side}_chain") or row.get(f"chain{suffix}", "")
    conf = row.get(f"{side}_conf_label") or row.get(f"conf_label{suffix}", "")
    return pdb, asm, chain, conf


def load_csv_pairs(csv_path: Path) -> Set[Tuple]:
    """Load valid pairs from CSV."""
    pairs = set()
    for row in load_csv(csv_path):
        a = _row_side(row, "a")[:3]
        b = _row_side(row, "b")[:3]
        pairs.add((a, b))
        pairs.add((b, a))
    return pairs


def build_valid_pairs_from_csv(
    csv_dir: Path, data: Dict[str, Dict]
) -> Dict[str, Dict[str, List[List[str]]]]:
    """Build ``valid_pairs`` directly from CSV rows (filtered clusters only)."""
    result: Dict[str, Dict[str, List[List[str]]]] = {sn: {} for sn in SET_NAMES}
    seen: Dict[str, Dict[str, Set[Tuple[str, ...]]]] = {
        sn: defaultdict(set) for sn in SET_NAMES
    }

    for set_name in SET_NAMES:
        rows = load_csv(csv_dir / f"{set_name}.csv")
        clusters = data.get(set_name, {})
        is_intrinsic = set_name == "intrinsic"

        for row in rows:
            cluster = get_cluster_name(row)
            if cluster not in clusters:
                continue

            a_pdb, a_asm, a_chain, _ = _row_side(row, "a")
            b_pdb, b_asm, b_chain, _ = _row_side(row, "b")
            if not all([a_pdb, a_asm, a_chain, b_pdb, b_asm, b_chain]):
                continue

            tag_a = make_tag(a_pdb, a_asm, a_chain, "m")
            tag_b = make_tag(
                b_pdb, b_asm, b_chain, "m" if is_intrinsic else "x"
            )
            pair_list = [tag_a, tag_b]
            dedupe_key: Tuple[str, ...] = (
                tuple(sorted(pair_list)) if is_intrinsic else (tag_a, tag_b)
            )
            if dedupe_key in seen[set_name][cluster]:
                continue
            seen[set_name][cluster].add(dedupe_key)
            result[set_name].setdefault(cluster, []).append(pair_list)

    return result


def _entry_id_for_tag(tag: str, conf: str) -> str:
    base = tag[:-2] if tag.endswith(("_m", "_x")) else tag
    return f"{base}_conf_{conf}"


def _entries_for_tags(
    tags: List[str], rows: List[dict], is_intrinsic: bool
) -> Tuple[List[str], List[str]]:
    """Rebuild ``apo``/``holo`` entry ids for tags that appear in CSV pair rows."""
    conf_by_base: Dict[str, str] = {}
    for row in rows:
        for side in ("a", "b"):
            pdb, asm, chain, conf = _row_side(row, side)
            if not (pdb and asm and chain):
                continue
            suffix = "m" if is_intrinsic or side == "a" else "x"
            tag = make_tag(pdb, asm, chain, suffix)
            base = tag[:-2] if tag.endswith(("_m", "_x")) else tag
            conf_by_base.setdefault(base, conf)

    apo_entries, holo_entries = [], []
    for tag in tags:
        base = tag[:-2] if tag.endswith(("_m", "_x")) else tag
        conf = conf_by_base.get(base)
        if conf is None:
            continue
        entry = _entry_id_for_tag(tag, conf)
        if tag.endswith("_m"):
            apo_entries.append(entry)
        elif tag.endswith("_x"):
            holo_entries.append(entry)
    return apo_entries, holo_entries


def restrict_to_valid_pairs(
    data: Dict[str, Dict],
    valid_pairs: Dict[str, Dict[str, List[List[str]]]],
    csv_dir: Path,
) -> Dict[str, Dict]:
    """Keep only clusters/tags that appear in ``valid_pairs`` (CSV pair scope)."""
    restricted: Dict[str, Dict] = {}
    for set_name in SET_NAMES:
        restricted[set_name] = {}
        rows = load_csv(csv_dir / f"{set_name}.csv")
        is_intrinsic = set_name == "intrinsic"
        cluster_rows = defaultdict(list)
        for row in rows:
            cluster_rows[get_cluster_name(row)].append(row)

        for cluster, pairs in valid_pairs.get(set_name, {}).items():
            if cluster not in data.get(set_name, {}):
                continue
            used_tags: Set[str] = set()
            for pair in pairs:
                used_tags.update(pair)

            info = json.loads(json.dumps(data[set_name][cluster]))
            if is_intrinsic:
                apo_tags = sorted(t for t in used_tags if t.endswith("_m"))
                holo_tags = []
            else:
                apo_tags = sorted(t for t in used_tags if t.endswith("_m"))
                holo_tags = sorted(t for t in used_tags if t.endswith("_x"))

            apo_entries, holo_entries = _entries_for_tags(
                apo_tags + holo_tags, cluster_rows.get(cluster, []), is_intrinsic
            )
            info["apo_tags"] = apo_tags
            info["holo_tags"] = holo_tags
            info["apo"] = apo_entries
            info["holo"] = holo_entries
            restricted[set_name][cluster] = info

    return restricted


# ============================================================================
# Process Sets
# ============================================================================
def process_intrinsic(rows: List[dict]) -> Dict[str, dict]:
    """Process intrinsic CSV (supports both formats)."""
    result = {}
    
    clusters = set(get_cluster_name(row) for row in rows)
    for cluster_name in clusters:
        cluster_rows = [r for r in rows if get_cluster_name(r) == cluster_name]
        
        entries_by_conf = defaultdict(set)
        for row in cluster_rows:
            # Support both formats
            for old_prefix, new_suffix in [('a', '_a'), ('b', '_b')]:
                pdb = (row.get(f'{old_prefix}_pdb') or row.get(f'pdb{new_suffix}', '')).lower()
                asm = row.get(f'{old_prefix}_assembly_id') or row.get(f'asm{new_suffix}', '')
                chain = row.get(f'{old_prefix}_chain') or row.get(f'chain{new_suffix}', '')
                conf = row.get(f'{old_prefix}_conf_label') or row.get(f'conf_label{new_suffix}', '')
                if pdb and asm and chain:
                    entries_by_conf[conf].add((pdb, asm, chain))
        
        apo_list, apo_tags = [], []
        for conf in sorted(entries_by_conf.keys()):
            pdb, asm, chain = sorted(entries_by_conf[conf])[0]
            apo_list.append(make_entry_id(pdb, asm, chain, conf))
            apo_tags.append(make_tag(pdb, asm, chain, "m"))
        
        result[cluster_name] = {
            "apo": apo_list,
            "holo": [],
            "apo_tags": apo_tags,
            "holo_tags": [],
        }
    
    return result


def process_induced_set(rows: List[dict]) -> Dict[str, dict]:
    """Process induced CSV (supports both formats)."""
    result = {}
    
    clusters = set(get_cluster_name(row) for row in rows)
    for cluster_name in clusters:
        cluster_rows = [r for r in rows if get_cluster_name(r) == cluster_name]
        
        apo_by_conf = defaultdict(set)
        holo_set = set()
        
        for row in cluster_rows:
            # a is apo (support both formats)
            a_pdb = (row.get('a_pdb') or row.get('pdb_a', '')).lower()
            a_asm = row.get('a_assembly_id') or row.get('asm_a', '')
            a_chain = row.get('a_chain') or row.get('chain_a', '')
            a_conf = row.get('a_conf_label') or row.get('conf_label_a', '')
            if a_pdb and a_asm and a_chain:
                apo_by_conf[a_conf].add((a_pdb, a_asm, a_chain))
            
            # b is holo
            b_pdb = (row.get('b_pdb') or row.get('pdb_b', '')).lower()
            b_asm = row.get('b_assembly_id') or row.get('asm_b', '')
            b_chain = row.get('b_chain') or row.get('chain_b', '')
            b_conf = row.get('b_conf_label') or row.get('conf_label_b', '')
            if b_pdb and b_asm and b_chain:
                holo_set.add((b_pdb, b_asm, b_chain, b_conf))
        
        apo_list, apo_tags = [], []
        for conf in sorted(apo_by_conf.keys()):
            pdb, asm, chain = sorted(apo_by_conf[conf])[0]
            apo_list.append(make_entry_id(pdb, asm, chain, conf))
            apo_tags.append(make_tag(pdb, asm, chain, "m"))
        
        holo_list, holo_tags = [], []
        for pdb, asm, chain, conf in sorted(holo_set):
            holo_list.append(make_entry_id(pdb, asm, chain, conf))
            holo_tags.append(make_tag(pdb, asm, chain, "x"))
        
        result[cluster_name] = {
            "apo": apo_list,
            "holo": holo_list,
            "apo_tags": apo_tags,
            "holo_tags": holo_tags,
        }
    
    return result


# ============================================================================
# Filtering
# ============================================================================
def filter_data(data: Dict[str, Dict], valid_centers: Set[str]) -> Dict[str, Dict]:
    filtered = {}
    
    for set_name, clusters in data.items():
        filtered[set_name] = {}
        is_induced = set_name in ['protein-induced', 'ligand-induced']
        
        for cluster_name, info in clusters.items():
            if valid_centers and cluster_name not in valid_centers:
                continue
            
            if is_induced:
                apo_confs = {get_conf(e) for e in info['apo']}
                holo_confs = {get_conf(e) for e in info['holo']}
                if apo_confs & holo_confs:
                    continue
                if len(info['apo']) < 1 or len(info['holo']) < 1:
                    continue
            else:
                if len(info['apo']) < 1:
                    continue
            
            filtered[set_name][cluster_name] = info
    
    return filtered


# ============================================================================
# Valid Pairs Generation (legacy helper; prefer build_valid_pairs_from_csv)
# ============================================================================
def generate_valid_pairs(data: Dict[str, Dict], csv_pairs: Dict[str, Set]) -> Dict[str, Dict]:
    result = {}
    
    for set_name, clusters in data.items():
        result[set_name] = {}
        pairs_set = csv_pairs.get(set_name, set())
        
        for cluster_name, info in clusters.items():
            valid = []
            
            if set_name == "intrinsic":
                for t1, t2 in combinations(info['apo_tags'], 2):
                    k1, k2 = tag_to_key(t1), tag_to_key(t2)
                    if (k1, k2) in pairs_set or (k2, k1) in pairs_set:
                        valid.append([t1, t2])
            else:
                for apo_tag in info['apo_tags']:
                    for holo_tag in info['holo_tags']:
                        k1, k2 = tag_to_key(apo_tag), tag_to_key(holo_tag)
                        if (k1, k2) in pairs_set or (k2, k1) in pairs_set:
                            valid.append([apo_tag, holo_tag])
            
            if valid:
                result[set_name][cluster_name] = valid
    
    return result

# ============================================================================
# Distogram enrichment (config: pipeline.distogram_enrich in config/config.yaml)
# ============================================================================
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _enrich_cfg() -> Dict:
    return pipeline_cfg.raw().get("distogram_enrich") or {}


def _resolve_path(val: object) -> Path | None:
    if not val or not isinstance(val, str):
        return None
    p = Path(val)
    return p if p.is_absolute() else _REPO_ROOT / p


def _map_cfg_path(s: str | None) -> str:
    p = _resolve_path(s)
    return str(p) if p is not None else ""


def get_distogram_path_pattern(
    method: str, method_type: str, cluster_id: str, yaml_tag: str
) -> str:
    """
    Resolve a distogram glob that exists on disk. Template list comes from
    ``config.pipeline.distogram_enrich.distogram`` (keys: af3, boltz1, boltz2, bioemu).
    Placeholders: {method_type} (intrinsic|ligand-induced|protein-induced), {cluster_id}, {yaml_tag}.
    """
    if method not in ("af3", "boltz1", "boltz2", "bioemu"):
        return ""
    de = _enrich_cfg()
    disto = de.get("distogram") or {}
    raw = disto.get(method)
    if raw is None or raw == []:
        return ""
    templates = [raw] if isinstance(raw, str) else list(raw)
    mt = (method_type or "").strip() or "intrinsic"
    for tmpl in templates:
        s = str(tmpl).format(
            method_type=mt, cluster_id=cluster_id, yaml_tag=yaml_tag
        )
        if s and glob(s):
            return s
    return ""


def get_chain_match_from_fasta(fasta_file: Path) -> Dict[str, str]:
    """
    Get chain ID mapping from FASTA file for chai method.
    Maps original chain ID to alphabet chain ID (A, B, C, ...).
    """
    chain_match = {}
    if not fasta_file.exists():
        return chain_match

    with open(fasta_file) as f:
        chain_ids = list(string.ascii_uppercase)
        idx = -1
        for line in f.readlines():
            if line.startswith(">"):
                idx += 1
                if idx < len(chain_ids):
                    chain_id = line.strip().split("=")[-1]
                    chain_match[chain_id] = chain_ids[idx]
    return chain_match


class MissingChainMappingEntry(KeyError):
    """Raised when the AF3/Boltz chain-mapping JSON has no entry for a
    given ``(method, set, cluster, yaml_tag)`` and the requested
    ``interested_chain``.

    There is no silent fallback: every prediction must be backed by an
    explicit entry in the chain-mapping JSON. The error message lists the
    exact key the lookup probed, so the missing entry can be patched into
    the JSON.
    """


def _require_chain_mapping(
    method: str,
    interested_chain: str,
    cluster_id: str,
    yaml_tag: str,
    map_set_name: str,
    mapping_json_path: Optional[str],
) -> str:
    """Look up the modeled chain id for ``interested_chain``; raise if absent."""
    if not mapping_json_path:
        raise MissingChainMappingEntry(
            f"{method}: chain-mapping JSON path is not configured (set "
            f"pipeline.distogram_enrich.{'af3' if method == 'af3' else 'boltz'}_chain_mappings)"
        )
    mapping = get_chain_mapping(cluster_id, yaml_tag, map_set_name, mapping_json_path)
    if mapping is not None and interested_chain in mapping:
        return mapping[interested_chain]
    key = f"{map_set_name}/{cluster_id}/{yaml_tag}"
    raise MissingChainMappingEntry(
        f"{method}: no entry in {mapping_json_path} for "
        f"interested_chain={interested_chain!r}. Key: {key}"
    )


def get_target_chain_for_method(
    method: str,
    yaml_tag: str,
    path_segment: str,
    cluster_id: str,
    map_set_name: str,
) -> str:
    """
    Get target chain ID for different prediction methods.
    ``path_segment`` comes from the prediction glob (set folder in the path);
    ``map_set_name`` is the answer-map set key (``intrinsic``, etc.) for chain JSON lookup.
    """
    de = _enrich_cfg()
    segment = (path_segment or "").strip() or "intrinsic"
    if method == "af3":
        interested_chain = extract_chain_from_yaml_tag(yaml_tag)
        mapping = get_chain_mapping(
            cluster_id, yaml_tag, map_set_name, _map_cfg_path(de.get("af3_chain_mappings"))
        )
        if mapping and interested_chain in mapping:
            return mapping[interested_chain]
        return interested_chain

    if method == "boltz2":
        interested_chain = extract_chain_from_yaml_tag(yaml_tag)
        mapping = get_chain_mapping(
            cluster_id, yaml_tag, map_set_name, _map_cfg_path(de.get("boltz_chain_mappings"))
        )
        if mapping and interested_chain in mapping:
            return mapping[interested_chain]
        return interested_chain

    if method == "boltz1":
        interested_chain = extract_chain_from_yaml_tag(yaml_tag)
        mapping = get_chain_mapping(
            cluster_id, yaml_tag, map_set_name, _map_cfg_path(de.get("boltz1_chain_mappings"))
        )
        if mapping and interested_chain in mapping:
            return mapping[interested_chain]
        return interested_chain

    if method == "chai":
        interested_chain_id = extract_chain_from_yaml_tag(yaml_tag)
        if segment == "ligand-induced":
            return "A"
        if segment == "intrinsic":
            return "A"
        root = _resolve_path(de.get("chai_fasta_root"))
        if root is None:
            raise MissingChainMappingEntry(
                "chai: chai_fasta_root is not configured (set "
                "pipeline.distogram_enrich.chai_fasta_root)"
            )
        fasta_path = root / segment / cluster_id / f"{yaml_tag}.fa"
        if not fasta_path.exists():
            raise MissingChainMappingEntry(
                f"chai: fasta not found for chain lookup: {fasta_path}"
            )
        chain_match = get_chain_match_from_fasta(fasta_path)
        if interested_chain_id not in chain_match:
            raise MissingChainMappingEntry(
                f"chai: interested_chain={interested_chain_id!r} not in fasta "
                f"chain_match {sorted(chain_match.keys())} ({fasta_path})"
            )
        return chain_match[interested_chain_id]

    if method == "bioemu":
        return "A"

    raise MissingChainMappingEntry(f"unsupported prediction method: {method!r}")


def extract_chain_from_yaml_tag(yaml_tag: str) -> str:
    """
    Extract chain ID from yaml tag.
    e.g., '2wrz_2_B1_m' -> 'B1'
    """
    parts = yaml_tag.split("_")
    if len(parts) >= 3:
        return parts[2]  # B1
    return ""


# Set names used in chain-mapping JSON keys.
@functools.lru_cache(maxsize=8)
def _load_chain_mapping_json_cached(path_str: str) -> Dict[str, Any]:
    """Read & cache an AF3/Boltz chain-mapping JSON (>1MB each).

    Without caching every prediction-method/cluster lookup re-parses the
    file, which dwarfs the rest of ``make_pairs`` enrichment.
    """
    if not path_str:
        return {}
    p = Path(path_str)
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: failed to read chain-mapping JSON {p}: {e}")
        return {}


def get_chain_mapping(
    cluster_id: str, yaml_tag: str, method_type: str, mapping_json_path: str
) -> Optional[Dict]:
    """
    Get modeled→target chain mapping from a JSON file (AF3 or Boltz layout).

    JSON key shape: ``"{set_name}/{cluster_id}/{yaml_tag}"``. Each entry must
    expose a ``"mapping"`` field mapping *output* (modeled) chain IDs to
    *target* (reference PDB) chain IDs. This function returns the reverse
    direction, ``{target_chain: output_chain}``, which is what callers want
    when they have an *interested* (target) chain from the yaml tag and need
    to know what chain to look up in the modeled CIF.

    The ``intrinsic`` set is keyed ``intrinsic`` in the answer map and in
    chain-mapping JSONs.
    """
    if not mapping_json_path:
        return None
    all_mappings = _load_chain_mapping_json_cached(str(mapping_json_path))
    if not all_mappings:
        return None
    set_key = (method_type or "").strip()
    entry = None
    for yt in tag_suffix_variants(yaml_tag):
        candidate = all_mappings.get(f"{set_key}/{cluster_id}/{yt}")
        if isinstance(candidate, dict):
            entry = candidate
            break
    if entry is None and set_key == "intrinsic":
        prefix = f"intrinsic/{cluster_id}/"
        for key, candidate in all_mappings.items():
            if key.startswith(prefix) and isinstance(candidate, dict):
                entry = candidate
                break
    if not isinstance(entry, dict):
        return None
    modeled = entry.get("mapping")
    if isinstance(modeled, dict) and modeled:
        return {v: k for k, v in modeled.items()}
    return None


def get_msa_path(cluster_id: str) -> str:
    """
    MSA file path: ``<data_root>/msas/<2-letter>/{cluster_id}.a3m`` (``pipeline.dirs.msas``).
    """
    try:
        two_letter = cluster_id.split("_")[0][1:3]
        msa_path = pipeline_cfg.dir("msas") / two_letter / f"{cluster_id}.a3m"
    except Exception:
        return ""
    if msa_path.exists():
        return str(msa_path)
    print(f"Warning: MSA file not found for {cluster_id}")
    return ""


def get_reference_cif_path(yaml_tag: str) -> str:
    """
    Get reference CIF file path for a yaml tag under ``pipeline.dirs.cif_asms``.
    """
    parts = yaml_tag.split("_")
    if len(parts) < 3:
        return ""

    pdb_id = parts[0].upper()
    asm_num = parts[1]
    first_two = pdb_id[1:3]
    rel = Path(first_two) / pdb_id / f"asm_{pdb_id.lower()}_{asm_num}.cif"

    try:
        cif_path = pipeline_cfg.dir("cif_asms") / rel
        if cif_path.exists():
            return str(cif_path)
    except Exception:
        pass

    print(f"Warning: Reference CIF file not found for yaml_tag {yaml_tag}")
    return ""


def extract_yaml_tag_from_pattern(pattern: str) -> str:
    """
    Extract yaml_tag from a prediction pattern.
    e.g., ``.../af3/.../intrinsic/8ABP_1/2wrz_2_B1_m/seed_...`` -> ``2wrz_2_B1_m``
    """

    match = re.search(
        r"/(?:intrinsic|ligand-induced|protein-induced|apo-monomers)/[^/]+/([^/]+)/",
        pattern,
    )
    if match:
        return match.group(1)
    # bioemu: .../intrinsic/{cluster}/pdbs/
    match = re.search(
        r"/(?:intrinsic|ligand-induced|protein-induced|apo-monomers)/([^/]+)/pdbs/",
        pattern,
    )
    if match:
        return ""
    return ""


def extract_method_type_from_pattern(pattern: str) -> str:
    """Extract set segment from a prediction path: intrinsic, ligand-induced, or protein-induced."""
    match = re.search(
        r"/(intrinsic|ligand-induced|protein-induced|apo-monomers)/[^/]+/", pattern
    )
    if match:
        seg = match.group(1)
        return "intrinsic" if seg == "apo-monomers" else seg
    return ""


def enhance_cluster_data(
    cluster_id: str,
    cluster_data: Dict,
    set_name: str,
    representative_sequences: Dict,
    enrich_methods: Optional[Set[str]] = None,
) -> Dict:
    """
    Enhance cluster data with additional information needed for distogram analysis.
    Maintains the original structure while adding analysis metadata.
    """
    # Create a deep copy to avoid modifying original
    enhanced_data = json.loads(json.dumps(cluster_data))
    de = _enrich_cfg()
    af3_chain_json = _map_cfg_path(de.get("af3_chain_mappings"))
    boltz_chain_json = _map_cfg_path(de.get("boltz_chain_mappings"))
    boltz1_chain_json = _map_cfg_path(de.get("boltz1_chain_mappings"))

    # Add MSA path at cluster level
    enhanced_data["msa_path"] = get_msa_path(cluster_id)

    # Add representative sequence if available
    if cluster_id in representative_sequences:
        enhanced_data["representative_sequence_id"] = representative_sequences[
            cluster_id
        ].get("header", "")

    # Enhance apo tags with reference information
    if "apo_tags" in enhanced_data:
        apo_refs = {}
        for tag in enhanced_data["apo_tags"]:
            apo_refs[tag] = {
                "reference_cif_path": get_reference_cif_path(tag),
                "target_chain": extract_chain_from_yaml_tag(tag),
            }
        enhanced_data["apo_references"] = apo_refs

    # Enhance holo tags with reference information
    if "holo_tags" in enhanced_data:
        holo_refs = {}
        for tag in enhanced_data["holo_tags"]:
            holo_refs[tag] = {
                "reference_cif_path": get_reference_cif_path(tag),
                "target_chain": extract_chain_from_yaml_tag(tag),
            }
        enhanced_data["holo_references"] = holo_refs

    # Enhance predictions with target chain and distogram path information
    # apo_predictions / holo_predictions: tag -> method -> info
    if "apo_predictions" in enhanced_data:
        enhanced_apo_by_tag: Dict[str, Dict[str, dict]] = {}
        for yaml_tag, conformation_data in enhanced_data["apo_predictions"].items():
            if not isinstance(conformation_data, dict):
                continue
            sample = next(iter(conformation_data.values()), None)
            if isinstance(sample, dict) and ("pattern" in sample or "yaml_tag" in sample):
                enhanced_apo_by_tag[yaml_tag] = _enhance_prediction_methods(
                    conformation_data,
                    method_label=f"apo_predictions/{yaml_tag}",
                    cluster_id=cluster_id,
                    set_name=set_name,
                    af3_chain_json=af3_chain_json,
                    boltz_chain_json=boltz_chain_json,
                    boltz1_chain_json=boltz1_chain_json,
                    default_yaml_tag=yaml_tag,
                    enrich_methods=enrich_methods,
                )
            else:
                raise click.ClickException(
                    f"Unexpected apo_predictions shape for cluster {cluster_id}, tag {yaml_tag!r}"
                )
        enhanced_data["apo_predictions"] = enhanced_apo_by_tag

    if "holo_predictions" in enhanced_data:
        for conformation, conformation_data in enhanced_data[
            "holo_predictions"
        ].items():
            enhanced_data["holo_predictions"][conformation] = _enhance_prediction_methods(
                conformation_data,
                method_label=f"holo_predictions/{conformation}",
                cluster_id=cluster_id,
                set_name=set_name,
                af3_chain_json=af3_chain_json,
                boltz_chain_json=boltz_chain_json,
                boltz1_chain_json=boltz1_chain_json,
                default_yaml_tag=conformation,
                enrich_methods=enrich_methods,
            )

    return enhanced_data


def _enhance_prediction_methods(
    conformation_data: Dict[str, dict],
    *,
    method_label: str,
    cluster_id: str,
    set_name: str,
    af3_chain_json: str,
    boltz_chain_json: str,
    boltz1_chain_json: str,
    default_yaml_tag: str,
    enrich_methods: Optional[Set[str]] = None,
) -> Dict[str, dict]:
    enhanced_conformation_data: Dict[str, dict] = {}
    for method, method_info in conformation_data.items():
        enhanced_method_info = json.loads(json.dumps(method_info))
        if enrich_methods is not None and method not in enrich_methods:
            enhanced_conformation_data[method] = enhanced_method_info
            continue

        if method == "bioemu":
            enhanced_method_info["target_chain"] = "A"
            pattern = get_distogram_path_pattern(
                method, set_name, cluster_id, yaml_tag=""
            )
            if pattern:
                enhanced_method_info["distogram_pattern"] = pattern
        else:
            pattern = method_info.get("pattern", "")
            disk_yaml_tag = (
                extract_yaml_tag_from_pattern(pattern)
                or method_info.get("yaml_tag")
                or default_yaml_tag
            )
            canonical_yaml_tag = (
                method_info.get("canonical_yaml_tag") or default_yaml_tag
            )
            lookup_yaml_tag = canonical_yaml_tag or disk_yaml_tag
            path_method_type = extract_method_type_from_pattern(pattern) or set_name
            chain_yaml_tag = disk_yaml_tag
            reference_yaml_tag = lookup_yaml_tag

            if disk_yaml_tag:
                enhanced_method_info["yaml_tag"] = disk_yaml_tag
                if canonical_yaml_tag and canonical_yaml_tag != disk_yaml_tag:
                    enhanced_method_info["canonical_yaml_tag"] = canonical_yaml_tag
                enhanced_method_info["reference_cif_path"] = get_reference_cif_path(
                    reference_yaml_tag
                )
                enhanced_method_info["target_chain"] = get_target_chain_for_method(
                    method, chain_yaml_tag, path_method_type, cluster_id, set_name
                )
                if method in ("af3", "boltz1", "boltz2"):
                    disto_pat = get_distogram_path_pattern(
                        method, path_method_type, cluster_id, disk_yaml_tag
                    )
                    if disto_pat:
                        enhanced_method_info["distogram_pattern"] = disto_pat
                    if method == "af3" and af3_chain_json:
                        af3_mapping = get_chain_mapping(
                            cluster_id, chain_yaml_tag, set_name, af3_chain_json
                        )
                        if af3_mapping:
                            enhanced_method_info["chain_mapping"] = af3_mapping
                    elif method == "boltz2" and boltz_chain_json:
                        boltz_mapping = get_chain_mapping(
                            cluster_id, chain_yaml_tag, set_name, boltz_chain_json
                        )
                        if boltz_mapping:
                            enhanced_method_info["chain_mapping"] = boltz_mapping
                    elif method == "boltz1" and boltz1_chain_json:
                        boltz1_mapping = get_chain_mapping(
                            cluster_id, chain_yaml_tag, set_name, boltz1_chain_json
                        )
                        if boltz1_mapping:
                            enhanced_method_info["chain_mapping"] = boltz1_mapping
            else:
                raise click.ClickException(
                    f"Could not extract yaml_tag from pattern for {method} "
                    f"in {method_label} (cluster {cluster_id}): {pattern!r}"
                )

        enhanced_conformation_data[method] = enhanced_method_info
    return enhanced_conformation_data


def enrich_seq_cluster_map(
    seq_cluster_data: Dict[str, Any],
    representative_sequences: Dict[str, Any],
    enrich_methods: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Enrich every cluster in a seq_cluster_to_answer_map (e.g. from make_pairs).
    The dynamics set is keyed ``intrinsic`` in the map and in
    ``distogram_enrich`` path templates.

  When ``enrich_methods`` is set, only those prediction methods get
  ``target_chain`` / distogram / chain-mapping fields; other methods keep
  their existing per-method dicts unchanged.
    """
    enhanced_data: Dict[str, Any] = {}
    for set_name, clusters in seq_cluster_data.items():
        enhanced_data[set_name] = {}
        for cluster_id, cluster_data in clusters.items():
            enhanced_data[set_name][cluster_id] = enhance_cluster_data(
                cluster_id,
                cluster_data,
                set_name,
                representative_sequences,
                enrich_methods=enrich_methods,
            )
    return enhanced_data

# ============================================================================
# Main CLI
# ============================================================================
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--csv-dir", type=click.Path(exists=True, file_okay=False), default="data/dataset", show_default=True)
@click.option("--clusters-json", type=click.Path(exists=True, dir_okay=False), default="data/clusters.json", show_default=True)
@click.option("--examples-dir", type=click.Path(file_okay=False), default="examples", show_default=True)
@click.option("--outdir", type=click.Path(file_okay=False), default="data/dataset", show_default=True)
@click.option(
    "--skip-enrichment",
    is_flag=True,
    default=False,
    help="Skip MSA/CIF/distogram/chain enrichment (prediction globs only).",
)
@click.option(
    "--enrich-methods",
    type=str,
    default=None,
    help="Comma-separated prediction methods to enrich (e.g. boltz2). "
    "Default: all methods with predictions.",
)
def main(csv_dir, clusters_json, examples_dir, outdir, skip_enrichment, enrich_methods):
    """Generate seq_cluster_to_answer_map.json and valid_pairs.json"""
    csv_dir = Path(csv_dir)
    clusters_json = Path(clusters_json)
    examples_dir = Path(examples_dir) if examples_dir else None
    out_dir = Path(outdir)
    
    click.echo("=" * 60)
    click.echo("make_pairs.py")
    click.echo("=" * 60)
    
    # 1. Load valid centers
    click.echo("\n[1] Loading valid centers from clusters.json...")
    valid_centers = load_valid_centers(clusters_json)
    click.echo(f"  Found {len(valid_centers)} valid centers")
    
    # 2. Load and process CSVs
    click.echo("\n[2] Loading CSVs...")
    data = {}
    
    rows = load_csv(csv_dir / "intrinsic.csv")
    data["intrinsic"] = process_intrinsic(rows)
    click.echo(f"  intrinsic: {len(data['intrinsic'])} clusters")
    
    for set_name in ["protein-induced", "ligand-induced"]:
        rows = load_csv(csv_dir / f"{set_name}.csv")
        data[set_name] = process_induced_set(rows)
        click.echo(f"  {set_name}: {len(data[set_name])} clusters")
    
    # 3. Filter
    click.echo("\n[3] Filtering (valid centers + conf overlap)...")
    data = filter_data(data, valid_centers)
    for set_name in SET_NAMES:
        click.echo(f"  {set_name}: {len(data.get(set_name, {}))} clusters")

    # 3b. Valid pairs from CSV rows only; trim clusters/tags to pair scope
    click.echo("\n[3b] Building valid pairs from CSV rows...")
    valid_pairs = build_valid_pairs_from_csv(csv_dir, data)
    data = restrict_to_valid_pairs(data, valid_pairs, csv_dir)
    valid_pairs = {
        set_name: {
            cluster: pairs
            for cluster, pairs in valid_pairs.get(set_name, {}).items()
            if cluster in data.get(set_name, {})
        }
        for set_name in SET_NAMES
    }
    for set_name in SET_NAMES:
        n_cl = len(data.get(set_name, {}))
        n_pairs = sum(len(v) for v in valid_pairs.get(set_name, {}).values())
        click.echo(f"  {set_name}: {n_cl} clusters, {n_pairs} CSV pairs")
    
    # 4. Add predictions
    click.echo("\n[4] Adding prediction paths...")
    if examples_dir and examples_dir.exists():
        click.echo(f"  Using examples from: {examples_dir}")
        data = add_predictions_to_data(data, examples_dir)
        model_counts = defaultdict(int)
        for clusters in data.values():
            for info in clusters.values():
                models_in_cluster: Set[str] = set()
                for tag_preds in (info.get("apo_predictions") or {}).values():
                    models_in_cluster.update(tag_preds.keys())
                for model in models_in_cluster:
                    model_counts[model] += 1
        for model, count in sorted(model_counts.items()):
            click.echo(f"    {model}: {count} clusters with predictions")
    else:
        click.echo("  [SKIP] examples directory not found")
        data = add_predictions_to_data(data, None)

    rep_path = pipeline_cfg.file("rep_seq")
    if skip_enrichment:
        click.echo("\n[4b] Skipping enrichment (--skip-enrichment)")
    elif rep_path is None or not rep_path.exists():
        raise click.ClickException(
            "Enrichment requires pipeline.files.rep_seq in config to point to an existing JSON file."
        )
    else:
        enrich_method_set: Optional[Set[str]] = None
        if enrich_methods:
            enrich_method_set = {m.strip() for m in enrich_methods.split(",") if m.strip()}
            click.echo(
                f"\n[4b] Enriching seq_cluster_to_answer_map "
                f"(methods: {', '.join(sorted(enrich_method_set))})..."
            )
        else:
            click.echo("\n[4b] Enriching seq_cluster_to_answer_map (MSA, CIF, distogram, chains)...")
        with open(rep_path) as fh:
            rep_data = json.load(fh)
        data = enrich_seq_cluster_map(data, rep_data, enrich_methods=enrich_method_set)
        click.echo("  Enrichment done.")

    # 5. Summary (valid_pairs already built from CSV in step 3b)
    click.echo("\n[5] Valid pairs (from CSV)...")
    for set_name in SET_NAMES:
        n_cl = len(valid_pairs.get(set_name, {}))
        total = sum(len(v) for v in valid_pairs.get(set_name, {}).values())
        click.echo(f"  {set_name}: {n_cl} clusters, {total} pairs")
    
    # 6. Save outputs
    click.echo("\n[6] Saving...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "seq_cluster_to_answer_map.json", 'w') as f:
        json.dump(data, f, indent=2)
    click.echo(f"  -> {out_dir / 'seq_cluster_to_answer_map.json'}")
    
    with open(out_dir / "valid_pairs.json", 'w') as f:
        json.dump(valid_pairs, f, indent=2)
    click.echo(f"  -> {out_dir / 'valid_pairs.json'}")
    
    click.echo("\nDone!")


if __name__ == "__main__":
    main()
