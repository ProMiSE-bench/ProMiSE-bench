#!/usr/bin/env python3
"""
Create intersection of Foldseek and MMseqs hits.

For each threshold combination, find target intersection and create new hit files.

Usage:
    python create_intersection_hits.py
"""

import os
from collections import defaultdict

import click

from utils._config import eval_cfg as E

# Paths (from config)
FOLDSEEK_BASE = str(E.dir("memorization_hits_foldseek"))
MMSEQS_BASE = str(E.dir("memorization_hits_mmseqs"))
OUTPUT_BASE = str(E.dir("memorization_hits_intersection"))

# Models
MODELS = ("af3", "boltz_2", "chai_1", "bioemu")

# Thresholds
TM_THRESHOLDS = (0.9,)
FIDENT_THRESHOLDS = (0.8,)

# Categories
CATEGORIES = ("intrinsic", "ligand-induced", "protein-induced")


def load_targets_from_file(tsv_file):
    """
    Load unique target PDB IDs (first 4 characters of 2nd column) from a TSV file.
    Returns: set of PDB IDs
    """
    if not os.path.exists(tsv_file):
        return set()
    
    targets = set()
    with open(tsv_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                target = parts[1]
                # Extract PDB ID (first 4 characters)
                pdb_id = target.split('_')[0][:4]
                targets.add(pdb_id)
    
    return targets


def load_hits_by_target(tsv_file):
    """
    Load hits grouped by target PDB ID.
    Returns: dict[pdb_id, list[hit_lines]]
    """
    if not os.path.exists(tsv_file):
        return {}
    
    hits_by_target = defaultdict(list)
    with open(tsv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                target = parts[1]
                # Extract PDB ID (first 4 characters)
                pdb_id = target.split('_')[0][:4]
                hits_by_target[pdb_id].append(line)
    
    return dict(hits_by_target)


def create_intersection_file(foldseek_file, mmseqs_file, output_file):
    """
    Create intersection of target PDB IDs and write to output file.
    Uses Foldseek hits but only for PDB IDs that exist in both files.
    
    Returns: number of intersection PDB IDs
    """
    # Load target PDB IDs from both files
    foldseek_targets = load_targets_from_file(foldseek_file)
    mmseqs_targets = load_targets_from_file(mmseqs_file)
    
    # Find intersection
    intersection_targets = foldseek_targets & mmseqs_targets
    
    # Always create output file and directory, even if empty
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # If no intersection, create empty file and return
    if not intersection_targets:
        open(output_file, 'w').close()
        return 0
    
    # Load Foldseek hits grouped by PDB ID
    foldseek_hits = load_hits_by_target(foldseek_file)
    
    # Write intersection hits to output file
    with open(output_file, 'w') as f:
        for pdb_id in sorted(intersection_targets):
            if pdb_id in foldseek_hits:
                for hit_line in foldseek_hits[pdb_id]:
                    f.write(hit_line + '\n')
    
    return len(intersection_targets)


def process_intersection(tm_threshold, fident_threshold, models):
    """
    Process intersection for a specific threshold combination.
    """
    foldseek_dir = f"hits_hits_foldseek_tm_{tm_threshold}"
    mmseqs_dir = f"hits_hits_mmseqs_fident_{fident_threshold}"
    output_dir = f"hits_tm_{tm_threshold}_fident_{fident_threshold}"
    
    total_intersections = 0
    total_files = 0
    
    for model in models:
        for category in CATEGORIES:
            foldseek_category_dir = os.path.join(FOLDSEEK_BASE, foldseek_dir, model, category)
            mmseqs_category_dir = os.path.join(MMSEQS_BASE, mmseqs_dir, model, category)
            output_category_dir = os.path.join(OUTPUT_BASE, output_dir, model, category)
            
            if not os.path.exists(foldseek_category_dir):
                click.echo(f"Warning: Foldseek directory not found: {foldseek_category_dir}")
                continue
            
            if not os.path.exists(mmseqs_category_dir):
                click.echo(f"Warning: MMseqs directory not found: {mmseqs_category_dir}")
                continue
            
            # Process each cluster directory
            category_intersections = 0
            category_files = 0
            
            for cluster_name in os.listdir(foldseek_category_dir):
                foldseek_cluster_dir = os.path.join(foldseek_category_dir, cluster_name)
                mmseqs_cluster_dir = os.path.join(mmseqs_category_dir, cluster_name)
                output_cluster_dir = os.path.join(output_category_dir, cluster_name)
                
                if not os.path.isdir(foldseek_cluster_dir):
                    continue
                
                if not os.path.exists(mmseqs_cluster_dir):
                    continue
                
                # Process each conf file
                for conf_file in os.listdir(foldseek_cluster_dir):
                    if not conf_file.startswith('conf_') or not conf_file.endswith('.tsv'):
                        continue
                    
                    foldseek_file = os.path.join(foldseek_cluster_dir, conf_file)
                    mmseqs_file = os.path.join(mmseqs_cluster_dir, conf_file)
                    output_file = os.path.join(output_cluster_dir, conf_file)
                    
                    num_intersection = create_intersection_file(foldseek_file, mmseqs_file, output_file)
                    
                    if num_intersection > 0:
                        category_intersections += num_intersection
                        category_files += 1
            
            total_intersections += category_intersections
            total_files += category_files

    return total_files, total_intersections


@click.command()
@click.option(
    "--tm-threshold",
    "tm_thresholds",
    multiple=True,
    type=float,
    default=TM_THRESHOLDS,
    show_default=True,
)
@click.option(
    "--fident-threshold",
    "fident_thresholds",
    multiple=True,
    type=float,
    default=FIDENT_THRESHOLDS,
    show_default=True,
)
@click.option(
    "--model",
    "models",
    multiple=True,
    type=click.Choice(MODELS, case_sensitive=False),
    help="Model to process. Repeat to select multiple. Defaults to all models.",
)
def main(
    tm_thresholds: tuple[float, ...],
    fident_thresholds: tuple[float, ...],
    models: tuple[str, ...],
) -> None:
    """Create intersections of Foldseek and MMseqs memorization hits."""
    # Create output base directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Process all combinations
    summary = []
    selected_models = tuple(model.lower() for model in (models or MODELS))

    for tm_threshold in tm_thresholds:
        for fident_threshold in fident_thresholds:
            total_files, total_intersections = process_intersection(
                tm_threshold,
                fident_threshold,
                selected_models,
            )
            summary.append({
                'tm': tm_threshold,
                'fident': fident_threshold,
                'files': total_files,
                'targets': total_intersections
            })

    for item in summary:
        click.echo(
            f"TM >= {item['tm']}, fident >= {item['fident']}: "
            f"files={item['files']}, targets={item['targets']}"
        )

    click.echo(f"Saved to: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
