"""Shared constants for conformation-success analysis."""

from __future__ import annotations

from typing import Dict, Tuple

# A sample "covers" a conformation iff TM(conf) >= TM_THRESHOLD AND
# TM(conf) is strictly greater than the TM of every other conformation
# for the same sample.
TM_THRESHOLD: float = 0.8

METHODS: Tuple[str, ...] = ("af3", "boltz1", "boltz2", "chai", "bioemu")

METHODS_WITH_CONFIDENCE: Tuple[str, ...] = ("af3", "boltz1", "boltz2", "chai")

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "af3": "AF3",
    "boltz1": "Boltz-1",
    "boltz2": "Boltz-2",
    "chai": "Chai-1",
    "bioemu": "Bioemu",
}

SET_TYPES: Tuple[str, ...] = ("intrinsic", "ligand-induced", "protein-induced")

SET_TYPE_COLORS: Dict[str, str] = {
    "intrinsic": "#8D9FCF",
    "ligand-induced": "#EEBEC3",
    "protein-induced": "#8CC6C0",
}
