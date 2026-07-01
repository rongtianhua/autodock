"""Lightweight covalent-warhead detection and annotation.

This module does **not** perform covalent docking. It detects electrophilic
warheads in ligands, maps them to compatible reactive residues, and produces
warnings/annotations that are carried through the docking pipeline. The actual
docking is still run with AutoDock Vina in non-covalent mode.

References:
    - Singh et al. (2025) "The covalent docking software landscape".
    - Baillie (2016) "Targeted covalent inhibitors for drug design".
    - Schirmeister & Kesselring (2020) "Covalent inhibitors of cysteine proteases".
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WarheadMatch:
    """A single warhead match in a ligand."""

    name: str
    smarts: str
    # Indices (in the input mol) of atoms that can form the covalent bond.
    reactive_atom_indices: tuple[int, ...]
    compatible_residues: tuple[str, ...]


@dataclass(slots=True)
class CovalentAnnotation:
    """Annotation produced for a ligand by :func:`detect_covalent_warheads`."""

    has_warhead: bool = False
    warhead_matches: list[WarheadMatch] = field(default_factory=list)
    recommended_residues: set[str] = field(default_factory=set)
    risk_level: str = "low"  # low | medium | high
    message: str = ""


#: Warhead SMARTS taxonomy. Order matters: more specific patterns should come
#: before generic ones to avoid a single atom being claimed by multiple warheads
#: when only the broadest pattern matches.
WARHEAD_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "maleimide",
        "smarts": "[C;H1:1]=[C:2]-C(=O)-N-C(=O)",
        "reactive_atoms": (0, 1),
        "compatible_residues": ("CYS",),
        "risk": "medium",
    },
    {
        # Michael acceptor: C=C-C(=O)-N (acrylamide, methacrylamide, crotonamide, ...)
        # Matches both terminal and substituted acrylamides.
        "name": "acrylamide",
        "smarts": "[C:1]=[C:2]-C(=O)-[N;H1,H2]",
        "reactive_atoms": (0, 1),
        "compatible_residues": ("CYS",),
        "risk": "medium",
    },
    {
        "name": "vinyl_sulfone",
        "smarts": "[C;H2:1]=[C;H1:2]-S(=O)(=O)",
        "reactive_atoms": (0, 1),
        "compatible_residues": ("CYS",),
        "risk": "medium",
    },
    {
        "name": "haloacetamide",
        "smarts": "[Cl,Br,I]-[CH2]-C(=O)-[N;H1,H2]",
        "reactive_atoms": (1,),
        "compatible_residues": ("CYS", "LYS", "HIS"),
        "risk": "high",
    },
    {
        "name": "sulfonyl_fluoride",
        "smarts": "[#16:1](=[O])(=[O])-[F]",
        "reactive_atoms": (0,),
        "compatible_residues": ("SER", "THR", "TYR", "LYS", "CYS"),
        "risk": "high",
    },
    {
        "name": "fluorosulfate",
        "smarts": "[O:1]-S(=O)(=O)-[F]",
        "reactive_atoms": (0,),
        "compatible_residues": ("TYR", "SER", "THR", "LYS"),
        "risk": "high",
    },
    {
        "name": "aldehyde",
        "smarts": "[C;H1,H2](=O)",
        "reactive_atoms": (0,),
        "compatible_residues": ("LYS", "CYS", "SER", "THR"),
        "risk": "medium",
    },
    {
        "name": "boronic_acid",
        "smarts": "[B;H0:1](-[O])(-[O])",
        "reactive_atoms": (0,),
        "compatible_residues": ("SER", "THR", "TYR"),
        "risk": "low",
    },
    {
        "name": "nitrile",
        "smarts": "[C:1]#[N]",
        "reactive_atoms": (0,),
        "compatible_residues": ("CYS", "SER", "THR"),
        "risk": "medium",
    },
    {
        "name": "epoxide",
        "smarts": "[C;R1:1]-[O;R1]-[C;R1:2]",
        "reactive_atoms": (0, 2),
        "compatible_residues": ("CYS", "ASP", "GLU", "HIS"),
        "risk": "medium",
    },
    {
        "name": "terminal_alkyne_amide",
        "smarts": "[C;H1:1]#[C]-C(=O)-[N;H0,H1]",
        "reactive_atoms": (0, 1),
        "compatible_residues": ("CYS",),
        "risk": "medium",
    },
    {
        # Generic alkyl halide (last because it is broad)
        "name": "alkyl_halide",
        "smarts": "[C;!$(C=C):1]-[Cl,Br,I]",
        "reactive_atoms": (0,),
        "compatible_residues": ("CYS", "LYS", "ASP", "GLU", "HIS", "MET"),
        "risk": "high",
    },
]

# Residues whose side chains can act as nucleophiles in covalent binding.
REACTIVE_RESIDUE_MAP: dict[str, tuple[str, ...]] = {
    "CYS": ("SG",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
    "LYS": ("NZ",),
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("NE2", "ND1"),
    "MET": ("SD",),
}


def _rdkit_available() -> bool:
    """Return True if RDKit can be imported."""
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


def detect_covalent_warheads(smiles: str) -> CovalentAnnotation:
    """Detect covalent warheads in a SMILES string.

    Args:
        smiles: Input SMILES.

    Returns:
        CovalentAnnotation describing any detected warheads. If RDKit is not
        available, returns an empty annotation.
    """
    annotation = CovalentAnnotation()
    if not _rdkit_available():
        logger.debug("RDKit not available; skipping covalent warhead detection")
        return annotation

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.debug("Could not parse SMILES for covalent detection: %s", smiles)
        return annotation

    matched_atom_ids: set[int] = set()
    for pattern in WARHEAD_PATTERNS:
        smarts = pattern["smarts"]
        with contextlib.suppress(Exception):
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            matches = mol.GetSubstructMatches(patt, useChirality=False)
            for match in matches:
                # Avoid double-counting when a broad pattern overlaps with a
                # specific one on the same reactive atoms.
                reactive_ids = tuple(match[i] for i in pattern["reactive_atoms"])
                if any(idx in matched_atom_ids for idx in reactive_ids):
                    continue
                for idx in reactive_ids:
                    matched_atom_ids.add(idx)
                annotation.warhead_matches.append(
                    WarheadMatch(
                        name=pattern["name"],
                        smarts=smarts,
                        reactive_atom_indices=reactive_ids,
                        compatible_residues=pattern["compatible_residues"],
                    )
                )
                annotation.recommended_residues.update(pattern["compatible_residues"])

    if annotation.warhead_matches:
        annotation.has_warhead = True
        annotation.risk_level = _aggregate_risk(annotation.warhead_matches)
        names = [m.name for m in annotation.warhead_matches]
        res = ", ".join(sorted(annotation.recommended_residues))
        annotation.message = (
            f"Covalent warhead(s) detected: {', '.join(names)}. "
            f"Compatible reactive residues: {res}. "
            f"AutoDock Vina will treat this ligand non-covalently; "
            f"reported affinities do not include covalent bond energy."
        )
    return annotation


def _aggregate_risk(matches: list[WarheadMatch]) -> str:
    """Return the highest risk level among matched warheads."""
    risk_order = {"low": 0, "medium": 1, "high": 2}
    max_risk = "low"
    risk_map = {p["name"]: p["risk"] for p in WARHEAD_PATTERNS}
    for match in matches:
        level = risk_map.get(match.name, "low")
        if risk_order.get(level, 0) > risk_order.get(max_risk, 0):
            max_risk = level
    return max_risk


def find_reactive_residues_in_receptor(
    receptor_pdbqt: str,
    residue_types: set[str] | None = None,
    center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Find reactive/nucleophilic residues in a receptor PDBQT file.

    Args:
        receptor_pdbqt: Path to receptor PDBQT.
        residue_types: Set of 3-letter residue codes to look for. If None, all
            known nucleophilic residues are searched.
        center: Optional (x, y, z) box center; used with ``box_size`` to limit
            the search to the docking box.
        box_size: Optional (x, y, z) box dimensions.

    Returns:
        List of dicts with keys: chain, resname, resnum, atomname, x, y, z.
    """
    if residue_types is None:
        residue_types = set(REACTIVE_RESIDUE_MAP.keys())

    results: list[dict[str, Any]] = []
    half_box = (
        (box_size[0] / 2.0, box_size[1] / 2.0, box_size[2] / 2.0)
        if box_size is not None
        else None
    )

    with open(receptor_pdbqt) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line) < 54:
                continue
            try:
                atomname = line[12:16].strip()
                resname = line[17:20].strip().upper()
                if resname not in residue_types:
                    continue
                expected_atoms = REACTIVE_RESIDUE_MAP.get(resname, ())
                if atomname not in expected_atoms:
                    continue
                chain = line[21].strip()
                resnum = line[22:26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            if center is not None and half_box is not None and (
                abs(x - center[0]) > half_box[0]
                or abs(y - center[1]) > half_box[1]
                or abs(z - center[2]) > half_box[2]
            ):
                continue

            results.append(
                {
                    "chain": chain,
                    "resname": resname,
                    "resnum": resnum,
                    "atomname": atomname,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
    return results


def check_reactive_geometry(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    reactive_atom_indices: tuple[int, ...],
    reactive_residue_atoms: list[dict[str, Any]],
    max_dist: float = 5.0,
) -> dict[str, Any]:
    """Check whether any ligand reactive atom is within ``max_dist`` Å of a
    receptor nucleophilic atom.

    This is a coarse geometric gate intended for annotation only. It does not
    enforce near-attack-conformation angles or covalent bond distances.

    Args:
        ligand_pdbqt: Path to ligand PDBQT.
        receptor_pdbqt: Path to receptor PDBQT.
        reactive_atom_indices: Ligand atom indices (0-based) to consider.
        reactive_residue_atoms: Output of :func:`find_reactive_residues_in_receptor`.
        max_dist: Distance threshold in Å.

    Returns:
        Dict with ``feasible`` (bool), ``min_dist`` (float), and ``closest_pair``.
    """
    import math

    ligand_atoms: list[tuple[int, float, float, float]] = []
    with open(ligand_pdbqt) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line) < 54:
                continue
            try:
                idx = int(line[6:11].strip()) - 1
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            if idx in reactive_atom_indices:
                ligand_atoms.append((idx, x, y, z))

    if not ligand_atoms or not reactive_residue_atoms:
        return {"feasible": False, "min_dist": float("inf"), "closest_pair": None}

    min_dist = float("inf")
    closest_pair: tuple[Any, Any] | None = None
    for lidx, lx, ly, lz in ligand_atoms:
        for ratom in reactive_residue_atoms:
            dx = lx - ratom["x"]
            dy = ly - ratom["y"]
            dz = lz - ratom["z"]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < min_dist:
                min_dist = d
                closest_pair = (lidx, ratom)

    return {
        "feasible": min_dist <= max_dist,
        "min_dist": min_dist,
        "closest_pair": closest_pair,
    }


def format_covalent_warning(annotation: CovalentAnnotation) -> str:
    """Return a human-readable warning string for logs/CLI output."""
    if not annotation.has_warhead:
        return ""
    return annotation.message
