"""
autodock.validation_params — Input validation and parameter sanitisation.
============================================================================
Centralised validation layer to ensure all user-facing parameters are
checked before they reach the docking engine.  This prevents cryptic
low-level errors and guarantees publication-grade reproducibility.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from autodock.core import (
    ConfigurationError,
    DockingCalculationError,
    PreparationError,
    VINA_DEFAULT_EXHAUSTIVENESS,
    VINA_DEFAULT_N_POSES,
    VINA_DEFAULT_ENERGY_RANGE,
    VINA_DEFAULT_TIMEOUT,
    _POCKET_MIN_DIM,
    _POCKET_MAX_DIM,
)

# ─────────────────────────────────────────────────────────────────────────────
# File validators
# ─────────────────────────────────────────────────────────────────────────────

def validate_file_exists(path: str | Path, label: str = "file") -> str:
    """Return absolute path if it exists, else raise."""
    p = Path(path)
    if not p.exists():
        raise DockingCalculationError(f"{label} not found: {p}")
    if not p.is_file():
        raise DockingCalculationError(f"{label} is not a file: {p}")
    return str(p.resolve())


def validate_pdbqt_file(path: str | Path, label: str = "PDBQT") -> str:
    """Validate file exists and contains at least one ATOM/HETATM record."""
    path = validate_file_exists(path, label)
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                return path
    raise PreparationError(f"{label} file contains no ATOM/HETATM records: {path}")


def validate_smiles(smiles: str) -> str:
    """Validate a SMILES string via RDKit."""
    try:
        from rdkit import Chem
    except ImportError:
        raise PreparationError("RDKit is required for SMILES validation.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PreparationError(f"Invalid SMILES string: {smiles}")
    return smiles


def validate_pdb_id(pdb_id: str) -> str:
    """Validate a PDB ID (4 alphanumeric characters)."""
    pid = pdb_id.strip().upper()
    if len(pid) != 4 or not pid.isalnum():
        raise ConfigurationError(f"Invalid PDB ID: {pdb_id} (expected 4 alphanumeric characters)")
    return pid


# ─────────────────────────────────────────────────────────────────────────────
# Numeric parameter validators
# ─────────────────────────────────────────────────────────────────────────────

def validate_exhaustiveness(value: int) -> int:
    v = int(value)
    if v < 1:
        raise ConfigurationError(f"exhaustiveness must be >= 1, got {v}")
    if v > 1024:
        raise ConfigurationError(f"exhaustiveness must be <= 1024, got {v}")
    return v


def validate_n_poses(value: int) -> int:
    v = int(value)
    if v < 1:
        raise ConfigurationError(f"n_poses must be >= 1, got {v}")
    if v > 1000:
        raise ConfigurationError(f"n_poses must be <= 1000, got {v}")
    return v


def validate_energy_range(value: float) -> float:
    v = float(value)
    if v <= 0:
        raise ConfigurationError(f"energy_range must be > 0, got {v}")
    if v > 100:
        raise ConfigurationError(f"energy_range must be <= 100, got {v}")
    return v


def validate_timeout(value: int) -> int:
    v = int(value)
    if v < 1:
        raise ConfigurationError(f"timeout must be >= 1 second, got {v}")
    if v > 86400:
        raise ConfigurationError(f"timeout must be <= 86400 seconds (1 day), got {v}")
    return v


def validate_seed(value: int | None) -> int | None:
    if value is None:
        return None
    v = int(value)
    if v < 0:
        raise ConfigurationError(f"seed must be non-negative, got {v}")
    if v > 2_147_483_647:
        raise ConfigurationError(f"seed must be <= 2^31-1, got {v}")
    return v


def validate_box_size(size: tuple[float, float, float]) -> tuple[float, float, float]:
    s = tuple(float(x) for x in size)
    if len(s) != 3:
        raise ConfigurationError(f"box_size must be a 3-tuple, got {s}")
    for dim in s:
        if dim < _POCKET_MIN_DIM:
            raise ConfigurationError(
                f"box_size dimension {dim} Å is below minimum {_POCKET_MIN_DIM} Å"
            )
        if dim > _POCKET_MAX_DIM:
            raise ConfigurationError(
                f"box_size dimension {dim} Å exceeds maximum {_POCKET_MAX_DIM} Å"
            )
    return s


def validate_center(center: tuple[float, float, float]) -> tuple[float, float, float]:
    c = tuple(float(x) for x in center)
    if len(c) != 3:
        raise ConfigurationError(f"center must be a 3-tuple, got {c}")
    return c


def validate_n_workers(value: int) -> int:
    v = int(value)
    if v == -1:
        return -1
    if v < 1:
        raise ConfigurationError(f"n_workers must be >= 1 or -1 (all cores), got {v}")
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Combined docking parameter validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_docking_params(
    receptor_pdbqt: str | Path,
    ligand_pdbqt: str | Path,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    exhaustiveness: int = VINA_DEFAULT_EXHAUSTIVENESS,
    n_poses: int = VINA_DEFAULT_N_POSES,
    energy_range: float = VINA_DEFAULT_ENERGY_RANGE,
    seed: int | None = None,
    timeout: int = VINA_DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Validate all docking parameters and return a sanitised dictionary.

    Raises:
        PreparationError: If structure files are missing or invalid.
        ConfigurationError: If numeric parameters are out of bounds.
    """
    return {
        "receptor_pdbqt": validate_pdbqt_file(receptor_pdbqt, "receptor"),
        "ligand_pdbqt": validate_pdbqt_file(ligand_pdbqt, "ligand"),
        "center": validate_center(center),
        "box_size": validate_box_size(box_size),
        "exhaustiveness": validate_exhaustiveness(exhaustiveness),
        "n_poses": validate_n_poses(n_poses),
        "energy_range": validate_energy_range(energy_range),
        "seed": validate_seed(seed),
        "timeout": validate_timeout(timeout),
    }
