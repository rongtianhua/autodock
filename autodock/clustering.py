"""
autodock.clustering — Pose clustering for docking result analysis.
==================================================================
Greedy RMSD-based clustering of docked poses to identify distinct
binding modes.  This is a publication-grade best practice: reporting
only the energy-lowest pose misses alternative binding modes that may
be only slightly higher in energy but biologically relevant.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from autodock.core import logger

# Optional RDKit dependency (probed at import time)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _HAVE_RDKIT_CLUSTERING = True
except Exception:
    _HAVE_RDKIT_CLUSTERING = False


def _parse_pose_to_mol(pose_str: str) -> Any | None:
    """Parse a PDBQT pose string to an RDKit mol (in memory, no temp files)."""
    if not _HAVE_RDKIT_CLUSTERING:
        return None

    # Strip MODEL/ENDMDL tags to get a clean PDB block
    lines = pose_str.splitlines()
    clean_lines = []
    for line in lines:
        if line.startswith(("MODEL", "ENDMDL")):
            continue
        # Skip model number lines (pure integers)
        if line.strip().isdigit():
            continue
        if line.startswith(("ATOM  ", "HETATM")):
            clean_lines.append(line)
    if not clean_lines:
        return None

    # Sanitize AutoDock atom types → element symbols for RDKit
    from autodock.utils import _AD4_ELEMENT_MAP, safe_pdb_slice

    sanitized = []
    for line in clean_lines:
        # Read last token for atom type (robust across generators)
        stripped_tail = line[71:].strip() if len(line) > 71 else ""
        ad_type = stripped_tail.split()[-1] if stripped_tail else ""
        elem = _AD4_ELEMENT_MAP.get(ad_type, ad_type)
        if not elem:
            atom_name = safe_pdb_slice(line, 12, 16)
            elem = atom_name[0] if atom_name else "C"
        # Reconstruct with element at cols 77-78 (0-based 76-77)
        new_line = line[:76] + f"{elem:>2}\n"
        sanitized.append(new_line)

    pdb_block = "".join(sanitized)
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True)
    return mol


def _rmsd_between_mols(mol1: Any, mol2: Any) -> float | None:
    """Compute RMSD between two RDKit mols (topology-aware + fallback)."""
    if mol1 is None or mol2 is None:
        return None
    # Attempt 1: topology-aware GetBestRMS
    try:
        return float(AllChem.GetBestRMS(mol1, mol2))
    except Exception:
        pass
    # Attempt 2: coordinate-based Kabsch on matched heavy atoms
    try:
        return _rmsd_kabsch_mols(mol1, mol2)
    except Exception:
        return None


def _rmsd_kabsch_mols(mol1: Any, mol2: Any) -> float | None:
    """Kabsch RMSD between two mols using element-type matching."""
    from scipy.optimize import linear_sum_assignment

    def _coords_elems(mol):
        conf = mol.GetConformer()
        coords = np.array(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ]
        )
        elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        return coords, elems

    c1, e1 = _coords_elems(mol1)
    c2, e2 = _coords_elems(mol2)

    unique_elems = set(e1) & set(e2)
    matched_1 = []
    matched_2 = []

    for elem in unique_elems:
        idx1 = [i for i, e in enumerate(e1) if e == elem]
        idx2 = [i for i, e in enumerate(e2) if e == elem]
        if len(idx1) != len(idx2):
            continue
        sub1 = c1[idx1]
        sub2 = c2[idx2]
        cost = np.linalg.norm(sub1[:, None, :] - sub2[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_1.extend([idx1[i] for i in row_ind])
        matched_2.extend([idx2[j] for j in col_ind])

    if not matched_1:
        return None

    # Kabsch
    P = c1[matched_1]
    Q = c2[matched_2]
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    Pc = P - P_mean
    Qc = Q - Q_mean
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Pr = Pc @ R
    return float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))


def cluster_poses(
    poses: list[str],
    energies: np.ndarray,
    rmsd_threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Greedy clustering of docked poses by RMSD.

    Algorithm:
        1. Sort poses by energy (most negative = best).
        2. Initialise first cluster with the lowest-energy pose.
        3. For each remaining pose, compute RMSD to the representative
           (lowest-energy member) of each existing cluster.
        4. If RMSD < threshold to any cluster representative, join that
           cluster; otherwise start a new cluster.

    Args:
        poses: List of PDBQT pose strings (one per MODEL).
        energies: NxM array from Vina (first column is affinity).
        rmsd_threshold: RMSD cutoff in Å (default 2.0 Å).

    Returns:
        List of cluster dicts sorted by representative energy:
        {
            "representative_index": int,
            "size": int,
            "member_indices": list[int],
            "representative_energy": float,
            "member_energies": list[float],
        }
    """
    if not poses or energies.size == 0:
        return []

    n = len(poses)
    if n != energies.shape[0]:
        logger.warning(f"Pose/energy count mismatch: {n} poses vs {energies.shape[0]} energies")
        n = min(n, energies.shape[0])

    # Parse all poses to RDKit mols in memory (no temp files)
    mols = [_parse_pose_to_mol(p) for p in poses[:n]]

    # Sort by energy (ascending: most negative first)
    sorted_indices = np.argsort(energies[:n, 0]).tolist()

    clusters: list[dict[str, Any]] = []

    for idx in sorted_indices:
        mol = mols[idx]
        energy = float(energies[idx, 0])
        assigned = False

        for cluster in clusters:
            rep_idx = cluster["representative_index"]
            rep_mol = mols[rep_idx]

            rmsd = _rmsd_between_mols(mol, rep_mol)
            if rmsd is None:
                # Cannot compute RMSD — skip clustering for this pair
                continue

            if rmsd < rmsd_threshold:
                cluster["member_indices"].append(idx)
                cluster["member_energies"].append(energy)
                cluster["size"] += 1
                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "representative_index": int(idx),
                    "size": 1,
                    "member_indices": [int(idx)],
                    "representative_energy": energy,
                    "member_energies": [energy],
                }
            )

    # Sort clusters by representative energy (best first)
    clusters.sort(key=lambda c: c["representative_energy"])
    logger.info(
        f"Pose clustering: {n} poses → {len(clusters)} clusters " f"(threshold={rmsd_threshold} Å)"
    )
    return clusters
