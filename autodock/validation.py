"""
autodock.validation — Pose validation and quality control.
==========================================================
PoseBusters checks, clash detection, RMSD calculation, and redocking validation.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from autodock.core import (
    _HAVE_RDKIT,
    CLASH_THRESHOLD_EXPLICIT_H,
    REDocking_RMSD_THRESHOLD,
    ValidationError,
    logger,
)
from autodock.utils import (
    _sanitize_pdbqt_for_rdkit,
    compute_bounding_box,
    extract_ligand_from_pdb,
    read_pdb_atoms,
)

# ─────────────────────────────────────────────────────────────────────────────
# PoseBusters Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_pose_with_posebusters(
    pose_pdbqt: str,
    receptor_pdb: str,
    ligand_ref_sdf: str | None = None,
) -> dict[str, Any]:
    """
    Validate a docked pose using PoseBusters.

    Args:
        pose_pdbqt: Docked pose PDBQT file.
        receptor_pdb: Receptor PDB file (conditioning molecule / protein).
        ligand_ref_sdf: Optional reference ligand SDF (not used by the dock config).

    Returns:
        Dict with PoseBusters checks and overall pass/fail.
    """
    try:
        from posebusters import PoseBusters
    except ImportError:
        logger.warning("PoseBusters not available — skipping validation")
        return {"available": False, "pass": None}

    if not _HAVE_RDKIT:
        logger.warning("RDKit not available — cannot convert PDBQT for PoseBusters")
        return {"available": False, "pass": None}

    # Convert PDBQT → temporary SDF because PoseBusters only accepts
    # .sdf, .mol, .mol2, or .pdb and needs bond information for chemistry checks.
    from rdkit import Chem

    sanitized_pdb = _sanitize_pdbqt_for_rdkit(pose_pdbqt)
    mol = Chem.MolFromPDBBlock(sanitized_pdb, removeHs=True)
    if mol is None:
        logger.warning("PoseBusters: RDKit could not parse sanitized PDBQT")
        return {"available": True, "pass": False, "error": "RDKit parse failure"}

    mol = Chem.AddHs(mol, addCoords=True)

    tmp_sdf = tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False)
    try:
        writer = Chem.SDWriter(tmp_sdf.name)
        writer.write(mol)
        writer.close()

        busters = PoseBusters(config="dock")
        try:
            results = busters.bust(tmp_sdf.name, mol_cond=receptor_pdb)
        except Exception as exc:
            logger.warning(f"PoseBusters validation failed: {exc}")
            return {"available": True, "pass": False, "error": str(exc)}
    finally:
        os.unlink(tmp_sdf.name)

    # PoseBusters returns a DataFrame with boolean columns.
    # Exclude checks that produce false negatives in the docking context:
    # - loading flags are not quality checks
    # - ring non-flatness is expected for chair/boat conformations
    # - cofactor/water distance/overlap checks depend on the specific pocket environment
    #   and can flag valid poses that happen to be near crystallographic additives
    _EXCLUDED_FROM_PASS = {
        "mol_pred_loaded",
        "mol_true_loaded",
        "mol_cond_loaded",
        "non-aromatic_ring_non-flatness",
        "minimum_distance_to_organic_cofactors",
        "minimum_distance_to_inorganic_cofactors",
        "minimum_distance_to_waters",
        "volume_overlap_with_organic_cofactors",
        "volume_overlap_with_inorganic_cofactors",
        "volume_overlap_with_waters",
    }
    checks = {}
    overall_pass = True
    for col in results.columns:
        if results[col].dtype != bool:
            continue
        val = bool(results[col].values[0])
        checks[col] = val
        if not val and col not in _EXCLUDED_FROM_PASS:
            overall_pass = False

    return {
        "available": True,
        "pass": overall_pass,
        "checks": checks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Clash Detection
# ─────────────────────────────────────────────────────────────────────────────


def compute_clash_score(
    pose_pdbqt: str,
    receptor_pdb: str,
    clash_threshold: float = CLASH_THRESHOLD_EXPLICIT_H,
) -> dict[str, Any]:
    """
    Compute receptor-ligand clash score.

    Returns:
        {
            "clash_score": max_overlap_Å,
            "n_clashes": int,
            "is_acceptable": bool,
            "mean_distance": float,
        }
    """
    rec_atoms = read_pdb_atoms(receptor_pdb)
    lig_atoms = read_pdb_atoms(pose_pdbqt)

    if not rec_atoms or not lig_atoms:
        return {
            "clash_score": None,
            "n_clashes": None,
            "is_acceptable": None,
            "mean_distance": None,
        }

    rec_coords = np.array([(a["x"], a["y"], a["z"]) for a in rec_atoms])

    # VDW radii (approximate, in Å) — Bondi radii for common bioorganic elements
    vdw = {
        "H": 1.2,
        "C": 1.7,
        "N": 1.55,
        "O": 1.52,
        "S": 1.8,
        "P": 1.8,
        "F": 1.47,
        "Cl": 1.75,
        "Br": 1.85,
        "I": 1.98,
        "B": 1.85,
        "Si": 2.1,
        "Se": 1.9,
        "Fe": 1.95,
        "Zn": 1.39,
        "Mg": 1.73,
        "Ca": 1.76,
        "Mn": 1.73,
        "Cu": 1.4,
        "Na": 1.02,
        "K": 1.76,
    }

    clashes = []
    min_dists = []
    for la in lig_atoms:
        lig_pt = np.array([la["x"], la["y"], la["z"]])
        lig_elem = la["element"].strip().upper() if la["element"] else "C"
        lig_r = vdw.get(lig_elem, 1.7)

        dists = np.linalg.norm(rec_coords - lig_pt, axis=1)
        min_dist = dists.min()
        min_dists.append(min_dist)

        # Find closest receptor atom
        closest_idx = dists.argmin()
        rec_elem = (
            rec_atoms[closest_idx]["element"].strip().upper()
            if rec_atoms[closest_idx]["element"]
            else "C"
        )
        rec_r = vdw.get(rec_elem, 1.7)
        sum_r = lig_r + rec_r

        # Overlap = sum_r - distance (positive = clash)
        overlap = sum_r - min_dist
        if overlap > 0.3:  # significant overlap
            clashes.append(overlap)

    if not min_dists:
        return {
            "clash_score": None,
            "n_clashes": 0,
            "is_acceptable": None,
            "mean_distance": None,
        }

    max_clash = max(clashes) if clashes else 0.0
    mean_dist = float(np.mean(min_dists))

    return {
        "clash_score": round(max_clash, 3),
        "n_clashes": len(clashes),
        "is_acceptable": max_clash <= clash_threshold,
        "mean_distance": round(mean_dist, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RMSD Calculation
# ─────────────────────────────────────────────────────────────────────────────


def compute_rmsd(
    pose1_pdbqt: str,
    pose2_pdbqt: str,
    heavy_atoms_only: bool = True,
) -> float | None:
    """
    Compute RMSD between two poses using RDKit.

    Args:
        pose1_pdbqt: First pose PDBQT.
        pose2_pdbqt: Second pose PDBQT.
        heavy_atoms_only: Exclude hydrogens from RMSD.

    Returns:
        RMSD in Å, or None if calculation fails.
    """
    if not _HAVE_RDKIT:
        logger.warning("RDKit not available — cannot compute RMSD")
        return None

    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Parse PDBQT to RDKit mols with 3D coords
    def _pdbqt_to_mol(path: str):
        pdb_block = _sanitize_pdbqt_for_rdkit(path)
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=heavy_atoms_only)
        return mol

    mol1 = _pdbqt_to_mol(pose1_pdbqt)
    mol2 = _pdbqt_to_mol(pose2_pdbqt)

    if mol1 is None or mol2 is None:
        logger.warning("Failed to parse one or both poses for RMSD")
        return None

    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        logger.warning(
            f"Atom count mismatch: {mol1.GetNumAtoms()} vs {mol2.GetNumAtoms()} — "
            f"RMSD may be unreliable"
        )

    try:
        rms = AllChem.GetBestRMS(mol1, mol2)
        return float(rms)
    except Exception as exc:
        logger.warning(f"RMSD calculation failed: {exc}")
        return None


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch algorithm for optimal RMSD between two Nx3 point sets."""
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    Pc = P - P_mean
    Qc = Q - Q_mean
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    # For the objective ||Pc @ R - Qc||^2, the optimal rotation is U @ Vt
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    Pr = Pc @ R
    return float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))


def compute_rmsd_coordinate_based(
    pose1_path: str,
    pose2_path: str,
    heavy_atoms_only: bool = True,
) -> float | None:
    """
    Compute RMSD between two poses without requiring matching atom ordering.

    Uses element-type grouping + Hungarian algorithm for correspondence,
    followed by Kabsch alignment.

    Returns:
        RMSD in Å, or None if calculation fails.
    """
    if not _HAVE_RDKIT:
        return None

    from rdkit import Chem
    from scipy.optimize import linear_sum_assignment

    # Try to parse files
    mol1 = None
    mol2 = None
    if pose1_path.endswith(".pdbqt"):
        block = _sanitize_pdbqt_for_rdkit(pose1_path)
        mol1 = Chem.MolFromPDBBlock(block, removeHs=heavy_atoms_only)
    else:
        mol1 = Chem.MolFromPDBFile(pose1_path, removeHs=heavy_atoms_only)

    if pose2_path.endswith(".pdbqt"):
        block = _sanitize_pdbqt_for_rdkit(pose2_path)
        mol2 = Chem.MolFromPDBBlock(block, removeHs=heavy_atoms_only)
    else:
        mol2 = Chem.MolFromPDBFile(pose2_path, removeHs=heavy_atoms_only)

    if mol1 is None or mol2 is None:
        logger.warning("Failed to parse one or both poses for coordinate RMSD")
        return None

    def _get_coords_elems(mol):
        conf = mol.GetConformer()
        coords = np.array(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ]
        )
        elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        return coords, elems

    c1, e1 = _get_coords_elems(mol1)
    c2, e2 = _get_coords_elems(mol2)

    unique_elems = set(e1) & set(e2)
    matched_1 = []
    matched_2 = []

    for elem in unique_elems:
        idx1 = [i for i, e in enumerate(e1) if e == elem]
        idx2 = [i for i, e in enumerate(e2) if e == elem]
        if len(idx1) != len(idx2):
            logger.debug(f"Element count mismatch for {elem}: {len(idx1)} vs {len(idx2)}")
            continue
        sub1 = c1[idx1]
        sub2 = c2[idx2]
        cost = np.linalg.norm(sub1[:, None, :] - sub2[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_1.extend([idx1[i] for i in row_ind])
        matched_2.extend([idx2[j] for j in col_ind])

    if not matched_1:
        logger.warning("No atoms could be matched for coordinate RMSD")
        return None

    try:
        return _kabsch_rmsd(c1[matched_1], c2[matched_2])
    except Exception as exc:
        logger.warning(f"Coordinate-based RMSD failed: {exc}")
        return None


def compute_rmsd_to_crystal(
    docked_pdbqt: str,
    crystal_ligand_pdb: str,
) -> float | None:
    """
    Compute RMSD between a docked pose and crystal ligand structure.

    Tries RDKit GetBestRMS first (topology-aware), then falls back to
    coordinate-based Hungarian/Kabsch method if atom ordering differs.

    Args:
        docked_pdbqt: Docked ligand PDBQT (or PDB file if path ends with .pdb).
        crystal_ligand_pdb: Crystal ligand PDB (extracted from holo structure).

    Returns:
        RMSD in Å.
    """
    if not _HAVE_RDKIT:
        return None

    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Parse docked pose
    if docked_pdbqt.lower().endswith(".pdb"):
        docked_mol = Chem.MolFromPDBFile(docked_pdbqt, removeHs=True)
    else:
        docked_pdb_block = _sanitize_pdbqt_for_rdkit(docked_pdbqt)
        docked_mol = Chem.MolFromPDBBlock(docked_pdb_block, removeHs=True)

    crystal_mol = Chem.MolFromPDBFile(crystal_ligand_pdb, removeHs=True)

    if docked_mol is None or crystal_mol is None:
        logger.warning("Failed to parse molecules for crystal RMSD")
        return None

    # Attempt 1: topology-aware GetBestRMS
    try:
        rms = AllChem.GetBestRMS(docked_mol, crystal_mol)
        return float(rms)
    except Exception as exc:
        logger.debug(f"GetBestRMS failed: {exc} — falling back to coordinate-based RMSD")

    # Attempt 2: coordinate-based matching (handles different atom orderings)
    return compute_rmsd_coordinate_based(docked_pdbqt, crystal_ligand_pdb)


def compute_best_rmsd_from_all_poses(
    all_poses_pdbqt: str,
    crystal_ligand_pdb: str,
) -> tuple[float | None, int]:
    """
    Compute the best (lowest) RMSD among all poses in a multi-MODEL PDBQT.

    Args:
        all_poses_pdbqt: PDBQT file containing multiple MODEL poses.
        crystal_ligand_pdb: Crystal ligand PDB reference.

    Returns:
        (best_rmsd, best_pose_index) where best_pose_index is 1-based.
        Returns (None, -1) if no poses could be evaluated.
    """
    if not _HAVE_RDKIT or not os.path.isfile(all_poses_pdbqt):
        return None, -1

    import re

    with open(all_poses_pdbqt) as fh:
        content = fh.read()

    # Split on MODEL lines
    models = re.split(r"MODEL\s+\d+\n", content)
    if len(models) <= 1:
        # Single pose — compute directly
        rmsd = compute_rmsd_to_crystal(all_poses_pdbqt, crystal_ligand_pdb)
        return (rmsd, 1) if rmsd is not None else (None, -1)

    best_rmsd = float("inf")
    best_idx = -1

    for idx, model_block in enumerate(models[1:], start=1):
        model_block = model_block.split("ENDMDL")[0]
        if not model_block.strip():
            continue

        # Write temporary single-pose PDBQT
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False)
        tmp.write("MODEL\n")
        tmp.write(model_block)
        tmp.write("ENDMDL\n")
        tmp.close()

        try:
            rmsd = compute_rmsd_to_crystal(tmp.name, crystal_ligand_pdb)
            if rmsd is not None and rmsd < best_rmsd:
                best_rmsd = rmsd
                best_idx = idx
        finally:
            os.unlink(tmp.name)

    if best_idx == -1:
        return None, -1
    return float(best_rmsd), best_idx


# ─────────────────────────────────────────────────────────────────────────────
# Redocking Validation
# ─────────────────────────────────────────────────────────────────────────────


def run_redocking_validation(
    holo_pdb: str,
    ligand_resname: str | None = None,
    chain_id: str | None = None,
    ligand_smiles: str | None = None,
    exhaustiveness: int = 32,
    n_poses: int = 20,
    seed: int | None = 42,
    output_dir: str = "./redock_validation",
    box_padding: float = 5.0,
    ligand_strategy: str | None = None,
    skip_consensus: bool = False,
    minimize: bool = False,
) -> dict[str, Any]:
    """
    Validate docking protocol by redocking the co-crystallized ligand.

    Supports two extraction modes:
      * ligand_resname: Extract HETATM records matching a residue name (e.g., "LIG")
      * chain_id: Extract an entire chain (e.g., "C") — needed for covalent / multi-fragment
        ligands like 6LU7 N3 inhibitor.

    Standard workflow:
      1. Extract crystal ligand from holo structure
      2. Prepare apo receptor (remove ligand + water)
      3. Prepare extracted ligand
      4. Define box from crystal ligand geometry
      5. Dock
      6. (Optional) OpenMM energy-minimise the best pose
      7. Compute RMSD between top pose and crystal

    Args:
        holo_pdb: PDB file containing protein-ligand complex.
        ligand_resname: Residue name of the co-crystallized ligand (HETATM mode).
        chain_id: Chain ID to extract (chain mode). Use this for peptide-like or
            multi-fragment ligands (e.g., 6LU7 chain C).
        ligand_smiles: Optional SMILES to use for ligand preparation. If None,
            the SMILES is derived from the extracted ligand structure.
        exhaustiveness: Vina exhaustiveness.
        n_poses: Number of poses.
        output_dir: Working directory.
        box_padding: Extra padding (Å) around crystal ligand bounding box.
        skip_consensus: If True, skip Vinardo consensus scoring (faster for benchmarks).
        minimize: If True, run OpenMM ligand-only energy minimisation on the
            best pose before RMSD evaluation.  This can rescue scoring failures
            by improving local geometry and hydrogen placement.

    Returns:
        Dict with rmsd, success flag, energies, and file paths.
    """
    from rdkit import Chem

    from autodock.docking import dock_ligand, dock_ligand_multi_conformer
    from autodock.preparation import find_top_pockets, prepare_ligand_adaptive, prepare_receptor
    from autodock.utils import (
        ensure_dir,
        extract_chain_from_pdb,
        pdb_chain_to_smiles,
    )

    ensure_dir(output_dir)

    crystal_ligand_pdb = os.path.join(output_dir, "crystal_ligand.pdb")
    crystal_mol = None
    crystal_smiles = ligand_smiles

    # ── 1. Extract crystal ligand ──────────────────────────────────────────
    if chain_id:
        # Chain mode: extract entire chain (e.g., 6LU7 chain C)
        extract_chain_from_pdb(holo_pdb, chain_id, crystal_ligand_pdb, include_connect=True)

        # Try RDKit direct read
        crystal_mol = Chem.MolFromPDBFile(crystal_ligand_pdb, removeHs=False)
        if crystal_mol is None:
            logger.warning(
                f"RDKit could not parse chain '{chain_id}' directly; trying obabel SMILES"
            )

        # Derive SMILES if not provided
        if crystal_smiles is None:
            crystal_smiles = pdb_chain_to_smiles(holo_pdb, chain_id)
            if crystal_smiles is None:
                raise ValidationError(
                    f"Could not derive SMILES for chain '{chain_id}' from {holo_pdb}"
                )
            logger.info(f"Chain '{chain_id}' SMILES: {crystal_smiles}")

    elif ligand_resname:
        # Traditional HETATM mode
        crystal_ligand_sdf = os.path.join(output_dir, "crystal_ligand.sdf")
        crystal_mol, _ = extract_ligand_from_pdb(holo_pdb, ligand_resname, crystal_ligand_sdf)
        if crystal_mol is None:
            raise ValidationError(f"Could not extract ligand '{ligand_resname}' from {holo_pdb}")
        crystal_smiles = Chem.MolToSmiles(crystal_mol)
        # Write PDB for RMSD reference
        from rdkit.Chem import rdmolfiles

        rdmolfiles.MolToPDBFile(crystal_mol, crystal_ligand_pdb)
    else:
        raise ValidationError("Either ligand_resname or chain_id must be provided")

    # ── 2. Prepare apo receptor ────────────────────────────────────────────
    apo_pdb = os.path.join(output_dir, "apo_receptor.pdb")

    with open(holo_pdb) as fh:
        lines = fh.readlines()

    filtered = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            # Remove water
            res_name = line[17:20].strip()
            if res_name in {"HOH", "WAT", "H2O", "DOD", "TIP", "SOL"}:
                continue
            # Remove target ligand/chain
            if chain_id and line[21].strip() == chain_id:
                continue
            if ligand_resname and line.startswith("HETATM") and ligand_resname in line[17:20]:
                continue
            # Remove all other HETATM (crystallographic additives, alternate ligands,
            # detergents, etc.) to ensure a clean apo receptor.
            # Retain common physiologically relevant metal ions and cofactors.
            if line.startswith("HETATM"):
                if res_name not in {
                    # Metal ions
                    "NA", "K", "CA", "MG", "ZN", "FE", "MN", "CO", "CU", "NI",
                    "CL", "BR", "IOD", "F",
                    # Common cofactors (minimal set)
                    "HEM", "FAD", "NAD", "NAP", "SAM", "ATP", "ADP", "AMP",
                    # Sulfate, phosphate
                    "SO4", "PO4", "GOL",
                }:
                    continue
            filtered.append(line)
        elif line.startswith("CONECT"):
            # Skip CONECTs involving removed atoms (optional simplification)
            filtered.append(line)
        else:
            filtered.append(line)

    with open(apo_pdb, "w") as fh:
        fh.writelines(filtered)

    receptor_pdbqt = os.path.join(output_dir, "apo_receptor.pdbqt")
    prepare_receptor(apo_pdb, receptor_pdbqt, remove_water=False, remove_hetatms=False)

    # ── 3. Prepare ligand (adaptive multi-conformer) ───────────────────────
    if crystal_smiles is None:
        raise ValidationError("No SMILES available for ligand preparation")

    ligand_prep_result = prepare_ligand_adaptive(crystal_smiles, output_dir, name="LIG", seed=seed)
    use_multi = isinstance(ligand_prep_result, list)
    if use_multi:
        conformer_pdbqts = ligand_prep_result
        ligand_pdbqt = conformer_pdbqts[0]  # reference path for results
        logger.info(f"Adaptive prep returned {len(conformer_pdbqts)} conformer(s)")
    else:
        ligand_pdbqt = ligand_prep_result
        conformer_pdbqts = None
        logger.info("Adaptive prep returned single conformer")

    # ── 4. Define box from crystal ligand ──────────────────────────────────
    # Try ligand-centered pocket detection first
    pockets = find_top_pockets(
        apo_pdb, ligand_pdb=crystal_ligand_pdb, max_pockets=1, use_p2rank=False
    )

    # Fallback: compute bounding box directly from crystal ligand PDB
    if not pockets:
        try:
            atoms = read_pdb_atoms(crystal_ligand_pdb)
            if atoms:
                center, size = compute_bounding_box(atoms)
                # Add padding
                size = tuple(s + 2 * box_padding for s in size)
                pockets = [{"center": center, "box_size": size}]
        except Exception as exc:
            logger.warning(f"Bounding-box fallback failed: {exc}")

    if not pockets:
        raise ValidationError("Could not define binding box from crystal ligand")

    center = pockets[0]["center"]
    box_size = pockets[0]["box_size"]
    logger.info(f"Redocking box: center={center}, size={box_size}")

    # ── 5. Dock ────────────────────────────────────────────────────────────
    if use_multi:
        result = dock_ligand_multi_conformer(
            receptor_pdbqt,
            conformer_pdbqts,
            center,
            box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            seed=seed,
            output_dir=output_dir,
            compound_name="redock",
            skip_consensus=skip_consensus,
        )
    else:
        result = dock_ligand(
            receptor_pdbqt,
            ligand_pdbqt,
            center,
            box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            seed=seed,
            output_dir=output_dir,
            compound_name="redock",
            skip_consensus=skip_consensus,
        )

    # ── 6. Optional OpenMM energy minimisation ─────────────────────────────
    minimized_pose_pdbqt = result.best_pose_pdbqt
    if minimize:
        from autodock.minimization import minimize_docked_pose

        ligand_sdf_path = None
        if ligand_resname:
            ligand_sdf_path = os.path.join(output_dir, "crystal_ligand.sdf")

        min_result = minimize_docked_pose(
            receptor_pdb=apo_pdb,
            ligand_pdbqt=result.best_pose_pdbqt,
            ligand_smiles=crystal_smiles,
            ligand_sdf=ligand_sdf_path if ligand_sdf_path and os.path.isfile(ligand_sdf_path) else None,
            output_pdb=os.path.join(output_dir, "docking_best_minimized.pdb"),
            max_iterations=500,
        )
        if min_result["success"]:
            minimized_pose_pdbqt = min_result["output_pdb"]
            logger.info(
                f"Minimised best pose: {min_result['initial_energy_kJ_mol']:.1f} → "
                f"{min_result['final_energy_kJ_mol']:.1f} kJ/mol"
            )
        else:
            logger.warning(
                f"Pose minimisation failed: {min_result.get('error', 'unknown')}"
            )

    # ── 7. Compute RMSD (raw + optional minimized) ─────────────────────────
    # Always compute raw RMSD from the un-minimized best pose for transparency
    rmsd_raw = compute_rmsd_to_crystal(result.best_pose_pdbqt, crystal_ligand_pdb)
    success_raw = rmsd_raw is not None and rmsd_raw < REDocking_RMSD_THRESHOLD

    rmsd_min = None
    success_min = None
    if minimize:
        rmsd_min = compute_rmsd_to_crystal(minimized_pose_pdbqt, crystal_ligand_pdb)
        success_min = rmsd_min is not None and rmsd_min < REDocking_RMSD_THRESHOLD

    # Primary reported rmsd/success: minimized when available, else raw
    rmsd = rmsd_min if minimize else rmsd_raw
    success = success_min if minimize else success_raw

    # Also compute best-achievable RMSD across all sampled poses
    best_rmsd = None
    best_rmsd_pose_idx = None
    if result.all_poses_pdbqt and os.path.isfile(result.all_poses_pdbqt):
        best_rmsd, best_rmsd_pose_idx = compute_best_rmsd_from_all_poses(
            result.all_poses_pdbqt, crystal_ligand_pdb
        )
        if best_rmsd is not None:
            logger.info(
                f"Redocking best-achievable RMSD: {best_rmsd:.2f} Å "
                f"(pose #{best_rmsd_pose_idx})"
            )

    rmsd_str = f"{rmsd:.2f} Å" if rmsd is not None else "N/A"
    raw_str = f"{rmsd_raw:.2f} Å" if rmsd_raw is not None else "N/A"
    if minimize and rmsd_min is not None:
        logger.info(
            f"Redocking RMSD: {raw_str} (raw) → {rmsd_str} (min) — "
            f"{'PASS' if success else 'FAIL'} (threshold: {REDocking_RMSD_THRESHOLD} Å)"
        )
    else:
        logger.info(
            f"Redocking RMSD: {rmsd_str} — {'PASS' if success else 'FAIL'} "
            f"(threshold: {REDocking_RMSD_THRESHOLD} Å)"
        )

    return {
        "rmsd": rmsd,
        "rmsd_raw": rmsd_raw,
        "rmsd_min": rmsd_min,
        "success": success,
        "success_raw": success_raw,
        "success_min": success_min,
        "threshold": REDocking_RMSD_THRESHOLD,
        "best_affinity": result.best_affinity,
        "center": center,
        "box_size": box_size,
        "apo_receptor": receptor_pdbqt,
        "ligand": ligand_pdbqt,
        "best_pose": result.best_pose_pdbqt,
        "minimized_pose": minimized_pose_pdbqt if minimize else None,
        "crystal_ligand": crystal_ligand_pdb,
        "best_rmsd": best_rmsd,
        "best_rmsd_pose_idx": best_rmsd_pose_idx,
    }
