"""
autodock.rescoring — Auxiliary pose re-scoring beyond AutoDock Vina.
========================================================
Light-weight, dependency-minimal rescoring methods that use only RDKit
(and optionally PLIP/ProLIF) to re-rank or complement Vina poses.

All functions operate on multi-MODEL PDBQT files produced by Vina and
return sorted score lists in a consistent format.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from typing import Any

from autodock.core import logger


def _split_poses(all_poses_pdbqt: str) -> list[tuple[int, str, float | None]]:
    """Split a multi-MODEL PDBQT into individual pose blocks.

    Returns a list of ``(1-based_index, pose_text, vina_energy)``.
    Energy is parsed from ``REMARK VINA RESULT:`` lines when present.
    """
    with open(all_poses_pdbqt) as fh:
        content = fh.read()
    models = re.split(r"MODEL\s+\d+\n", content)
    poses: list[tuple[int, str, float | None]] = []
    for idx, block in enumerate(models[1:], start=1):
        if not block.strip():
            continue
        energy: float | None = None
        for line in block.splitlines():
            if line.startswith("REMARK VINA RESULT:"):
                try:
                    parts = line.split()
                    energy = float(parts[3])
                except (IndexError, ValueError):
                    pass
                break
        poses.append((idx, block, energy))
    return poses


def _pdbqt_to_mol(pdbqt_text: str) -> Any | None:
    """Convert a single-model PDBQT text block to an RDKit Mol.

    Returns ``None`` if parsing fails.
    """
    try:
        from rdkit import Chem
    except ImportError:
        logger.debug("RDKit not available for rescoring")
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as tf:
        tf.write(pdbqt_text)
        tmp = tf.name
    try:
        # PDBQT is close enough to PDB for RDKit's PDB parser
        mol = Chem.MolFromPDBFile(tmp, removeHs=False)
        if mol is None:
            # Fallback: try without hydrogens
            mol = Chem.MolFromPDBFile(tmp, removeHs=True)
        return mol
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def _mol_from_file(path: str) -> Any | None:
    """Load a PDBQT/PDB file into an RDKit Mol."""
    try:
        from rdkit import Chem
    except ImportError:
        return None

    try:
        mol = Chem.MolFromPDBFile(path, removeHs=False)
        if mol is None:
            mol = Chem.MolFromPDBFile(path, removeHs=True)
        return mol
    except OSError:
        return None


def shape_similarity_scores(
    all_poses_pdbqt: str,
    reference_pdbqt: str,
) -> list[tuple[int, float, float | None]]:
    """Score each pose by 3-D shape Tanimoto similarity to a reference pose.

    Args:
        all_poses_pdbqt: Multi-MODEL PDBQT from Vina.
        reference_pdbqt: Reference ligand PDBQT (e.g. crystal pose).

    Returns:
        List of ``(pose_index, shape_tanimoto, vina_energy)`` sorted by
        descending Tanimoto score.  Pose indices are 1-based to match
        Vina output numbering.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdShapeHelpers
    except ImportError as exc:
        logger.warning(f"Shape rescoring unavailable: {exc}")
        return []

    ref_mol = _mol_from_file(reference_pdbqt)
    if ref_mol is None:
        logger.warning("Shape rescoring: could not parse reference ligand")
        return []

    if ref_mol.GetNumConformers() == 0:
        logger.warning("Shape rescoring: reference ligand has no 3D coordinates")
        return []

    ref_conf_id = ref_mol.GetConformer().GetId()
    poses = _split_poses(all_poses_pdbqt)
    if not poses:
        logger.warning("Shape rescoring: no poses found in PDBQT")
        return []

    scores: list[tuple[int, float, float | None]] = []
    for idx, block, energy in poses:
        pose_mol = _pdbqt_to_mol(block)
        if pose_mol is None or pose_mol.GetNumConformers() == 0:
            logger.debug(f"Pose {idx}: could not parse for shape scoring")
            scores.append((idx, 0.0, energy))
            continue

        pose_conf_id = pose_mol.GetConformer().GetId()
        try:
            from rdkit.Chem import rdMolAlign

            # Align pose to reference using O3A (shape-based, no substructure match needed)
            # Try CrippenO3A first (works with any atom types, no MMFF params needed)
            # Fall back to O3A on heavy atoms only if CrippenO3A fails.
            aligned = False
            try:
                o3a = rdMolAlign.GetCrippenO3A(
                    pose_mol, ref_mol, prbCid=pose_conf_id, refCid=ref_conf_id
                )
                if o3a is not None:
                    o3a.Align()
                    aligned = True
            except Exception:
                pass

            if not aligned:
                # Fallback: align on heavy atoms only
                pose_ha = Chem.RemoveHs(pose_mol)
                ref_ha = Chem.RemoveHs(ref_mol)
                if pose_ha.GetNumAtoms() > 0 and ref_ha.GetNumAtoms() > 0:
                    ha_conf_id = pose_ha.GetConformer().GetId()
                    ref_ha_conf_id = ref_ha.GetConformer().GetId()
                    o3a = rdMolAlign.GetO3A(
                        pose_ha, ref_ha, prbCid=ha_conf_id, refCid=ref_ha_conf_id
                    )
                    if o3a is not None:
                        o3a.Align()
                        # Transfer alignment back to original molecule
                        conf = pose_mol.GetConformer(pose_conf_id)
                        for i in range(pose_ha.GetNumAtoms()):
                            pos = pose_ha.GetConformer(ha_conf_id).GetAtomPosition(i)
                            conf.SetAtomPosition(i, pos)
                        aligned = True

            if aligned:
                tanimoto = rdShapeHelpers.ShapeTanimotoDist(
                    pose_mol, ref_mol, pose_conf_id, ref_conf_id
                )
                similarity = 1.0 - tanimoto
            else:
                similarity = 0.0

            scores.append((idx, similarity, energy))
            logger.debug(f"Pose {idx}: shape Tanimoto={similarity:.3f}")
        except Exception as exc:
            logger.debug(f"Pose {idx}: shape scoring failed ({exc})")
            scores.append((idx, 0.0, energy))

    # Sort by descending similarity, then by energy (more negative = better)
    scores.sort(key=lambda x: (-x[1], x[2] if x[2] is not None else 0.0))
    return scores


def strain_energy_scores(
    all_poses_pdbqt: str,
) -> list[tuple[int, float, float | None]]:
    """Score each pose by internal strain energy (MMFF94, UFF fallback).

    Very high strain energy suggests an unrealistic ligand conformation.
    This is a *penalty* score — it should be combined with binding energy,
    not used as a standalone ranking criterion.

    Args:
        all_poses_pdbqt: Multi-MODEL PDBQT from Vina.

    Returns:
        List of ``(pose_index, strain_energy_kcal_mol, vina_energy)`` sorted
        by ascending strain energy (most stable first).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        logger.warning(f"Strain-energy rescoring unavailable: {exc}")
        return []

    poses = _split_poses(all_poses_pdbqt)
    if not poses:
        logger.warning("Strain rescoring: no poses found in PDBQT")
        return []

    scores: list[tuple[int, float, float | None]] = []
    for idx, block, energy in poses:
        pose_mol = _pdbqt_to_mol(block)
        if pose_mol is None:
            scores.append((idx, float("inf"), energy))
            continue

        # Add explicit Hs for force-field energy calculation
        mol_h = Chem.AddHs(pose_mol, addCoords=True)
        if mol_h.GetNumConformers() == 0:
            scores.append((idx, float("inf"), energy))
            continue

        # Try MMFF94 first, fall back to UFF
        strain: float | None = None
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
                if ff is not None:
                    strain = ff.CalcEnergy()
        except Exception:
            pass

        if strain is None:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol_h)
                if ff is not None:
                    strain = ff.CalcEnergy()
            except Exception:
                pass

        if strain is None:
            strain = float("inf")

        scores.append((idx, strain, energy))
        logger.debug(f"Pose {idx}: strain energy={strain:.1f} kcal/mol")

    # Sort by ascending strain (most stable first), then by Vina energy
    scores.sort(key=lambda x: (x[1], x[2] if x[2] is not None else 0.0))
    return scores


def combined_rescoring(
    all_poses_pdbqt: str,
    reference_pdbqt: str | None = None,
    methods: list[str] | None = None,
    receptor_pdb: str | None = None,
) -> dict[str, list[tuple[int, float, float | None]]]:
    """Run multiple auxiliary rescoring methods on a pose ensemble.

    Args:
        all_poses_pdbqt: Multi-MODEL PDBQT from Vina.
        reference_pdbqt: Reference ligand PDBQT (required for ``"shape"``).
        methods: List of method names.  Supported:
            * ``"shape"`` — 3-D shape Tanimoto vs. reference
            * ``"strain"`` — internal MMFF94/UFF strain energy
            * ``"ifp"`` — interaction-fingerprint Tanimoto (requires *receptor_pdb*)
        receptor_pdb: Receptor PDB file (required for ``"ifp"``).

    Returns:
        Dict mapping method name to sorted score list.
    """
    if methods is None:
        methods = ["shape", "strain"]

    results: dict[str, list[tuple[int, float, float | None]]] = {}
    for method in methods:
        if method == "shape":
            if reference_pdbqt is None:
                logger.warning("Shape rescoring skipped: no reference PDBQT provided")
                continue
            results["shape"] = shape_similarity_scores(all_poses_pdbqt, reference_pdbqt)
        elif method == "strain":
            results["strain"] = strain_energy_scores(all_poses_pdbqt)
        elif method == "ifp":
            if receptor_pdb is None or reference_pdbqt is None:
                logger.warning("IFP rescoring skipped: receptor_pdb and reference_pdbqt required")
                continue
            try:
                from autodock.interactions import ifp_similarity_scores

                results["ifp"] = ifp_similarity_scores(
                    receptor_pdb, all_poses_pdbqt, reference_pdbqt
                )
            except Exception as exc:
                logger.warning(f"IFP rescoring failed: {exc}")
        else:
            logger.warning(f"Unknown rescoring method: {method}")
    return results


def select_best_by_method(
    scores: list[tuple[int, float, float | None]],
    method: str = "max",
) -> tuple[int, float] | None:
    """Select the best pose from a sorted score list.

    Args:
        scores: Sorted list from any rescoring function.
        method: ``"max"`` for descending-optimal (similarity),
            ``"min"`` for ascending-optimal (energy).

    Returns:
        ``(pose_index, best_score)`` or ``None`` if empty.
    """
    if not scores:
        return None
    best = min(scores, key=lambda x: x[1]) if method == "min" else scores[0]
    return best[0], best[1]
