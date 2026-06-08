"""
autodock.rescoring — Auxiliary pose re-scoring beyond AutoDock Vina.
========================================================
Provides interaction-fingerprint (IFP) re-scoring and optional
OpenMM-based simplified MM-GBSA rescoring.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Optional OpenMM / OpenFF imports ────────────────────────────────────────
try:
    from openff.toolkit import Molecule as OpenFFMolecule

    _HAVE_OPENFF = True
except ImportError:
    _HAVE_OPENFF = False

try:
    from openmm import (
        Vec3,
        VerletIntegrator,
        app,
        unit,
    )
    from openmmforcefields.generators import SystemGenerator

    _HAVE_OPENMM = True
except ImportError:
    _HAVE_OPENMM = False

try:
    from rdkit import Chem

    _HAVE_RDKIT = True
except ImportError:
    _HAVE_RDKIT = False


# ── Public API ─────────────────────────────────────────────────────────────


def combined_rescoring(
    all_poses_pdbqt: str,
    reference_pdbqt: str | None = None,
    methods: list[str] | None = None,
    receptor_pdb: str | None = None,
) -> dict[str, list[tuple[int, float, float | None]]]:
    """Run auxiliary rescoring methods on a pose ensemble.

    Args:
        all_poses_pdbqt: Multi-MODEL PDBQT from Vina.
        reference_pdbqt: Reference ligand PDBQT (required for ``"ifp"``).
        methods: List of method names.  Currently supported:
            * ``"ifp"`` — interaction-fingerprint Tanimoto (requires *receptor_pdb*)
            * ``"mmgbsa"`` — simplified OpenMM MM-GBSA (requires *receptor_pdb*)
        receptor_pdb: Receptor PDB file (required for ``"ifp"`` and ``"mmgbsa"``).

    Returns:
        Dict mapping method name to sorted score list.
    """
    if methods is None:
        methods = []

    results: dict[str, list[tuple[int, float, float | None]]] = {}
    for method in methods:
        if method == "ifp":
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
        elif method == "mmgbsa":
            logger.warning(
                "MM-GBSA rescoring should be called via _run_mmgbsa_rescoring "
                "which requires ligand_smiles; skipping in combined_rescoring"
            )
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


# ── MM-GBSA rescoring ──────────────────────────────────────────────────────


def _perturb_zero_charges(offmol) -> None:
    """Perturb all-zero partial charges so they are recognised as user-provided.

    SMIRNOFFTemplateGenerator._molecule_has_user_charges() returns *False*
    when every charge is ~0, causing a fallback to am1bcc which may be
    unavailable.  This helper adds ±1e-6 e perturbations (sum conserved)
    so the charges are accepted as user-provided.
    """
    from openff.units import unit as off_unit

    for i, atom in enumerate(offmol.atoms):
        sign = 1.0 if i % 2 == 0 else -1.0
        atom.partial_charge = off_unit.Quantity(
            atom.partial_charge.m + sign * 1e-6, off_unit.elementary_charge
        )
    # Re-normalise so total charge stays exact
    total = sum(a.partial_charge.m for a in offmol.atoms)
    offmol.atoms[0].partial_charge = off_unit.Quantity(
        offmol.atoms[0].partial_charge.m - total, off_unit.elementary_charge
    )


def _run_mmgbsa_rescoring(
    receptor_pdb: str,
    all_poses_pdbqt: str,
    ligand_smiles: str,
) -> list[tuple[int, float, float | None]] | None:
    """Simplified MM-GBSA rescoring using OpenMM + implicit solvent.

    Computes approximate binding energy for each pose::

        ΔG ≈ E(complex) – E(receptor) – E(ligand)

    Uses AMBER14 protein force field + OpenFF 2.2.0 small-molecule force
    field + OBC2 GBSA implicit solvent (``implicit/obc2.xml``).

    Args:
        receptor_pdb: Apo receptor PDB file.
        all_poses_pdbqt: Multi-MODEL PDBQT from Vina.
        ligand_smiles: SMILES string for the ligand.

    Returns:
        List of ``(pose_index, binding_energy_kcal_mol, vina_energy)``
        sorted by ascending energy (most-negative = best), or *None* on
        failure.  *pose_index* is 1-based.
    """
    if not (_HAVE_OPENMM and _HAVE_OPENFF and _HAVE_RDKIT):
        logger.warning("MM-GBSA skipped: OpenMM/OpenFF/RDKit not all available")
        return None

    try:
        from pdbfixer import PDBFixer
    except ImportError:
        logger.warning("MM-GBSA skipped: PDBFixer not available")
        return None

    # ── 1. Parse poses ────────────────────────────────────────────────────
    poses = _parse_poses(all_poses_pdbqt)
    if not poses:
        logger.warning("MM-GBSA skipped: no poses found")
        return None

    # ── 2. Prepare receptor (once) ────────────────────────────────────────
    try:
        fixer = PDBFixer(filename=receptor_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        receptor_topology = fixer.topology
        receptor_positions = fixer.positions
    except (OSError, ValueError, ImportError) as exc:
        logger.warning(f"MM-GBSA receptor preparation failed: {exc}")
        return None

    # ── 3. Build base ligand from SMILES (once) ───────────────────────────
    try:
        offmol_base = OpenFFMolecule.from_smiles(ligand_smiles, allow_undefined_stereo=True)
        offmol_base.assign_partial_charges("gasteiger")
        # Gasteiger can produce NaN for unusual chemotypes (e.g. phosphonates,
        # sulfonic acids).  Detect and fall back to formal charges.
        if any(np.isnan(a.partial_charge.m) for a in offmol_base.atoms):
            logger.warning(
                "MM-GBSA: Gasteiger charges contain NaN — falling back to formal charges"
            )
            offmol_base.assign_partial_charges("formal_charge")
            # SMIRNOFFTemplateGenerator._molecule_has_user_charges() treats
            # all-zero charges as "not user-provided" and re-runs am1bcc.
            # Perturb charges by ±1e-6 e so they are recognised as user charges.
            _perturb_zero_charges(offmol_base)
        ligand_topology = offmol_base.to_topology().to_openmm()
        ligand_n = offmol_base.n_atoms
    except (ValueError, TypeError, RuntimeError, ImportError) as exc:
        logger.warning(f"MM-GBSA ligand build failed: {exc}")
        return None

    # ── 4. Create system generator with implicit solvent ──────────────────
    try:
        system_generator = SystemGenerator(
            forcefields=["amber14-all.xml", "implicit/obc2.xml"],
            small_molecule_forcefield="openff-2.2.0",
        )
        system_generator.add_molecules([offmol_base])
    except (ValueError, TypeError, RuntimeError, ImportError) as exc:
        logger.warning(f"MM-GBSA system generator failed: {exc}")
        return None

    # ── 5. Build complex topology (once) ──────────────────────────────────
    try:
        modeller = app.Modeller(receptor_topology, receptor_positions)
        modeller.add(ligand_topology, [Vec3(0, 0, 0) * unit.angstrom] * ligand_n)
        complex_topology = modeller.topology
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.warning(f"MM-GBSA complex topology failed: {exc}")
        return None

    # ── 6. Create systems ─────────────────────────────────────────────────
    try:
        # Pass molecules explicitly so SystemGenerator re-uses the partial
        # charges we already assigned (including formal_charge fallback).
        complex_system = system_generator.create_system(complex_topology, molecules=[offmol_base])
        receptor_system = system_generator.create_system(receptor_topology)
        ligand_system = system_generator.create_system(ligand_topology, molecules=[offmol_base])
    except (ValueError, TypeError, RuntimeError, ImportError) as exc:
        logger.warning(f"MM-GBSA system creation failed: {exc}")
        return None

    # ── 7. Create simulations ─────────────────────────────────────────────
    try:
        complex_sim = app.Simulation(complex_topology, complex_system, VerletIntegrator(0.001))
        receptor_sim = app.Simulation(receptor_topology, receptor_system, VerletIntegrator(0.001))
        ligand_sim = app.Simulation(ligand_topology, ligand_system, VerletIntegrator(0.001))
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.warning(f"MM-GBSA simulation creation failed: {exc}")
        return None

    # ── 8. Compute receptor energy (once) ─────────────────────────────────
    try:
        receptor_sim.context.setPositions(receptor_positions)
        e_receptor = (
            receptor_sim.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilocalorie_per_mole)
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.warning(f"MM-GBSA receptor energy failed: {exc}")
        return None

    # ── 9. Score each pose ────────────────────────────────────────────────
    scores: list[tuple[int, float, float | None]] = []

    for pose_idx, (pose_block, vina_energy) in enumerate(poses, start=1):
        # Extract ligand coordinates from this pose
        ligand_positions = _extract_ligand_coords_from_pose(pose_block, offmol_base, ligand_smiles)
        if ligand_positions is None:
            continue

        # Merge receptor + ligand coordinates
        complex_positions = list(receptor_positions) + list(ligand_positions)

        try:
            complex_sim.context.setPositions(complex_positions)
            e_complex = (
                complex_sim.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilocalorie_per_mole)
            )

            ligand_sim.context.setPositions(ligand_positions)
            e_ligand = (
                ligand_sim.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilocalorie_per_mole)
            )

            binding_energy = e_complex - e_receptor - e_ligand
            scores.append((pose_idx, binding_energy, vina_energy))
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.debug(f"MM-GBSA pose {pose_idx} energy failed: {exc}")
            continue

    if not scores:
        logger.warning("MM-GBSA: no poses scored successfully")
        return None

    scores.sort(key=lambda x: x[1])
    logger.info(
        f"MM-GBSA: scored {len(scores)} poses, best ΔG={scores[0][1]:.2f} kcal/mol "
        f"(pose #{scores[0][0]})"
    )
    return scores


def _parse_poses(all_poses_pdbqt: str) -> list[tuple[str, float | None]]:
    """Split Vina multi-MODEL PDBQT into individual pose blocks with energies."""
    with open(all_poses_pdbqt) as fh:
        content = fh.read()
    models = re.split(r"MODEL\s+\d+\n", content)
    poses: list[tuple[str, float | None]] = []
    for block in models[1:]:
        block = block.split("ENDMDL")[0]
        energy: float | None = None
        for line in block.splitlines():
            if line.startswith("REMARK VINA RESULT:"):
                with contextlib.suppress(IndexError, ValueError):
                    energy = float(line.split()[3])
                break
        poses.append((block, energy))
    return poses


def _extract_ligand_coords_from_pose(
    pose_block: str,
    offmol_base: Any,
    ligand_smiles: str,
) -> list | None:
    """Extract 3D coordinates from a PDBQT pose block and map to offmol atoms.

    Returns a list of OpenMM Vec3 positions matching offmol_base atom order,
    or *None* if mapping fails.
    """
    from autodock.utils import _sanitize_pdbqt_block_for_rdkit

    pdb_block = _sanitize_pdbqt_block_for_rdkit(pose_block)
    docked_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
    if docked_mol is None:
        return None

    docked_no_h = Chem.RemoveHs(docked_mol)
    template_no_h = Chem.RemoveHs(Chem.MolFromSmiles(ligand_smiles))
    if template_no_h is None:
        return None

    # Query = template, target = docked  →  match[i] = docked idx for template atom i
    match = docked_no_h.GetSubstructMatch(template_no_h)
    if not match:
        logger.debug("MM-GBSA: substructure match failed for pose")
        return None

    docked_conf = docked_no_h.GetConformer()
    coords: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(offmol_base.n_atoms)]

    # Map heavy atoms
    for template_idx, docked_idx in enumerate(match):
        pos = docked_conf.GetAtomPosition(docked_idx)
        coords[template_idx] = [pos.x, pos.y, pos.z]

    # Place hydrogens near their bonded heavy atoms using OpenFF topology
    rng = np.random.default_rng(42)
    for a in offmol_base.atoms:
        if a.atomic_number == 1:
            for bond in offmol_base.bonds:
                if bond.atom1_index == a.molecule_atom_index:
                    parent = bond.atom2_index
                elif bond.atom2_index == a.molecule_atom_index:
                    parent = bond.atom1_index
                else:
                    continue
                if coords[parent] != [0.0, 0.0, 0.0]:
                    coords[a.molecule_atom_index] = [
                        coords[parent][0] + rng.normal(0, 0.3),
                        coords[parent][1] + rng.normal(0, 0.3),
                        coords[parent][2] + rng.normal(0, 0.3),
                    ]
                    break

    return [Vec3(x, y, z) * unit.angstrom for x, y, z in coords]
