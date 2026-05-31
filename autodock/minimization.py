"""
autodock.minimization — OpenMM energy minimization for docked poses.
===========================================================
Post-processes docked ligand poses with OpenMM + OpenFF force fields
to improve local geometry, relieve steric clashes, and enhance
PoseBusters chemical validity.

Requires:
    - openmm
    - openmmforcefields
    - openff-toolkit
    - rdkit

Because AmberTools is not installed, small-molecule partial charges
are assigned with the RDKit Gasteiger method (via OpenFF toolkit).

**Note on receptor handling**
-------------------------------
Full protein–ligand complex minimization requires a *continuous* receptor
PDB chain (no missing internal residues).  Many benchmark structures contain
gaps, which cause PDBFixer/OpenMM to create spurious long-range bonds and
crash minimization.  Therefore the default mode minimises the ligand *in
vacuo* while keeping heavy-atom coordinates from the docking pose.  This
still improves bond lengths, angles, and hydrogen placement, which is the
primary goal for PoseBusters post-processing.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from autodock.core import logger

# ── Optional imports ──────────────────────────────────────────────────────
try:
    from openff.toolkit import Molecule as OpenFFMolecule
    from openmmforcefields.generators import SystemGenerator

    _HAVE_OPENFF = True
except ImportError:
    _HAVE_OPENFF = False

try:
    from openmm import (
        CustomExternalForce,
        Vec3,
        VerletIntegrator,
        app,
        unit,
    )

    _HAVE_OPENMM = True
except ImportError:
    _HAVE_OPENMM = False

try:
    from rdkit import Chem

    _HAVE_RDKIT = True
except ImportError:
    _HAVE_RDKIT = False


# ── Public API ────────────────────────────────────────────────────────────


def minimize_docked_pose(
    receptor_pdb: str,
    ligand_pdbqt: str,
    output_pdb: str | None = None,
    ligand_smiles: str | None = None,
    ligand_sdf: str | None = None,
    include_receptor: bool = False,
    max_iterations: int = 500,
    restraint_k: float = 10000.0,
    force_field: str = "amber14-all.xml",
    small_molecule_forcefield: str = "openff-2.2.0",
    ph: float = 7.4,
) -> dict[str, Any]:
    """
    Energy-minimize a docked ligand pose.

    Parameters
    ----------
    receptor_pdb
        Path to apo receptor PDB file.  Only used when
        *include_receptor* is *True*.
    ligand_pdbqt
        Path to docked ligand PDBQT file (may lack explicit H).
    output_pdb
        Path for minimized ligand PDB output.  If *None*, a temporary
        file is created.
    ligand_smiles
        SMILES string for the ligand.  Used to build the full topology
        with explicit hydrogens.  Required if *ligand_sdf* is not given.
    ligand_sdf
        Path to an SDF/MOL file with the full ligand (including H).
        If provided, this takes precedence over *ligand_smiles*.
    include_receptor
        If *True*, include the receptor in the OpenMM system and
        minimise the complex.  **Warning**: this requires a continuous
        receptor chain without missing internal residues; otherwise
        minimisation will fail with infinite energy.  Default is
        *False* (ligand-only minimisation).
    max_iterations
        Maximum L-BFGS minimization steps.
    restraint_k
        Restraint force constant for ligand heavy atoms during
        ligand-only minimisation (kJ mol⁻¹ nm⁻²).
    force_field
        OpenMM XML force field for the receptor (only used when
        *include_receptor* is *True*).
    small_molecule_forcefield
        OpenFF force field for the ligand (e.g. ``openff-2.2.0``).

    Returns
    -------
    dict
        ``{"output_pdb": str, "initial_energy_kJ_mol": float,
          "final_energy_kJ_mol": float, "success": bool, "error": str|None}``
    """
    if not _HAVE_OPENFF:
        return {"success": False, "error": "OpenFF toolkit not available"}
    if not _HAVE_OPENMM:
        return {"success": False, "error": "OpenMM not available"}
    if not _HAVE_RDKIT:
        return {"success": False, "error": "RDKit not available"}

    if output_pdb is None:
        fd, output_pdb = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)

    try:
        # ── 1. Build full ligand molecule with coordinates ───────────────
        offmol, ligand_positions = _build_ligand(
            ligand_pdbqt=ligand_pdbqt,
            ligand_smiles=ligand_smiles,
            ligand_sdf=ligand_sdf,
        )
        if offmol is None:
            return {
                "success": False,
                "error": "Failed to build ligand molecule",
                "output_pdb": output_pdb,
            }

        # ── 2. Create OpenMM system ──────────────────────────────────────
        if include_receptor:
            result = _minimize_complex(
                offmol,
                ligand_positions,
                receptor_pdb,
                output_pdb,
                max_iterations,
                restraint_k,
                force_field,
                small_molecule_forcefield,
                ph,
            )
        else:
            result = _minimize_ligand_only(
                offmol,
                ligand_positions,
                output_pdb,
                max_iterations,
                restraint_k,
                small_molecule_forcefield,
            )

        return result

    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.warning(f"OpenMM minimization failed: {exc}")
        return {
            "success": False,
            "error": str(exc),
            "output_pdb": output_pdb,
        }


# ── Internal minimisation backends ────────────────────────────────────────


def _minimize_ligand_only(
    offmol,
    ligand_positions,
    output_pdb: str,
    max_iterations: int,
    restraint_k: float,
    small_molecule_forcefield: str,
) -> dict[str, Any]:
    """Minimise ligand in vacuo with heavy-atom position restraints."""
    system_generator = SystemGenerator(
        forcefields=[],
        small_molecule_forcefield=small_molecule_forcefield,
    )
    system_generator.add_molecules([offmol])

    topology = offmol.to_topology().to_openmm()
    system = system_generator.create_system(topology)

    # Restrain heavy atoms to their docked coordinates
    restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", restraint_k * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    heavy_indices = [a.molecule_atom_index for a in offmol.atoms if a.atomic_number > 1]
    for i in heavy_indices:
        restraint.addParticle(
            i,
            [ligand_positions[i][0], ligand_positions[i][1], ligand_positions[i][2]],
        )
    system.addForce(restraint)

    integrator = VerletIntegrator(0.001)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(ligand_positions)

    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    simulation.minimizeEnergy(maxIterations=max_iterations)

    state = simulation.context.getState(getEnergy=True)
    final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    minimized_positions = simulation.context.getState(getPositions=True).getPositions()

    with open(output_pdb, "w") as fh:
        app.PDBFile.writeFile(topology, minimized_positions, fh)

    logger.info(
        f"Ligand-only minimisation: E_initial={initial_energy:.1f} → "
        f"E_final={final_energy:.1f} kJ/mol"
    )

    return {
        "success": True,
        "output_pdb": output_pdb,
        "initial_energy_kJ_mol": float(initial_energy),
        "final_energy_kJ_mol": float(final_energy),
        "error": None,
    }


def _minimize_complex(
    offmol,
    ligand_positions,
    receptor_pdb: str,
    output_pdb: str,
    max_iterations: int,
    restraint_k: float,
    force_field: str,
    small_molecule_forcefield: str,
    ph: float = 7.4,
) -> dict[str, Any]:
    """Minimise ligand in complex with receptor (requires continuous chain)."""
    try:
        from pdbfixer import PDBFixer
    except (ImportError, OSError, ValueError):
        return {
            "success": False,
            "error": "PDBFixer not available for complex minimisation",
            "output_pdb": output_pdb,
        }

    fixer = PDBFixer(filename=receptor_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    receptor_n = fixer.topology.getNumAtoms()

    system_generator = SystemGenerator(
        forcefields=[force_field],
        small_molecule_forcefield=small_molecule_forcefield,
    )
    system_generator.add_molecules([offmol])

    modeller = app.Modeller(fixer.topology, fixer.positions)
    ligand_topology = offmol.to_topology().to_openmm()
    modeller.add(ligand_topology, ligand_positions)

    system = system_generator.create_system(modeller.topology)

    # Fix receptor, minimise ligand
    positions = modeller.positions
    restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", restraint_k * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    for i in range(receptor_n):
        restraint.addParticle(i, [positions[i][0], positions[i][1], positions[i][2]])
    system.addForce(restraint)

    integrator = VerletIntegrator(0.001)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    simulation.minimizeEnergy(maxIterations=max_iterations)

    state = simulation.context.getState(getEnergy=True)
    final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    ligand_start = receptor_n
    ligand_positions_out = minimized_positions[ligand_start : ligand_start + offmol.n_atoms]

    with open(output_pdb, "w") as fh:
        app.PDBFile.writeFile(ligand_topology, ligand_positions_out, fh)

    logger.info(
        f"Complex minimisation: E_initial={initial_energy:.1f} → E_final={final_energy:.1f} kJ/mol"
    )

    return {
        "success": True,
        "output_pdb": output_pdb,
        "initial_energy_kJ_mol": float(initial_energy),
        "final_energy_kJ_mol": float(final_energy),
        "error": None,
    }


# ── Ligand building helpers ───────────────────────────────────────────────


def _build_ligand(
    ligand_pdbqt: str,
    ligand_smiles: str | None,
    ligand_sdf: str | None,
) -> tuple[Any | None, list]:
    """
    Build an OpenFF Molecule with 3D coordinates from a docked PDBQT.

    If *ligand_sdf* is provided, the full molecule (with H) is read from
    there and Kabsch-aligned to the docked heavy-atom coordinates.
    Otherwise *ligand_smiles* is used to generate the topology.
    """
    from autodock.utils import _sanitize_pdbqt_for_rdkit

    docked_pdb_block = _sanitize_pdbqt_for_rdkit(ligand_pdbqt)
    docked_mol = Chem.MolFromPDBBlock(docked_pdb_block, removeHs=False)
    if docked_mol is None:
        return None, []

    docked_no_h = Chem.RemoveHs(docked_mol)
    docked_conf = docked_no_h.GetConformer()
    docked_coords = np.array(
        [
            [
                docked_conf.GetAtomPosition(i).x,
                docked_conf.GetAtomPosition(i).y,
                docked_conf.GetAtomPosition(i).z,
            ]
            for i in range(docked_no_h.GetNumAtoms())
        ]
    )

    # ── Case A: use provided SDF ──────────────────────────────────────
    if ligand_sdf and os.path.isfile(ligand_sdf):
        supplier = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
        template_mol = next(supplier)
        if template_mol is None:
            return _build_ligand_from_smiles(docked_mol, ligand_smiles)

        template_no_h = Chem.RemoveHs(template_mol)
        match = template_no_h.GetSubstructMatch(docked_no_h)
        if not match:
            logger.warning(
                "Substructure match failed between template SDF and docked PDBQT; "
                "falling back to SMILES"
            )
            return _build_ligand_from_smiles(docked_mol, ligand_smiles)

        # Kabsch alignment of template onto docked pose
        template_conf = template_no_h.GetConformer()
        template_coords = np.array(
            [
                [
                    template_conf.GetAtomPosition(i).x,
                    template_conf.GetAtomPosition(i).y,
                    template_conf.GetAtomPosition(i).z,
                ]
                for i in range(template_no_h.GetNumAtoms())
            ]
        )

        matched_template = template_coords[list(match), :]

        t_center = matched_template.mean(axis=0)
        d_center = docked_coords.mean(axis=0)
        matched_c = matched_template - t_center
        docked_c = docked_coords - d_center

        H = docked_c.T @ matched_c
        U, S, Vt = np.linalg.svd(H)
        # NOTE: H = docked_c.T @ matched_c  (not the more common Q.T @ P).
        # For this convention the optimal rotation that sends matched_c →
        # docked_c is R = Vt.T @ U.T.
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        full_coords = np.array(
            [
                [
                    template_mol.GetConformer().GetAtomPosition(i).x,
                    template_mol.GetConformer().GetAtomPosition(i).y,
                    template_mol.GetConformer().GetAtomPosition(i).z,
                ]
                for i in range(template_mol.GetNumAtoms())
            ]
        )
        aligned_coords = (full_coords - t_center) @ R + d_center

        aligned_mol = Chem.RWMol(template_mol)
        for i in range(aligned_mol.GetNumAtoms()):
            aligned_mol.GetConformer().SetAtomPosition(i, aligned_coords[i].tolist())

        fd, tmp_sdf = tempfile.mkstemp(suffix=".sdf")
        os.close(fd)
        w = Chem.SDWriter(tmp_sdf)
        w.write(aligned_mol)
        w.close()

        try:
            offmol = OpenFFMolecule.from_file(tmp_sdf, allow_undefined_stereo=True)
        finally:
            os.remove(tmp_sdf)

        offmol.assign_partial_charges("gasteiger")
        ligand_positions = [Vec3(x, y, z) for x, y, z in offmol.conformers[0].m] * unit.angstrom
        return offmol, ligand_positions

    # ── Case B: use SMILES ────────────────────────────────────────────
    return _build_ligand_from_smiles(docked_mol, ligand_smiles)


def _build_ligand_from_smiles(
    docked_mol: Chem.Mol, ligand_smiles: str | None
) -> tuple[Any | None, list]:
    """Build OpenFF molecule from SMILES, map docked coordinates."""
    if not ligand_smiles:
        return None, []

    try:
        offmol = OpenFFMolecule.from_smiles(ligand_smiles, allow_undefined_stereo=True)
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.warning(f"OpenFF Molecule.from_smiles failed: {exc}")
        return None, []

    # Map docked heavy-atom coordinates onto OpenFF molecule
    docked_no_h = Chem.RemoveHs(docked_mol)
    template_no_h = Chem.RemoveHs(Chem.MolFromSmiles(ligand_smiles))
    match = template_no_h.GetSubstructMatch(docked_no_h)
    if not match:
        logger.warning("Substructure match failed for SMILES-based ligand build")
        return None, []

    docked_conf = docked_no_h.GetConformer()
    coords = np.zeros((offmol.n_atoms, 3))

    template_heavy = [a.GetIdx() for a in template_no_h.GetAtoms() if a.GetAtomicNum() > 1]
    docked_heavy = [a.GetIdx() for a in docked_mol.GetAtoms() if a.GetAtomicNum() > 1]
    for i in range(len(match)):
        docked_idx = docked_heavy[i]
        template_idx = template_heavy[match[i]]
        pos = docked_conf.GetAtomPosition(docked_idx)
        coords[template_idx] = [pos.x, pos.y, pos.z]

    # Hydrogen initial guess: offset slightly from bonded heavy atom
    rng = np.random.default_rng(42)
    for a in offmol.atoms:
        if a.atomic_number == 1:
            for bond in offmol.bonds:
                if bond.atom1_index == a.molecule_atom_index:
                    parent = bond.atom2_index
                elif bond.atom2_index == a.molecule_atom_index:
                    parent = bond.atom1_index
                else:
                    continue
                if coords[parent].any():
                    coords[a.molecule_atom_index] = coords[parent] + rng.normal(0, 0.3, 3)
                    break

    offmol.add_conformer(coords * unit.angstrom)
    offmol.assign_partial_charges("gasteiger")

    ligand_positions = [Vec3(x, y, z) for x, y, z in offmol.conformers[0].m] * unit.angstrom
    return offmol, ligand_positions
