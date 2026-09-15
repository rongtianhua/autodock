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

import contextlib
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


def _pdb2pqr_protonate(pdb_path: str, output_pdb: str, ph: float = 7.4) -> bool:
    """Run PDB2PQR to protonate a PDB file with PROPKA-corrected states."""
    from autodock.core import find_conda_tool, safe_subprocess

    pdb2pqr_bin = find_conda_tool("pdb2pqr")
    if not pdb2pqr_bin:
        return False
    try:
        success, _, stderr = safe_subprocess(
            [
                pdb2pqr_bin,
                "--ff=AMBER",
                f"--with-ph={ph}",
                "--pdb-output",
                output_pdb,
                "--titration-state-method=propka",
                "--noopt",
                "--keep-chain",
                pdb_path,
                os.devnull,
            ],
            timeout=300,
        )
        return success and os.path.getsize(output_pdb) > 100
    except (OSError, ValueError, RuntimeError, TypeError):
        return False


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
    offmol: Any,
    ligand_positions: list[Any],
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
    offmol: Any,
    ligand_positions: list[Any],
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


def _coordinate_assignment_match(
    template_mol: Chem.Mol,
    docked_mol: Chem.Mol,
    max_dist: float = 4.0,
) -> list[int] | None:
    """Element-aware optimal assignment between template and docked heavy atoms.

    Returns ``match`` where ``match[docked_idx] = template_idx`` (indices into
    each mol's atom order), or ``None`` if no plausible assignment exists.

    Runs two rounds of Hungarian optimal assignment with a Kabsch alignment
    in between: the first (rough) assignment aligns the template onto the
    docked pose, the second assignment on aligned coordinates resolves
    near-symmetric ambiguities.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        logger.warning("scipy not available — coordinate-based match fallback disabled")
        return None

    n_t = template_mol.GetNumAtoms()
    n_d = docked_mol.GetNumAtoms()
    if n_t != n_d or n_t == 0:
        return None
    if not template_mol.GetNumConformers() or not docked_mol.GetNumConformers():
        return None

    elems_t = [a.GetAtomicNum() for a in template_mol.GetAtoms()]
    elems_d = [a.GetAtomicNum() for a in docked_mol.GetAtoms()]

    t_conf = template_mol.GetConformer()
    d_conf = docked_mol.GetConformer()
    t_coords = np.asarray(t_conf.GetPositions(), dtype=float)
    d_coords = np.asarray(d_conf.GetPositions(), dtype=float)

    def _assign(t_c: np.ndarray, d_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cost = np.full((n_t, n_d), 1e6)
        for i in range(n_t):
            same = [j for j in range(n_d) if elems_t[i] == elems_d[j]]
            if same:
                diff = t_c[i][None, :] - d_c[same]
                cost[i, same] = np.einsum("ij,ij->i", diff, diff)
        return linear_sum_assignment(cost)

    def _kabsch(t_sel: np.ndarray, d_sel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotation (row-vector convention) + centers sending t_sel → d_sel."""
        t_cen = t_sel.mean(axis=0)
        d_cen = d_sel.mean(axis=0)
        H = (d_sel - d_cen).T @ (t_sel - t_cen)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        return R, t_cen, d_cen

    # Template adjacency from explicit bonds; docked adjacency from a distance
    # cutoff (the docked PDBQT topology may lack bond orders, but which atoms
    # are bonded is recovered reliably from coordinates).
    template_bonds = set()
    for b in template_mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        template_bonds.add((min(a1, a2), max(a1, a2)))
    docked_adj = np.zeros((n_d, n_d), dtype=bool)
    for i in range(n_d):
        for j in range(i + 1, n_d):
            if elems_d[i] == 1 and elems_d[j] == 1:
                continue
            if np.linalg.norm(d_coords[i] - d_coords[j]) < 1.8:
                docked_adj[i, j] = docked_adj[j, i] = True

    def _score(match: list[int]) -> tuple[float, int]:
        """(max residual distance after optimal proper rotation, adjacency violations)."""
        t_sel = t_coords[np.array(match)]
        R, t_cen, d_cen = _kabsch(t_sel, d_coords)
        resid = np.linalg.norm((t_sel - t_cen) @ R + d_cen - d_coords, axis=1)
        violations = 0
        for a1, a2 in template_bonds:
            d1, d2 = match.index(a1), match.index(a2)
            if not docked_adj[d1, d2]:
                violations += 1
        return float(resid.max()), violations

    candidates: list[tuple[list[int], float, int]] = []

    def _icp(start: np.ndarray) -> None:
        """Assign → Kabsch-align → re-assign, collecting plausible matches."""
        t_work = start.copy()
        for _round in range(6):
            rows, cols = _assign(t_work, d_coords)
            if len(rows) != n_t:
                return
            dists = np.linalg.norm(t_work[rows] - d_coords[cols], axis=1)
            if dists.max() <= max_dist * 2:
                cand = [0] * n_d
                for r, c in zip(rows, cols, strict=True):
                    cand[c] = int(r)
                resid_max, viol = _score(cand)
                if resid_max <= max_dist * 2:
                    candidates.append((cand, resid_max, viol))
            if dists.max() <= max_dist:
                return
            R, t_cen, d_cen = _kabsch(t_work[rows], d_coords[cols])
            t_work = (t_work - t_cen) @ R + d_cen

    # Multi-start ICP. The raw start converges for asymmetric molecules; for
    # near-symmetric ones (phenol, catechol) a bad first pairing can reflect
    # the assignment and stall, so also seed from PCA principal-axis
    # alignments (all four proper-rotation sign combinations).
    starts = [t_coords]
    try:
        ct = t_coords.mean(axis=0)
        cd = d_coords.mean(axis=0)
        _, vt = np.linalg.eigh(np.cov((t_coords - ct).T))
        _, vd = np.linalg.eigh(np.cov((d_coords - cd).T))
        for s1, s2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            signs = np.diag([1.0, float(s1), float(s2)])
            R0 = vt @ signs @ vd.T
            starts.append((t_coords - ct) @ R0 + cd)
    except np.linalg.LinAlgError:
        pass

    for start in starts:
        _icp(start)
        if any(v == 0 and r <= max_dist for _, r, v in candidates):
            break

    if not candidates:
        return None

    # Prefer adjacency-consistent matches; break ties by residual distance.
    # A small non-zero violation count is tolerated (with a warning): the
    # whole reason this fallback exists is that the docked PDBQT topology is
    # imperfect, so one spurious/missing inferred bond must not sink an
    # otherwise geometrically consistent match. The residual gate is what
    # actually protects against wrong assignments.
    candidates.sort(key=lambda c: (c[2], c[1]))
    best, resid_max, viol = candidates[0]
    if resid_max > max_dist * 2:
        logger.warning(
            f"Coordinate assignment best residual {resid_max:.1f} Å exceeds "
            f"2× threshold — rejecting match"
        )
        return None
    if viol > 0:
        logger.warning(
            f"Coordinate assignment has {viol} bond-adjacency violation(s) — "
            "docked topology inference is imperfect; accepting the geometrically "
            "consistent match"
        )
    elif resid_max > max_dist:
        logger.warning(
            f"Coordinate assignment converged to max distance {resid_max:.1f} Å "
            f"(threshold {max_dist:.1f} Å) — accepting best match"
        )
    return best


def _match_heavy_atoms(
    template_no_h: Chem.Mol,
    docked_no_h: Chem.Mol,
) -> list[int] | None:
    """Map template heavy atoms onto docked heavy atoms (same molecule).

    Primary path: exact RDKit substructure match. Fallback: element-aware
    optimal coordinate assignment — robust when the PDBQT-derived docked
    topology disagrees with the template on bond order / aromaticity
    perception (common for conjugated systems: flavonoids, quinolines,
    indoles), where substructure matching reliably fails.

    Returns ``match`` where ``match[docked_idx] = template_idx``, or ``None``.
    """
    match = template_no_h.GetSubstructMatch(docked_no_h)
    if match:
        return list(match)

    n_t = template_no_h.GetNumAtoms()
    if n_t != docked_no_h.GetNumAtoms() or n_t == 0:
        return None

    # Ensure the template has 3D coordinates (SMILES templates do not).
    template_3d = template_no_h
    if not template_no_h.GetNumConformers():
        from rdkit.Chem import AllChem

        template_3d = Chem.Mol(template_no_h)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        try:
            status = AllChem.EmbedMolecule(template_3d, params)
        except (RuntimeError, ValueError):
            status = -1
        if status != 0:
            logger.warning("Template coordinate embedding failed — cannot fall back")
            return None
        with contextlib.suppress(RuntimeError, ValueError):
            AllChem.MMFFOptimizeMolecule(template_3d, maxIters=200)

    match = _coordinate_assignment_match(template_3d, docked_no_h)
    if match is not None:
        logger.info(
            "Substructure match failed — recovered atom mapping via "
            "element-aware coordinate assignment"
        )
    return match


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
        match = _match_heavy_atoms(template_no_h, docked_no_h)
        if match is None:
            logger.warning(
                "Atom matching failed between template SDF and docked PDBQT; "
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
    # Substructure match with coordinate-assignment fallback (the docked
    # PDBQT topology inferred by RDKit can disagree with the SMILES template
    # on bond order / aromaticity for conjugated systems).
    match = _match_heavy_atoms(template_no_h, docked_no_h)
    if match is None:
        logger.warning("Substructure match failed for SMILES-based ligand build")
        return None, []

    # match[i] is the template atom index matching the i-th heavy atom of
    # docked_no_h (GetSubstructMatch query-atom order). template_no_h /
    # docked_no_h are hydrogen-stripped mols, so their atom indices ARE the
    # heavy-atom indices — do NOT index into the parent (possibly explicit-H)
    # docked_mol here, or GetAtomPosition() raises a RangeError.
    docked_conf = docked_no_h.GetConformer()
    coords = np.zeros((offmol.n_atoms, 3))
    for i in range(len(match)):
        pos = docked_conf.GetAtomPosition(i)
        coords[match[i]] = [pos.x, pos.y, pos.z]

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
