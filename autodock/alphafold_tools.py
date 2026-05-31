"""
autodock.alphafold_tools — AlphaFold-specific quality assessment and relaxation.
================================================================================
Implements two critical steps for using AlphaFold structures in docking:

1. **Quality assessment** — parses per-residue pLDDT from the B-factor column,
   identifies high/low confidence regions, and flags problematic domains.

2. **MD relaxation** — runs a short OpenMM implicit-solvent MD (200 ps NVT +
   1 ns production) to relieve the torsional strain present in AlphaFold
   predictions, which cannot be used directly for rigid docking.

References:
    - Jumper et al. (2021) Nature 596:583–589 (AlphaFold pLDDT interpretation)
    - Heo & Feig (2022) bioRxiv (AF relaxation best practices)
    - Eastman et al. (2017) PLoS Comput. Biol. (OpenMM)
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np

from autodock.core import MDError, logger
from autodock.utils import ensure_dir

# ─────────────────────────────────────────────────────────────────────────────
# 1. pLDDT Quality Assessment
# ─────────────────────────────────────────────────────────────────────────────


class PLDDTThresholds:
    """AlphaFold pLDDT confidence thresholds (Jumper et al. 2021)."""

    VERY_HIGH = 90.0  # comparable to experimental ~2.0 Å
    HIGH = 70.0  # acceptable for docking
    LOW = 50.0  # low confidence, exclude from pocket definition


def assess_alphafold_quality(
    structure_path: str,
    plddt_threshold_high: float = PLDDTThresholds.HIGH,
    plddt_threshold_low: float = PLDDTThresholds.LOW,
) -> dict[str, Any]:
    """
    Assess the quality of an AlphaFold-predicted structure from pLDDT scores.

    AlphaFold writes per-residue pLDDT confidence scores into the B-factor
    column (PDB format: cols 61-66; mmCIF: ``atom_site.auth_B_iso_or_equiv``).

    Args:
        structure_path: Path to AlphaFold PDB or mmCIF file.
        plddt_threshold_high: pLDDT above this is "high confidence"
            (default 90.0, as in Jumper et al. 2021).
        plddt_threshold_low: pLDDT below this is "low confidence"
            (default 50.0).

    Returns:
        Dict with keys:
        - ``mean_plddt``: mean per-residue pLDDT
        - ``median_plddt``: median per-residue pLDDT
        - ``high_conf_pct``: percentage of residues with pLDDT ≥ threshold_high
        - ``low_conf_pct``: percentage of residues with pLDDT < threshold_low
        - ``n_residues``: total number of residues assessed
        - ``low_confidence_regions``: list of (chain, start_res, end_res, min_pLDDT)
          for contiguous segments where pLDDT < threshold_low
        - ``suitable_for_docking``: bool — True if mean_pLDDT ≥ threshold_high
          and low_conf_pct < 20%
        - ``warning``: str or None
    """
    ext = os.path.splitext(structure_path)[1].lower()
    if ext in (".cif", ".pdbx"):
        plddt_values, residues = _parse_plddt_from_cif(structure_path)
    else:
        plddt_values, residues = _parse_plddt_from_pdb(structure_path)

    if not plddt_values:
        return {
            "mean_plddt": None,
            "median_plddt": None,
            "high_conf_pct": None,
            "low_conf_pct": None,
            "n_residues": 0,
            "low_confidence_regions": [],
            "suitable_for_docking": False,
            "warning": "No ATOM/HETATM records found — is this a valid structure file?",
        }

    n_total = len(plddt_values)
    mean_p = float(np.mean(plddt_values))
    median_p = float(np.median(plddt_values))
    high_conf = sum(1 for v in plddt_values if v >= plddt_threshold_high) / n_total * 100
    low_conf = sum(1 for v in plddt_values if v < plddt_threshold_low) / n_total * 100

    # Identify contiguous low-confidence regions
    low_regions: list[dict[str, Any]] = []
    if residues:
        _in_low = False
        _start_res = None
        _chain = None
        _min_p = 100.0
        for plddt, (chain, resi) in zip(plddt_values, residues, strict=False):
            if plddt < plddt_threshold_low:
                if not _in_low:
                    _in_low = True
                    _start_res = resi
                    _chain = chain
                    _min_p = plddt
                else:
                    _min_p = min(_min_p, plddt)
            else:
                if _in_low:
                    low_regions.append(
                        {
                            "chain": _chain,
                            "start": _start_res,
                            "end": resi,
                            "min_plddt": _min_p,
                        }
                    )
                    _in_low = False
        if _in_low:
            low_regions.append(
                {
                    "chain": _chain,
                    "start": _start_res,
                    "end": residues[-1][1],
                    "min_plddt": _min_p,
                }
            )

    # Determine suitability
    # Jumper 2021 & Heo & Feig 2022: pLDDT > 70 is acceptable for docking.
    suitable = mean_p >= plddt_threshold_high and low_conf < 20.0
    warning = None
    if not suitable:
        if mean_p < plddt_threshold_high:
            warning = (
                f"Low overall confidence (mean pLDDT={mean_p:.1f} < "
                f"{plddt_threshold_high:.0f}).  Consider SWISS-MODEL homology "
                f"modelling or alternative experimental structures."
            )
        elif low_conf >= 20:
            regions_str = "; ".join(
                f"{r['chain']}:{r['start']}-{r['end']} (pLDDT={r['min_plddt']:.0f})"
                for r in low_regions[:5]
            )
            warning = (
                f"{low_conf:.0f}% residues in low-confidence regions. "
                f"Exclude these from pocket definition: {regions_str}"
            )

    result = {
        "mean_plddt": mean_p,
        "median_plddt": median_p,
        "high_conf_pct": high_conf,
        "low_conf_pct": low_conf,
        "n_residues": n_total,
        "low_confidence_regions": low_regions,
        "suitable_for_docking": suitable,
        "warning": warning,
    }

    if warning:
        logger.warning(f"AlphaFold quality: {warning}")
    else:
        logger.info(
            f"AlphaFold quality: mean pLDDT={mean_p:.1f}, "
            f"high={high_conf:.0f}%, low={low_conf:.0f}% — suitable for docking"
        )

    return result


def _parse_plddt_from_pdb(pdb_path: str) -> tuple[list[float], list[tuple[str, int]]]:
    """Extract per-residue pLDDT from PDB B-factor column (cols 61-66)."""
    plddt_vals: list[float] = []
    residues: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            # PDB columns: 61-66 = B-factor (0-based: 60-66)
            bfactor_str = line[60:66].strip()
            try:
                plddt = float(bfactor_str)
            except (ValueError, IndexError):
                continue
            chain = line[21:22].strip() or "A"
            resi = int(line[22:26].strip())
            key = (chain, resi)
            # Prefer CA atom's pLDDT; fallback to the first atom of the residue.
            # This avoids O(n²) set-comprehension and correctly prioritises CA.
            atom_name = line[12:16].strip()
            if key not in seen:
                seen.add(key)
                plddt_vals.append(plddt)
                residues.append((chain, resi))
            elif atom_name == "CA":
                # Replace first-atom pLDDT with CA pLDDT
                idx = residues.index(key)
                plddt_vals[idx] = plddt

    return plddt_vals, residues


def _parse_plddt_from_cif(cif_path: str) -> tuple[list[float], list[tuple[str, int]]]:
    """Extract per-residue pLDDT from mmCIF B_iso_or_equiv column."""
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError(f"gemmi required for mmCIF parsing: {exc}")

    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()
    try:
        atom_site = block.find_mmcif_category("_atom_site")
    except (ValueError, TypeError, RuntimeError, AttributeError):
        return [], []

    col_label = atom_site.find_column("label_atom_id")
    col_auth_B = atom_site.find_column("auth_B_iso_or_equiv")
    col_auth_seq = atom_site.find_column("auth_seq_id")
    col_auth_asym = atom_site.find_column("auth_asym_id")

    if col_auth_B is None or col_label is None:
        return [], []

    plddt_vals: list[float] = []
    residues: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    for i in range(len(col_auth_B)):
        atom_name = col_label[i].strip() if i < len(col_label) else ""
        b_str = col_auth_B[i].strip() if i < len(col_auth_B) else ""
        try:
            plddt = float(b_str)
        except (ValueError, TypeError):
            continue
        chain = col_auth_asym[i].strip() if col_auth_asym and i < len(col_auth_asym) else "A"
        resi_str = col_auth_seq[i].strip() if col_auth_seq and i < len(col_auth_seq) else "0"
        try:
            resi = int(resi_str)
        except ValueError:
            continue
        key = (chain, resi)
        if atom_name == "CA" or key not in seen:
            if key in seen:
                continue
            seen.add(key)
            plddt_vals.append(plddt)
            residues.append((chain, resi))

    return plddt_vals, residues


# ─────────────────────────────────────────────────────────────────────────────
# 2. MD Relaxation
# ─────────────────────────────────────────────────────────────────────────────


def _kabsch_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """Kabsch RMSD between two N×3 coordinate arrays (in nm).

    Centres both sets, finds the optimal rotation via SVD, and returns the
    least-squares RMSD.  This is the standard structural-alignment RMSD used
    in MD analysis (not the coordinate-difference RMSD which is inflated by
    global translation/rotation).
    """
    # Centre
    mc = mobile - mobile.mean(axis=0)
    rc = reference - reference.mean(axis=0)
    # Covariance matrix
    H = mc.T @ rc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Correct reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Rotate mobile onto reference
    aligned = mc @ R.T
    return float(np.sqrt(np.mean(np.sum((aligned - rc) ** 2, axis=1))))


def relax_alphafold_structure(
    input_structure: str,
    output_pdb: str | None = None,
    output_dir: str = "./af_relaxed",
    nvt_ns: float = 0.2,
    production_ns: float = 1.0,
    temperature_k: float = 300.0,
    ph: float = 7.4,
    restraint_c_alpha: bool = True,
    restraint_k: float = 5.0,
    forcefield: str = "amber14-all.xml",
    platform_name: str | None = None,
) -> dict[str, Any]:
    """
    Relax an AlphaFold-predicted structure with short MD in implicit solvent.

    AlphaFold predictions contain local torsional strain because the network
    predicts residue geometry independently before stitching.  This function
    runs: energy minimise → 200 ps NVT @ 300K → 1 ns production @ 300K with
    weak Cα restraints (5 kcal/mol/Å²) to relieve strain while preserving the
    global fold.

    Args:
        input_structure: Path to AlphaFold PDB or mmCIF file.
        output_pdb: Output path for the relaxed PDB.  If None, written
            inside ``output_dir`` as ``relaxed.pdb``.
        output_dir: Output directory for trajectory files.
        nvt_ns: NVT equilibration length in ns (default 0.2).
        production_ns: Production length in ns (default 1.0).
        temperature_k: Simulation temperature (default 300 K).
        ph: pH for hydrogen placement with PDBFixer (default 7.4).
        restraint_c_alpha: Restrain Cα atoms during production (default True).
        restraint_k: Cα restraint force constant in kcal/mol/Å² (default 5.0).
        forcefield: OpenMM force field (default ``"amber14-all.xml"``).
        platform_name: OpenMM platform (None = auto).

    Returns:
        Dict with keys: ``output_pdb``, ``initial_rmsd``, ``final_rmsd``,
        ``rmsd_vs_time``, ``final_energy_kj_mol``, ``n_residues``, ``success``.
    """
    try:
        import openmm.app as app
        import openmm.unit as unit
        from openmm import CustomExternalForce, LangevinMiddleIntegrator
    except ImportError as exc:
        raise MDError(f"OpenMM not available: {exc}")

    ensure_dir(output_dir)

    if output_pdb is None:
        output_pdb = os.path.join(output_dir, "relaxed.pdb")

    # 1. Load structure
    ext = os.path.splitext(input_structure)[1].lower()
    if ext in (".cif", ".pdbx"):
        try:
            import gemmi
        except ImportError as exc:
            raise MDError(f"gemmi required for mmCIF: {exc}")
        doc = gemmi.cif.read(input_structure)
        block = doc.sole_block()
        structure = gemmi.make_structure_from_block(block)
        pdb_str = structure.make_pdb_string()
        import tempfile

        tmp_in = tempfile.mktemp(suffix="_af_raw.pdb")
        try:
            with open(tmp_in, "w") as f:
                f.write(pdb_str)
            pdb = app.PDBFile(tmp_in)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_in)
    else:
        pdb = app.PDBFile(input_structure)

    # 2. Use PDBFixer to fill missing atoms (common in AF structures)
    try:
        # Write to temp for PDBFixer
        import tempfile

        from pdbfixer import PDBFixer

        tmp_pdb = tempfile.mktemp(suffix="_af_fix.pdb")
        try:
            with open(tmp_pdb, "w") as f:
                app.PDBFile.writeFile(pdb.topology, pdb.positions, f)
            fixer = PDBFixer(filename=tmp_pdb)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(ph)
            topology = fixer.topology
            positions = fixer.positions
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_pdb)
    except (ImportError, OSError, ValueError, TypeError, RuntimeError):
        logger.warning("PDBFixer unavailable or failed — using raw AlphaFold positions")
        topology = pdb.topology
        positions = pdb.positions

    # 3. Build system
    system = _build_af_system(
        topology,
        forcefield=forcefield,
        restraint_c_alpha=restraint_c_alpha,
        restraint_k=restraint_k,
    )

    # 5. Minimise
    dt = 0.002 * unit.picoseconds
    integrator = LangevinMiddleIntegrator(temperature_k * unit.kelvin, 1.0 / unit.picosecond, dt)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # Set Cα restraint reference positions after context has positions
    _ca_indices: list[int] = []
    if restraint_c_alpha:
        for force_i in range(system.getNumForces()):
            force = system.getForce(force_i)
            if isinstance(force, CustomExternalForce):
                ca_idx = 0
                for atom in topology.atoms():
                    if atom.name == "CA" and atom.element.symbol == "C":
                        _ca_indices.append(atom.index)
                        p = positions[atom.index]
                        force.setParticleParameters(ca_idx, [p.x, p.y, p.z])
                        ca_idx += 1
                force.updateParametersInContext(simulation.context)
                break

    logger.info("AF relax: energy minimising (500 steps L-BFGS)...")
    simulation.minimizeEnergy(maxIterations=500)

    # Reference positions for RMSD tracking (before production)
    state0 = simulation.context.getState(getPositions=True)
    ref_positions = state0.getPositions(asNumpy=True)
    ref_pdb = os.path.join(output_dir, "reference.pdb")
    with open(ref_pdb, "w") as f:
        app.PDBFile.writeFile(topology, state0.getPositions(), f)

    # 6. NVT equilibration
    dt_ps = dt.value_in_unit(unit.picoseconds)
    nvt_steps = max(500, int(nvt_ns * 1_000_000.0 / dt_ps))
    nvt_steps = max(nvt_steps, 500)
    logger.info(f"AF relax: NVT equilibration ({nvt_ns} ns)...")
    simulation.step(nvt_steps)

    # 7. Production
    prod_steps = max(5000, int(production_ns * 1_000_000.0 / dt_ps))
    prod_steps = max(prod_steps, 5000)
    traj_interval = max(1, prod_steps // 100)  # ~100 frames

    rmsd_trace: list[float] = []
    _nan_detected = False
    logger.info(f"AF relax: production MD ({production_ns} ns)...")
    for _step_i in range(0, prod_steps, traj_interval):
        simulation.step(traj_interval)
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        # NaN / divergence guard
        if np.isnan(pos).any() or np.isinf(pos).any():
            _nan_detected = True
            logger.error(
                "AF relax: NaN/Inf coordinates detected during production MD — "
                "simulation diverged. Stopping relaxation."
            )
            break
        # Kabsch-aligned RMSD to minimised structure (Cα only)
        if _ca_indices:
            ca_pos = pos[_ca_indices, :]
            ca_ref = ref_positions[_ca_indices, :]
            rmsd_nm = _kabsch_rmsd(ca_pos, ca_ref)
        else:
            rmsd_nm = _kabsch_rmsd(pos, ref_positions)
        rmsd_A = rmsd_nm * 10.0
        rmsd_trace.append(rmsd_A)

    if _nan_detected:
        return {
            "output_pdb": output_pdb,
            "initial_rmsd": 0.0,
            "final_rmsd": 0.0,
            "rmsd_vs_time": rmsd_trace,
            "final_energy_kj_mol": 0.0,
            "n_residues": sum(1 for r in topology.residues()),
            "success": False,
            "error": "NaN/Inf coordinates during MD — simulation diverged",
        }

    # 8. Extract final frame
    final_state = simulation.context.getState(getPositions=True)
    final_positions = final_state.getPositions()
    with open(output_pdb, "w") as f:
        app.PDBFile.writeFile(topology, final_positions, f)

    # 8. Final energy
    final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    final_energy_kj = final_energy.value_in_unit(unit.kilojoules_per_mole)

    n_residues = sum(1 for r in topology.residues())
    initial_rmsd = rmsd_trace[0] if rmsd_trace else 0.0
    final_rmsd = rmsd_trace[-1] if rmsd_trace else 0.0

    logger.info(
        f"AF relax complete: initial RMSD={initial_rmsd:.2f} Å, "
        f"final RMSD={final_rmsd:.2f} Å, energy={final_energy_kj:.0f} kJ/mol"
    )

    return {
        "output_pdb": output_pdb,
        "initial_rmsd": initial_rmsd,
        "final_rmsd": final_rmsd,
        "rmsd_vs_time": rmsd_trace,
        "final_energy_kj_mol": final_energy_kj,
        "n_residues": n_residues,
        "success": True,
    }


def _build_af_system(
    topology: Any,
    forcefield: str = "amber14-all.xml",
    implicit_solvent: str = "amber14_obc2.xml",
    restraint_c_alpha: bool = True,
    restraint_k: float = 5.0,
) -> Any:
    """Build OpenMM system for AlphaFold relaxation with Cα restraints.

    Uses AMBER14 + GB-OBC2 implicit solvent (standard for AF relaxation per
    Jumper 2021 / ColabFold).  Explicit solvent was removed because the prior
    implementation did not add a water box, producing a physically incorrect
    vacuum+PME system.
    """
    import openmm.app as _app
    import openmm.unit as _u
    from openmm import CustomExternalForce

    ff = _app.ForceField(forcefield, implicit_solvent)
    system = ff.createSystem(
        topology,
        nonbondedMethod=_app.CutoffNonPeriodic,
        constraints=_app.HBonds,
    )

    if restraint_c_alpha:
        k_val = restraint_k * _u.kilocalories_per_mole / _u.angstrom**2
        rest_force = CustomExternalForce("k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
        rest_force.addGlobalParameter("k", k_val)
        rest_force.addPerParticleParameter("x0")
        rest_force.addPerParticleParameter("y0")
        rest_force.addPerParticleParameter("z0")

        for atom in topology.atoms():
            # Restrict to protein Cα (carbon); exclude calcium (element Ca)
            if atom.name == "CA" and atom.element.symbol == "C":
                rest_force.addParticle(atom.index, [0.0, 0.0, 0.0])

        system.addForce(rest_force)

    return system
