"""
autodock.md_simulation — Short MD simulation for pose stability assessment.
============================================================================
Uses OpenMM + openmmforcefields to run short molecular dynamics simulations
on docked complexes. Provides ligand/receptor RMSD, RMSF, and H-bond
persistence analysis from the trajectory.

Requirements:
    - openmm
    - openmmforcefields
    - mdanalysis (optional, for analysis)
    - numpy
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from autodock.core import logger, MDError
from autodock.utils import ensure_dir, _sanitize_pdbqt_for_rdkit


def _pdbqt_to_pdb(pdbqt_path: str, output_pdb: str) -> str:
    """Convert PDBQT to standard PDB using RDKit (sanitizing atom types)."""
    from rdkit import Chem

    pdb_block = _sanitize_pdbqt_for_rdkit(pdbqt_path)
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
    if mol is None:
        raise MDError(f"Could not parse PDBQT: {pdbqt_path}")
    Chem.MolToPDBFile(mol, output_pdb)
    return output_pdb


def _merge_receptor_ligand_pdb(receptor_pdb: str, ligand_pdb: str, output_pdb: str) -> str:
    """Concatenate receptor and ligand PDB files into a single complex PDB."""
    with open(receptor_pdb, "r") as f:
        rec_lines = [l for l in f if l.startswith(("ATOM  ", "HETATM", "TER   ", "END"))]
    with open(ligand_pdb, "r") as f:
        lig_lines = [l for l in f if l.startswith(("ATOM  ", "HETATM", "TER   "))]

    rec_lines = [l for l in rec_lines if not l.startswith("END")]

    with open(output_pdb, "w") as f:
        f.writelines(rec_lines)
        f.write("TER\n")
        f.writelines(lig_lines)
        f.write("END\n")
    return output_pdb


def run_md_stability(
    receptor_pdb: str,
    ligand_pdbqt: str,
    output_dir: str = "./md_results",
    n_steps: int = 500_000,
    dt_fs: float = 2.0,
    temperature_k: float = 300.0,
    friction_coeff: float = 1.0,
    pressure_bar: float = 1.0,
    nvt_steps: int = 50_000,
    npt_steps: int = 50_000,
    minimize: bool = True,
    save_interval: int = 5_000,
    platform_name: str | None = None,
    solvent_model: str = "implicit",
) -> dict[str, Any]:
    """
    Run a short MD simulation on a receptor-ligand complex to assess stability.

    Args:
        receptor_pdb: Receptor PDB file.
        ligand_pdbqt: Docked ligand PDBQT file.
        output_dir: Output directory.
        n_steps: Number of production MD steps.
        dt_fs: Timestep in femtoseconds.
        temperature_k: Temperature in Kelvin.
        friction_coeff: Langevin friction coefficient (1/ps).
        pressure_bar: Pressure for Monte Carlo barostat (bar).
        nvt_steps: NVT equilibration steps.
        npt_steps: NPT equilibration steps.
        minimize: Whether to perform energy minimization.
        save_interval: Save trajectory frame every N steps.
        platform_name: OpenMM platform. None = auto-select best.
        solvent_model: "implicit" (GBn2, fast) or "explicit" (TIP3P, more accurate).

    Returns:
        Dict with trajectory path, analysis results, and RMSD values.
    """
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError as exc:
        raise MDError(f"OpenMM not available: {exc}")

    ensure_dir(output_dir)

    # 1. Prepare complex PDB
    logger.info("Preparing complex for MD simulation...")
    ligand_pdb = os.path.join(output_dir, "ligand.pdb")
    _pdbqt_to_pdb(ligand_pdbqt, ligand_pdb)

    complex_pdb = os.path.join(output_dir, "complex.pdb")
    _merge_receptor_ligand_pdb(receptor_pdb, ligand_pdb, complex_pdb)

    # 2. Load PDB
    pdb = app.PDBFile(complex_pdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # 3. Identify ligand residues
    standard_residues = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "CYX",
        "ASH", "GLH", "LYN", "HOH", "WAT", "H2O", "NA", "CL",
        "K", "CA", "MG", "ZN", "SOL", "DOD", "TIP",
    }
    ligand_residues = [r for r in modeller.topology.residues() if r.name not in standard_residues]
    if not ligand_residues:
        # Fallback: residues with few atoms
        ligand_residues = [r for r in modeller.topology.residues() if len(list(r.atoms())) < 5]

    ligand_resnames = {r.name for r in ligand_residues}
    logger.info(f"Identified ligand residues: {ligand_resnames}")

    # 4. Build force field with small-molecule support
    logger.info("Building force field...")
    try:
        from openmmforcefields.generators import SystemGenerator

        if solvent_model == "explicit":
            forcefield_xmls = ["amber/protein.ff14SB.xml", "amber/tip3p_standard.xml"]
            nonbonded_method = app.PME
        else:
            forcefield_xmls = ["amber/protein.ff14SB.xml", "implicit/gbn2.xml"]
            nonbonded_method = app.CutoffNonPeriodic

        # Try to parameterize ligand with GAFF via openmmforcefields
        _have_gaff = False
        if ligand_residues:
            try:
                from openmmforcefields.generators import GAFFTemplateGenerator
                from rdkit import Chem

                ligand_mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=False)
                if ligand_mol:
                    gaff = GAFFTemplateGenerator(molecules=ligand_mol)
                    forcefield_xmls = list(forcefield_xmls) + [gaff.forcefield]
                    _have_gaff = True
                    logger.info("Ligand parameterized with GAFF via openmmforcefields")
            except Exception as exc:
                logger.warning(f"GAFF parameterization failed: {exc}")

        system_generator = SystemGenerator(
            forcefields=forcefield_xmls,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )
    except ImportError:
        # Fallback to basic force field without small-molecule support
        logger.warning("openmmforcefields not available — using basic Amber FF (ligand may not be parameterized)")
        if solvent_model == "explicit":
            forcefield = app.ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml")
            nonbonded_method = app.PME
        else:
            forcefield = app.ForceField("amber/protein.ff14SB.xml", "implicit/gbn2.xml")
            nonbonded_method = app.CutoffNonPeriodic
        system_generator = None

    # 5. Solvate (explicit solvent only)
    if solvent_model == "explicit":
        logger.info("Adding explicit solvent and ions...")
        if system_generator:
            modeller.addSolvent(
                system_generator.forcefield,
                padding=1.0 * unit.nanometer,
                ionicStrength=0.15 * unit.molar,
            )
        else:
            modeller.addSolvent(
                forcefield,
                padding=1.0 * unit.nanometer,
                ionicStrength=0.15 * unit.molar,
            )

    # 6. Create system
    logger.info("Creating OpenMM system...")
    if system_generator:
        system = system_generator.create_system(modeller.topology)
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        friction_coeff / unit.picosecond,
        dt_fs * unit.femtoseconds,
    )

    # 7. Platform selection
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)
    else:
        platform = openmm.Platform.getPlatformByName("CPU")
        for p_name in ["Metal", "OpenCL", "CUDA"]:
            try:
                platform = openmm.Platform.getPlatformByName(p_name)
                logger.info(f"Using OpenMM platform: {p_name}")
                break
            except Exception:
                continue
        else:
            logger.info("Using OpenMM platform: CPU")

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # 8. Minimization
    if minimize:
        logger.info("Energy minimization...")
        simulation.minimizeEnergy(maxIterations=500)

    # 9. NVT equilibration
    if nvt_steps > 0:
        logger.info(f"NVT equilibration ({nvt_steps} steps, {nvt_steps * dt_fs / 1e6:.2f} ns)...")
        simulation.step(nvt_steps)

    # 10. NPT equilibration
    if npt_steps > 0 and solvent_model == "explicit":
        logger.info(f"NPT equilibration ({npt_steps} steps, {npt_steps * dt_fs / 1e6:.2f} ns)...")
        system.addForce(openmm.MonteCarloBarostat(pressure_bar * unit.bar, temperature_k * unit.kelvin))
        simulation.context.reinitialize(preserveState=True)
        simulation.step(npt_steps)

    # 11. Production run
    logger.info(f"Production MD ({n_steps} steps, {n_steps * dt_fs / 1e6:.2f} ns)...")
    traj_dcd = os.path.join(output_dir, "trajectory.dcd")
    simulation.reporters.append(app.DCDReporter(traj_dcd, save_interval))
    simulation.reporters.append(
        app.StateDataReporter(
            os.path.join(output_dir, "md_log.txt"),
            save_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            speed=True,
        )
    )
    simulation.step(n_steps)

    final_pdb = os.path.join(output_dir, "final_structure.pdb")
    with open(final_pdb, "w") as f:
        app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f,
        )

    logger.info(f"MD complete. Trajectory: {traj_dcd}")

    # 12. Analyze
    analysis = analyze_md_trajectory(traj_dcd, complex_pdb, ligand_resnames, output_dir)

    return {
        "trajectory": traj_dcd,
        "final_structure": final_pdb,
        "output_dir": output_dir,
        "ligand_residues": list(ligand_resnames),
        **analysis,
    }


def analyze_md_trajectory(
    traj_dcd: str,
    topology_pdb: str,
    ligand_resnames: set[str],
    output_dir: str,
) -> dict[str, Any]:
    """
    Analyze MD trajectory for ligand stability metrics.

    Returns:
        Dict with ligand RMSD, receptor RMSD, RMSF, and H-bond data.
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis import rms, align
    except ImportError:
        logger.warning("MDAnalysis not available — skipping trajectory analysis")
        return {}

    u = mda.Universe(topology_pdb, traj_dcd)

    protein = u.select_atoms("protein")
    ca = u.select_atoms("protein and name CA")
    ligand = u.select_atoms(" or ".join(f"resname {r}" for r in ligand_resnames)) if ligand_resnames else None
    if ligand is None or len(ligand) == 0:
        ligand = u.select_atoms("not protein and not water and not resname NA CL K CA MG ZN")

    results = {}

    # Align trajectory on protein Cα
    if len(ca) > 0:
        try:
            align.AlignTraj(u, u, select="protein and name CA", in_memory=True).run()
        except Exception as exc:
            logger.debug(f"Trajectory alignment failed: {exc}")

    # Ligand RMSD
    if ligand is not None and len(ligand) > 0:
        try:
            lig_rmsd = rms.RMSD(ligand, ligand, ref_frame=0).run()
            results["ligand_rmsd_mean"] = round(float(np.mean(lig_rmsd.results.rmsd[:, 2])), 3)
            results["ligand_rmsd_max"] = round(float(np.max(lig_rmsd.results.rmsd[:, 2])), 3)
            results["ligand_rmsd_std"] = round(float(np.std(lig_rmsd.results.rmsd[:, 2])), 3)
        except Exception as exc:
            logger.warning(f"Ligand RMSD analysis failed: {exc}")

    # Receptor Cα RMSD
    if len(ca) > 0:
        try:
            rec_rmsd = rms.RMSD(ca, ca, ref_frame=0).run()
            results["receptor_ca_rmsd_mean"] = round(float(np.mean(rec_rmsd.results.rmsd[:, 2])), 3)
            results["receptor_ca_rmsd_max"] = round(float(np.max(rec_rmsd.results.rmsd[:, 2])), 3)
        except Exception as exc:
            logger.warning(f"Receptor RMSD analysis failed: {exc}")

    # Receptor Cα RMSF
    if len(ca) > 0:
        try:
            rmsf = rms.RMSF(ca).run()
            results["receptor_ca_rmsf_mean"] = round(float(np.mean(rmsf.results.rmsf)), 3)
            results["receptor_ca_rmsf_max"] = round(float(np.max(rmsf.results.rmsf)), 3)
        except Exception as exc:
            logger.warning(f"Receptor RMSF analysis failed: {exc}")

    # H-bond analysis
    if ligand is not None and len(ligand) > 0 and len(protein) > 0:
        try:
            from MDAnalysis.analysis import hydrogenbonds
            hbonds = hydrogenbonds.HydrogenBondAnalysis(
                universe=u,
                donors_sel="protein",
                hydrogens_sel="protein",
                acceptors_sel=f"({' or '.join(f'resname {r}' for r in ligand_resnames)})",
                d_a_cutoff=3.5,
                d_h_a_angle_cutoff=150,
            )
            hbonds.run()
            n_per_frame = [len(f) for f in hbonds.results.hbonds]
            results["n_hbonds_mean"] = round(float(np.mean(n_per_frame)), 2)
            results["n_hbonds_max"] = int(np.max(n_per_frame))
        except Exception as exc:
            logger.warning(f"H-bond analysis failed: {exc}")

    logger.info(f"MD analysis: {results}")
    return results
