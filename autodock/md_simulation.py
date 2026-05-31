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
from typing import Any

import numpy as np

from autodock.core import MDError, logger
from autodock.utils import _sanitize_pdbqt_for_rdkit, ensure_dir


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
    with open(receptor_pdb) as f:
        rec_lines = [line for line in f if line.startswith(("ATOM  ", "HETATM", "TER   ", "END"))]
    with open(ligand_pdb) as f:
        lig_lines = [line for line in f if line.startswith(("ATOM  ", "HETATM", "TER   "))]

    rec_lines = [line for line in rec_lines if not line.startswith("END")]

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
    n_steps: int | None = None,
    production_ns: float = 10.0,
    dt_fs: float = 2.0,
    temperature_k: float = 300.0,
    friction_coeff: float = 1.0,
    pressure_bar: float = 1.0,
    nvt_steps: int | None = None,
    nvt_ns: float = 0.1,
    npt_steps: int | None = None,
    npt_ns: float = 0.1,
    minimize: bool = True,
    local_minimize_radius: float = 5.0,
    restrain_backbone: bool = True,
    restraint_k: float = 1000.0,
    save_interval: int = 5_000,
    platform_name: str | None = None,
    solvent_model: str = "implicit",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Run MD simulation on a receptor-ligand complex to assess stability.

    Args:
        receptor_pdb: Receptor PDB file.
        ligand_pdbqt: Docked ligand PDBQT file.
        output_dir: Output directory.
        n_steps: Deprecated — use production_ns instead. Number of production MD steps.
        production_ns: Production simulation length in nanoseconds (default 10 ns).
        dt_fs: Timestep in femtoseconds.
        temperature_k: Temperature in Kelvin.
        friction_coeff: Langevin friction coefficient (1/ps).
        pressure_bar: Pressure for Monte Carlo barostat (bar).
        nvt_steps: Deprecated — use nvt_ns instead. NVT equilibration steps.
        nvt_ns: NVT equilibration length in nanoseconds (default 0.1 ns).
        npt_steps: Deprecated — use npt_ns instead. NPT equilibration steps.
        npt_ns: NPT equilibration length in nanoseconds (default 0.1 ns).
        minimize: Whether to perform energy minimization.
        local_minimize_radius: If > 0, only minimize ligand + receptor residues within
            this distance (Å). Set to 0 for full-system minimization.
        restrain_backbone: Apply position restraints to protein Cα atoms during
            NVT/NPT equilibration.
        restraint_k: Restraint force constant in kJ/mol/nm².
        save_interval: Save trajectory frame every N steps.
        platform_name: OpenMM platform. None = auto-select best.
        solvent_model: "implicit" (GBn2, fast) or "explicit" (TIP3P, more accurate).
        seed: Random seed for the Langevin integrator.  If *None*, OpenMM
            uses a non-deterministic seed (platform-dependent).  Set to an
            integer for reproducible MD trajectories.

    Returns:
        Dict with trajectory path, analysis results, and RMSD values.
    """
    # Resolve deprecated step parameters to ns-based defaults
    if n_steps is not None:
        production_ns = n_steps * dt_fs / 1_000_000.0
    if nvt_steps is not None:
        nvt_ns = nvt_steps * dt_fs / 1_000_000.0
    if npt_steps is not None:
        npt_ns = npt_steps * dt_fs / 1_000_000.0

    n_steps_int = int(round(production_ns * 1_000_000.0 / dt_fs))
    nvt_steps_int = int(round(nvt_ns * 1_000_000.0 / dt_fs))
    npt_steps_int = int(round(npt_ns * 1_000_000.0 / dt_fs))
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
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "HID",
        "HIE",
        "HIP",
        "CYX",
        "ASH",
        "GLH",
        "LYN",
        "HOH",
        "WAT",
        "H2O",
        "NA",
        "CL",
        "K",
        "CA",
        "MG",
        "ZN",
        "SOL",
        "DOD",
        "TIP",
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
            except (RuntimeError, ValueError, TypeError, ImportError) as exc:
                logger.warning(f"GAFF parameterization failed: {exc}")

        system_generator = SystemGenerator(
            forcefields=forcefield_xmls,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )
    except ImportError:
        # Fallback to basic force field without small-molecule support
        logger.warning(
            "openmmforcefields not available"
            " — using basic Amber FF (ligand may not be parameterized)"
        )
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
    if seed is not None:
        integrator.setRandomNumberSeed(seed)

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
            except openmm.OpenMMException:
                continue
        else:
            logger.info("Using OpenMM platform: CPU")

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # 8. Minimization (optionally local: ligand + nearby residues only)
    if minimize:
        if local_minimize_radius > 0.0 and ligand_residues:
            logger.info(f"Local energy minimization (ligand + {local_minimize_radius} Å shell)...")
            # Collect ligand atom indices and positions
            ligand_atoms = set()
            ligand_positions = []
            for atom in modeller.topology.atoms():
                if atom.residue in ligand_residues:
                    ligand_atoms.add(atom.index)
                    ligand_positions.append(
                        modeller.positions[atom.index].value_in_unit(unit.nanometer)
                    )
            ligand_positions = np.array(ligand_positions)

            # Find receptor atoms within radius of any ligand atom
            minimize_atoms = set(ligand_atoms)
            for atom in modeller.topology.atoms():
                if atom.index in ligand_atoms:
                    continue
                pos = np.array(modeller.positions[atom.index].value_in_unit(unit.nanometer))
                distances = np.linalg.norm(ligand_positions - pos, axis=1)
                if np.any(distances < local_minimize_radius * 0.1):  # Å -> nm
                    minimize_atoms.add(atom.index)

            simulation.minimizeEnergy(maxIterations=500)
        else:
            logger.info("Energy minimization...")
            simulation.minimizeEnergy(maxIterations=500)

    # 9. NVT equilibration with optional backbone restraints
    restraint_force = None
    if restrain_backbone and nvt_steps_int > 0:
        logger.info("Adding Cα backbone restraints for equilibration...")
        restraint_force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        restraint_force.addGlobalParameter(
            "k", restraint_k * unit.kilojoules_per_mole / unit.nanometer**2
        )
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")
        positions_nm = (
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
        )
        for atom in modeller.topology.atoms():
            if atom.name == "CA" and atom.residue.name in standard_residues:
                idx = atom.index
                restraint_force.addParticle(idx, positions_nm[idx].tolist())
        system.addForce(restraint_force)
        simulation.context.reinitialize(preserveState=True)

    if nvt_steps_int > 0:
        logger.info(
            f"NVT equilibration ({nvt_steps_int} steps, {nvt_steps_int * dt_fs / 1e6:.2f} ns)..."
        )
        simulation.step(nvt_steps_int)

    # 10. NPT equilibration
    if npt_steps_int > 0 and solvent_model == "explicit":
        logger.info(
            f"NPT equilibration ({npt_steps_int} steps, {npt_steps_int * dt_fs / 1e6:.2f} ns)..."
        )
        system.addForce(
            openmm.MonteCarloBarostat(pressure_bar * unit.bar, temperature_k * unit.kelvin)
        )
        simulation.context.reinitialize(preserveState=True)
        simulation.step(npt_steps_int)

    # Remove backbone restraints before production
    if restraint_force is not None:
        logger.info("Removing backbone restraints for production...")
        system.removeForce(system.getNumForces() - 1)
        simulation.context.reinitialize(preserveState=True)

    # 11. Production run
    logger.info(f"Production MD ({n_steps_int} steps, {n_steps_int * dt_fs / 1e6:.2f} ns)...")
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
    simulation.step(n_steps_int)

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
        Dict with ligand/receptor RMSD, RMSF, COM drift, contact map,
        H-bond data, PCA, and clustering results.
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis import align, rms
    except ImportError:
        logger.warning("MDAnalysis not available — skipping trajectory analysis")
        return {}

    u = mda.Universe(topology_pdb, traj_dcd)

    protein = u.select_atoms("protein")
    ca = u.select_atoms("protein and name CA")
    ligand = (
        u.select_atoms(" or ".join(f"resname {r}" for r in ligand_resnames))
        if ligand_resnames
        else None
    )
    if ligand is None or len(ligand) == 0:
        ligand = u.select_atoms("not protein and not water and not resname NA CL K CA MG ZN")

    results: dict[str, Any] = {}

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

    # Ligand RMSF (per-atom)
    if ligand is not None and len(ligand) > 0:
        try:
            lig_rmsf = rms.RMSF(ligand).run()
            results["ligand_rmsf_mean"] = round(float(np.mean(lig_rmsf.results.rmsf)), 3)
            results["ligand_rmsf_max"] = round(float(np.max(lig_rmsf.results.rmsf)), 3)
        except Exception as exc:
            logger.warning(f"Ligand RMSF analysis failed: {exc}")

    # Ligand COM drift
    if ligand is not None and len(ligand) > 0:
        try:
            com_traj = np.array([ligand.center_of_geometry() for ts in u.trajectory])
            com_drift = np.linalg.norm(com_traj - com_traj[0], axis=1)
            results["ligand_com_drift_mean"] = round(float(np.mean(com_drift)), 3)
            results["ligand_com_drift_max"] = round(float(np.max(com_drift)), 3)
        except Exception as exc:
            logger.warning(f"Ligand COM drift analysis failed: {exc}")

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

    # Contact map: protein residues within 4.5 Å of ligand (mean over trajectory)
    if ligand is not None and len(ligand) > 0 and len(protein) > 0:
        try:
            contact_counts: dict[str, int] = {}
            for ts in u.trajectory:
                dist_matrix = mda.lib.distances.distance_array(
                    protein.residues.atoms.positions,
                    ligand.positions,
                    box=ts.dimensions,
                )
                min_dist_per_residue = np.min(dist_matrix, axis=1)
                close_mask = min_dist_per_residue < 4.5
                for i, res in enumerate(protein.residues):
                    if close_mask[i]:
                        key = f"{res.resname}{res.resid}"
                        contact_counts[key] = contact_counts.get(key, 0) + 1
            # Normalize by number of frames
            n_frames = len(u.trajectory)
            contact_freq = {
                k: round(v / n_frames, 3)
                for k, v in sorted(contact_counts.items(), key=lambda x: -x[1])
            }
            results["contact_map"] = contact_freq
            results["n_contacting_residues"] = len(contact_freq)
        except Exception as exc:
            logger.warning(f"Contact map analysis failed: {exc}")

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

    # PCA on ligand conformations (optional — requires scipy)
    if ligand is not None and len(ligand) > 0 and len(u.trajectory) > 10:
        try:
            from MDAnalysis.analysis.pca import PCA

            pca = PCA(ligand, select="all").run()
            variance = pca.results.variance
            results["pca_explained_variance_pc1"] = round(float(variance[0]), 3)
            results["pca_explained_variance_pc2"] = round(float(variance[1]), 3)
        except Exception as exc:
            logger.debug(f"PCA analysis failed: {exc}")

    # Clustering: hierarchical clustering on ligand RMSD matrix
    if ligand is not None and len(ligand) > 0 and len(u.trajectory) > 5:
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            n_frames = len(u.trajectory)
            rmsd_matrix = np.zeros((n_frames, n_frames))
            for i, _ts_i in enumerate(u.trajectory):
                ref_pos = ligand.positions.copy()
                for j, _ts_j in enumerate(u.trajectory):
                    if j >= i:
                        break
                    rmsd_matrix[i, j] = rmsd_matrix[j, i] = rms.rmsd(
                        ligand.positions, ref_pos, superposition=True
                    )
            # Convert to condensed distance matrix
            dists = squareform(rmsd_matrix)
            Z = linkage(dists, method="average")
            clusters = fcluster(Z, t=2.0, criterion="distance")
            results["n_clusters"] = int(len(set(clusters)))
            results["cluster_sizes"] = [int(np.sum(clusters == c)) for c in sorted(set(clusters))]
        except Exception as exc:
            logger.debug(f"Clustering analysis failed: {exc}")

    logger.info(f"MD analysis: {results}")
    return results
