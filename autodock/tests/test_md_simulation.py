"""
Comprehensive tests for autodock.md_simulation.

All heavy external dependencies (OpenMM, openmmforcefields, RDKit, MDAnalysis)
are mocked so these tests run quickly without requiring those packages.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autodock.core import MDError
from autodock.md_simulation import (
    _merge_receptor_ligand_pdb,
    _pdbqt_to_pdb,
    analyze_md_trajectory,
    run_md_stability,
)

# ─────────────────────────────────────────────────────────────────────────────
# _pdbqt_to_pdb
# ─────────────────────────────────────────────────────────────────────────────


def test_pdbqt_to_pdb_success(tmp_path):
    pdbqt_path = tmp_path / "ligand.pdbqt"
    output_pdb = tmp_path / "ligand.pdb"
    pdbqt_path.write_text("dummy")

    mock_mol = MagicMock()
    mock_chem = MagicMock()
    mock_chem.MolFromPDBBlock.return_value = mock_mol

    with patch.dict(sys.modules, {"rdkit": MagicMock(Chem=mock_chem)}):
        with patch("autodock.md_simulation._sanitize_pdbqt_for_rdkit", return_value="pdb block"):
            result = _pdbqt_to_pdb(str(pdbqt_path), str(output_pdb))

    assert result == str(output_pdb)
    mock_chem.MolFromPDBBlock.assert_called_once_with("pdb block", removeHs=False)
    mock_chem.MolToPDBFile.assert_called_once_with(mock_mol, str(output_pdb))


def test_pdbqt_to_pdb_parse_failure(tmp_path):
    pdbqt_path = tmp_path / "ligand.pdbqt"
    output_pdb = tmp_path / "ligand.pdb"
    pdbqt_path.write_text("dummy")

    mock_chem = MagicMock()
    mock_chem.MolFromPDBBlock.return_value = None

    with patch.dict(sys.modules, {"rdkit": MagicMock(Chem=mock_chem)}):
        with patch("autodock.md_simulation._sanitize_pdbqt_for_rdkit", return_value="pdb block"):
            with pytest.raises(MDError, match="Could not parse PDBQT"):
                _pdbqt_to_pdb(str(pdbqt_path), str(output_pdb))


# ─────────────────────────────────────────────────────────────────────────────
# _merge_receptor_ligand_pdb
# ─────────────────────────────────────────────────────────────────────────────


def test_merge_receptor_ligand_pdb_success(tmp_path):
    receptor_pdb = tmp_path / "receptor.pdb"
    ligand_pdb = tmp_path / "ligand.pdb"
    output_pdb = tmp_path / "complex.pdb"

    receptor_pdb.write_text(
        "ATOM    1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N  \nTER\nEND\n"
    )
    ligand_pdb.write_text(
        "HETATM  1  C   LIG A   2      12.000   7.000  -7.000  1.00  0.00           C  \nTER\n"
    )

    result = _merge_receptor_ligand_pdb(str(receptor_pdb), str(ligand_pdb), str(output_pdb))
    assert result == str(output_pdb)

    content = output_pdb.read_text()
    lines = content.splitlines(keepends=True)
    assert lines[0].startswith("ATOM")
    assert "TER\n" in lines
    assert any(line.startswith("HETATM") for line in lines)
    assert lines[-1] == "END\n"
    # Receptor END should be stripped; only one END at the very end
    end_count = sum(1 for line in lines if line == "END\n")
    assert end_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# run_md_stability helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_receptor_pdb(path: Path) -> None:
    path.write_text(
        "ATOM    1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N  \nEND\n"
    )


def _setup_openmm_mocks(gaff_fail: bool = False):
    """Create comprehensive mocks for OpenMM and related deps."""
    mock_unit = MagicMock()
    mock_app = MagicMock()
    mock_openmm = MagicMock()

    mock_openmm.app = mock_app
    mock_openmm.unit = mock_unit

    # Topology with a non-standard residue so ligand detection works
    mock_residue = MagicMock()
    mock_residue.name = "LIG"
    mock_topology = MagicMock()
    mock_topology.residues.return_value = [mock_residue]

    mock_pdbfile = MagicMock()
    mock_pdbfile.topology = mock_topology
    mock_pdbfile.positions = []

    mock_modeller = MagicMock()
    mock_modeller.topology = mock_topology
    mock_modeller.positions = []

    mock_app.PDBFile.return_value = mock_pdbfile
    mock_app.Modeller.return_value = mock_modeller
    mock_app.PME = "PME"
    mock_app.CutoffNonPeriodic = "CutoffNonPeriodic"
    mock_app.HBonds = "HBonds"

    mock_system = MagicMock()
    mock_system_generator = MagicMock()
    mock_system_generator.create_system.return_value = mock_system
    mock_system_generator.forcefield = MagicMock()

    mock_generators = MagicMock()
    mock_generators.SystemGenerator.return_value = mock_system_generator

    if gaff_fail:
        mock_generators.GAFFTemplateGenerator.side_effect = RuntimeError("GAFF failed")
    else:
        mock_gaff = MagicMock()
        mock_gaff.forcefield = "gaff_xml"
        mock_generators.GAFFTemplateGenerator.return_value = mock_gaff

    mock_simulation = MagicMock()
    mock_simulation.reporters = []
    mock_app.Simulation.return_value = mock_simulation

    mock_platform = MagicMock()
    mock_openmm.Platform.getPlatformByName.return_value = mock_platform
    mock_openmm.LangevinMiddleIntegrator.return_value = MagicMock()
    mock_openmm.MonteCarloBarostat.return_value = MagicMock()

    mock_rdkit_chem = MagicMock()
    mock_rdkit_chem.MolFromPDBFile.return_value = MagicMock()

    modules = {
        "openmm": mock_openmm,
        "openmm.app": mock_app,
        "openmm.unit": mock_unit,
        "openmmforcefields": MagicMock(generators=mock_generators),
        "openmmforcefields.generators": mock_generators,
        "rdkit": MagicMock(Chem=mock_rdkit_chem),
    }

    return modules, mock_app, mock_openmm, mock_simulation, mock_system_generator


# ─────────────────────────────────────────────────────────────────────────────
# run_md_stability
# ─────────────────────────────────────────────────────────────────────────────


def test_run_md_stability_openmm_import_failure(tmp_path):
    receptor_pdb = tmp_path / "receptor.pdb"
    _make_receptor_pdb(receptor_pdb)
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    ligand_pdbqt.write_text("dummy")

    original_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name == "openmm":
            raise ImportError("No module named 'openmm'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", mock_import):
        with pytest.raises(MDError, match="OpenMM not available"):
            run_md_stability(str(receptor_pdb), str(ligand_pdbqt), output_dir=str(tmp_path / "md"))


def test_run_md_stability_implicit_solvent_success(tmp_path):
    receptor_pdb = tmp_path / "receptor.pdb"
    _make_receptor_pdb(receptor_pdb)
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    ligand_pdbqt.write_text("dummy")
    output_dir = tmp_path / "md"

    modules, mock_app, mock_openmm, mock_simulation, mock_system_generator = _setup_openmm_mocks()

    with (
        patch("autodock.md_simulation._pdbqt_to_pdb"),
        patch("autodock.md_simulation._merge_receptor_ligand_pdb"),
        patch("autodock.md_simulation.analyze_md_trajectory", return_value={"rmsd": 1.0}),
        patch.dict(sys.modules, modules),
    ):
        result = run_md_stability(
            str(receptor_pdb),
            str(ligand_pdbqt),
            output_dir=str(output_dir),
            n_steps=100,
            nvt_steps=10,
            npt_steps=0,
            solvent_model="implicit",
        )

    assert result["trajectory"] == str(output_dir / "trajectory.dcd")
    assert result["final_structure"] == str(output_dir / "final_structure.pdb")
    assert result["ligand_residues"] == ["LIG"]
    assert result["rmsd"] == 1.0

    mock_system_generator.create_system.assert_called_once()
    mock_simulation.minimizeEnergy.assert_called_once_with(maxIterations=500)
    assert mock_simulation.step.call_count >= 1
    mock_app.DCDReporter.assert_called_once()
    mock_app.StateDataReporter.assert_called_once()


def test_run_md_stability_explicit_solvent(tmp_path):
    receptor_pdb = tmp_path / "receptor.pdb"
    _make_receptor_pdb(receptor_pdb)
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    ligand_pdbqt.write_text("dummy")
    output_dir = tmp_path / "md"

    modules, mock_app, mock_openmm, mock_simulation, mock_system_generator = _setup_openmm_mocks()

    with (
        patch("autodock.md_simulation._pdbqt_to_pdb"),
        patch("autodock.md_simulation._merge_receptor_ligand_pdb"),
        patch("autodock.md_simulation.analyze_md_trajectory", return_value={}),
        patch.dict(sys.modules, modules),
    ):
        result = run_md_stability(
            str(receptor_pdb),
            str(ligand_pdbqt),
            output_dir=str(output_dir),
            n_steps=100,
            nvt_steps=10,
            npt_steps=10,
            solvent_model="explicit",
        )

    # Explicit solvent should call addSolvent and add MonteCarloBarostat
    mock_app.Modeller.return_value.addSolvent.assert_called_once()
    # reinitialize is called when adding restraints, during NPT, and when removing restraints
    assert mock_simulation.context.reinitialize.call_count >= 1
    mock_simulation.context.reinitialize.assert_called_with(preserveState=True)
    assert "trajectory" in result


def test_run_md_stability_gaff_fallback(tmp_path):
    receptor_pdb = tmp_path / "receptor.pdb"
    _make_receptor_pdb(receptor_pdb)
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    ligand_pdbqt.write_text("dummy")
    output_dir = tmp_path / "md"

    modules, mock_app, mock_openmm, mock_simulation, mock_system_generator = _setup_openmm_mocks(
        gaff_fail=True
    )

    with (
        patch("autodock.md_simulation._pdbqt_to_pdb"),
        patch("autodock.md_simulation._merge_receptor_ligand_pdb"),
        patch("autodock.md_simulation.analyze_md_trajectory", return_value={}),
        patch.dict(sys.modules, modules),
    ):
        result = run_md_stability(
            str(receptor_pdb),
            str(ligand_pdbqt),
            output_dir=str(output_dir),
            n_steps=100,
            nvt_steps=0,
            npt_steps=0,
            solvent_model="implicit",
        )

    # GAFF failure should not crash; simulation should still proceed
    mock_generators = modules["openmmforcefields"].generators
    mock_generators.GAFFTemplateGenerator.assert_called_once()
    mock_system_generator.create_system.assert_called_once()
    assert "trajectory" in result


def test_run_md_stability_minimize_false(tmp_path):
    receptor_pdb = tmp_path / "receptor.pdb"
    _make_receptor_pdb(receptor_pdb)
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    ligand_pdbqt.write_text("dummy")
    output_dir = tmp_path / "md"

    modules, mock_app, mock_openmm, mock_simulation, mock_system_generator = _setup_openmm_mocks()

    with (
        patch("autodock.md_simulation._pdbqt_to_pdb"),
        patch("autodock.md_simulation._merge_receptor_ligand_pdb"),
        patch("autodock.md_simulation.analyze_md_trajectory", return_value={}),
        patch.dict(sys.modules, modules),
    ):
        run_md_stability(
            str(receptor_pdb),
            str(ligand_pdbqt),
            output_dir=str(output_dir),
            n_steps=100,
            nvt_steps=0,
            npt_steps=0,
            minimize=False,
        )

    mock_simulation.minimizeEnergy.assert_not_called()


def test_run_md_stability_custom_platform(tmp_path):
    receptor_pdb = tmp_path / "receptor.pdb"
    _make_receptor_pdb(receptor_pdb)
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    ligand_pdbqt.write_text("dummy")
    output_dir = tmp_path / "md"

    modules, mock_app, mock_openmm, mock_simulation, mock_system_generator = _setup_openmm_mocks()

    with (
        patch("autodock.md_simulation._pdbqt_to_pdb"),
        patch("autodock.md_simulation._merge_receptor_ligand_pdb"),
        patch("autodock.md_simulation.analyze_md_trajectory", return_value={}),
        patch.dict(sys.modules, modules),
    ):
        run_md_stability(
            str(receptor_pdb),
            str(ligand_pdbqt),
            output_dir=str(output_dir),
            n_steps=100,
            nvt_steps=0,
            npt_steps=0,
            platform_name="CUDA",
        )

    mock_openmm.Platform.getPlatformByName.assert_called_once_with("CUDA")


# ─────────────────────────────────────────────────────────────────────────────
# analyze_md_trajectory
# ─────────────────────────────────────────────────────────────────────────────


def test_analyze_md_trajectory_import_failure(tmp_path):
    traj_dcd = tmp_path / "traj.dcd"
    topology_pdb = tmp_path / "top.pdb"
    traj_dcd.write_text("dummy")
    topology_pdb.write_text("dummy")

    with patch.dict(sys.modules, {"MDAnalysis": None}):
        result = analyze_md_trajectory(str(traj_dcd), str(topology_pdb), {"LIG"}, str(tmp_path))

    assert result == {}


def test_analyze_md_trajectory_success(tmp_path):
    traj_dcd = tmp_path / "traj.dcd"
    topology_pdb = tmp_path / "top.pdb"
    traj_dcd.write_text("dummy")
    topology_pdb.write_text("dummy")

    mock_protein = MagicMock()
    mock_protein.__len__ = MagicMock(return_value=100)
    mock_ca = MagicMock()
    mock_ca.__len__ = MagicMock(return_value=10)
    mock_ligand = MagicMock()
    mock_ligand.__len__ = MagicMock(return_value=5)

    def select_atoms_side_effect(selection):
        mapping = {
            "protein": mock_protein,
            "protein and name CA": mock_ca,
            "resname LIG": mock_ligand,
            "not protein and not water and not resname NA CL K CA MG ZN": mock_ligand,
        }
        return mapping.get(selection, MagicMock())

    mock_u = MagicMock()
    mock_u.select_atoms.side_effect = select_atoms_side_effect

    mock_mda = MagicMock()
    mock_mda.Universe.return_value = mock_u

    mock_align = MagicMock()
    mock_align_traj = MagicMock()
    mock_align_traj.run.return_value = None
    mock_align.AlignTraj.return_value = mock_align_traj

    mock_lig_rmsd = MagicMock()
    mock_lig_rmsd.results.rmsd = np.array([[0, 0, 1.0], [1, 1, 1.5]])
    mock_lig_rmsd.run.return_value = mock_lig_rmsd

    mock_rec_rmsd = MagicMock()
    mock_rec_rmsd.results.rmsd = np.array([[0, 0, 0.5], [1, 1, 0.6]])
    mock_rec_rmsd.run.return_value = mock_rec_rmsd

    mock_rmsf = MagicMock()
    mock_rmsf.results.rmsf = np.array([0.5, 0.6, 0.7])
    mock_rmsf.run.return_value = mock_rmsf

    mock_rms = MagicMock()

    def rmsd_side_effect(*args, **kwargs):
        if args[0] is mock_ligand:
            return mock_lig_rmsd
        return mock_rec_rmsd

    mock_rms.RMSD.side_effect = rmsd_side_effect
    mock_rms.RMSF.return_value = mock_rmsf

    mock_hbonds = MagicMock()
    mock_hbonds.results.hbonds = [[1, 2], [3, 4]]
    mock_hbonds.run.return_value = mock_hbonds

    mock_hbonds_module = MagicMock()
    mock_hbonds_module.HydrogenBondAnalysis.return_value = mock_hbonds

    mock_mda_analysis = MagicMock()
    mock_mda_analysis.align = mock_align
    mock_mda_analysis.rms = mock_rms
    mock_mda_analysis.hydrogenbonds = mock_hbonds_module

    modules = {
        "MDAnalysis": mock_mda,
        "MDAnalysis.analysis": mock_mda_analysis,
        "MDAnalysis.analysis.align": mock_align,
        "MDAnalysis.analysis.rms": mock_rms,
        "MDAnalysis.analysis.hydrogenbonds": mock_hbonds_module,
    }

    with patch.dict(sys.modules, modules):
        result = analyze_md_trajectory(str(traj_dcd), str(topology_pdb), {"LIG"}, str(tmp_path))

    assert result["ligand_rmsd_mean"] == 1.25
    assert result["ligand_rmsd_max"] == 1.5
    assert result["ligand_rmsd_std"] == 0.25
    assert result["receptor_ca_rmsd_mean"] == 0.55
    assert result["receptor_ca_rmsd_max"] == 0.6
    assert result["receptor_ca_rmsf_mean"] == 0.6
    assert result["receptor_ca_rmsf_max"] == 0.7
    assert result["n_hbonds_mean"] == 2.0
    assert result["n_hbonds_max"] == 2


def test_run_md_stability_deprecated_n_steps(tmp_path):
    """Deprecated n_steps / nvt_steps / npt_steps are converted to ns."""
    receptor_pdb = tmp_path / "receptor.pdb"
    _make_receptor_pdb(receptor_pdb)
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    ligand_pdbqt.write_text("dummy")
    output_dir = tmp_path / "md"

    modules, mock_app, mock_openmm, mock_simulation, mock_system_generator = _setup_openmm_mocks()

    with (
        patch("autodock.md_simulation._pdbqt_to_pdb"),
        patch("autodock.md_simulation._merge_receptor_ligand_pdb"),
        patch("autodock.md_simulation.analyze_md_trajectory", return_value={}),
        patch.dict(sys.modules, modules),
    ):
        result = run_md_stability(
            str(receptor_pdb),
            str(ligand_pdbqt),
            output_dir=str(output_dir),
            n_steps=1000,
            nvt_steps=500,
            npt_steps=0,
        )
    assert "trajectory" in result


def test_run_md_stability_openmmforcefields_missing(tmp_path):
    """Fallback to basic Amber FF when openmmforcefields is unavailable."""
    receptor_pdb = tmp_path / "receptor.pdb"
    _make_receptor_pdb(receptor_pdb)
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    ligand_pdbqt.write_text("dummy")
    output_dir = tmp_path / "md"

    modules, mock_app, mock_openmm, mock_simulation, mock_system_generator = _setup_openmm_mocks()
    # Remove openmmforcefields to trigger the ImportError fallback
    modules["openmmforcefields"] = None
    modules["openmmforcefields.generators"] = None

    with (
        patch("autodock.md_simulation._pdbqt_to_pdb"),
        patch("autodock.md_simulation._merge_receptor_ligand_pdb"),
        patch("autodock.md_simulation.analyze_md_trajectory", return_value={}),
        patch.dict(sys.modules, modules),
    ):
        result = run_md_stability(
            str(receptor_pdb),
            str(ligand_pdbqt),
            output_dir=str(output_dir),
            n_steps=100,
            nvt_steps=0,
            npt_steps=0,
            solvent_model="implicit",
        )
    assert "trajectory" in result
    mock_app.ForceField.assert_called_once()


def test_run_md_stability_local_minimize_radius(tmp_path):
    """Local minimization path: only minimize ligand + nearby receptor atoms."""
    receptor_pdb = tmp_path / "receptor.pdb"
    _make_receptor_pdb(receptor_pdb)
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    ligand_pdbqt.write_text("dummy")
    output_dir = tmp_path / "md"

    modules, mock_app, mock_openmm, mock_simulation, mock_system_generator = _setup_openmm_mocks()

    # Set up a ligand residue and atoms with positions
    mock_lig_res = MagicMock()
    mock_lig_res.name = "LIG"

    mock_lig_atom = MagicMock()
    mock_lig_atom.residue = mock_lig_res
    mock_lig_atom.index = 0

    mock_rec_atom = MagicMock()
    mock_rec_atom.residue = MagicMock()
    mock_rec_atom.residue.name = "ALA"
    mock_rec_atom.index = 1

    mock_topology = modules["openmm.app"].PDBFile.return_value.topology
    mock_topology.residues.return_value = [mock_lig_res]
    mock_topology.atoms.return_value = [mock_lig_atom, mock_rec_atom]

    mock_modeller = modules["openmm.app"].Modeller.return_value
    # Positions in nm: ligand at origin, receptor at 0.3 nm (3 Å)
    mock_pos_lig = MagicMock()
    mock_pos_lig.value_in_unit.return_value = [0.0, 0.0, 0.0]
    mock_pos_rec = MagicMock()
    mock_pos_rec.value_in_unit.return_value = [0.3, 0.0, 0.0]
    mock_modeller.positions = [mock_pos_lig, mock_pos_rec]

    with (
        patch("autodock.md_simulation._pdbqt_to_pdb"),
        patch("autodock.md_simulation._merge_receptor_ligand_pdb"),
        patch("autodock.md_simulation.analyze_md_trajectory", return_value={}),
        patch.dict(sys.modules, modules),
    ):
        result = run_md_stability(
            str(receptor_pdb),
            str(ligand_pdbqt),
            output_dir=str(output_dir),
            n_steps=100,
            nvt_steps=0,
            npt_steps=0,
            local_minimize_radius=5.0,
            minimize=True,
        )
    assert "trajectory" in result
    mock_simulation.minimizeEnergy.assert_called_once_with(maxIterations=500)


def test_analyze_md_trajectory_contact_map_pca_clustering(tmp_path):
    """Contact map, PCA, and clustering branches of analyze_md_trajectory."""
    traj_dcd = tmp_path / "traj.dcd"
    topology_pdb = tmp_path / "top.pdb"
    traj_dcd.write_text("dummy")
    topology_pdb.write_text("dummy")

    mock_residue = MagicMock()
    mock_residue.resname = "ALA"
    mock_residue.resid = 1

    mock_protein = MagicMock()
    mock_protein.__len__ = MagicMock(return_value=100)
    mock_protein_residues = MagicMock()
    mock_protein_residues.atoms.positions = np.array([[0.0, 0.0, 0.0]])
    mock_protein_residues.__iter__ = MagicMock(return_value=iter([mock_residue]))
    mock_protein.residues = mock_protein_residues

    mock_ca = MagicMock()
    mock_ca.__len__ = MagicMock(return_value=10)

    mock_ligand = MagicMock()
    mock_ligand.__len__ = MagicMock(return_value=5)
    mock_ligand.positions = np.array([[0.0, 0.0, 0.0]])

    def select_atoms_side_effect(selection):
        mapping = {
            "protein": mock_protein,
            "protein and name CA": mock_ca,
            "resname LIG": mock_ligand,
            "not protein and not water and not resname NA CL K CA MG ZN": mock_ligand,
        }
        return mapping.get(selection, MagicMock())

    mock_u = MagicMock()
    mock_u.select_atoms.side_effect = select_atoms_side_effect
    # 11 frames triggers both PCA (>10) and clustering (>5)
    mock_u.trajectory = [MagicMock() for _ in range(11)]

    mock_mda = MagicMock()
    mock_mda.Universe.return_value = mock_u

    mock_align = MagicMock()
    mock_align_traj = MagicMock()
    mock_align_traj.run.return_value = None
    mock_align.AlignTraj.return_value = mock_align_traj

    mock_rms = MagicMock()
    mock_rms.RMSD.return_value.run.return_value = MagicMock(
        results=MagicMock(rmsd=np.array([[0, 0, 1.0]]))
    )
    mock_rms.RMSF.return_value.run.return_value = MagicMock(results=MagicMock(rmsf=np.array([0.5])))
    mock_rms.rmsd.return_value = 0.5

    mock_hbonds = MagicMock()
    mock_hbonds.results.hbonds = []
    mock_hbonds.run.return_value = mock_hbonds

    mock_hbonds_module = MagicMock()
    mock_hbonds_module.HydrogenBondAnalysis.return_value = mock_hbonds

    mock_distances = MagicMock()
    mock_distances.distance_array.return_value = np.array([[3.0]])

    mock_mda_lib = MagicMock()
    mock_mda_lib.distances = mock_distances
    mock_mda.lib = mock_mda_lib

    mock_pca = MagicMock()
    mock_pca.results.variance = np.array([50.0, 30.0])
    mock_pca.run.return_value = mock_pca

    mock_pca_module = MagicMock()
    mock_pca_module.PCA.return_value = mock_pca

    mock_scipy_linkage = MagicMock(return_value=np.array([[0, 1, 0.5, 2]]))
    mock_scipy_fcluster = MagicMock(return_value=np.array([1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1]))
    mock_scipy_squareform = MagicMock(return_value=np.array([0.5]))

    mock_mda_analysis = MagicMock()
    mock_mda_analysis.align = mock_align
    mock_mda_analysis.rms = mock_rms
    mock_mda_analysis.hydrogenbonds = mock_hbonds_module
    mock_mda_analysis.pca = mock_pca_module

    modules = {
        "MDAnalysis": mock_mda,
        "MDAnalysis.analysis": mock_mda_analysis,
        "MDAnalysis.analysis.align": mock_align,
        "MDAnalysis.analysis.rms": mock_rms,
        "MDAnalysis.analysis.hydrogenbonds": mock_hbonds_module,
        "MDAnalysis.analysis.pca": mock_pca_module,
        "MDAnalysis.lib": mock_mda_lib,
        "MDAnalysis.lib.distances": mock_distances,
        "scipy.cluster.hierarchy": MagicMock(
            linkage=mock_scipy_linkage, fcluster=mock_scipy_fcluster
        ),
        "scipy.spatial.distance": MagicMock(squareform=mock_scipy_squareform),
    }

    with patch.dict(sys.modules, modules):
        result = analyze_md_trajectory(str(traj_dcd), str(topology_pdb), {"LIG"}, str(tmp_path))

    assert "contact_map" in result
    assert "n_contacting_residues" in result
    assert "pca_explained_variance_pc1" in result
    assert result["pca_explained_variance_pc1"] == 50.0
    assert "pca_explained_variance_pc2" in result
    assert result["pca_explained_variance_pc2"] == 30.0
    assert "n_clusters" in result
    assert "cluster_sizes" in result
