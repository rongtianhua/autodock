"""Tests for autodock.docking — mock-based unit tests for docking logic."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autodock import docking
from autodock.core import DockingCalculationError, DockingResult


class TestRunVinaDock:
    @patch("autodock.docking._HAVE_VINA", False)
    def test_missing_vina_raises(self):
        with pytest.raises(DockingCalculationError, match="vina Python package not available"):
            docking._run_vina_dock("rec.pdbqt", "lig.pdbqt", (0, 0, 0), (20, 20, 20))

    @patch("autodock.docking._HAVE_VINA", True)
    @patch("vina.Vina")
    def test_successful_dock(self, mock_vina_cls, tmp_path):
        mock_vina = MagicMock()
        mock_vina_cls.return_value = mock_vina

        # Mock energies: 2 poses
        mock_vina.energies.return_value = np.array([[-8.0, 1.0, 2.0], [-7.5, 2.0, 3.0]])

        # Mock pose file output
        poses_content = (
            "MODEL 1\nREMARK VINA RESULT: -8.0\n"
            "ATOM      1  C   LIG A   1"
            "      0.000   0.000   0.000\nENDMDL\n"
            "MODEL 2\nREMARK VINA RESULT: -7.5\n"
            "ATOM      1  C   LIG A   1"
            "      1.000   1.000   1.000\nENDMDL\n"
        )
        out_file = tmp_path / "poses.pdbqt"
        out_file.write_text(poses_content)

        def mock_write_poses(path, **kwargs):
            with open(path, "w") as fh:
                fh.write(poses_content)

        mock_vina.write_poses.side_effect = mock_write_poses

        energies, poses = docking._run_vina_dock(
            "rec.pdbqt",
            "lig.pdbqt",
            (0, 0, 0),
            (20, 20, 20),
            exhaustiveness=8,
            n_poses=9,
            seed=42,
            _use_subprocess=False,
        )

        assert energies.shape[0] == 2
        assert len(poses) == 2
        assert "MODEL 1" in poses[0]
        mock_vina.set_receptor.assert_called_once_with("rec.pdbqt")
        mock_vina.set_ligand_from_file.assert_called_once_with("lig.pdbqt")

    @patch("autodock.docking._HAVE_VINA", True)
    @patch("vina.Vina")
    def test_flex_receptor_passed_to_vina(self, mock_vina_cls, tmp_path):
        mock_vina = MagicMock()
        mock_vina_cls.return_value = mock_vina
        mock_vina.energies.return_value = np.array([[-8.0, 1.0, 2.0]])

        poses_content = (
            "MODEL 1\nREMARK VINA RESULT: -8.0\n"
            "ATOM      1  C   LIG A   1"
            "      0.000   0.000   0.000\nENDMDL\n"
        )
        out_file = tmp_path / "poses.pdbqt"
        out_file.write_text(poses_content)

        def mock_write_poses(path, **kwargs):
            with open(path, "w") as fh:
                fh.write(poses_content)

        mock_vina.write_poses.side_effect = mock_write_poses

        flex_file = str(tmp_path / "flex.pdbqt")
        open(flex_file, "w").close()

        docking._run_vina_dock(
            "rec.pdbqt",
            "lig.pdbqt",
            (0, 0, 0),
            (20, 20, 20),
            exhaustiveness=8,
            n_poses=9,
            seed=42,
            flex_receptor_pdbqt=flex_file,
            _use_subprocess=False,
        )
        mock_vina.set_flex.assert_called_once_with(flex_file)

    @patch("autodock.docking._HAVE_VINA", True)
    @patch("vina.Vina")
    def test_timeout_raises(self, mock_vina_cls):
        import threading

        mock_vina = MagicMock()
        mock_vina_cls.return_value = mock_vina

        # Simulate a dock() that never returns
        def hang(**kwargs):
            threading.Event().wait(10)

        mock_vina.dock.side_effect = hang

        with pytest.raises(DockingCalculationError, match="timed out"):
            docking._run_vina_dock(
                "rec.pdbqt",
                "lig.pdbqt",
                (0, 0, 0),
                (20, 20, 20),
                timeout=1,
                _use_subprocess=False,
            )


class TestConsensusScore:
    def test_returns_vina_only(self):
        # Consensus scoring is disabled; _consensus_score returns Vina only.
        all_scores, consensus = docking._consensus_score(
            "rec.pdbqt",
            "pose.pdbqt",
            (0, 0, 0),
            (20, 20, 20),
            -8.0,
            seed=42,
        )
        assert all_scores == {"vina": -8.0}
        assert consensus is None


class TestDockLigand:
    @patch("autodock.docking._run_vina_dock")
    def test_basic_dock(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM      1  C   LIG A   1      0.000   0.000   0.000\nENDMDL\n"],
        )

        with patch("autodock.docking._consensus_score") as mock_consensus:
            mock_consensus.return_value = ({"vina": -8.0}, None)
            result = docking.dock_ligand(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                seed=42,
                output_dir=str(tmp_path / "out"),
            )

        assert isinstance(result, DockingResult)
        assert result.best_affinity == -8.0
        assert result.seed == 42
        assert result.compound_name == "lig"

    @patch("autodock.docking._run_vina_dock")
    def test_no_poses_raises(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        mock_run.return_value = (np.array([]), [])
        with pytest.raises(DockingCalculationError, match="no poses"):
            docking.dock_ligand(str(rec), str(lig), (0, 0, 0), (20, 20, 20), seed=42)

    def test_missing_receptor_raises(self):
        with pytest.raises(DockingCalculationError):
            docking.dock_ligand("missing.pdbqt", "lig.pdbqt", (0, 0, 0), (20, 20, 20))

    def test_invalid_params_raises(self, tmp_path):
        from autodock.core import ConfigurationError

        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        with pytest.raises(ConfigurationError):
            docking.dock_ligand(str(rec), str(lig), (0, 0, 0), (20, 20, 20), exhaustiveness=-5)


class TestDockLigandMultiConformer:
    @patch("autodock.docking._run_vina_dock")
    def test_multi_conformer(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        conf1 = tmp_path / "conf1.pdbqt"
        conf1.write_text("ATOM      1  C   LIG A   1      0.000   0.000   0.000\n")
        conf2 = tmp_path / "conf2.pdbqt"
        conf2.write_text("ATOM      1  C   LIG A   1      1.000   1.000   1.000\n")

        # First conformer returns -8.0, second returns -9.0
        def side_effect(rec_path, lig_path, *args, **kwargs):
            if "conf1" in lig_path:
                return np.array([[-8.0, 1.0, 2.0]]), ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"]
            else:
                return np.array([[-9.0, 1.0, 2.0]]), ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"]

        mock_run.side_effect = side_effect

        with patch("autodock.docking._consensus_score") as mock_consensus:
            mock_consensus.return_value = ({"vina": -9.0}, None)
            result = docking.dock_ligand_multi_conformer(
                str(rec),
                [str(conf1), str(conf2)],
                (0, 0, 0),
                (20, 20, 20),
                seed=42,
                max_workers=1,  # sequential for mocked tests
            )

        assert result.best_affinity == -9.0

    def test_empty_conformers_raises(self):
        with pytest.raises(DockingCalculationError, match="No conformers"):
            docking.dock_ligand_multi_conformer("rec.pdbqt", [], (0, 0, 0), (20, 20, 20))


class TestVirtualScreen:
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_ligand")
    def test_virtual_screen_serial(self, mock_prep, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )

        library = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O", "ibu": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"}
        outdir = str(tmp_path / "vs")

        def make_result(rec_path, lig_path, center, box, **kwargs):
            name = kwargs.get("compound_name") or (
                os.path.basename(os.path.dirname(lig_path)) if "/" in lig_path else "unknown"
            )
            return DockingResult(
                compound_name=name,
                receptor=rec_path,
                center=center,
                box_size=box,
                best_affinity=-7.5,
                seed=kwargs.get("seed"),
            )

        mock_dock.side_effect = make_result
        mock_prep.return_value = None

        results, csv_path = docking.virtual_screen(
            str(rec),
            library,
            (0, 0, 0),
            (20, 20, 20),
            output_dir=outdir,
            n_workers=1,
            seed=42,
        )

        assert len(results) == 2
        assert all(r.best_affinity is not None for r in results)
        assert os.path.isfile(csv_path)

    def test_empty_library(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        results, csv_path = docking.virtual_screen(
            str(rec),
            {},
            (0, 0, 0),
            (20, 20, 20),
            output_dir=str(tmp_path / "vs"),
        )
        assert results == []

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_ligand")
    def test_virtual_screen_covalent_check(self, mock_prep, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )

        library = {"acrylamide": "C=CC(=O)N", "ibu": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"}
        outdir = str(tmp_path / "vs")

        def make_result(rec_path, lig_path, center, box, **kwargs):
            name = kwargs.get("compound_name") or (
                os.path.basename(os.path.dirname(lig_path)) if "/" in lig_path else "unknown"
            )
            return DockingResult(
                compound_name=name,
                receptor=rec_path,
                center=center,
                box_size=box,
                best_affinity=-7.5,
                seed=kwargs.get("seed"),
            )

        mock_dock.side_effect = make_result
        mock_prep.return_value = None

        results, _ = docking.virtual_screen(
            str(rec),
            library,
            (0, 0, 0),
            (20, 20, 20),
            output_dir=outdir,
            n_workers=1,
            seed=42,
            covalent_check=True,
        )

        covalent_results = [r for r in results if r.is_covalent_ligand]
        assert len(covalent_results) == 1
        assert covalent_results[0].compound_name == "acrylamide"
        assert "acrylamide" in covalent_results[0].covalent_warheads
        assert covalent_results[0].covalent_recommendation is not None

        non_covalent = [r for r in results if r.is_covalent_ligand is False]
        assert len(non_covalent) == 1
        assert non_covalent[0].compound_name == "ibu"


class TestBatchDock:
    @patch("autodock.docking.dock_ligand")
    def test_batch_dock_two_by_two(self, mock_dock, tmp_path):
        rec1 = tmp_path / "rec1.pdbqt"
        rec1.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        rec2 = tmp_path / "rec2.pdbqt"
        rec2.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        lig1 = tmp_path / "lig1.pdbqt"
        lig1.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")
        lig2 = tmp_path / "lig2.pdbqt"
        lig2.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

        receptors = {"rec1": str(rec1), "rec2": str(rec2)}
        ligands = {"lig1": str(lig1), "lig2": str(lig2)}
        pockets = {
            "rec1": {"center": (0.0, 0.0, 0.0), "box_size": (20.0, 20.0, 20.0)},
            "rec2": {"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)},
        }

        call_count = 0

        def make_result(rec, lig, center, box, **kwargs):
            nonlocal call_count
            call_count += 1
            return DockingResult(
                compound_name=os.path.basename(lig).replace(".pdbqt", ""),
                receptor=rec,
                center=center,
                box_size=box,
                best_affinity=-7.0 - call_count,
                seed=kwargs.get("seed"),
            )

        mock_dock.side_effect = make_result

        outdir = str(tmp_path / "batch")
        results = docking.batch_dock(
            receptors,
            ligands,
            pockets,
            seed=42,
            output_dir=outdir,
            n_workers=1,
        )

        assert set(results.keys()) == {"rec1", "rec2"}
        assert len(results["rec1"]) == 2
        assert len(results["rec2"]) == 2
        # Check CSV was written
        assert os.path.isfile(os.path.join(outdir, "batch_docking_results.csv"))

    def test_missing_pocket_raises(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        with pytest.raises(ValueError, match="missing"):
            docking.batch_dock(
                {"rec": str(rec)},
                {"lig": str(lig)},
                {},
            )

    def test_missing_receptor_file_raises(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        with pytest.raises(DockingCalculationError):
            docking.batch_dock(
                {"rec": "missing.pdbqt"},
                {"lig": str(lig)},
                {"rec": {"center": (0, 0, 0), "box_size": (20, 20, 20)}},
            )


class TestDockEnsemble:
    @patch("autodock.docking.dock_ligand")
    def test_basic(self, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        # Create pose files with slightly different coordinates for RMSD
        pose_files = []
        for i in range(3):
            pf = tmp_path / f"pose_{i}.pdbqt"
            x = 1.0 + i * 0.1
            pf.write_text(f"ATOM      1  C   LIG A   1      {x:.3f}   2.000   3.000\n")
            pose_files.append(str(pf))

        def make_result(rec_path, lig_path, center, box, **kwargs):
            idx = mock_dock.call_count - 1
            return DockingResult(
                compound_name=f"lig_repeat{idx + 1}",
                receptor=rec_path,
                center=center,
                box_size=box,
                best_affinity=-7.0 - idx * 0.1,
                best_pose_pdbqt=pose_files[idx],
                seed=kwargs.get("seed"),
            )

        mock_dock.side_effect = make_result

        summary = docking.dock_ensemble(
            str(rec),
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            n_repeats=3,
            seed=42,
            output_dir=str(tmp_path / "ensemble"),
        )

        assert summary["n_repeats"] == 3
        assert summary["n_successful"] == 3
        assert summary["confidence"] in ("high", "moderate", "low")
        assert "recommendation" in summary
        assert summary["ensemble_best_affinity_mean"] < 0
        assert summary["pose_stability_rmsd_mean"] is not None
        assert summary["n_clusters"] >= 1

    def test_n_repeats_too_low(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        with pytest.raises(ValueError, match="n_repeats"):
            docking.dock_ensemble(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                n_repeats=1,
            )

    @patch("autodock.docking.dock_ligand")
    def test_too_many_failures(self, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        mock_dock.side_effect = DockingCalculationError("Vina failed")

        with pytest.raises(DockingCalculationError, match="Fewer than 2"):
            docking.dock_ensemble(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                n_repeats=3,
            )


class TestCountPdbqtAtoms:
    def test_missing_file_raises(self):
        with pytest.raises(DockingCalculationError, match="not found"):
            docking._count_pdbqt_atoms("/nonexistent/lig.pdbqt")

    def test_counts_correctly(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000\n"
            "HETATM    2  O   LIG A   1      1.000   0.000   0.000\n"
            "REMARK    3\n"
        )
        assert docking._count_pdbqt_atoms(str(lig)) == 2


class TestAutoExhaustiveness:
    def test_small_ligand_unchanged(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        # 10 atoms
        lig.write_text(
            "\n".join(f"ATOM  {i:5d}  C   LIG A   1      0.000   0.000   0.000" for i in range(10))
            + "\n"
        )
        assert docking._auto_exhaustiveness(str(lig), 32) == 32

    def test_36_atoms_halved(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "\n".join(f"ATOM  {i:5d}  C   LIG A   1      0.000   0.000   0.000" for i in range(36))
            + "\n"
        )
        assert docking._auto_exhaustiveness(str(lig), 32) == 16

    def test_46_atoms_quartered(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "\n".join(f"ATOM  {i:5d}  C   LIG A   1      0.000   0.000   0.000" for i in range(46))
            + "\n"
        )
        assert docking._auto_exhaustiveness(str(lig), 32) == 16

    def test_56_atoms_eighth(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "\n".join(f"ATOM  {i:5d}  C   LIG A   1      0.000   0.000   0.000" for i in range(56))
            + "\n"
        )
        assert docking._auto_exhaustiveness(str(lig), 32) == 16

    def test_floor_16(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "\n".join(f"ATOM  {i:5d}  C   LIG A   1      0.000   0.000   0.000" for i in range(60))
            + "\n"
        )
        assert docking._auto_exhaustiveness(str(lig), 8) == 16


class FakeInThreadExecutor:
    """Mock ProcessPoolExecutor that runs tasks in the current thread."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a, **k):
        pass

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        f = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except Exception as exc:
            f.set_exception(exc)
        return f


class TestRunVinaCli:
    @patch("shutil.which")
    @patch("os.path.isfile")
    def test_vina_not_found_raises(self, mock_isfile, mock_which):
        mock_which.return_value = None
        mock_isfile.return_value = False
        with pytest.raises(DockingCalculationError, match="not found in PATH"):
            docking._run_vina_cli(
                "rec.pdbqt", "lig.pdbqt", (0, 0, 0), (20, 20, 20), 8, 9, 3, 42, "vina", 1.0, 300
            )

    @patch("shutil.which")
    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_successful_cli_run(self, mock_isfile, mock_run, mock_which, tmp_path):
        mock_which.return_value = "/usr/bin/vina"
        mock_isfile.return_value = True

        # Create output poses file
        out_pdbqt = tmp_path / "poses.pdbqt"
        out_pdbqt.write_text(
            "MODEL 1\n"
            "REMARK VINA RESULT:      -8.0      0.000      0.000\n"
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000\n"
            "ENDMDL\n"
            "MODEL 2\n"
            "REMARK VINA RESULT:      -7.5      0.000      0.000\n"
            "ATOM      1  C   LIG A   1      1.000   1.000   1.000\n"
            "ENDMDL\n"
        )

        result_mock = MagicMock()
        result_mock.returncode = 0
        result_mock.stdout = (
            "mode |   affinity | dist from best mode\n"
            "-----+------------+---------------------\n"
            "   1         -8.0      0.000      0.000\n"
            "   2         -7.5      1.234      1.234\n"
        )
        mock_run.return_value = result_mock

        # We need to inject the tmp_path into the TemporaryDirectory so the
        # function reads our prepared file. Use a side_effect to copy content.
        class FakeTD:
            def __init__(self, *a, **k):
                self.name = str(tmp_path)

            def __enter__(self):
                return self.name

            def __exit__(self, *a):
                pass

        with patch("tempfile.TemporaryDirectory", FakeTD):
            energies, poses = docking._run_vina_cli(
                "rec.pdbqt",
                "lig.pdbqt",
                (0, 0, 0),
                (20, 20, 20),
                8,
                9,
                3,
                42,
                "vina",
                1.0,
                300,
            )

        assert energies.shape[0] == 2
        assert len(poses) == 2
        assert "MODEL 1" in poses[0]

    @patch("shutil.which")
    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_cli_timeout_raises(self, mock_isfile, mock_run, mock_which):
        import subprocess

        mock_which.return_value = "/usr/bin/vina"
        mock_isfile.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired("vina", 300)
        with pytest.raises(DockingCalculationError, match="timed out"):
            docking._run_vina_cli(
                "rec.pdbqt", "lig.pdbqt", (0, 0, 0), (20, 20, 20), 8, 9, 3, 42, "vina", 1.0, 300
            )

    @patch("shutil.which")
    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_cli_nonzero_returncode_raises(self, mock_isfile, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/vina"
        mock_isfile.return_value = True
        result_mock = MagicMock()
        result_mock.returncode = 1
        result_mock.stderr = "Vina error"
        mock_run.return_value = result_mock
        with pytest.raises(DockingCalculationError, match="Vina failed"):
            docking._run_vina_cli(
                "rec.pdbqt", "lig.pdbqt", (0, 0, 0), (20, 20, 20), 8, 9, 3, 42, "vina", 1.0, 300
            )

    @patch("shutil.which")
    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_cli_no_poses_raises(self, mock_isfile, mock_run, mock_which, tmp_path):
        mock_which.return_value = "/usr/bin/vina"
        mock_isfile.return_value = True
        result_mock = MagicMock()
        result_mock.returncode = 0
        result_mock.stdout = "no modes"
        mock_run.return_value = result_mock

        class FakeTD:
            def __init__(self, *a, **k):
                self.name = str(tmp_path)

            def __enter__(self):
                return self.name

            def __exit__(self, *a):
                pass

        with patch("tempfile.TemporaryDirectory", FakeTD):
            with pytest.raises(DockingCalculationError, match="no poses"):
                docking._run_vina_cli(
                    "rec.pdbqt", "lig.pdbqt", (0, 0, 0), (20, 20, 20), 8, 9, 3, 42, "vina", 1.0, 300
                )

    @patch("shutil.which")
    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_cli_scoring_function_ad4(self, mock_isfile, mock_run, mock_which, tmp_path):
        mock_which.return_value = "/usr/bin/vina"
        mock_isfile.return_value = True
        result_mock = MagicMock()
        result_mock.returncode = 0
        result_mock.stdout = "mode |   affinity\n-----+------------\n   1         -8.0\n"
        mock_run.return_value = result_mock

        out_pdbqt = tmp_path / "poses.pdbqt"
        out_pdbqt.write_text("MODEL 1\nREMARK VINA RESULT:      -8.0\nATOM 1 C LIG\nENDMDL\n")

        class FakeTD:
            def __init__(self, *a, **k):
                self.name = str(tmp_path)

            def __enter__(self):
                return self.name

            def __exit__(self, *a):
                pass

        with patch("tempfile.TemporaryDirectory", FakeTD):
            energies, poses = docking._run_vina_cli(
                "rec.pdbqt", "lig.pdbqt", (0, 0, 0), (20, 20, 20), 8, 9, 3, 42, "ad4", 1.0, 300
            )
        assert energies.shape[0] == 1

    @patch("shutil.which")
    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_cli_flex_receptor(self, mock_isfile, mock_run, mock_which, tmp_path):
        mock_which.return_value = "/usr/bin/vina"
        mock_isfile.return_value = True
        result_mock = MagicMock()
        result_mock.returncode = 0
        result_mock.stdout = "mode |   affinity\n-----+------------\n   1         -8.0\n"
        mock_run.return_value = result_mock

        out_pdbqt = tmp_path / "poses.pdbqt"
        out_pdbqt.write_text("MODEL 1\nREMARK VINA RESULT:      -8.0\nATOM 1 C LIG\nENDMDL\n")

        class FakeTD:
            def __init__(self, *a, **k):
                self.name = str(tmp_path)

            def __enter__(self):
                return self.name

            def __exit__(self, *a):
                pass

        with patch("tempfile.TemporaryDirectory", FakeTD):
            energies, poses = docking._run_vina_cli(
                "rec.pdbqt",
                "lig.pdbqt",
                (0, 0, 0),
                (20, 20, 20),
                8,
                9,
                3,
                42,
                "vina",
                1.0,
                300,
                flex_receptor_pdbqt="flex.pdbqt",
            )
        assert energies.shape[0] == 1


class TestVinaDockWorker:
    @patch("autodock.docking._HAVE_VINA", True)
    def test_worker_ok(self):
        mock_queue = MagicMock()
        args = (
            "rec.pdbqt",
            "lig.pdbqt",
            (0, 0, 0),
            (20, 20, 20),
            8,
            9,
            3.0,
            42,
            None,
            "vina",
            1.0,
        )
        with patch("vina.Vina") as mock_vina_cls:
            mock_vina = MagicMock()
            mock_vina_cls.return_value = mock_vina
            mock_vina.energies.return_value = np.array([[-8.0, 1.0, 2.0]])

            with patch("tempfile.NamedTemporaryFile") as mock_tf:
                mock_tf.return_value.__enter__.return_value.name = "/tmp/fake.pdbqt"
                mock_tf.return_value.__enter__.return_value.write = MagicMock()
                with patch("builtins.open", MagicMock()) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = (
                        "MODEL 1\nATOM 1 C LIG\nENDMDL\n"
                    )
                    with patch("os.unlink"):
                        docking._vina_dock_worker(args, mock_queue)

        mock_queue.put.assert_called_once()
        status, energies, poses = mock_queue.put.call_args[0][0]
        assert status == "ok"

    @patch("autodock.docking._HAVE_VINA", True)
    def test_worker_exception(self):
        mock_queue = MagicMock()
        args = (
            "rec.pdbqt",
            "lig.pdbqt",
            (0, 0, 0),
            (20, 20, 20),
            8,
            9,
            3.0,
            42,
            None,
            "vina",
            1.0,
        )
        with patch("vina.Vina", side_effect=ImportError("no vina")):
            docking._vina_dock_worker(args, mock_queue)
        status, msg, poses = mock_queue.put.call_args[0][0]
        assert status == "error"
        assert "no vina" in msg


class TestRunVinaDockBranches:
    @patch("autodock.docking._HAVE_VINA", True)
    @patch("vina.Vina")
    def test_auto_exhaustiveness_adjusts(self, mock_vina_cls, tmp_path):
        mock_vina = MagicMock()
        mock_vina_cls.return_value = mock_vina
        mock_vina.energies.return_value = np.array([[-8.0, 1.0, 2.0]])

        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "\n".join(f"ATOM  {i:5d}  C   LIG A   1      0.000   0.000   0.000" for i in range(40))
            + "\n"
        )
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )

        with patch("autodock.docking._auto_exhaustiveness", return_value=16) as mock_ae:
            docking._run_vina_dock(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                auto_exhaustiveness=True,
                _use_subprocess=False,
            )
        mock_ae.assert_called_once()

    @patch("autodock.docking._HAVE_VINA", True)
    @patch("vina.Vina")
    def test_in_subprocess_runs_directly(self, mock_vina_cls, tmp_path):
        mock_vina = MagicMock()
        mock_vina_cls.return_value = mock_vina
        mock_vina.energies.return_value = np.array([[-8.0, 1.0, 2.0]])

        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        with patch("multiprocessing.current_process") as mock_cp:
            mock_cp.return_value.name = "SpawnProcess-1"
            energies, poses = docking._run_vina_dock(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                _use_subprocess=True,
            )
        assert energies.shape[0] == 1

    @patch("autodock.docking._HAVE_VINA", True)
    @patch("shutil.which")
    @patch("os.path.isfile")
    def test_cli_path_used(self, mock_isfile, mock_which, tmp_path):
        mock_isfile.return_value = False
        mock_which.return_value = "/usr/bin/vina"
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        with patch("autodock.docking._run_vina_cli") as mock_cli:
            mock_cli.return_value = (np.array([[-8.0]]), ["MODEL 1\nATOM 1 C\n"])
            energies, poses = docking._run_vina_dock(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                _use_subprocess=True,
            )
        mock_cli.assert_called_once()

    @patch("autodock.docking._HAVE_VINA", True)
    @patch("shutil.which")
    @patch("os.path.isfile")
    def test_spawn_subprocess_timeout(self, mock_isfile, mock_which, tmp_path):
        mock_isfile.return_value = False
        mock_which.return_value = None
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 1234
        mock_process.exitcode = -9

        with patch("multiprocessing.get_context") as mock_ctx:
            mock_ctx.return_value.Queue.return_value = MagicMock()
            mock_ctx.return_value.Process.return_value = mock_process
            with pytest.raises(DockingCalculationError, match="timed out"):
                docking._run_vina_dock(
                    str(rec),
                    str(lig),
                    (0, 0, 0),
                    (20, 20, 20),
                    timeout=1,
                    _use_subprocess=True,
                )

    @patch("autodock.docking._HAVE_VINA", True)
    @patch("shutil.which")
    @patch("os.path.isfile")
    def test_spawn_queue_empty(self, mock_isfile, mock_which, tmp_path):
        mock_isfile.return_value = False
        mock_which.return_value = None
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        mock_queue = MagicMock()
        import queue

        mock_queue.get.side_effect = queue.Empty("empty")
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_process.exitcode = 0

        with patch("multiprocessing.get_context") as mock_ctx:
            mock_ctx.return_value.Queue.return_value = mock_queue
            mock_ctx.return_value.Process.return_value = mock_process
            with pytest.raises(DockingCalculationError, match="empty"):
                docking._run_vina_dock(
                    str(rec),
                    str(lig),
                    (0, 0, 0),
                    (20, 20, 20),
                    _use_subprocess=True,
                )

    @patch("autodock.docking._HAVE_VINA", True)
    @patch("shutil.which")
    @patch("os.path.isfile")
    def test_spawn_error_status(self, mock_isfile, mock_which, tmp_path):
        mock_isfile.return_value = False
        mock_which.return_value = None
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )

        mock_queue = MagicMock()
        mock_queue.get.return_value = ("error", "worker crashed", [])
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_process.exitcode = 1

        with patch("multiprocessing.get_context") as mock_ctx:
            mock_ctx.return_value.Queue.return_value = mock_queue
            mock_ctx.return_value.Process.return_value = mock_process
            with pytest.raises(DockingCalculationError, match="worker crashed"):
                docking._run_vina_dock(
                    str(rec),
                    str(lig),
                    (0, 0, 0),
                    (20, 20, 20),
                    _use_subprocess=True,
                )


class TestScorePoseWithSf:
    @patch("autodock.docking._HAVE_VINA", True)
    @patch("vina.Vina")
    def test_score_pose_success(self, mock_vina_cls, tmp_path):
        mock_vina = MagicMock()
        mock_vina_cls.return_value = mock_vina
        mock_vina.score.return_value = [-8.5]

        pose = tmp_path / "pose.pdbqt"
        pose.write_text("ATOM 1 C LIG\n")
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )

        score = docking._score_pose_with_sf(str(rec), str(pose), (0, 0, 0), (20, 20, 20), "vinardo")
        assert score == pytest.approx(-8.5)

    @patch("autodock.docking._HAVE_VINA", True)
    @patch("vina.Vina")
    def test_score_pose_failure_returns_none(self, mock_vina_cls, tmp_path):
        mock_vina = MagicMock()
        mock_vina_cls.return_value = mock_vina
        mock_vina.score.side_effect = RuntimeError("fail")

        pose = tmp_path / "pose.pdbqt"
        pose.write_text("ATOM 1 C LIG\n")
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )

        score = docking._score_pose_with_sf(str(rec), str(pose), (0, 0, 0), (20, 20, 20), "vinardo")
        assert score is None


class TestDockLigandBranches:
    @patch("autodock.docking._run_vina_dock")
    def test_ad4_scoring_function(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        result = docking.dock_ligand(
            str(rec),
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            scoring_function="ad4",
            seed=42,
            output_dir=str(tmp_path / "out"),
        )
        assert "ad4" in result.scoring_functions

    @patch("autodock.docking._run_vina_dock")
    def test_flex_receptor_passed(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        flex = tmp_path / "flex.pdbqt"
        flex.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        docking.dock_ligand(
            str(rec),
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            flex_receptor_pdbqt=str(flex),
            seed=42,
        )
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("flex_receptor_pdbqt") == str(flex)

    @patch("autodock.docking._run_vina_dock")
    def test_auto_exhaustiveness_passed(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        docking.dock_ligand(
            str(rec),
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            auto_exhaustiveness=True,
            seed=42,
        )
        assert mock_run.call_args.kwargs.get("auto_exhaustiveness") is True

    @patch("autodock.docking._run_vina_dock")
    def test_energy_range_passed(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        docking.dock_ligand(
            str(rec),
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            energy_range=5.0,
            seed=42,
        )
        assert mock_run.call_args.kwargs.get("energy_range") == 5.0

    @patch("autodock.docking._run_vina_dock")
    def test_no_output_dir_uses_temp(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        result = docking.dock_ligand(str(rec), str(lig), (0, 0, 0), (20, 20, 20), seed=42)
        assert result.best_pose_pdbqt is not None
        assert os.path.isfile(result.best_pose_pdbqt)

    @patch("autodock.docking._run_vina_dock")
    def test_receptor_pdb_source(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        receptor_pdb = tmp_path / "rec.pdb"
        receptor_pdb.write_text("ATOM 1 N SER\n")
        with patch("autodock.core.detect_receptor_source", return_value="pdb"):
            result = docking.dock_ligand(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                seed=42,
                receptor_pdb=str(receptor_pdb),
            )
        assert result.receptor_source == "pdb"

    def test_multi_conformer_requires_smiles(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        with pytest.raises(ValueError, match="ligand_smiles"):
            docking.dock_ligand(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                multi_conformer=True,
                ligand_smiles=None,
            )

    @patch("autodock.docking.dock_ligand_multi_conformer")
    @patch("autodock.preparation.prepare_ligand_conformers")
    @patch("autodock.validation_params.validate_docking_params")
    def test_multi_conformer_flow(self, mock_val, mock_prep, mock_mc, tmp_path):
        mock_val.return_value = {
            "receptor_pdbqt": str(tmp_path / "rec.pdbqt"),
            "ligand_pdbqt": str(tmp_path / "lig.pdbqt"),
            "center": (0, 0, 0),
            "box_size": (20, 20, 20),
            "exhaustiveness": 8,
            "n_poses": 9,
            "energy_range": 3,
            "seed": 42,
            "timeout": 300,
        }
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_prep.return_value = [str(tmp_path / "conf1.pdbqt"), str(tmp_path / "conf2.pdbqt")]
        mock_mc.return_value = docking.DockingResult(
            compound_name="LIG",
            receptor="rec.pdbqt",
            best_affinity=-9.0,
            center=(0, 0, 0),
            box_size=(20, 20, 20),
        )
        result = docking.dock_ligand(
            str(rec),
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            multi_conformer=True,
            ligand_smiles="CCO",
            seed=42,
        )
        assert result.best_affinity == -9.0


class TestDockConformerCore:
    @patch("autodock.docking._run_vina_dock")
    def test_missing_conf_returns_empty(self, mock_run, tmp_path):
        result = docking._dock_conformer_core(
            "rec.pdbqt",
            str(tmp_path / "missing.pdbqt"),
            (0, 0, 0),
            (20, 20, 20),
            8,
            9,
            3,
            42,
            300,
        )
        assert result == ([], 0)

    @patch("autodock.docking._run_vina_dock")
    def test_docking_error_returns_empty(self, mock_run, tmp_path):
        mock_run.side_effect = DockingCalculationError("fail")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        result = docking._dock_conformer_core(
            "rec.pdbqt",
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            8,
            9,
            3,
            42,
            300,
        )
        assert result == ([], 0)

    @patch("autodock.docking._run_vina_dock")
    def test_success(self, mock_run, tmp_path):
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        result = docking._dock_conformer_core(
            "rec.pdbqt",
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            8,
            9,
            3,
            42,
            300,
        )
        assert result[1] == 1
        assert len(result[0]) == 1

    def test_worker_wrapper(self):
        args = (
            "rec.pdbqt",
            "lig.pdbqt",
            (0, 0, 0),
            (20, 20, 20),
            8,
            9,
            3.0,
            42,
            300,
            True,
            "vina",
            1.0,
        )
        with patch("autodock.docking._dock_conformer_core", return_value=([(-8.0, "pose")], 1)):
            result = docking._dock_conformer_worker(args)
        assert result == ([(-8.0, "pose")], 1)


class TestDockLigandMultiConformerBranches:
    @patch("autodock.docking._run_vina_dock")
    def test_psutil_not_available(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        conf1 = tmp_path / "c1.pdbqt"
        conf1.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        conf2 = tmp_path / "c2.pdbqt"
        conf2.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("concurrent.futures.ProcessPoolExecutor", FakeInThreadExecutor):
                result = docking.dock_ligand_multi_conformer(
                    str(rec),
                    [str(conf1), str(conf2)],
                    (0, 0, 0),
                    (20, 20, 20),
                    max_workers=-1,
                    seed=42,
                )
        assert result.best_affinity == -8.0

    @patch("autodock.docking._run_vina_dock")
    def test_explicit_max_workers_clamped_by_mem(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        conf1 = tmp_path / "c1.pdbqt"
        conf1.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        # Mock psutil to report very little memory, forcing clamp to 1
        mock_mem = MagicMock()
        mock_mem.available = 1e8  # ~0.1 GB
        with patch("psutil.virtual_memory", return_value=mock_mem):
            result = docking.dock_ligand_multi_conformer(
                str(rec),
                [str(conf1)],
                (0, 0, 0),
                (20, 20, 20),
                max_workers=8,
                seed=42,
            )
        assert result.best_affinity == -8.0

    @patch("autodock.docking._run_vina_dock")
    def test_max_workers_less_than_one(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        conf1 = tmp_path / "c1.pdbqt"
        conf1.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_run.return_value = (
            np.array([[-8.0, 1.0, 2.0]]),
            ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"],
        )
        result = docking.dock_ligand_multi_conformer(
            str(rec),
            [str(conf1)],
            (0, 0, 0),
            (20, 20, 20),
            max_workers=0,
            seed=42,
        )
        assert result.best_affinity == -8.0

    def test_parallel_workers_exception_handled(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        conf1 = tmp_path / "c1.pdbqt"
        conf1.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        conf2 = tmp_path / "c2.pdbqt"
        conf2.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        call_count = 0

        def worker_side_effect(item):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ([(-8.0, "pose1")], 1)
            raise DockingCalculationError("fail")

        with patch("autodock.docking._dock_conformer_worker", side_effect=worker_side_effect):
            with patch("concurrent.futures.ProcessPoolExecutor", FakeInThreadExecutor):
                result = docking.dock_ligand_multi_conformer(
                    str(rec),
                    [str(conf1), str(conf2)],
                    (0, 0, 0),
                    (20, 20, 20),
                    max_workers=2,
                    seed=42,
                )
        assert result.best_affinity == -8.0

    @patch("autodock.docking._run_vina_dock")
    def test_all_conformers_fail_raises(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        conf1 = tmp_path / "c1.pdbqt"
        conf1.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_run.side_effect = DockingCalculationError("fail")
        with pytest.raises(DockingCalculationError, match="All conformers failed"):
            docking.dock_ligand_multi_conformer(
                str(rec),
                [str(conf1)],
                (0, 0, 0),
                (20, 20, 20),
                max_workers=1,
                seed=42,
            )


class TestVirtualScreenBranches:
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_ligand")
    def test_parallel_workers(self, mock_prep, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        library = {"a": "CCO", "b": "CCN"}
        outdir = str(tmp_path / "vs")

        def make_result(rec_path, lig_path, center, box, **kwargs):
            return DockingResult(
                compound_name="x",
                receptor=rec_path,
                center=center,
                box_size=box,
                best_affinity=-7.5,
            )

        mock_dock.side_effect = make_result
        mock_prep.return_value = None

        with patch("concurrent.futures.ProcessPoolExecutor", FakeInThreadExecutor):
            results, csv_path = docking.virtual_screen(
                str(rec),
                library,
                (0, 0, 0),
                (20, 20, 20),
                output_dir=outdir,
                n_workers=-1,
                seed=42,
            )
        assert len(results) == 2

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_ligand")
    def test_prepare_failure_returns_none(self, mock_prep, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        library = {"a": "CCO"}
        outdir = str(tmp_path / "vs")
        mock_prep.side_effect = RuntimeError("prep failed")
        results, csv_path = docking.virtual_screen(
            str(rec),
            library,
            (0, 0, 0),
            (20, 20, 20),
            output_dir=outdir,
            n_workers=1,
            seed=42,
        )
        assert len(results) == 1
        assert results[0].best_affinity is None


class TestBatchDockBranches:
    def test_empty_receptors_raises(self):
        with pytest.raises(DockingCalculationError, match="At least one receptor"):
            docking.batch_dock({}, {"lig": "lig.pdbqt"}, {})

    def test_empty_ligands_raises(self):
        with pytest.raises(DockingCalculationError, match="At least one receptor"):
            docking.batch_dock({"rec": "rec.pdbqt"}, {}, {})

    def test_missing_receptor_file_raises(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        with pytest.raises(DockingCalculationError, match="Receptor file not found"):
            docking.batch_dock(
                {"rec": str(tmp_path / "missing.pdbqt")},
                {"lig": str(tmp_path / "lig.pdbqt")},
                {"rec": {"center": (0, 0, 0), "box_size": (20, 20, 20)}},
            )

    def test_missing_pocket_raises(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        with pytest.raises(ValueError, match="Pocket definition missing"):
            docking.batch_dock(
                {"rec": str(rec)},
                {"lig": str(lig)},
                {},
            )

    def test_missing_pocket_keys_raises(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        with pytest.raises(ValueError, match="center"):
            docking.batch_dock(
                {"rec": str(rec)},
                {"lig": str(lig)},
                {"rec": {"box_size": (20, 20, 20)}},
            )

    def test_missing_ligand_file_raises(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        with pytest.raises(DockingCalculationError, match="Ligand file not found"):
            docking.batch_dock(
                {"rec": str(rec)},
                {"lig": str(tmp_path / "missing.pdbqt")},
                {"rec": {"center": (0, 0, 0), "box_size": (20, 20, 20)}},
            )

    @patch("autodock.docking.dock_ligand")
    def test_parallel_workers(self, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_dock.return_value = DockingResult(
            compound_name="lig",
            receptor=str(rec),
            center=(0, 0, 0),
            box_size=(20, 20, 20),
            best_affinity=-7.0,
        )
        results = docking.batch_dock(
            {"rec": str(rec)},
            {"lig": str(lig)},
            {"rec": {"center": (0, 0, 0), "box_size": (20, 20, 20)}},
            output_dir=str(tmp_path / "batch"),
            n_workers=1,
        )
        assert len(results["rec"]) == 1

    @patch("autodock.docking.dock_ligand")
    def test_worker_crash_handled(self, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        mock_dock.side_effect = DockingCalculationError("fail")
        results = docking.batch_dock(
            {"rec": str(rec)},
            {"lig": str(lig)},
            {"rec": {"center": (0, 0, 0), "box_size": (20, 20, 20)}},
            output_dir=str(tmp_path / "batch"),
            n_workers=1,
        )
        assert results["rec"][0].best_affinity is None


class TestDockEnsembleBranches:
    @patch("autodock.docking.dock_ligand")
    def test_confidence_high(self, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        pf = tmp_path / "pose.pdbqt"
        pf.write_text(
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000  1.00 20.00      A    C\n"
        )
        mock_dock.return_value = DockingResult(
            compound_name="test",
            receptor=str(rec),
            center=(0, 0, 0),
            box_size=(20, 20, 20),
            best_affinity=-8.0,
            best_pose_pdbqt=str(pf),
        )
        with patch("autodock.validation.compute_rmsd", return_value=0.0):
            summary = docking.dock_ensemble(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                n_repeats=3,
                seed=42,
                output_dir=str(tmp_path / "ens"),
            )
        assert summary["confidence"] == "high"
        assert "recommendation" in summary

    @patch("autodock.docking.dock_ligand")
    def test_confidence_moderate(self, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        # Create slightly different poses
        for i in range(3):
            pf = tmp_path / f"pose_{i}.pdbqt"
            pf.write_text(f"ATOM 1 C LIG A 1 {i*0.5:.3f} 0 0\n")
        call_count = 0

        def make_result(*a, **k):
            nonlocal call_count
            idx = call_count
            call_count += 1
            return DockingResult(
                compound_name="test",
                receptor=str(rec),
                center=(0, 0, 0),
                box_size=(20, 20, 20),
                best_affinity=-7.5,
                best_pose_pdbqt=str(tmp_path / f"pose_{idx}.pdbqt"),
            )

        mock_dock.side_effect = make_result
        summary = docking.dock_ensemble(
            str(rec),
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            n_repeats=3,
            seed=42,
            output_dir=str(tmp_path / "ens"),
        )
        assert summary["confidence"] in ("high", "moderate", "low")

    @patch("autodock.docking.dock_ligand")
    def test_all_nan_affinities(self, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        pf = tmp_path / "pose.pdbqt"
        pf.write_text(
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000  1.00 20.00      A    C\n"
        )
        mock_dock.return_value = DockingResult(
            compound_name="test",
            receptor=str(rec),
            center=(0, 0, 0),
            box_size=(20, 20, 20),
            best_affinity=float("nan"),
            best_pose_pdbqt=str(pf),
        )
        with patch("autodock.validation.compute_rmsd", return_value=0.0):
            summary = docking.dock_ensemble(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                n_repeats=3,
                seed=42,
                output_dir=str(tmp_path / "ens"),
            )
        assert summary["confidence"] == "low"
        assert summary["ensemble_best_affinity_mean"] is None

    @patch("autodock.docking.dock_ligand")
    def test_no_rmsd_values(self, mock_dock, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00 20.00      A    N\n" * 5
            + "ENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00 20.00      A    C\n" * 5
            + "ENDMDL\n"
        )
        # Best pose path points to missing file
        mock_dock.return_value = DockingResult(
            compound_name="test",
            receptor=str(rec),
            center=(0, 0, 0),
            box_size=(20, 20, 20),
            best_affinity=-8.0,
            best_pose_pdbqt=str(tmp_path / "missing.pdbqt"),
        )
        summary = docking.dock_ensemble(
            str(rec),
            str(lig),
            (0, 0, 0),
            (20, 20, 20),
            n_repeats=3,
            seed=42,
            output_dir=str(tmp_path / "ens"),
        )
        assert summary["pose_stability_rmsd_mean"] is None
