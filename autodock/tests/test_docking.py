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
            "MODEL 1\nREMARK VINA RESULT: -8.0\nATOM      1  C   LIG A   1      0.000   0.000   0.000\nENDMDL\n"
            "MODEL 2\nREMARK VINA RESULT: -7.5\nATOM      1  C   LIG A   1      1.000   1.000   1.000\nENDMDL\n"
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
    @patch("autodock.docking._score_pose_with_sf")
    def test_single_score(self, mock_score):
        mock_score.return_value = None
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

    @patch("autodock.docking._score_pose_with_sf")
    def test_median_consensus(self, mock_score):
        # First call returns vinardo score
        mock_score.return_value = -7.5
        all_scores, consensus = docking._consensus_score(
            "rec.pdbqt",
            "pose.pdbqt",
            (0, 0, 0),
            (20, 20, 20),
            -8.0,
            seed=42,
        )
        assert "vinardo" in all_scores
        assert consensus == -7.5  # median of [-8.0, -7.5]


class TestDockLigand:
    @patch("autodock.docking._run_vina_dock")
    def test_basic_dock(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

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
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

        mock_run.return_value = (np.array([]), [])
        with pytest.raises(DockingCalculationError, match="no poses"):
            docking.dock_ligand(str(rec), str(lig), (0, 0, 0), (20, 20, 20), seed=42)

    def test_missing_receptor_raises(self):
        with pytest.raises(DockingCalculationError):
            docking.dock_ligand("missing.pdbqt", "lig.pdbqt", (0, 0, 0), (20, 20, 20))

    def test_invalid_params_raises(self, tmp_path):
        from autodock.core import ConfigurationError

        rec = tmp_path / "rec.pdbqt"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")
        with pytest.raises(ConfigurationError):
            docking.dock_ligand(str(rec), str(lig), (0, 0, 0), (20, 20, 20), exhaustiveness=-5)


class TestDockLigandMultiConformer:
    @patch("autodock.docking._run_vina_dock")
    def test_multi_conformer(self, mock_run, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
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
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")

        library = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O", "ibu": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"}
        outdir = str(tmp_path / "vs")

        def make_result(rec_path, lig_path, center, box, **kwargs):
            name = os.path.basename(os.path.dirname(lig_path)) if "/" in lig_path else "unknown"
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
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        results, csv_path = docking.virtual_screen(
            str(rec),
            {},
            (0, 0, 0),
            (20, 20, 20),
            output_dir=str(tmp_path / "vs"),
        )
        assert results == []


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
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

        with pytest.raises(ValueError, match="missing"):
            docking.batch_dock(
                {"rec": str(rec)},
                {"lig": str(lig)},
                {},
            )

    def test_missing_receptor_file_raises(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")
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
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

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
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")
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
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

        mock_dock.side_effect = DockingCalculationError("Vina failed")

        with pytest.raises(DockingCalculationError, match="Fewer than 2"):
            docking.dock_ensemble(
                str(rec),
                str(lig),
                (0, 0, 0),
                (20, 20, 20),
                n_repeats=3,
            )
