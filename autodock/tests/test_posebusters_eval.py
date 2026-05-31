"""Tests for autodock.posebusters_eval — PoseBusters benchmark evaluation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from autodock import posebusters_eval as pbe
from autodock.core import ValidationError


class TestLoadPosebustersIds:
    def test_normal_lines(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("5SAK_ZRY\n5SB2_6W4\n")
        result = pbe.load_posebusters_ids(str(f))
        assert result == [("5SAK", "ZRY"), ("5SB2", "6W4")]

    def test_skips_comments_and_blank_lines(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("\n# comment\n5SAK_ZRY\n\n5SB2_6W4\n  \n")
        result = pbe.load_posebusters_ids(str(f))
        assert result == [("5SAK", "ZRY"), ("5SB2", "6W4")]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("")
        result = pbe.load_posebusters_ids(str(f))
        assert result == []


class TestRunSinglePosebuster:
    @patch("autodock.utils.download_pdb")
    @patch("autodock.posebusters_eval.auto_detect_ligand_resname")
    @patch("autodock.posebusters_eval.run_redocking_validation")
    @patch("autodock.posebusters_eval.validate_pose_with_posebusters")
    def test_success_path(
        self,
        mock_pb,
        mock_redock,
        mock_detect,
        mock_download,
        tmp_path,
    ):
        mock_detect.return_value = True
        outdir = tmp_path / "5SAK"
        outdir.mkdir()
        holo_pdb = outdir / "5SAK.pdb"
        holo_pdb.write_text("ATOM\n")
        best_pose = outdir / "docking_best.pdbqt"
        best_pose.write_text("REMARK\n")

        mock_redock.return_value = {
            "success": True,
            "rmsd": 1.2,
            "best_affinity": -8.5,
            "best_pose": str(best_pose),
        }
        mock_pb.return_value = {"available": True, "pass": True}

        item = {
            "pdb_id": "5SAK",
            "ccd": "ZRY",
            "output_dir": str(outdir),
            "exhaustiveness": 32,
            "n_poses": 20,
            "seed": 42,
        }
        result = pbe._run_single_posebuster(item)

        assert result["pdb_id"] == "5SAK"
        assert result["success"] is True
        assert result["rmsd"] == 1.2
        assert result["best_affinity"] == -8.5
        assert result["posebusters_pass"] is True
        assert result["posebusters_available"] is True
        mock_download.assert_not_called()  # PDB already exists
        mock_redock.assert_called_once()
        mock_pb.assert_called_once_with(str(best_pose), str(holo_pdb))

    @patch("autodock.utils.download_pdb")
    @patch("autodock.posebusters_eval.auto_detect_ligand_resname")
    @patch("autodock.posebusters_eval.run_redocking_validation")
    def test_download_failure(
        self,
        mock_redock,
        mock_detect,
        mock_download,
        tmp_path,
    ):
        mock_download.side_effect = RuntimeError("network down")

        item = {
            "pdb_id": "5SAK",
            "ccd": "ZRY",
            "output_dir": str(tmp_path / "5SAK"),
            "exhaustiveness": 32,
            "n_poses": 20,
            "seed": 42,
        }
        result = pbe._run_single_posebuster(item)

        assert result["pdb_id"] == "5SAK"
        assert result["success"] is False
        assert "download" in result["error"]
        mock_redock.assert_not_called()

    @patch("autodock.utils.download_pdb")
    @patch("autodock.posebusters_eval.auto_detect_ligand_resname")
    @patch("autodock.posebusters_eval.run_redocking_validation")
    def test_no_ligand_detected(
        self,
        mock_redock,
        mock_detect,
        mock_download,
        tmp_path,
    ):
        outdir = tmp_path / "5SAK"
        outdir.mkdir()
        holo_pdb = outdir / "5SAK.pdb"
        holo_pdb.write_text("ATOM\n")
        mock_detect.return_value = False

        item = {
            "pdb_id": "5SAK",
            "ccd": "ZRY",
            "output_dir": str(outdir),
            "exhaustiveness": 32,
            "n_poses": 20,
            "seed": 42,
        }
        result = pbe._run_single_posebuster(item)

        assert result["pdb_id"] == "5SAK"
        assert result["success"] is False
        assert "No ligand detected" in result["error"]
        mock_redock.assert_not_called()

    @patch("autodock.utils.download_pdb")
    @patch("autodock.posebusters_eval.auto_detect_ligand_resname")
    @patch("autodock.posebusters_eval.run_redocking_validation")
    def test_redocking_validation_error(
        self,
        mock_redock,
        mock_detect,
        mock_download,
        tmp_path,
    ):
        outdir = tmp_path / "5SAK"
        outdir.mkdir()
        holo_pdb = outdir / "5SAK.pdb"
        holo_pdb.write_text("ATOM\n")
        mock_detect.return_value = True
        mock_redock.side_effect = ValidationError("preparation failed")

        item = {
            "pdb_id": "5SAK",
            "ccd": "ZRY",
            "output_dir": str(outdir),
            "exhaustiveness": 32,
            "n_poses": 20,
            "seed": 42,
        }
        result = pbe._run_single_posebuster(item)

        assert result["pdb_id"] == "5SAK"
        assert result["success"] is False
        assert "preparation failed" in result["error"]

    @patch("autodock.utils.download_pdb")
    @patch("autodock.posebusters_eval.auto_detect_ligand_resname")
    @patch("autodock.posebusters_eval.run_redocking_validation")
    def test_redocking_generic_exception(
        self,
        mock_redock,
        mock_detect,
        mock_download,
        tmp_path,
    ):
        outdir = tmp_path / "5SAK"
        outdir.mkdir()
        holo_pdb = outdir / "5SAK.pdb"
        holo_pdb.write_text("ATOM\n")
        mock_detect.return_value = True
        mock_redock.side_effect = RuntimeError("unexpected crash")

        item = {
            "pdb_id": "5SAK",
            "ccd": "ZRY",
            "output_dir": str(outdir),
            "exhaustiveness": 32,
            "n_poses": 20,
            "seed": 42,
        }
        result = pbe._run_single_posebuster(item)

        assert result["pdb_id"] == "5SAK"
        assert result["success"] is False
        assert "unexpected crash" in result["error"]

    @patch("autodock.utils.download_pdb")
    @patch("autodock.posebusters_eval.auto_detect_ligand_resname")
    @patch("autodock.posebusters_eval.run_redocking_validation")
    @patch("autodock.posebusters_eval.validate_pose_with_posebusters")
    def test_posebusters_failure(
        self,
        mock_pb,
        mock_redock,
        mock_detect,
        mock_download,
        tmp_path,
    ):
        outdir = tmp_path / "5SAK"
        outdir.mkdir()
        holo_pdb = outdir / "5SAK.pdb"
        holo_pdb.write_text("ATOM\n")
        best_pose = outdir / "docking_best.pdbqt"
        best_pose.write_text("REMARK\n")
        mock_detect.return_value = True
        mock_redock.return_value = {
            "success": True,
            "rmsd": 1.2,
            "best_affinity": -8.5,
            "best_pose": str(best_pose),
        }
        mock_pb.side_effect = RuntimeError("pb module missing")

        item = {
            "pdb_id": "5SAK",
            "ccd": "ZRY",
            "output_dir": str(outdir),
            "exhaustiveness": 32,
            "n_poses": 20,
            "seed": 42,
        }
        result = pbe._run_single_posebuster(item)

        assert result["pdb_id"] == "5SAK"
        assert result["success"] is True
        assert result["rmsd"] == 1.2
        assert result["posebusters_pass"] is None
        assert result["posebusters_available"] is False


class TestRunPosebustersEvaluation:
    @patch("autodock.posebusters_eval._run_single_posebuster")
    def test_serial_execution(self, mock_run, tmp_path):
        mock_run.side_effect = [
            {
                "pdb_id": "5SAK",
                "success": True,
                "rmsd": 1.2,
                "best_affinity": -8.5,
                "posebusters_pass": True,
                "posebusters_available": True,
            },
            {
                "pdb_id": "5SB2",
                "success": True,
                "rmsd": 2.1,
                "best_affinity": -7.5,
                "posebusters_pass": False,
                "posebusters_available": True,
            },
        ]

        id_file = tmp_path / "ids.txt"
        id_file.write_text("5SAK_ZRY\n5SB2_6W4\n")
        outdir = tmp_path / "results"

        summary = pbe.run_posebusters_evaluation(
            id_list_path=str(id_file),
            output_dir=str(outdir),
            n_workers=1,
            seed=42,
        )

        assert summary["n_total"] == 2
        assert summary["n_success"] == 2
        assert summary["success_rate"] == 1.0
        assert summary["posebusters_pass_rate"] == 0.5
        assert summary["posebusters_pass_count"] == 1
        assert summary["mean_rmsd"] == pytest.approx(1.65)
        assert summary["median_rmsd"] == pytest.approx(1.65)
        assert summary["parameters"]["seed"] == 42
        assert len(summary["per_target"]) == 2
        assert mock_run.call_count == 2

        json_path = outdir / "posebusters_summary.json"
        assert json_path.exists()
        with open(json_path) as fh:
            loaded = json.load(fh)
        assert loaded["n_total"] == 2

    @patch("concurrent.futures.as_completed")
    @patch("concurrent.futures.ProcessPoolExecutor")
    def test_parallel_execution(self, mock_executor_cls, mock_as_completed, tmp_path):
        """Mock ProcessPoolExecutor to avoid actual subprocess spawning."""
        mock_future_1 = MagicMock()
        mock_future_1.result.return_value = {
            "pdb_id": "5SAK",
            "success": True,
            "rmsd": 1.2,
            "best_affinity": -8.5,
            "posebusters_pass": True,
            "posebusters_available": True,
        }
        mock_future_2 = MagicMock()
        mock_future_2.result.return_value = {
            "pdb_id": "5SB2",
            "success": True,
            "rmsd": 2.1,
            "best_affinity": -7.5,
            "posebusters_pass": False,
            "posebusters_available": True,
        }

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = [mock_future_1, mock_future_2]
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor_cls.return_value = mock_executor
        mock_as_completed.side_effect = lambda futures: futures

        id_file = tmp_path / "ids.txt"
        id_file.write_text("5SAK_ZRY\n5SB2_6W4\n")
        outdir = tmp_path / "results"

        summary = pbe.run_posebusters_evaluation(
            id_list_path=str(id_file),
            output_dir=str(outdir),
            n_workers=2,
            seed=42,
        )

        assert summary["n_total"] == 2
        assert summary["n_success"] == 2
        assert summary["success_rate"] == 1.0
        assert summary["posebusters_pass_rate"] == 0.5
        assert summary["posebusters_pass_count"] == 1
        # mp_context is added for macOS fork-safety; accept either form
        assert mock_executor_cls.call_count == 1
        call_kwargs = mock_executor_cls.call_args[1]
        assert call_kwargs.get("max_workers") == 2
        assert mock_executor.submit.call_count == 2

    @patch("autodock.posebusters_eval._run_single_posebuster")
    def test_max_targets_limit(self, mock_run, tmp_path):
        mock_run.return_value = {
            "pdb_id": "5SAK",
            "success": True,
            "rmsd": 1.2,
            "best_affinity": -8.5,
            "posebusters_pass": True,
            "posebusters_available": True,
        }

        id_file = tmp_path / "ids.txt"
        id_file.write_text("5SAK_ZRY\n5SB2_6W4\n5SD5_XXX\n")
        outdir = tmp_path / "results"

        summary = pbe.run_posebusters_evaluation(
            id_list_path=str(id_file),
            output_dir=str(outdir),
            n_workers=1,
            max_targets=1,
        )

        assert summary["n_total"] == 1
        assert summary["n_success"] == 1
        mock_run.assert_called_once()

    @patch("autodock.posebusters_eval._run_single_posebuster")
    def test_empty_list(self, mock_run, tmp_path):
        id_file = tmp_path / "ids.txt"
        id_file.write_text("")
        outdir = tmp_path / "results"

        summary = pbe.run_posebusters_evaluation(
            id_list_path=str(id_file),
            output_dir=str(outdir),
            n_workers=1,
        )

        assert summary["n_total"] == 0
        assert summary["n_success"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["mean_rmsd"] is None
        assert summary["posebusters_pass_rate"] == 0.0
        assert summary["per_target"] == []
        mock_run.assert_not_called()

        json_path = outdir / "posebusters_summary.json"
        assert json_path.exists()
