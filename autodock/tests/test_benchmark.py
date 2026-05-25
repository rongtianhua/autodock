"""Tests for autodock.benchmark — redocking benchmark logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autodock import benchmark as bm


class TestRunRedockingBenchmark:
    @patch("autodock.utils.download_pdb")
    @patch("autodock.benchmark.run_redocking_validation")
    def test_two_targets_one_success(self, mock_redock, mock_download, tmp_path):
        mock_download.return_value = "dummy.pdb"
        mock_redock.side_effect = [
            {"success": True, "rmsd": 1.2, "best_affinity": -8.0, "threshold": 2.0},
            {"success": False, "rmsd": 3.5, "best_affinity": -7.0, "threshold": 2.0},
        ]

        targets = [
            {"pdb_id": "1ABC", "ligand_resname": "LIG", "family": "kinase", "name": "Test1"},
            {"pdb_id": "2DEF", "ligand_resname": "LIG", "family": "protease", "name": "Test2"},
        ]
        summary = bm.run_redocking_benchmark(
            targets=targets,
            output_dir=str(tmp_path / "bench"),
            seed=42,
            n_workers=1,
        )

        assert summary["n_total"] == 2
        assert summary["n_success"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["median_rmsd"] == 1.2
        assert "by_family" in summary
        assert summary["by_family"]["kinase"]["n_success"] == 1
        assert summary["by_family"]["protease"]["n_success"] == 0
        assert summary["json_path"] is not None

    @patch("autodock.utils.download_pdb")
    @patch("autodock.benchmark.run_redocking_validation")
    def test_all_targets_fail_gracefully(self, mock_redock, mock_download, tmp_path):
        mock_download.return_value = "dummy.pdb"
        mock_redock.side_effect = Exception("redock failed")

        targets = [
            {"pdb_id": "1ABC", "ligand_resname": "LIG", "family": "kinase"},
        ]
        summary = bm.run_redocking_benchmark(
            targets=targets,
            output_dir=str(tmp_path / "bench"),
            seed=42,
            n_workers=1,
        )

        assert summary["n_total"] == 1
        assert summary["n_success"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["median_rmsd"] is None

    def test_default_targets_not_empty(self):
        assert len(bm.DEFAULT_BENCHMARK_TARGETS) >= 20
        for t in bm.DEFAULT_BENCHMARK_TARGETS:
            assert "pdb_id" in t
            assert "family" in t

    def test_summary_statistics(self):
        # Direct test of stats compilation logic via mocked _run_single_benchmark
        with patch("autodock.benchmark._run_single_benchmark") as mock_run:
            mock_run.side_effect = [
                {"pdb_id": "1A", "family": "kinase", "success": True, "rmsd": 1.0, "best_affinity": -8.0},
                {"pdb_id": "1B", "family": "kinase", "success": True, "rmsd": 2.0, "best_affinity": -7.5},
                {"pdb_id": "1C", "family": "protease", "success": False, "rmsd": None, "error": "fail"},
            ]
            summary = bm.run_redocking_benchmark(
                targets=[
                    {"pdb_id": "1A", "ligand_resname": "LIG", "family": "kinase"},
                    {"pdb_id": "1B", "ligand_resname": "LIG", "family": "kinase"},
                    {"pdb_id": "1C", "ligand_resname": "LIG", "family": "protease"},
                ],
                output_dir="/tmp/bench_test",
                n_workers=1,
            )
            assert summary["mean_rmsd"] == 1.5
            assert summary["median_rmsd"] == 1.5
            assert summary["by_family"]["kinase"]["success_rate"] == 1.0
            assert summary["by_family"]["protease"]["success_rate"] == 0.0
