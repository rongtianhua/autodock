"""Tests for autodock.benchmark (lightweight, no redocking)."""

from unittest.mock import patch

import pytest

from autodock import benchmark


def _hetatm(serial, name, resname, chain, resseq, x, y, z, occ, bfac):
    # PDB ATOM/HETATM: cols 13-16 = name, 17 = alt-loc, 18-20 = resname
    return (
        f"HETATM{serial:5d} {name:<4s} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}"
    )


def _pdb_atom(serial, name, resname, chain, resseq, x, y, z, occ, bfac):
    return (
        f"ATOM  {serial:5d} {name:<4s} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}"
    )


class TestConstants:
    """Importing the module covers the large constant data structures."""

    def test_default_targets_length(self):
        assert len(benchmark.DEFAULT_BENCHMARK_TARGETS) == 20
        assert benchmark.DEFAULT_BENCHMARK_TARGETS[0]["pdb_id"] == "1C5Z"

    def test_default_targets_used_when_none(self, tmp_path, monkeypatch):
        """Cover line 244: targets=None defaults to DEFAULT_BENCHMARK_TARGETS."""
        _calls = []

        def _fake(item):
            _calls.append(item["target"]["pdb_id"])
            return {
                "pdb_id": item["target"]["pdb_id"],
                "family": "x",
                "success": True,
                "rmsd": 1.0,
            }

        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake)
        out = tmp_path / "bench"
        summary = benchmark.run_redocking_benchmark(
            targets=None,
            output_dir=str(out),
            n_workers=1,
        )
        assert summary["n_total"] == len(benchmark.DEFAULT_BENCHMARK_TARGETS)

    def test_hard_target_overrides(self):
        assert "1GWX" in benchmark.HARD_TARGET_OVERRIDES
        assert benchmark.HARD_TARGET_OVERRIDES["1GWX"]["exhaustiveness"] == 16

    def test_standard_residues_contains_ala(self):
        assert "ALA" in benchmark._STANDARD_RESIDUES
        assert "HOH" not in benchmark._STANDARD_RESIDUES

    def test_non_ligand_hets_contains_hoh(self):
        assert "HOH" in benchmark._NON_LIGAND_HETS
        assert "SO4" in benchmark._NON_LIGAND_HETS


def _fake_run_basic(item):
    return {
        "pdb_id": item["target"]["pdb_id"],
        "family": item["target"].get("family", "unknown"),
        "name": item["target"].get("name", ""),
        "success": True,
        "rmsd": 1.2,
        "success_raw": True,
        "rmsd_raw": 1.5,
        "best_rmsd": 0.8,
        "best_affinity": -8.5,
    }


def _fake_run_scoring_fail(item):
    pdb = item["target"]["pdb_id"]
    return {
        "pdb_id": pdb,
        "family": "kinase",
        "name": pdb,
        "success": False,
        "rmsd": None,
        "success_raw": False,
        "rmsd_raw": None,
        "best_rmsd": 1.0,
        "best_affinity": -7.0,
    }


_rescue_calls = []


def _fake_run_rescue(item):
    pdb = item["target"]["pdb_id"]
    _rescue_calls.append(pdb)
    if pdb == "PASS":
        return {
            "pdb_id": pdb,
            "family": "x",
            "success": True,
            "rmsd": 1.0,
            "success_raw": True,
            "rmsd_raw": 1.2,
            "best_rmsd": 0.9,
        }
    if pdb == "RESC":
        return {
            "pdb_id": pdb,
            "family": "x",
            "success": True,
            "rmsd": 1.0,
            "success_raw": False,
            "rmsd_raw": None,
            "best_rmsd": 0.9,
        }
    if pdb == "DEGR":
        return {
            "pdb_id": pdb,
            "family": "x",
            "success": False,
            "rmsd": None,
            "success_raw": True,
            "rmsd_raw": 1.2,
            "best_rmsd": 0.9,
        }
    return {
        "pdb_id": pdb,
        "family": "x",
        "success": False,
        "rmsd": None,
        "success_raw": False,
        "rmsd_raw": None,
        "best_rmsd": 5.0,
    }


def _fake_run_parallel(item):
    return {
        "pdb_id": item["target"]["pdb_id"],
        "family": "x",
        "success": True,
        "rmsd": 1.0,
        "success_raw": True,
        "rmsd_raw": 1.1,
        "best_rmsd": 0.9,
    }


class TestRunRedockingBenchmark:
    """Tests for run_redocking_benchmark stats compilation (mocked worker)."""

    def test_basic_stats_and_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake_run_basic)
        out = tmp_path / "bench"
        summary = benchmark.run_redocking_benchmark(
            targets=benchmark.DEFAULT_BENCHMARK_TARGETS[:2],
            output_dir=str(out),
            n_workers=1,
            minimize=True,
        )
        assert summary["n_total"] == 2
        assert summary["n_success"] == 2
        assert summary["success_rate"] == 1.0
        assert summary["mean_rmsd"] == pytest.approx(1.2, abs=1e-6)
        assert summary["n_success_best"] == 2
        assert summary["n_scoring_failures"] == 0
        assert "by_family" in summary
        assert out.joinpath("benchmark_summary.json").exists()

    def test_scoring_failure_detection(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake_run_scoring_fail)
        out = tmp_path / "bench"
        summary = benchmark.run_redocking_benchmark(
            targets=[{"pdb_id": "1ABC", "family": "kinase", "name": "Test"}],
            output_dir=str(out),
            n_workers=1,
            minimize=True,
        )
        assert summary["n_success"] == 0
        assert summary["n_success_best"] == 1
        assert summary["n_scoring_failures"] == 1
        assert summary["scoring_failure_pdb_ids"] == ["1ABC"]

    def test_rescued_and_degraded_counts(self, tmp_path, monkeypatch):
        _rescue_calls.clear()
        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake_run_rescue)
        out = tmp_path / "bench"
        targets = [
            {"pdb_id": "PASS", "family": "x"},
            {"pdb_id": "RESC", "family": "x"},
            {"pdb_id": "DEGR", "family": "x"},
            {"pdb_id": "FAIL", "family": "x"},
        ]
        summary = benchmark.run_redocking_benchmark(
            targets=targets, output_dir=str(out), n_workers=1, minimize=True
        )
        assert summary["n_rescued"] == 1
        assert summary["n_degraded"] == 1
        assert summary["n_success_raw"] == 2

    def test_parallel_workers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake_run_parallel)
        out = tmp_path / "bench"
        summary = benchmark.run_redocking_benchmark(
            targets=[{"pdb_id": "1A", "family": "x"}, {"pdb_id": "1B", "family": "x"}],
            output_dir=str(out),
            n_workers=-1,
            minimize=False,
        )
        assert summary["n_success"] == 2

    def test_empty_targets(self, tmp_path):
        out = tmp_path / "bench"
        summary = benchmark.run_redocking_benchmark(targets=[], output_dir=str(out), n_workers=1)
        assert summary["n_total"] == 0
        assert summary["success_rate"] == 0.0


class TestAutoDetectLigandResname:
    """Tests for auto_detect_ligand_resname."""

    def test_detects_from_hetatm(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        lines = [
            _hetatm(1, "C1 ", "N3 ", "A", 1, 0.0, 0.0, 0.0, 1.0, 20.0),
            _hetatm(2, "C2 ", "N3 ", "A", 1, 1.0, 0.0, 0.0, 1.0, 20.0),
            _hetatm(3, "O  ", "HOH", "A", 2, 5.0, 0.0, 0.0, 1.0, 30.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        assert benchmark.auto_detect_ligand_resname(str(pdb)) == "N3"

    def test_detects_non_standard_atom_residue(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        lines = [
            _pdb_atom(1, "CA ", "ALA", "A", 1, 0.0, 0.0, 0.0, 1.0, 20.0),
            _pdb_atom(2, "CA ", "N3 ", "A", 2, 1.0, 0.0, 0.0, 1.0, 20.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        assert benchmark.auto_detect_ligand_resname(str(pdb)) == "N3"

    def test_returns_none_when_no_ligand(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        lines = [
            _pdb_atom(1, "CA ", "ALA", "A", 1, 0.0, 0.0, 0.0, 1.0, 20.0),
            _hetatm(2, "O  ", "HOH", "A", 2, 5.0, 0.0, 0.0, 1.0, 30.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        assert benchmark.auto_detect_ligand_resname(str(pdb)) is None

    def test_empty_file_returns_none(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("")
        assert benchmark.auto_detect_ligand_resname(str(pdb)) is None

    def test_prefers_largest_ligand(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        # Use N3 (not in _NON_LIGAND_HETS) vs GOL (in _NON_LIGAND_HETS)
        lines = [
            _hetatm(1, "C1 ", "N3 ", "A", 1, 0.0, 0.0, 0.0, 1.0, 20.0),
            _hetatm(2, "C2 ", "N3 ", "A", 1, 1.0, 0.0, 0.0, 1.0, 20.0),
            _hetatm(3, "C3 ", "N3 ", "A", 1, 2.0, 0.0, 0.0, 1.0, 20.0),
            _hetatm(4, "O  ", "GOL", "A", 2, 5.0, 0.0, 0.0, 1.0, 30.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        assert benchmark.auto_detect_ligand_resname(str(pdb)) == "N3"

    def test_skips_non_ligand_hets(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        lines = [
            _hetatm(1, "O  ", "HOH", "A", 1, 0.0, 0.0, 0.0, 1.0, 20.0),
            _hetatm(2, "O  ", "SO4", "A", 2, 1.0, 0.0, 0.0, 1.0, 20.0),
            _hetatm(3, "C1 ", "N3 ", "A", 3, 5.0, 0.0, 0.0, 1.0, 30.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        assert benchmark.auto_detect_ligand_resname(str(pdb)) == "N3"


class TestRunRepeatDocking:
    def test_empty_targets(self, tmp_path, monkeypatch):
        summary = benchmark.run_repeat_docking(
            targets=[],
            output_dir=str(tmp_path / "repeat"),
            n_repeats=1,
        )
        assert summary["per_target"] == []

    def test_repeat_with_mocked_worker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake_run_basic)
        targets = [{"pdb_id": "1A", "family": "x", "name": "Test"}]
        out = tmp_path / "repeat"
        summary = benchmark.run_repeat_docking(
            targets=targets,
            output_dir=str(out),
            n_repeats=2,
        )
        assert len(summary["per_target"]) == 1
        pt = summary["per_target"][0]
        assert pt["pdb_id"] == "1A"
        assert pt["n_total"] == 2
        assert pt["n_success"] == 2
        assert pt["success_rate"] == 1.0
        assert out.joinpath("repeat_docking_summary.json").exists()

    def test_repeat_catches_worker_exception(self, tmp_path, monkeypatch):
        def _crash(item):
            raise RuntimeError("boom")

        monkeypatch.setattr(benchmark, "_run_single_benchmark", _crash)
        targets = [{"pdb_id": "1A", "family": "x", "name": "Test"}]
        out = tmp_path / "repeat"
        summary = benchmark.run_repeat_docking(
            targets=targets,
            output_dir=str(out),
            n_repeats=1,
        )
        assert len(summary["per_target"]) == 1
        pt = summary["per_target"][0]
        assert pt["n_success"] == 0
        assert pt["success_rate"] == 0.0
        assert len(pt["errors"]) == 1
        assert "boom" in pt["errors"][0]

    def test_repeat_with_best_affinity(self, tmp_path, monkeypatch):
        def _fake_with_affinity(item):
            return {
                "pdb_id": item["target"]["pdb_id"],
                "family": "x",
                "success": True,
                "rmsd": 1.0,
                "best_affinity": -7.5,
            }

        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake_with_affinity)
        targets = [{"pdb_id": "1A", "family": "x", "name": "Test"}]
        out = tmp_path / "repeat"
        summary = benchmark.run_repeat_docking(
            targets=targets,
            output_dir=str(out),
            n_repeats=2,
        )
        pt = summary["per_target"][0]
        assert pt["mean_affinity"] is not None
        assert pt["sd_affinity"] is not None


class TestRunRedockingBenchmarkEdgeCases:
    def test_ifp_and_cascade_message_paths(self, tmp_path, monkeypatch):
        def _fake_with_ifp(item):
            return {
                "pdb_id": item["target"]["pdb_id"],
                "family": "x",
                "success": True,
                "rmsd": 1.0,
                "success_raw": True,
                "rmsd_raw": 1.2,
                "best_rmsd": 0.9,
                "ifp_best_rmsd": 0.9,
                "ifp_best_pose_idx": 0,
                "ifp_best_score": 1.0,
                "cascade_ifp_success": True,
                "cascade_mmgbsa_success": False,
            }

        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake_with_ifp)
        out = tmp_path / "bench"
        summary = benchmark.run_redocking_benchmark(
            targets=[{"pdb_id": "1A", "family": "x"}],
            output_dir=str(out),
            n_workers=1,
            minimize=True,
        )
        assert summary["n_total"] == 1
        assert summary["ifp_rmsds"] == [0.9]
        assert len(summary["cascade_ifp_successes"]) == 1

    def test_csv_write_failure(self, tmp_path, monkeypatch):
        """Cover the OSError/TypeError branch when writing CSV."""

        def _fake(item):
            return {
                "pdb_id": item["target"]["pdb_id"],
                "family": "x",
                "success": True,
                "rmsd": 1.0,
            }

        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake)
        out = tmp_path / "bench"
        # Monkeypatch pd.DataFrame.to_csv to raise
        import pandas as pd

        original_to_csv = pd.DataFrame.to_csv

        def _broken_to_csv(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_csv", _broken_to_csv)
        summary = benchmark.run_redocking_benchmark(
            targets=[{"pdb_id": "1A", "family": "x"}],
            output_dir=str(out),
            n_workers=1,
        )
        assert summary.get("csv_path") is None
        monkeypatch.setattr(pd.DataFrame, "to_csv", original_to_csv)

    def test_no_median_rmsd_message(self, tmp_path, monkeypatch):
        def _fake(item):
            return {
                "pdb_id": item["target"]["pdb_id"],
                "family": "x",
                "success": False,
                "rmsd": None,
            }

        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake)
        out = tmp_path / "bench"
        summary = benchmark.run_redocking_benchmark(
            targets=[{"pdb_id": "1A", "family": "x"}],
            output_dir=str(out),
            n_workers=1,
        )
        assert summary["n_total"] == 1
        assert summary["n_success"] == 0

    def test_run_repeat_docking_defaults(self, tmp_path, monkeypatch):
        """Cover run_repeat_docking default target selection (1C5Z, 3EL8, 1T46)."""
        _calls = []

        def _fake(item):
            _calls.append(item["target"]["pdb_id"])
            return {
                "pdb_id": item["target"]["pdb_id"],
                "family": "x",
                "success": True,
                "rmsd": 1.0,
            }

        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake)
        out = tmp_path / "repeat"
        # n_repeats=0 should clamp to 1
        summary = benchmark.run_repeat_docking(
            targets=None,
            output_dir=str(out),
            n_repeats=0,
        )
        assert len(summary["per_target"]) == 3
        assert set(_calls) == {"1C5Z", "3EL8", "1T46"}

    def test_run_repeat_docking_repeat_clamping(self, tmp_path, monkeypatch):
        def _fake(item):
            return {
                "pdb_id": item["target"]["pdb_id"],
                "family": "x",
                "success": True,
                "rmsd": 1.0,
            }

        monkeypatch.setattr(benchmark, "_run_single_benchmark", _fake)
        targets = [{"pdb_id": "1A", "family": "x", "name": "Test"}]
        out = tmp_path / "repeat"
        # n_repeats larger than available seeds should clamp
        summary = benchmark.run_repeat_docking(
            targets=targets,
            output_dir=str(out),
            n_repeats=100,
        )
        pt = summary["per_target"][0]
        assert pt["n_total"] <= len(benchmark._REPEAT_SEEDS)


class TestRunSingleBenchmark:
    """Direct tests for _run_single_benchmark error paths."""

    @patch("autodock.utils.download_pdb")
    def test_download_failure(self, mock_dl, tmp_path):
        from autodock.core import StructureFetchError

        mock_dl.side_effect = StructureFetchError("no network")
        item = {
            "target": {"pdb_id": "1A", "family": "x", "name": "Test"},
            "output_dir": str(tmp_path / "out"),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is False
        assert "download" in result["error"]

    @patch("autodock.utils.download_pdb")
    def test_no_ligand_detected(self, mock_dl, tmp_path):
        outdir = tmp_path / "out"
        outdir.mkdir()
        holo = outdir / "1A.pdb"
        holo.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        item = {
            "target": {"pdb_id": "1A", "family": "x", "name": "Test"},
            "output_dir": str(outdir),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is False
        assert "No ligand detected" in result["error"]

    @patch("autodock.benchmark.run_redocking_validation")
    def test_auto_detect_ligand_success(self, mock_val, tmp_path):
        """Cover lines 697-700: auto-detect ligand returns a resname."""
        outdir = tmp_path / "out"
        outdir.mkdir()
        holo = outdir / "1A.pdb"
        holo.write_text("HETATM    1  C   XYZ A   1      0.000   0.000   0.000\n")
        mock_val.return_value = {"success": True, "rmsd": 1.0}
        # No ligand_resname and no chain_id → triggers auto-detect
        item = {
            "target": {"pdb_id": "1A", "family": "x", "name": "Test"},
            "output_dir": str(outdir),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is True

    @patch("autodock.benchmark.run_redocking_validation")
    def test_validation_error(self, mock_val, tmp_path):
        from autodock.core import ValidationError

        outdir = tmp_path / "out"
        outdir.mkdir()
        holo = outdir / "1A.pdb"
        holo.write_text("HETATM    1  C   LIG A   1      0.000   0.000   0.000\n")
        mock_val.side_effect = ValidationError("bad input")
        item = {
            "target": {
                "pdb_id": "1A",
                "family": "x",
                "name": "Test",
                "ligand_resname": "LIG",
            },
            "output_dir": str(outdir),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is False
        assert "bad input" in result["error"]

    @patch("autodock.benchmark.run_redocking_validation")
    def test_docking_error(self, mock_val, tmp_path):
        from autodock.core import DockingError

        outdir = tmp_path / "out"
        outdir.mkdir()
        holo = outdir / "1A.pdb"
        holo.write_text("HETATM    1  C   LIG A   1      0.000   0.000   0.000\n")
        mock_val.side_effect = DockingError("docking failed")
        item = {
            "target": {
                "pdb_id": "1A",
                "family": "x",
                "name": "Test",
                "ligand_resname": "LIG",
            },
            "output_dir": str(outdir),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is False
        assert "docking failed" in result["error"]

    @patch("autodock.benchmark.run_redocking_validation")
    def test_generic_exception(self, mock_val, tmp_path):
        outdir = tmp_path / "out"
        outdir.mkdir()
        holo = outdir / "1A.pdb"
        holo.write_text("HETATM    1  C   LIG A   1      0.000   0.000   0.000\n")
        mock_val.side_effect = RuntimeError("boom")
        item = {
            "target": {
                "pdb_id": "1A",
                "family": "x",
                "name": "Test",
                "ligand_resname": "LIG",
            },
            "output_dir": str(outdir),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is False
        assert "boom" in result["error"]

    @patch("autodock.benchmark.run_redocking_validation")
    def test_success_path(self, mock_val, tmp_path):
        outdir = tmp_path / "out"
        outdir.mkdir()
        holo = outdir / "1A.pdb"
        holo.write_text("HETATM    1  C   LIG A   1      0.000   0.000   0.000\n")
        mock_val.return_value = {
            "success": True,
            "rmsd": 1.2,
            "best_affinity": -8.5,
            "best_rmsd": 0.9,
            "best_rmsd_pose_idx": 0,
            "top_n_check": 3,
            "top_n_best_rmsd": 0.8,
            "top_n_best_pose_idx": 1,
            "top_n_success": True,
            "ifp_best_rmsd": None,
            "ifp_best_pose_idx": None,
            "ifp_best_score": None,
            "cascade": False,
            "cascade_results": {},
            "cascade_ifp_rmsd": None,
            "cascade_ifp_success": None,
            "cascade_mmgbsa_rmsd": None,
            "cascade_mmgbsa_success": None,
            "threshold": 2.0,
            "pocket_method": "crystal",
            "pocket_source": "crystal",
        }
        item = {
            "target": {
                "pdb_id": "1A",
                "family": "x",
                "name": "Test",
                "ligand_resname": "LIG",
            },
            "output_dir": str(outdir),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is True
        assert result["rmsd"] == 1.2
        assert result["best_affinity"] == -8.5

    @patch("autodock.utils.download_pdb")
    @patch("autodock.benchmark.run_redocking_validation")
    def test_cif_download_returned(self, mock_val, mock_dl, tmp_path):
        """Cover lines 680-681: when download_pdb returns a .cif file."""
        outdir = tmp_path / "out"
        outdir.mkdir()
        # Place cif outside outdir so os.path.exists(holo_cif) is False
        cif_path = tmp_path / "1A.cif"
        cif_path.write_text("data_1A\n")
        mock_dl.return_value = str(cif_path)
        mock_val.return_value = {"success": True, "rmsd": 1.0}
        item = {
            "target": {"pdb_id": "1A", "family": "x", "name": "Test", "ligand_resname": "LIG"},
            "output_dir": str(outdir),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is True

    @patch("autodock.benchmark.run_redocking_validation")
    def test_existing_cif_no_pdb(self, mock_val, tmp_path):
        """Cover lines 693, 699-700: existing .cif but no .pdb."""
        outdir = tmp_path / "out"
        outdir.mkdir()
        cif = outdir / "1A.cif"
        cif.write_text("data_1A\n")
        # Do NOT create 1A.pdb so the elif branch triggers
        mock_val.return_value = {"success": True, "rmsd": 1.0}
        item = {
            "target": {"pdb_id": "1A", "family": "x", "name": "Test", "ligand_resname": "LIG"},
            "output_dir": str(outdir),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is True

    @patch("autodock.benchmark.run_redocking_validation")
    def test_hard_target_override(self, mock_val, tmp_path):
        """Cover lines 738-741: pdb_id in HARD_TARGET_OVERRIDES."""
        outdir = tmp_path / "out"
        outdir.mkdir()
        holo = outdir / "1D4K.pdb"
        holo.write_text("HETATM    1  C   LIG A   1      0.000   0.000   0.000\n")
        mock_val.return_value = {"success": True, "rmsd": 1.0}
        item = {
            "target": {"pdb_id": "1D4K", "family": "x", "name": "Test", "ligand_resname": "LIG"},
            "output_dir": str(outdir),
            "exhaustiveness": 8,
            "n_poses": 9,
            "seed": 42,
            "skip_consensus": True,
            "minimize": False,
            "pocket_method": "crystal",
            "interaction_method": "plip",
            "auto_exhaustiveness": True,
            "top_n_check": 3,
            "use_flexible_receptor": False,
            "rescoring_methods": None,
            "cascade": False,
            "cascade_n_poses": 50,
            "remove_water": True,
            "remove_hetatms": True,
            "predict_pka": True,
            "fix_protonation": True,
            "cache_dir": None,
        }
        result = benchmark._run_single_benchmark(item)
        assert result["success"] is True
