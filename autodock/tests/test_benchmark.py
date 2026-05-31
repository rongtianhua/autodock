"""Tests for autodock.benchmark (lightweight, no redocking)."""

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
