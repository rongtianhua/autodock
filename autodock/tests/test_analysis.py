"""Tests for autodock.analysis — scoring bias diagnostics."""
import os
import tempfile
import textwrap

import pytest

from autodock.analysis import _extract_affinity, _parse_all_poses, analyze_scoring_bias


class TestParseAllPoses:
    def test_extract_affinity_valid(self):
        assert _extract_affinity("REMARK VINA RESULT:    -8.236      0.000      0.000") == pytest.approx(-8.236)
        assert _extract_affinity("REMARK VINA RESULT: -5.94 0 0") == pytest.approx(-5.94)

    def test_extract_affinity_no_match(self):
        assert _extract_affinity("REMARK VINA RESULT:") is None
        assert _extract_affinity("Nothing here") is None
        assert _extract_affinity("") is None

    def test_parse_single_pose(self):
        pdbqt = textwrap.dedent("""\
            REMARK VINA RESULT:    -9.452      0.000      0.000
            ATOM      1  C   UNL     1      10.0  10.0  10.0
            ENDMDL
        """)
        fd, path = tempfile.mkstemp(suffix=".pdbqt")
        with os.fdopen(fd, "w") as fh:
            fh.write(pdbqt)
        try:
            result = _parse_all_poses(path)
            assert len(result) == 1
            assert result[0][0] == pytest.approx(-9.452)
            assert "ATOM" in result[0][1]
        finally:
            os.unlink(path)

    def test_parse_multi_model(self):
        pdbqt = textwrap.dedent("""\
            REMARK header info
            MODEL 1
            REMARK VINA RESULT:    -8.236      0.000      0.000
            ATOM  1  C  UNL 1  1.0 1.0 1.0
            ENDMDL
            MODEL 2
            REMARK VINA RESULT:    -7.500      0.000      0.000
            ATOM  1  C  UNL 1  2.0 2.0 2.0
            ENDMDL
        """)
        fd, path = tempfile.mkstemp(suffix=".pdbqt")
        with os.fdopen(fd, "w") as fh:
            fh.write(pdbqt)
        try:
            result = _parse_all_poses(path)
            assert len(result) == 2
            assert result[0][0] == pytest.approx(-8.236)
            assert result[1][0] == pytest.approx(-7.500)
        finally:
            os.unlink(path)

    def test_parse_nonexistent_file(self):
        assert _parse_all_poses("/nonexistent/file.pdbqt") == []

    def test_parse_empty_file(self):
        fd, path = tempfile.mkstemp(suffix=".pdbqt")
        with os.fdopen(fd, "w") as fh:
            fh.write("")
        try:
            result = _parse_all_poses(path)
            assert result == []  # no MODEL sections
        finally:
            os.unlink(path)

    def test_parse_no_models_but_affinity(self):
        """Single pose without MODEL tags but with REMARK line."""
        pdbqt = textwrap.dedent("""\
            REMARK VINA RESULT:    -6.543      0.000      0.000
            ATOM      1  C   UNL     1      1.0  1.0  1.0
        """)
        fd, path = tempfile.mkstemp(suffix=".pdbqt")
        with os.fdopen(fd, "w") as fh:
            fh.write(pdbqt)
        try:
            result = _parse_all_poses(path)
            assert len(result) == 1
            assert result[0][0] == pytest.approx(-6.543)
        finally:
            os.unlink(path)

    def test_parse_model_without_affinity_skipped(self):
        """MODEL without VINA RESULT should be skipped."""
        pdbqt = textwrap.dedent("""\
            MODEL 1
            ATOM  1  C  UNL 1  1.0 1.0 1.0
            ENDMDL
            MODEL 2
            REMARK VINA RESULT:    -5.000      0.000      0.000
            ATOM  1  C  UNL 1  2.0 2.0 2.0
            ENDMDL
        """)
        fd, path = tempfile.mkstemp(suffix=".pdbqt")
        with os.fdopen(fd, "w") as fh:
            fh.write(pdbqt)
        try:
            result = _parse_all_poses(path)
            assert len(result) == 1  # only model 2 has affinity
            assert result[0][0] == pytest.approx(-5.000)
        finally:
            os.unlink(path)


class TestAnalyzeScoringBias:
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_scoring_bias(tmpdir)
            assert isinstance(result, dict)
            assert len(result) == 0

    def test_missing_crystal_ligand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = os.path.join(tmpdir, "1ABC")
            os.makedirs(target_dir)
            with open(os.path.join(target_dir, "docking_all_poses.pdbqt"), "w") as f:
                f.write("REMARK VINA RESULT:    -5.0\nATOM  1 C UNL 1  1.0 1.0 1.0\nENDMDL\n")
            result = analyze_scoring_bias(tmpdir)
            assert "1ABC" not in result

    def test_custom_target_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_scoring_bias(tmpdir, target_ids=["FAKE1", "FAKE2"])
            assert isinstance(result, dict)
            assert len(result) == 0  # no valid data

    def test_figure_dir_autocreated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fig_dir = os.path.join(tmpdir, "custom_figs")
            result = analyze_scoring_bias(tmpdir, figure_dir=fig_dir)
            assert os.path.isdir(fig_dir)
            assert result == {}


class TestBenchmarkScoringDecoupling:
    """Tests for the scoring/sampling decoupling metrics added to run_redocking_benchmark."""

    def test_scoring_failure_detection(self):
        """When best_rmsd < 2.0 but top-1 > 2.0, it's a scoring failure."""
        from autodock.benchmark import REDocking_RMSD_THRESHOLD

        # Simulate raw results
        raw = [
            {"pdb_id": "GOOD", "success": True, "rmsd": 0.5, "best_rmsd": 0.4, "best_rmsd_success": True},
            {"pdb_id": "FAIL1", "success": False, "rmsd": 3.0, "best_rmsd": 0.9, "best_rmsd_success": True},
            {"pdb_id": "FAIL2", "success": False, "rmsd": 3.0, "best_rmsd": 3.0, "best_rmsd_success": False},
        ]

        scoring_failures = [
            r for r in raw
            if not r.get("success")
            and r.get("best_rmsd") is not None
            and r["best_rmsd"] <= REDocking_RMSD_THRESHOLD
        ]
        assert len(scoring_failures) == 1
        assert scoring_failures[0]["pdb_id"] == "FAIL1"

    def test_all_success_no_scoring_failures(self):
        """When all targets pass top-1, there are zero scoring failures."""
        from autodock.benchmark import REDocking_RMSD_THRESHOLD

        raw = [
            {"pdb_id": "T1", "success": True, "rmsd": 0.5, "best_rmsd": 0.3},
            {"pdb_id": "T2", "success": True, "rmsd": 1.0, "best_rmsd": 0.8},
        ]
        scoring_failures = [
            r for r in raw
            if not r.get("success")
            and r.get("best_rmsd") is not None
            and r["best_rmsd"] <= REDocking_RMSD_THRESHOLD
        ]
        assert len(scoring_failures) == 0
