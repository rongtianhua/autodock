"""Tests for autodock.analysis."""

from unittest.mock import MagicMock, patch

import pytest

from autodock import analysis


class TestExtractAffinity:
    """Tests for _extract_affinity."""

    def test_extracts_from_remark(self):
        text = "REMARK VINA RESULT:      -8.236      0.000      0.000"
        assert analysis._extract_affinity(text) == pytest.approx(-8.236)

    def test_returns_none_when_missing(self):
        assert analysis._extract_affinity("ATOM 1 C") is None

    def test_extracts_positive_affinity(self):
        text = "REMARK VINA RESULT:       2.500      0.000      0.000"
        assert analysis._extract_affinity(text) == pytest.approx(2.5)


class TestParseAllPoses:
    """Tests for _parse_all_poses."""

    def test_empty_file_returns_empty(self, tmp_path):
        pdbqt = tmp_path / "poses.pdbqt"
        pdbqt.write_text("")
        assert analysis._parse_all_poses(str(pdbqt)) == []

    def test_missing_file_returns_empty(self, tmp_path):
        assert analysis._parse_all_poses(str(tmp_path / "missing.pdbqt")) == []

    def test_single_pose_no_model(self, tmp_path):
        pdbqt = tmp_path / "poses.pdbqt"
        content = "REMARK VINA RESULT:      -7.500      0.000      0.000\nATOM 1 C\n"
        pdbqt.write_text(content)
        results = analysis._parse_all_poses(str(pdbqt))
        assert len(results) == 1
        assert results[0][0] == pytest.approx(-7.5)

    def test_multi_model_parsing(self, tmp_path):
        pdbqt = tmp_path / "poses.pdbqt"
        content = (
            "REMARK VINA RESULT:      -7.500      0.000      0.000\n"
            "MODEL 1\n"
            "REMARK VINA RESULT:      -8.000      0.000      0.000\n"
            "ATOM 1 C\n"
            "ENDMDL\n"
            "MODEL 2\n"
            "REMARK VINA RESULT:      -6.500      0.000      0.000\n"
            "ATOM 2 O\n"
            "ENDMDL\n"
        )
        pdbqt.write_text(content)
        results = analysis._parse_all_poses(str(pdbqt))
        assert len(results) == 2
        assert results[0][0] == pytest.approx(-8.0)
        assert results[1][0] == pytest.approx(-6.5)

    def test_skips_empty_models(self, tmp_path):
        pdbqt = tmp_path / "poses.pdbqt"
        content = (
            "MODEL 1\n"
            "REMARK VINA RESULT:      -8.000      0.000      0.000\n"
            "ATOM 1 C\n"
            "ENDMDL\n"
            "MODEL 2\n"
            "ENDMDL\n"
        )
        pdbqt.write_text(content)
        results = analysis._parse_all_poses(str(pdbqt))
        assert len(results) == 1


class TestAnalyzeScoringBias:
    """Tests for analyze_scoring_bias."""

    def test_skips_missing_files(self, tmp_path):
        out = tmp_path / "benchmark"
        out.mkdir()
        (out / "1ABC").mkdir()
        # No docking_all_poses.pdbqt or crystal_ligand.pdb
        with patch.object(analysis, "compute_rmsd_to_crystal"):
            results = analysis.analyze_scoring_bias(str(out), target_ids=["1ABC"])
        assert "1ABC" not in results

    def test_analyzes_valid_target(self, tmp_path):
        out = tmp_path / "benchmark"
        target_dir = out / "1ABC"
        target_dir.mkdir(parents=True)

        # Write all_poses file
        poses = target_dir / "docking_all_poses.pdbqt"
        content = (
            "MODEL 1\n"
            "REMARK VINA RESULT:      -8.000      0.000      0.000\n"
            "ATOM 1 C\n"
            "ENDMDL\n"
            "MODEL 2\n"
            "REMARK VINA RESULT:      -6.000      0.000      0.000\n"
            "ATOM 2 O\n"
            "ENDMDL\n"
        )
        poses.write_text(content)

        # Write crystal ligand
        crystal = target_dir / "crystal_ligand.pdb"
        crystal.write_text("ATOM 1 C\n")

        # Pre-import matplotlib.pyplot so patch() can find it

        with patch.object(analysis, "compute_rmsd_to_crystal", return_value=1.5):
            with patch("matplotlib.pyplot") as mock_plt:
                fig = MagicMock()
                ax = MagicMock()
                mock_plt.subplots.return_value = (fig, ax)
                results = analysis.analyze_scoring_bias(str(out), target_ids=["1ABC"])

        assert "1ABC" in results
        assert len(results["1ABC"]["poses"]) == 2
        assert results["1ABC"]["top1_affinity"] == pytest.approx(-8.0)
        assert results["1ABC"]["best_rmsd"] == pytest.approx(1.5)
        assert results["1ABC"]["top1_vs_best"] == pytest.approx(0.0)

    def test_matplotlib_unavailable_skips_plot(self, tmp_path):
        out = tmp_path / "benchmark"
        target_dir = out / "1ABC"
        target_dir.mkdir(parents=True)

        poses = target_dir / "docking_all_poses.pdbqt"
        poses.write_text(
            "MODEL 1\n"
            "REMARK VINA RESULT:      -8.000      0.000      0.000\n"
            "ATOM 1 C\n"
            "ENDMDL\n"
        )
        crystal = target_dir / "crystal_ligand.pdb"
        crystal.write_text("ATOM 1 C\n")

        with patch.object(analysis, "compute_rmsd_to_crystal", return_value=1.0):
            with patch.dict("sys.modules", {"matplotlib": None}):
                results = analysis.analyze_scoring_bias(str(out), target_ids=["1ABC"])
        assert "1ABC" in results
        assert "figure_path" not in results["1ABC"]
