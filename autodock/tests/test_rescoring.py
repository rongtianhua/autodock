"""Tests for autodock.rescoring auxiliary scoring methods."""

from __future__ import annotations

import os
import tempfile

import pytest

from autodock.rescoring import (
    _split_poses,
    combined_rescoring,
    select_best_by_method,
    shape_similarity_scores,
    strain_energy_scores,
)

# Minimal multi-MODEL PDBQT for unit tests
_MINI_POSES_PDBQT = """MODEL 1
REMARK VINA RESULT:      -6.5      0.000      0.000
ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   UNL     1       1.500   0.000   0.000  1.00  0.00     0.000 C
ENDMDL
MODEL 2
REMARK VINA RESULT:      -5.0      0.000      0.000
ATOM      1  C   UNL     1       0.100   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   UNL     1       1.600   0.000   0.000  1.00  0.00     0.000 C
ENDMDL
MODEL 3
REMARK VINA RESULT:      -4.0      0.000      0.000
ATOM      1  C   UNL     1       5.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   UNL     1       6.500   0.000   0.000  1.00  0.00     0.000 C
ENDMDL
"""

_REF_PDBQT = """ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   UNL     1       1.500   0.000   0.000  1.00  0.00     0.000 C
"""


@pytest.fixture
def mini_poses_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
        f.write(_MINI_POSES_PDBQT)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def ref_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
        f.write(_REF_PDBQT)
        path = f.name
    yield path
    os.unlink(path)


class TestSplitPoses:
    def test_splits_correctly(self, mini_poses_path):
        poses = _split_poses(mini_poses_path)
        assert len(poses) == 3
        # 1-based indexing
        assert poses[0][0] == 1
        assert poses[1][0] == 2
        assert poses[2][0] == 3

    def test_extracts_vina_energy(self, mini_poses_path):
        poses = _split_poses(mini_poses_path)
        assert poses[0][2] == pytest.approx(-6.5)


class TestShapeSimilarityScores:
    @pytest.mark.requires_rdkit
    def test_shape_scores_returned(self, mini_poses_path, ref_path):
        scores = shape_similarity_scores(mini_poses_path, ref_path)
        assert len(scores) == 3
        # Pose 1 is identical to reference → highest similarity
        assert scores[0][0] == 1
        assert scores[0][1] > 0.9
        # Pose 2 is slightly shifted → lower but still high similarity
        assert scores[1][0] == 2
        assert scores[1][1] > 0.5
        # Pose 3 is far away → lowest similarity
        assert scores[-1][0] == 3
        assert scores[-1][1] < 0.5

    def test_missing_reference_warns(self, mini_poses_path):
        scores = shape_similarity_scores(mini_poses_path, "/nonexistent/path.pdbqt")
        assert scores == []


class TestStrainEnergyScores:
    @pytest.mark.requires_rdkit
    def test_strain_scores_returned(self, mini_poses_path):
        scores = strain_energy_scores(mini_poses_path)
        assert len(scores) == 3
        # All identical geometries → similar strain
        for _, strain, energy in scores:
            assert strain > 0
            assert energy is not None


class TestCombinedRescoring:
    @pytest.mark.requires_rdkit
    def test_combined_runs_multiple_methods(self, mini_poses_path, ref_path):
        results = combined_rescoring(
            mini_poses_path,
            reference_pdbqt=ref_path,
            methods=["shape", "strain"],
        )
        assert "shape" in results
        assert "strain" in results
        assert len(results["shape"]) == 3
        assert len(results["strain"]) == 3

    def test_ifp_requires_receptor(self, mini_poses_path, ref_path):
        # Without receptor_pdb, IFP should be skipped
        results = combined_rescoring(
            mini_poses_path,
            reference_pdbqt=ref_path,
            methods=["ifp"],
        )
        assert "ifp" not in results

    def test_unknown_method_warns(self, mini_poses_path, ref_path):
        results = combined_rescoring(
            mini_poses_path,
            reference_pdbqt=ref_path,
            methods=["unknown_method"],
        )
        assert "unknown_method" not in results


class TestSelectBestByMethod:
    def test_max_method(self):
        scores = [(1, 0.9, -5.0), (2, 0.8, -4.0), (3, 0.7, -6.0)]
        best = select_best_by_method(scores, method="max")
        assert best == (1, 0.9)

    def test_min_method(self):
        scores = [(1, 10.0, -5.0), (2, 5.0, -4.0), (3, 15.0, -6.0)]
        best = select_best_by_method(scores, method="min")
        assert best == (2, 5.0)

    def test_empty_returns_none(self):
        assert select_best_by_method([]) is None
