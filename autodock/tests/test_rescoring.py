"""Tests for autodock.rescoring auxiliary scoring methods."""

from __future__ import annotations

import os
import tempfile

import pytest

from autodock.rescoring import combined_rescoring, select_best_by_method


@pytest.fixture
def mini_poses_path():
    content = """MODEL 1
REMARK VINA RESULT:      -6.5      0.000      0.000
ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   UNL     1       1.500   0.000   0.000  1.00  0.00     0.000 C
ENDMDL
MODEL 2
REMARK VINA RESULT:      -5.0      0.000      0.000
ATOM      1  C   UNL     1       0.100   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   UNL     1       1.600   0.000   0.000  1.00  0.00     0.000 C
ENDMDL
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def ref_path():
    content = """ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   UNL     1       1.500   0.000   0.000  1.00  0.00     0.000 C
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


class TestCombinedRescoring:
    def test_ifp_requires_receptor(self, mini_poses_path, ref_path):
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

    def test_empty_methods(self, mini_poses_path):
        results = combined_rescoring(mini_poses_path)
        assert results == {}


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
