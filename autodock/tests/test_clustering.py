"""Tests for autodock.clustering — pose clustering logic."""

from __future__ import annotations

import numpy as np

from autodock import clustering


class TestClusterPoses:
    def test_empty_input(self):
        assert clustering.cluster_poses([], np.array([])) == []

    def test_single_pose(self):
        poses = ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"]
        energies = np.array([[-8.0, 1.0, 2.0]])
        clusters = clustering.cluster_poses(poses, energies, rmsd_threshold=2.0)
        assert len(clusters) == 1
        assert clusters[0]["size"] == 1
        assert clusters[0]["representative_energy"] == -8.0

    def test_two_identical_poses_one_cluster(self):
        # Identical poses should fall into the same cluster
        atom_line = (
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000  1.00  0.00           C  \n"
        )
        poses = [
            f"MODEL 1\n{atom_line}ENDMDL\n",
            f"MODEL 1\n{atom_line}ENDMDL\n",
        ]
        energies = np.array([[-8.0, 1.0, 2.0], [-7.5, 1.5, 2.5]])
        clusters = clustering.cluster_poses(poses, energies, rmsd_threshold=2.0)
        assert len(clusters) == 1
        assert clusters[0]["size"] == 2

    def test_two_distant_poses_two_clusters(self):
        # Use 2 different atoms with different internal geometry so RMSD > 0 even after alignment
        p1 = (
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000  1.00  0.00           C  \n"
            "ATOM      2  N   LIG A   1      1.000   0.000   0.000  1.00  0.00           N  \n"
        )
        p2 = (
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000  1.00  0.00           C  \n"
            "ATOM      2  N   LIG A   1      5.000   0.000   0.000  1.00  0.00           N  \n"
        )
        poses = [
            f"MODEL 1\n{p1}ENDMDL\n",
            f"MODEL 1\n{p2}ENDMDL\n",
        ]
        energies = np.array([[-8.0, 1.0, 2.0], [-7.5, 1.5, 2.5]])
        clusters = clustering.cluster_poses(poses, energies, rmsd_threshold=2.0)
        assert len(clusters) == 2
        assert clusters[0]["size"] == 1
        assert clusters[1]["size"] == 1

    def test_representative_is_lowest_energy(self):
        atom_line = (
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000  1.00  0.00           C  \n"
        )
        poses = [
            f"MODEL 1\n{atom_line}ENDMDL\n",
            f"MODEL 1\n{atom_line}ENDMDL\n",
        ]
        energies = np.array([[-7.0, 1.0, 2.0], [-8.0, 1.5, 2.5]])
        clusters = clustering.cluster_poses(poses, energies, rmsd_threshold=2.0)
        # Representative should be pose 1 (energy -8.0), even though it was docked second
        assert clusters[0]["representative_energy"] == -8.0

    def test_mismatch_lengths_warning(self, caplog):
        poses = ["MODEL 1\nATOM 1 C LIG\nENDMDL\n"]
        energies = np.array([[-8.0, 1.0, 2.0], [-7.0, 1.5, 2.5]])
        with caplog.at_level("WARNING"):
            clusters = clustering.cluster_poses(poses, energies, rmsd_threshold=2.0)
        assert len(clusters) == 1
        assert "mismatch" in caplog.text
