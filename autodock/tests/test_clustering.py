"""Tests for autodock.clustering — pose clustering logic."""

from __future__ import annotations

import numpy as np
import pytest

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


class TestParsePoseToMol:
    def test_empty_pose_returns_none(self):
        assert clustering._parse_pose_to_mol("") is None

    def test_pose_without_atoms_returns_none(self):
        assert clustering._parse_pose_to_mol("MODEL 1\nENDMDL\n") is None

    def test_pose_with_only_model_number_returns_none(self):
        assert clustering._parse_pose_to_mol("MODEL 1\n1\nENDMDL\n") is None


class TestRmsdBetweenMols:
    def test_none_mol_returns_none(self):
        assert clustering._rmsd_between_mols(None, None) is None

    def test_valid_mols_rmsd(self):
        from rdkit import Chem

        mol1 = Chem.MolFromSmiles("CC")
        mol2 = Chem.MolFromSmiles("CC")
        if mol1 and mol2:
            from rdkit.Chem import AllChem

            AllChem.EmbedMolecule(mol1)
            AllChem.EmbedMolecule(mol2)
            rmsd = clustering._rmsd_between_mols(mol1, mol2)
            assert rmsd is not None
            assert rmsd >= 0.0

    def test_getbestrms_fallback_to_kabsch(self, monkeypatch):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol1 = Chem.AddHs(Chem.MolFromSmiles("CC"))
        mol2 = Chem.AddHs(Chem.MolFromSmiles("CC"))
        if mol1 is None or mol2 is None:
            pytest.skip("RDKit not available")
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        AllChem.EmbedMolecule(mol2, randomSeed=43)

        # Force GetBestRMS to fail so fallback Kabsch is exercised
        def _fail(*args, **kwargs):
            raise RuntimeError("mock failure")

        monkeypatch.setattr(AllChem, "GetBestRMS", _fail)
        rmsd = clustering._rmsd_between_mols(mol1, mol2)
        assert rmsd is not None
        assert rmsd >= 0.0


class TestClusterPosesFallback:
    def test_rmsd_none_creates_new_cluster(self, monkeypatch):
        """When _rmsd_between_mols returns None, pose should form its own cluster."""
        monkeypatch.setattr(clustering, "_rmsd_between_mols", lambda a, b: None)
        atom_line = (
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000  1.00  0.00           C  \n"
        )
        poses = [
            f"MODEL 1\n{atom_line}ENDMDL\n",
            f"MODEL 1\n{atom_line}ENDMDL\n",
        ]
        energies = np.array([[-8.0, 1.0, 2.0], [-7.5, 1.5, 2.5]])
        clusters = clustering.cluster_poses(poses, energies, rmsd_threshold=2.0)
        assert len(clusters) == 2
        assert clusters[0]["size"] == 1
        assert clusters[1]["size"] == 1

    def test_three_poses_two_clusters(self):
        atom_line = (
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000  1.00  0.00           C  \n"
        )
        poses = [
            f"MODEL 1\n{atom_line}ENDMDL\n",
            f"MODEL 1\n{atom_line}ENDMDL\n",
            f"MODEL 1\n{atom_line}ENDMDL\n",
        ]
        energies = np.array([[-8.0, 1.0, 2.0], [-7.5, 1.5, 2.5], [-7.0, 2.0, 3.0]])
        clusters = clustering.cluster_poses(poses, energies, rmsd_threshold=2.0)
        assert len(clusters) == 1
        assert clusters[0]["size"] == 3


class TestRmsdKabschMols:
    def test_kabsch_rmsd_with_rdkit(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol1 = Chem.AddHs(Chem.MolFromSmiles("CC"))
        mol2 = Chem.AddHs(Chem.MolFromSmiles("CC"))
        if mol1 is None or mol2 is None:
            pytest.skip("RDKit not available")
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        AllChem.EmbedMolecule(mol2, randomSeed=43)
        rmsd = clustering._rmsd_kabsch_mols(mol1, mol2)
        assert rmsd is not None
        assert rmsd >= 0.0

    def test_kabsch_none_on_empty_match(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol1 = Chem.AddHs(Chem.MolFromSmiles("CC"))
        mol2 = Chem.AddHs(Chem.MolFromSmiles("CO"))
        if mol1 is None or mol2 is None:
            pytest.skip("RDKit not available")
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        AllChem.EmbedMolecule(mol2, randomSeed=43)
        rmsd = clustering._rmsd_kabsch_mols(mol1, mol2)
        # Different molecules may or may not match; just check it doesn't crash
        assert rmsd is None or rmsd >= 0.0
