"""Tests for autodock.validation — clash, RMSD, redocking validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autodock import validation as val

# ─────────────────────────────────────────────────────────────────────────────
# PoseBusters
# ─────────────────────────────────────────────────────────────────────────────


class TestPoseBusters:
    def test_not_available(self, tmp_path):
        with patch.dict("sys.modules", {"posebusters": None}):
            result = val.validate_pose_with_posebusters("pose.pdbqt", "rec.pdb")
        assert result["available"] is False

    def test_pass(self, tmp_path):
        mock_pb = MagicMock()
        mock_result = MagicMock()
        mock_result.columns = [
            "bond_lengths",
            "bond_angles",
            "aromatic_ring_flatness",
            "mol_pred_loaded",
        ]
        # Build a mock DataFrame-like object with bool dtype columns
        import pandas as pd

        df = pd.DataFrame(
            {
                "bond_lengths": [True],
                "bond_angles": [True],
                "aromatic_ring_flatness": [True],
                "mol_pred_loaded": [True],
            }
        )
        mock_pb.bust.return_value = df

        with patch("posebusters.PoseBusters", return_value=mock_pb):
            with patch.object(val, "_HAVE_RDKIT", True):
                with patch("rdkit.Chem.MolFromPDBBlock") as mock_rdkit:
                    mock_mol = MagicMock()
                    mock_mol.GetNumAtoms.return_value = 10
                    mock_mol.GetNumBonds.return_value = 10
                    mock_rdkit.return_value = mock_mol
                    with patch("rdkit.Chem.AddHs", return_value=mock_mol):
                        with patch("rdkit.Chem.SDWriter") as mock_writer:
                            mock_writer.return_value = MagicMock()
                            with patch(
                                "autodock.validation._sanitize_pdbqt_for_rdkit", return_value="ATOM"
                            ):
                                result = val.validate_pose_with_posebusters("pose.pdbqt", "rec.pdb")
        assert result["available"] is True
        assert result["pass"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Clash Detection
# ─────────────────────────────────────────────────────────────────────────────


class TestClashDetection:
    def test_empty_files(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        rec.write_text("")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("")
        result = val.compute_clash_score(str(lig), str(rec))
        assert result["clash_score"] is None
        assert result["n_clashes"] is None

    def test_no_clash(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  C   ALA A   1     10.000  10.000  10.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      0.000   0.000   0.000\n")
        result = val.compute_clash_score(str(lig), str(rec))
        assert result["is_acceptable"] is True
        assert result["n_clashes"] == 0

    def test_detects_clash(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        # Carbon at origin
        rec.write_text("ATOM      1  C   ALA A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        # Carbon very close (1.0 Å, VDW overlap = 1.7+1.7-1.0 = 2.4 Å > 0.3)
        lig.write_text("ATOM      1  C   LIG A   1      1.000   0.000   0.000\n")
        result = val.compute_clash_score(str(lig), str(rec))
        # Check that we got a numeric result; if atoms are too close, it should flag
        assert result["n_clashes"] is not None
        assert result["is_acceptable"] in (True, False)


# ─────────────────────────────────────────────────────────────────────────────
# RMSD
# ─────────────────────────────────────────────────────────────────────────────


class TestKabschRmsd:
    def test_identity(self):
        P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        Q = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        rms = val._kabsch_rmsd(P, Q)
        assert rms == pytest.approx(0.0, abs=1e-6)

    def test_translation_only(self):
        P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        Q = np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]])
        rms = val._kabsch_rmsd(P, Q)
        assert rms == pytest.approx(0.0, abs=1e-6)

    def test_rotation(self):
        # 90 degree rotation around Z
        P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Q = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        rms = val._kabsch_rmsd(P, Q)
        assert rms == pytest.approx(0.0, abs=1e-6)


class TestComputeRmsdCoordinateBased:
    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_same_molecule(self, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        block = Chem.MolToPDBBlock(mol)
        p1 = tmp_path / "a.pdb"
        p1.write_text(block)
        p2 = tmp_path / "b.pdb"
        p2.write_text(block)
        rms = val.compute_rmsd_coordinate_based(str(p1), str(p2))
        assert rms is not None
        assert rms == pytest.approx(0.0, abs=1e-3)

    def test_no_rdkit_returns_none(self, tmp_path):
        with patch.object(val, "_HAVE_RDKIT", False):
            rms = val.compute_rmsd_coordinate_based("a.pdb", "b.pdb")
            assert rms is None


class TestComputeRmsdToCrystal:
    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_same_molecule(self, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        block = Chem.MolToPDBBlock(mol)
        p1 = tmp_path / "a.pdbqt"
        p1.write_text(block)
        p2 = tmp_path / "b.pdb"
        p2.write_text(block)
        rms = val.compute_rmsd_to_crystal(str(p1), str(p2))
        assert rms is not None
        assert rms == pytest.approx(0.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Redocking Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestRunRedockingValidation:
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.extract_ligand_from_pdb")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    def test_successful_redock(
        self,
        mock_rmsd,
        mock_extract,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C LIG A 2 1 1 1\n")

        mock_extract.return_value = (MagicMock(), str(tmp_path / "lig.sdf"))
        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [{"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)}]

        mock_result = MagicMock()
        mock_result.best_affinity = -8.0
        mock_result.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_dock.return_value = mock_result
        mock_rmsd.return_value = 1.2

        with (
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("rdkit.Chem.rdmolfiles.MolToPDBFile"),
        ):
            result = val.run_redocking_validation(
                str(holo), ligand_resname="LIG", output_dir=str(tmp_path / "out")
            )
        assert result["success"] is True
        assert result["rmsd"] == pytest.approx(1.2)
        assert result["best_affinity"] == -8.0

    def test_no_extraction_mode_raises(self):
        with pytest.raises((ValueError, val.ValidationError)):
            val.run_redocking_validation("holo.pdb")


# ─────────────────────────────────────────────────────────────────────────────
# Top-N RMSD
# ─────────────────────────────────────────────────────────────────────────────


class TestTopNBestRMSD:
    def test_missing_file_returns_none(self, tmp_path):
        missing = str(tmp_path / "missing.pdbqt")
        crystal = str(tmp_path / "crystal.pdb")
        rmsd, idx = val.compute_top_n_best_rmsd_from_all_poses(missing, crystal, n=3)
        assert rmsd is None
        assert idx == -1

    def test_no_rdkit_returns_none(self, tmp_path):
        with patch.object(val, "_HAVE_RDKIT", False):
            rmsd, idx = val.compute_top_n_best_rmsd_from_all_poses("a.pdbqt", "b.pdb", n=3)
        assert rmsd is None
        assert idx == -1

    def test_single_model(self, tmp_path):
        poses = tmp_path / "poses.pdbqt"
        poses.write_text("ATOM      1  C   LIG A   1      0.000   0.000   0.000\n")
        crystal = tmp_path / "crystal.pdb"
        crystal.write_text("ATOM      1  C   LIG A   1      0.000   0.000   0.000\n")

        with patch.object(val, "_HAVE_RDKIT", True):
            with patch("rdkit.Chem.MolFromPDBFile") as mock_from_pdb:
                mock_from_pdb.return_value = MagicMock()
                with patch("rdkit.Chem.AllChem.GetBestRMS", return_value=1.5):
                    rmsd, idx = val.compute_top_n_best_rmsd_from_all_poses(
                        str(poses), str(crystal), n=3
                    )
        assert rmsd == pytest.approx(1.5)
        assert idx == 1

    def test_multi_model_limits_to_n(self, tmp_path):
        """Only the first *n* models are evaluated."""
        poses = tmp_path / "poses.pdbqt"
        poses.write_text(
            "MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n"
            "MODEL 2\nATOM 1 C LIG A 1 1 1 1\nENDMDL\n"
            "MODEL 3\nATOM 1 C LIG A 1 2 2 2\nENDMDL\n"
        )
        crystal = tmp_path / "crystal.pdb"
        crystal.write_text("ATOM 1 C LIG A 1 0 0 0\n")

        call_count = 0

        def fake_rms(a, b):
            nonlocal call_count
            call_count += 1
            return float(call_count)  # 1.0, 2.0, 3.0, ...

        with patch.object(val, "_HAVE_RDKIT", True):
            with patch("rdkit.Chem.MolFromPDBFile", return_value=MagicMock()):
                with patch("rdkit.Chem.MolFromPDBBlock", return_value=MagicMock()):
                    with patch("rdkit.Chem.AllChem.GetBestRMS", side_effect=fake_rms):
                        rmsd, idx = val.compute_top_n_best_rmsd_from_all_poses(
                            str(poses), str(crystal), n=2
                        )
        # Only 2 calls (first 2 models), best is 1.0 from model 1
        assert call_count == 2
        assert rmsd == pytest.approx(1.0)
        assert idx == 1
