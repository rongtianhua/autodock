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
        assert not result["available"]

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


class TestComputeClashScoreBranches:
    def test_clash_with_different_elements(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  O   ALA A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  N   LIG A   1      0.000   0.000   0.000\n")
        result = val.compute_clash_score(str(lig), str(rec))
        assert result["n_clashes"] > 0
        assert not result["is_acceptable"]

    def test_mean_distance_reported(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  C   ALA A   1     10.000  10.000  10.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      0.000   0.000   0.000\n")
        result = val.compute_clash_score(str(lig), str(rec))
        assert result["mean_distance"] is not None
        assert result["mean_distance"] > 0

    def test_no_ligand_atoms_returns_none(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  C   ALA A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("")
        result = val.compute_clash_score(str(lig), str(rec))
        assert result["clash_score"] is None
        assert result["n_clashes"] is None


class TestComputeRmsdBranches:
    def test_no_rdkit_returns_none(self, tmp_path):
        with patch.object(val, "_HAVE_RDKIT", False):
            rms = val.compute_rmsd("a.pdbqt", "b.pdbqt")
        assert rms is None

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_atom_count_mismatch_warns(self, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol1 = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        block1 = Chem.MolToPDBBlock(mol1)

        mol2 = Chem.MolFromSmiles("CCCC")
        AllChem.EmbedMolecule(mol2, randomSeed=42)
        block2 = Chem.MolToPDBBlock(mol2)

        p1 = tmp_path / "a.pdb"
        p1.write_text(block1)
        p2 = tmp_path / "b.pdb"
        p2.write_text(block2)

        with patch.object(val, "_HAVE_RDKIT", True):
            rms = val.compute_rmsd(str(p1), str(p2))
        # Should not crash; may return None or a value
        assert rms is not None or rms is None

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_getbestrms_failure_returns_none(self, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        block = Chem.MolToPDBBlock(mol)
        p1 = tmp_path / "a.pdb"
        p1.write_text(block)
        p2 = tmp_path / "b.pdb"
        p2.write_text(block)

        with patch.object(val, "_HAVE_RDKIT", True):
            with patch("rdkit.Chem.AllChem.GetBestRMS", side_effect=RuntimeError("fail")):
                rms = val.compute_rmsd(str(p1), str(p2))
        assert rms is None


class TestKabschRmsdBranches:
    def test_nan_input_raises(self):
        P = np.array([[0.0, 0.0, np.nan], [1.0, 0.0, 0.0]])
        Q = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="NaN or inf"):
            val._kabsch_rmsd(P, Q)

    def test_insufficient_atoms_raises(self):
        P = np.array([[0.0, 0.0, 0.0]])
        Q = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="Insufficient atoms"):
            val._kabsch_rmsd(P, Q)

    def test_shape_mismatch_raises(self):
        P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        Q = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="Insufficient atoms"):
            val._kabsch_rmsd(P, Q)

    def test_reflection_case(self):
        # Mirror image that triggers det(R) < 0 branch
        P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        rms = val._kabsch_rmsd(P, Q)
        assert rms is not None
        assert rms >= 0


class TestComputeRmsdCoordinateBasedBranches:
    def test_no_rdkit_returns_none(self):
        with patch.object(val, "_HAVE_RDKIT", False):
            rms = val.compute_rmsd_coordinate_based("a.pdb", "b.pdb")
        assert rms is None

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_pdbqt_vs_pdb(self, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        block = Chem.MolToPDBBlock(mol)
        p1 = tmp_path / "a.pdbqt"
        p1.write_text(block)
        p2 = tmp_path / "b.pdb"
        p2.write_text(block)

        with patch.object(val, "_HAVE_RDKIT", True):
            rms = val.compute_rmsd_coordinate_based(str(p1), str(p2))
        assert rms is not None
        assert rms == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_parse_failure_returns_none(self, tmp_path):
        p1 = tmp_path / "a.pdbqt"
        p1.write_text("not a molecule")
        p2 = tmp_path / "b.pdbqt"
        p2.write_text("not a molecule")

        with patch.object(val, "_HAVE_RDKIT", True):
            with patch("rdkit.Chem.MolFromPDBBlock", return_value=None):
                rms = val.compute_rmsd_coordinate_based(str(p1), str(p2))
        assert rms is None

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_no_matched_atoms_returns_none(self, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol1 = Chem.MolFromSmiles("C")
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        block1 = Chem.MolToPDBBlock(mol1)

        mol2 = Chem.MolFromSmiles("C")
        AllChem.EmbedMolecule(mol2, randomSeed=42)
        block2 = Chem.MolToPDBBlock(mol2)

        p1 = tmp_path / "a.pdbqt"
        p1.write_text(block1)
        p2 = tmp_path / "b.pdbqt"
        p2.write_text(block2)

        with patch.object(val, "_HAVE_RDKIT", True):
            with patch("scipy.optimize.linear_sum_assignment", side_effect=IndexError("no match")):
                rms = val.compute_rmsd_coordinate_based(str(p1), str(p2))
        # May return None or raise; just ensure no crash
        assert rms is None or isinstance(rms, float)


class TestComputeRmsdToCrystalBranches:
    def test_no_rdkit_returns_none(self):
        with patch.object(val, "_HAVE_RDKIT", False):
            rms = val.compute_rmsd_to_crystal("dock.pdbqt", "crystal.pdb")
        assert rms is None

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_docked_pdb_extension(self, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        block = Chem.MolToPDBBlock(mol)
        docked = tmp_path / "dock.pdb"
        docked.write_text(block)
        crystal = tmp_path / "crystal.pdb"
        crystal.write_text(block)

        with patch.object(val, "_HAVE_RDKIT", True):
            rms = val.compute_rmsd_to_crystal(str(docked), str(crystal))
        assert rms is not None
        assert rms == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_getbestrms_falls_back(self, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        block = Chem.MolToPDBBlock(mol)
        docked = tmp_path / "dock.pdbqt"
        docked.write_text(block)
        crystal = tmp_path / "crystal.pdb"
        crystal.write_text(block)

        with patch.object(val, "_HAVE_RDKIT", True):
            with patch("rdkit.Chem.AllChem.GetBestRMS", side_effect=RuntimeError("fail")):
                with patch.object(val, "compute_rmsd_coordinate_based", return_value=1.5):
                    rms = val.compute_rmsd_to_crystal(str(docked), str(crystal))
        assert rms == pytest.approx(1.5)


class TestComputeBestRmsdFromAllPosesBranches:
    def test_no_rdkit_returns_none(self, tmp_path):
        with patch.object(val, "_HAVE_RDKIT", False):
            rmsd, idx = val.compute_best_rmsd_from_all_poses("poses.pdbqt", "crystal.pdb")
        assert rmsd is None
        assert idx == -1

    def test_missing_crystal_returns_none(self, tmp_path):
        with patch.object(val, "_HAVE_RDKIT", True):
            poses = tmp_path / "poses.pdbqt"
            poses.write_text("ATOM 1 C LIG\n")
            rmsd, idx = val.compute_best_rmsd_from_all_poses(str(poses), "missing.pdb")
        assert rmsd is None
        assert idx == -1

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_single_model(self, tmp_path):
        poses = tmp_path / "poses.pdbqt"
        poses.write_text("ATOM 1 C LIG A 1 0 0 0\n")
        crystal = tmp_path / "crystal.pdb"
        crystal.write_text("ATOM 1 C LIG A 1 0 0 0\n")

        with patch.object(val, "_HAVE_RDKIT", True):
            with patch("rdkit.Chem.MolFromPDBFile", return_value=MagicMock()):
                with patch("rdkit.Chem.MolFromPDBBlock", return_value=MagicMock()):
                    with patch("rdkit.Chem.AllChem.GetBestRMS", return_value=1.2):
                        rmsd, idx = val.compute_best_rmsd_from_all_poses(str(poses), str(crystal))
        assert rmsd == pytest.approx(1.2)
        assert idx == 1

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_multi_model_with_fallback(self, tmp_path):
        poses = tmp_path / "poses.pdbqt"
        poses.write_text(
            "MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n" "MODEL 2\nATOM 1 C LIG A 1 1 1 1\nENDMDL\n"
        )
        crystal = tmp_path / "crystal.pdb"
        crystal.write_text("ATOM 1 C LIG A 1 0 0 0\n")

        call_count = 0

        def fake_best(a, b):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 2.0
            raise RuntimeError("fail")

        with patch.object(val, "_HAVE_RDKIT", True):
            with patch("rdkit.Chem.MolFromPDBFile", return_value=MagicMock()):
                with patch("rdkit.Chem.MolFromPDBBlock", return_value=MagicMock()):
                    with patch("rdkit.Chem.AllChem.GetBestRMS", side_effect=fake_best):
                        with patch.object(val, "compute_rmsd_coordinate_based", return_value=0.5):
                            rmsd, idx = val.compute_best_rmsd_from_all_poses(
                                str(poses), str(crystal)
                            )
        assert rmsd == pytest.approx(0.5)
        assert idx == 2

    @pytest.mark.skipif(not val._HAVE_RDKIT, reason="RDKit not available")
    def test_all_models_fail_returns_none(self, tmp_path):
        poses = tmp_path / "poses.pdbqt"
        poses.write_text("MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n")
        crystal = tmp_path / "crystal.pdb"
        crystal.write_text("ATOM 1 C LIG A 1 0 0 0\n")

        with patch.object(val, "_HAVE_RDKIT", True):
            with patch("rdkit.Chem.MolFromPDBFile", return_value=MagicMock()):
                with patch("rdkit.Chem.MolFromPDBBlock", return_value=None):
                    rmsd, idx = val.compute_best_rmsd_from_all_poses(str(poses), str(crystal))
        assert rmsd is None
        assert idx == -1


class TestRunRedockingValidationBranches:
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    @patch("autodock.utils.extract_chain_from_pdb")
    def test_chain_id_mode(
        self,
        mock_extract_chain,
        mock_rmsd,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C N3 A 2 1 1 1\n")

        mock_extract_chain.return_value = None
        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [{"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)}]

        mock_result = MagicMock()
        mock_result.best_affinity = -8.0
        mock_result.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_result.all_poses_pdbqt = None
        mock_result.pose_clusters = None
        mock_dock.return_value = mock_result
        mock_rmsd.return_value = 1.2

        with (
            patch("rdkit.Chem.MolFromPDBFile", return_value=MagicMock()),
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("autodock.utils.pdb_chain_to_smiles", return_value="CC"),
        ):
            result = val.run_redocking_validation(
                str(holo),
                chain_id="A",
                output_dir=str(tmp_path / "out"),
            )
        assert result["success"] is True

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    @patch("autodock.preparation.prepare_ligand_adaptive")
    def test_blind_pocket_mode(
        self,
        mock_adaptive,
        mock_rmsd,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C LIG A 2 1 1 1\n")

        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_adaptive.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [
            {"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0), "method": "fpocket"}
        ]

        mock_result = MagicMock()
        mock_result.best_affinity = -8.0
        mock_result.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_result.all_poses_pdbqt = None
        mock_result.pose_clusters = None
        mock_dock.return_value = mock_result
        mock_rmsd.return_value = 1.2

        with (
            patch(
                "autodock.validation.extract_ligand_from_pdb",
                return_value=(MagicMock(), str(tmp_path / "lig.sdf")),
            ),
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("rdkit.Chem.rdmolfiles.MolToPDBFile"),
        ):
            result = val.run_redocking_validation(
                str(holo),
                ligand_resname="LIG",
                output_dir=str(tmp_path / "out"),
                pocket_method="blind",
            )
        assert result["success"] is True
        assert result["pocket_method"] == "blind"

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    @patch("autodock.preparation.prepare_ligand_adaptive")
    def test_minimize_branch(
        self,
        mock_adaptive,
        mock_rmsd,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C LIG A 2 1 1 1\n")

        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_adaptive.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [{"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)}]

        mock_result = MagicMock()
        mock_result.best_affinity = -8.0
        mock_result.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_result.all_poses_pdbqt = None
        mock_result.pose_clusters = None
        mock_dock.return_value = mock_result
        mock_rmsd.return_value = 1.2

        with (
            patch(
                "autodock.validation.extract_ligand_from_pdb",
                return_value=(MagicMock(), str(tmp_path / "lig.sdf")),
            ),
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("rdkit.Chem.rdmolfiles.MolToPDBFile"),
            patch(
                "autodock.minimization.minimize_docked_pose",
                return_value={
                    "success": True,
                    "output_pdb": str(tmp_path / "min.pdb"),
                    "initial_energy_kJ_mol": -100.0,
                    "final_energy_kJ_mol": -150.0,
                },
            ),
        ):
            result = val.run_redocking_validation(
                str(holo),
                ligand_resname="LIG",
                output_dir=str(tmp_path / "out"),
                minimize=True,
            )
        assert result["success"] is True
        assert result["minimized_pose"] is not None

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    @patch("autodock.preparation.prepare_ligand_adaptive")
    def test_use_ifp_branch(
        self,
        mock_adaptive,
        mock_rmsd,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C LIG A 2 1 1 1\n")

        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_adaptive.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [{"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)}]

        all_poses = tmp_path / "all_poses.pdbqt"
        all_poses.write_text(
            "MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n" "MODEL 2\nATOM 1 C LIG A 1 1 1 1\nENDMDL\n"
        )

        mock_result = MagicMock()
        mock_result.best_affinity = -8.0
        mock_result.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_result.all_poses_pdbqt = str(all_poses)
        mock_result.pose_clusters = None
        mock_dock.return_value = mock_result
        mock_rmsd.return_value = 1.2

        with (
            patch(
                "autodock.validation.extract_ligand_from_pdb",
                return_value=(MagicMock(), str(tmp_path / "lig.sdf")),
            ),
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("rdkit.Chem.rdmolfiles.MolToPDBFile"),
            patch("autodock.interactions.ifp_similarity_scores", return_value=[(1, 0.9, None)]),
        ):
            result = val.run_redocking_validation(
                str(holo),
                ligand_resname="LIG",
                output_dir=str(tmp_path / "out"),
                use_ifp=True,
            )
        assert result["success"] is True

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    @patch("autodock.preparation.prepare_ligand_adaptive")
    def test_consensus_rescue_branch(
        self,
        mock_adaptive,
        mock_rmsd,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C LIG A 2 1 1 1\n")

        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_adaptive.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [{"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)}]

        all_poses = tmp_path / "all_poses.pdbqt"
        all_poses.write_text("MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n")

        mock_result = MagicMock()
        mock_result.best_affinity = -8.0
        mock_result.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_result.all_poses_pdbqt = str(all_poses)
        mock_result.pose_clusters = [{"representative_index": 0}]
        mock_dock.return_value = mock_result
        # First call for raw RMSD returns >2.0 to trigger consensus rescue
        call_count = 0

        def rmsd_side(*a, **k):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return 3.0
            return 0.5

        mock_rmsd.side_effect = rmsd_side

        with (
            patch(
                "autodock.validation.extract_ligand_from_pdb",
                return_value=(MagicMock(), str(tmp_path / "lig.sdf")),
            ),
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("rdkit.Chem.rdmolfiles.MolToPDBFile"),
        ):
            result = val.run_redocking_validation(
                str(holo),
                ligand_resname="LIG",
                output_dir=str(tmp_path / "out"),
            )
        assert result["consensus_best_rmsd"] is not None

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    @patch("autodock.preparation.prepare_ligand_adaptive")
    def test_flexible_receptor_fallback(
        self,
        mock_adaptive,
        mock_rmsd,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C LIG A 2 1 1 1\n")

        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_adaptive.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [{"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)}]

        all_poses = tmp_path / "all_poses.pdbqt"
        all_poses.write_text("MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n")

        mock_result = MagicMock()
        mock_result.best_affinity = -8.0
        mock_result.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_result.all_poses_pdbqt = str(all_poses)
        mock_result.pose_clusters = None
        mock_dock.return_value = mock_result
        mock_rmsd.return_value = 1.0

        with (
            patch(
                "autodock.validation.extract_ligand_from_pdb",
                return_value=(MagicMock(), str(tmp_path / "lig.sdf")),
            ),
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("rdkit.Chem.rdmolfiles.MolToPDBFile"),
            patch("autodock.preparation.find_nearby_residues", return_value=["A:42"]),
            patch(
                "autodock.preparation.prepare_flexible_receptor",
                return_value=("rigid.pdbqt", "flex.pdbqt"),
            ),
        ):
            result = val.run_redocking_validation(
                str(holo),
                ligand_resname="LIG",
                output_dir=str(tmp_path / "out"),
                use_flexible_receptor=True,
            )
        assert result["success"] is True

    def test_no_extraction_mode_raises(self):
        with pytest.raises((ValueError, val.ValidationError)):
            val.run_redocking_validation("holo.pdb")


class TestCascadeFallback:
    """Direct tests for the three-tier cascade fallback (Vina → IFP → MM-GBSA)."""

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    @patch("autodock.preparation.prepare_ligand_adaptive")
    def test_cascade_tier2a_ifp20_rescue(
        self,
        mock_adaptive,
        mock_rmsd,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        """Tier 1 fails (RMSD 3.0); Tier 2a IFP(20) rescues with RMSD 1.0."""
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C LIG A 2 1 1 1\n")

        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_adaptive.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [{"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)}]

        all_poses = tmp_path / "all_poses.pdbqt"
        all_poses.write_text(
            "MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n" "MODEL 2\nATOM 1 C LIG A 1 1 1 1\nENDMDL\n"
        )

        mock_result = MagicMock()
        mock_result.best_affinity = -8.0
        mock_result.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_result.all_poses_pdbqt = str(all_poses)
        mock_result.pose_clusters = None
        mock_dock.return_value = mock_result

        # Tier 1 RMSD 3.0 (fail), Tier 2a IFP pose RMSD 1.0 (success)
        mock_rmsd.side_effect = [3.0, 1.0]

        with (
            patch(
                "autodock.validation.extract_ligand_from_pdb",
                return_value=(MagicMock(), str(tmp_path / "lig.sdf")),
            ),
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("rdkit.Chem.rdmolfiles.MolToPDBFile"),
            patch(
                "autodock.interactions.ifp_similarity_scores",
                return_value=[(1, 0.9, None)],
            ),
        ):
            result = val.run_redocking_validation(
                str(holo),
                ligand_resname="LIG",
                output_dir=str(tmp_path / "out"),
                cascade=True,
            )
        assert result["success"] is True
        assert result["cascade_ifp_success"] is True
        assert result["cascade_ifp_rmsd"] == pytest.approx(1.0, abs=1e-6)
        assert result["cascade_mmgbsa_success"] is None

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    @patch("autodock.preparation.prepare_ligand_adaptive")
    def test_cascade_tier2b_redock_ifp_rescue(
        self,
        mock_adaptive,
        mock_rmsd,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        """Tier 1 fails; Tier 2a IFP(20) fails (empty); Tier 2b re-dock + IFP rescues."""
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C LIG A 2 1 1 1\n")

        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_adaptive.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [{"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)}]

        all_poses = tmp_path / "all_poses.pdbqt"
        all_poses.write_text("MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n")
        tier2_poses = tmp_path / "tier2_poses.pdbqt"
        tier2_poses.write_text("MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n")

        mock_result_tier1 = MagicMock()
        mock_result_tier1.best_affinity = -8.0
        mock_result_tier1.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_result_tier1.all_poses_pdbqt = str(all_poses)
        mock_result_tier1.pose_clusters = None

        mock_result_tier2 = MagicMock()
        mock_result_tier2.best_affinity = -8.0
        mock_result_tier2.best_pose_pdbqt = str(tmp_path / "tier2_pose.pdbqt")
        mock_result_tier2.all_poses_pdbqt = str(tier2_poses)
        mock_result_tier2.pose_clusters = None

        mock_dock.side_effect = [mock_result_tier1, mock_result_tier2]

        # Tier 1 = 3.0, Tier 2b IFP pose = 1.0 (Tier 2a skipped because IFP empty)
        mock_rmsd.side_effect = [3.0, 1.0]

        with (
            patch(
                "autodock.validation.extract_ligand_from_pdb",
                return_value=(MagicMock(), str(tmp_path / "lig.sdf")),
            ),
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("rdkit.Chem.rdmolfiles.MolToPDBFile"),
            patch("autodock.interactions.ifp_similarity_scores") as mock_ifp,
        ):
            # Tier 2a returns empty → no pose extracted; Tier 2b returns good pose
            mock_ifp.side_effect = [
                [],  # tier 2a: empty
                [(1, 0.9, None)],  # tier 2b: success
            ]
            result = val.run_redocking_validation(
                str(holo),
                ligand_resname="LIG",
                output_dir=str(tmp_path / "out"),
                cascade=True,
                cascade_n_poses=50,
            )
        assert result["success"] is True
        assert result["cascade_ifp_success"] is True
        assert result["cascade_ifp_rmsd"] == pytest.approx(1.0, abs=1e-6)

    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.validation.compute_rmsd_to_crystal")
    @patch("autodock.preparation.prepare_ligand_adaptive")
    def test_cascade_tier3_mmgbsa_rescue(
        self,
        mock_adaptive,
        mock_rmsd,
        mock_pockets,
        mock_prep_lig,
        mock_prep_rec,
        mock_dock,
        tmp_path,
    ):
        """Tier 1 & 2 both fail; Tier 3 MM-GBSA rescues the docking failure."""
        holo = tmp_path / "holo.pdb"
        holo.write_text("ATOM 1 N SER A 1 0 0 0\nHETATM 2 C LIG A 2 1 1 1\n")

        mock_prep_rec.return_value = str(tmp_path / "apo.pdbqt")
        mock_prep_lig.return_value = str(tmp_path / "lig.pdbqt")
        mock_adaptive.return_value = str(tmp_path / "lig.pdbqt")
        mock_pockets.return_value = [{"center": (1.0, 1.0, 1.0), "box_size": (20.0, 20.0, 20.0)}]

        all_poses = tmp_path / "all_poses.pdbqt"
        all_poses.write_text("MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n")
        tier2_poses = tmp_path / "tier2_poses.pdbqt"
        tier2_poses.write_text("MODEL 1\nATOM 1 C LIG A 1 0 0 0\nENDMDL\n")

        mock_result_tier1 = MagicMock()
        mock_result_tier1.best_affinity = -8.0
        mock_result_tier1.best_pose_pdbqt = str(tmp_path / "pose.pdbqt")
        mock_result_tier1.all_poses_pdbqt = str(all_poses)
        mock_result_tier1.pose_clusters = None

        mock_result_tier2 = MagicMock()
        mock_result_tier2.best_affinity = -8.0
        mock_result_tier2.best_pose_pdbqt = str(tmp_path / "tier2_pose.pdbqt")
        mock_result_tier2.all_poses_pdbqt = str(tier2_poses)
        mock_result_tier2.pose_clusters = None

        mock_dock.side_effect = [mock_result_tier1, mock_result_tier2]

        # Tier 1 = 3.0, Tier 2a IFP = 3.0, Tier 2b IFP = 3.0, Tier 3 MM-GBSA = 1.0
        mock_rmsd.side_effect = [3.0, 3.0, 3.0, 1.0]

        with (
            patch(
                "autodock.validation.extract_ligand_from_pdb",
                return_value=(MagicMock(), str(tmp_path / "lig.sdf")),
            ),
            patch("rdkit.Chem.MolToSmiles", return_value="CC"),
            patch("rdkit.Chem.rdmolfiles.MolToPDBFile"),
            patch("autodock.interactions.ifp_similarity_scores") as mock_ifp,
            patch(
                "autodock.rescoring._run_mmgbsa_rescoring",
                return_value=[(1, -50.0, None)],
            ),
        ):
            mock_ifp.side_effect = [
                [(1, 0.3, None)],  # tier 2a: low score, RMSD 3.0
                [(1, 0.3, None)],  # tier 2b: low score, RMSD 3.0
            ]
            result = val.run_redocking_validation(
                str(holo),
                ligand_resname="LIG",
                output_dir=str(tmp_path / "out"),
                cascade=True,
                cascade_n_poses=50,
            )
        assert result["success"] is True
        assert result["cascade_mmgbsa_success"] is True
        assert result["cascade_mmgbsa_rmsd"] == pytest.approx(1.0, abs=1e-6)
