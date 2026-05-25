"""Tests for autodock.preparation — receptor/ligand prep and pocket detection."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from autodock import preparation as prep
from autodock.core import PreparationError


# ─────────────────────────────────────────────────────────────────────────────
# Receptor Preparation
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareReceptor:
    def test_missing_file_raises(self):
        with pytest.raises(PreparationError, match="not found"):
            prep.prepare_receptor("/nonexistent/file.pdb", "out.pdbqt")

    def test_cif_without_gemmi_raises(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.write_text("data_test\n")
        with patch.dict("sys.modules", {"gemmi": None}):
            with pytest.raises(PreparationError, match="gemmi"):
                prep.prepare_receptor(str(cif), "out.pdbqt")

    @patch("autodock.preparation.find_conda_tool")
    @patch("autodock.preparation.safe_subprocess")
    def test_filter_waters_and_hetatms(self, mock_subprocess, mock_find, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2  O   HOH A   2      1.000   1.000   1.000\n"
            "HETATM    3  O   SO4 A   3      2.000   2.000   2.000\n"
        )
        out = tmp_path / "rec.pdbqt"

        # Mock meeko success
        mock_polymer = MagicMock()
        mock_templates = MagicMock()
        mock_mk = MagicMock()
        with patch("meeko.ResidueChemTemplates") as mock_tmpl_cls, \
             patch("meeko.MoleculePreparation") as mock_mk_cls, \
             patch("meeko.Polymer") as mock_poly_cls, \
             patch("meeko.PDBQTWriterLegacy") as mock_writer:
            mock_tmpl_cls.create_from_defaults.return_value = mock_templates
            mock_mk_cls.return_value = mock_mk
            mock_poly_cls.from_pdb_string.return_value = mock_polymer
            mock_writer.write_from_polymer.return_value = ("REMARK  mock\nATOM 1 N", None)

            result = prep.prepare_receptor(str(pdb), str(out), remove_water=True, remove_hetatms=True)
            assert out.exists()
            content = out.read_text()
            assert "ATOM" in content

    def test_keep_residues_filter(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "ATOM      2  CA  ALA A   2      1.000   1.000   1.000\n"
            "ATOM      3  C   GLY A   3      2.000   2.000   2.000\n"
        )
        out = tmp_path / "rec.pdbqt"
        with patch("meeko.ResidueChemTemplates"), \
             patch("meeko.MoleculePreparation"), \
             patch("meeko.Polymer") as mock_poly_cls, \
             patch("meeko.PDBQTWriterLegacy") as mock_writer:
            mock_poly_cls.from_pdb_string.return_value = MagicMock()
            mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
            prep.prepare_receptor(str(pdb), str(out), keep_residues={"SER"})
            # Should not raise; Polymer called with filtered content
            assert mock_poly_cls.from_pdb_string.called


# ─────────────────────────────────────────────────────────────────────────────
# Ligand Preparation
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareLigand:
    def test_invalid_smiles_raises(self, tmp_path):
        out = tmp_path / "lig.pdbqt"
        with pytest.raises(PreparationError, match="parse SMILES"):
            prep.prepare_ligand("NOT_A_SMILES!!!", str(out))

    @patch("rdkit.Chem.MolFromSmiles")
    @patch("rdkit.Chem.AddHs")
    @patch("rdkit.Chem.AllChem.ETKDGv3")
    @patch("rdkit.Chem.AllChem.EmbedMolecule")
    @patch("rdkit.Chem.AllChem.MMFFOptimizeMolecule")
    @patch("rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges")
    @patch("meeko.MoleculePreparation")
    @patch("meeko.PDBQTWriterLegacy")
    def test_prepare_ligand_mock(self, mock_writer_cls, mock_mk_cls, mock_charges, mock_mmff, mock_embed, mock_etkdg, mock_addhs, mock_molfrom, tmp_path):
        mock_mol = MagicMock()
        mock_molfrom.return_value = mock_mol
        mock_addhs.return_value = mock_mol
        mock_embed.return_value = 0
        mock_mmff.return_value = None

        mock_mk = MagicMock()
        mock_mk_cls.return_value = mock_mk
        mock_setup = MagicMock()
        mock_mk.prepare.return_value = mock_setup

        mock_writer_cls.write_string.return_value = ("ATOM 1 C LIG\n", True, "")

        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand("CCO", str(out), name="LIG", seed=42)
        assert out.exists()
        assert result == str(out.resolve())


class TestPrepareLigandConformers:
    @patch("autodock.preparation.prepare_ligand")
    def test_generates_n_conformers(self, mock_prep, tmp_path):
        mock_prep.return_value = "dummy.pdbqt"
        outdir = tmp_path / "conformers"
        paths = prep.prepare_ligand_conformers("CCO", str(outdir), n_conformers=5, seed_start=10)
        assert len(paths) == 5
        assert mock_prep.call_count == 5
        # Check seeds are sequential
        seeds = [call.kwargs.get("seed") or call.args[2] for call in mock_prep.call_args_list]
        assert seeds == [10, 11, 12, 13, 14]


# ─────────────────────────────────────────────────────────────────────────────
# Pocket Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBoxSize:
    def test_basic(self):
        assert prep._compute_box_size((10.0, 10.0, 10.0), padding=5.0) == (20.0, 20.0, 20.0)

    def test_minimum_10A(self):
        # Small pockets: 1.0 + 2*5 = 11.0 after padding and rounding
        assert prep._compute_box_size((1.0, 1.0, 1.0)) == (11.0, 11.0, 11.0)

    def test_rounding(self):
        # 13.2 + 10 = 23.2 -> rounds to 23.0 (nearest 0.5)
        result = prep._compute_box_size((13.2, 13.2, 13.2), padding=5.0)
        assert all(r == 23.0 for r in result)


class TestPreparePdbForFpocket:
    def test_strips_water(self, tmp_path):
        inp = tmp_path / "in.pdb"
        inp.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2  O   HOH A   2      1.000   1.000   1.000\n"
        )
        out = tmp_path / "out.pdb"
        prep._prepare_pdb_for_fpocket(str(inp), str(out))
        lines = out.read_text().splitlines()
        assert len(lines) == 1
        assert "SER" in lines[0]


class TestParseFpocketInfo:
    def test_parse_typical(self, tmp_path):
        info = tmp_path / "test_info.txt"
        info.write_text(
            "Pocket 1 :\n"
            "Druggability Score : 0.75\n"
            "Volume : 450.5\n"
            "Depth : 8.2\n"
            "Number of mouth openings : 3\n"
            "Number of apolar alpha sphere : 12\n"
            "Number of polar alpha sphere : 8\n"
        )
        # Create corresponding PQR so center/dims are computed
        pqr = tmp_path / "pocket1_vert.pqr"
        pqr.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000\n"
            "ATOM      2  C   LIG A   1      4.000   5.000   6.000\n"
        )
        pockets = prep._parse_fpocket_info(str(info))
        assert len(pockets) == 1
        assert pockets[0]["num"] == 1
        assert pockets[0]["druggability"] == 0.75
        assert pockets[0]["volume"] == 450.5
        assert pockets[0]["depth"] == 8.2
        assert pockets[0]["openings"] == 3

    def test_missing_file(self):
        assert prep._parse_fpocket_info("/nonexistent/info.txt") == []

    def test_no_pockets(self, tmp_path):
        info = tmp_path / "empty_info.txt"
        info.write_text("No pockets found\n")
        assert prep._parse_fpocket_info(str(info)) == []


class TestFindTopPockets:
    def test_reference_ligand_path(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdb"
        lig.write_text("HETATM    1  C   LIG A   1      5.000   5.000   5.000  1.00  0.00           C\n")

        pockets = prep.find_top_pockets(str(pdb), ligand_pdb=str(lig))
        assert len(pockets) >= 1
        assert pockets[0]["center"] == (5.0, 5.0, 5.0)
