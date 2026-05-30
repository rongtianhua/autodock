"""Tests for autodock.preparation — receptor/ligand prep and pocket detection."""

from __future__ import annotations

from pathlib import Path
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
        out = tmp_path / "out.pdbqt"
        with patch.dict("sys.modules", {"gemmi": None}):
            with pytest.raises(PreparationError, match="gemmi"):
                prep.prepare_receptor(str(cif), str(out))

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
        with (
            patch("meeko.ResidueChemTemplates") as mock_tmpl_cls,
            patch("meeko.MoleculePreparation") as mock_mk_cls,
            patch("meeko.Polymer") as mock_poly_cls,
            patch("meeko.PDBQTWriterLegacy") as mock_writer,
        ):
            mock_tmpl_cls.create_from_defaults.return_value = mock_templates
            mock_mk_cls.return_value = mock_mk
            mock_poly_cls.from_pdb_string.return_value = mock_polymer
            mock_writer.write_from_polymer.return_value = ("REMARK  mock\nATOM 1 N", None)

            prep.prepare_receptor(str(pdb), str(out), remove_water=True, remove_hetatms=True)
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
        with (
            patch("meeko.ResidueChemTemplates"),
            patch("meeko.MoleculePreparation"),
            patch("meeko.Polymer") as mock_poly_cls,
            patch("meeko.PDBQTWriterLegacy") as mock_writer,
        ):
            mock_poly_cls.from_pdb_string.return_value = MagicMock()
            mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
            prep.prepare_receptor(str(pdb), str(out), keep_residues={"SER"})
            # Should not raise; Polymer called with filtered content
            assert mock_poly_cls.from_pdb_string.called

    def test_keep_waters_near_metal(self, tmp_path):
        """Functional waters coordinating metal ions are retained."""
        pdb = tmp_path / "rec.pdb"
        # Zn at origin; water1 at 2.0 Å (should retain); water2 at 5.0 Å (should remove)
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2 ZN   ZN  A 100      0.000   0.000   0.000\n"
            "HETATM    3  O   HOH A 101      2.000   0.000   0.000\n"
            "HETATM    4  O   HOH A 102      5.000   0.000   0.000\n"
        )
        out = tmp_path / "rec.pdbqt"
        with (
            patch("meeko.ResidueChemTemplates"),
            patch("meeko.MoleculePreparation"),
            patch("meeko.Polymer") as mock_poly_cls,
            patch("meeko.PDBQTWriterLegacy") as mock_writer,
        ):
            mock_poly_cls.from_pdb_string.return_value = MagicMock()
            mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
            prep.prepare_receptor(
                str(pdb),
                str(out),
                remove_water=True,
                remove_hetatms=False,
                keep_waters_near_metal=True,
            )
            call_args = mock_poly_cls.from_pdb_string.call_args
            pdb_content = call_args[0][0]
            assert "HOH A 101" in pdb_content
            assert "HOH A 102" not in pdb_content
            assert "ZN  A 100" in pdb_content

    def test_output_report_json(self, tmp_path):
        """JSON report is written when output_report_json is provided."""
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        out = tmp_path / "rec.pdbqt"
        report = tmp_path / "report.json"
        with (
            patch("meeko.ResidueChemTemplates"),
            patch("meeko.MoleculePreparation"),
            patch("meeko.Polymer") as mock_poly_cls,
            patch("meeko.PDBQTWriterLegacy") as mock_writer,
        ):
            mock_poly_cls.from_pdb_string.return_value = MagicMock()
            mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
            prep.prepare_receptor(str(pdb), str(out), output_report_json=str(report))
            assert report.exists()
            import json

            data = json.loads(report.read_text())
            assert data["input_file"] == str(pdb)
            assert data["output_pdbqt"] == str(out)
            assert "parameters" in data
            assert data["parameters"]["ph"] == 7.4


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
    @patch("rdkit.Chem.AllChem.EmbedMultipleConfs")
    @patch("rdkit.Chem.AllChem.MMFFOptimizeMoleculeConfs")
    @patch("rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges")
    @patch("meeko.MoleculePreparation")
    @patch("meeko.PDBQTWriterLegacy")
    @patch("rdkit.Chem.Conformer")
    @patch("rdkit.Chem.Mol")
    @patch("rdkit.Geometry")
    def test_prepare_ligand_mock(
        self,
        mock_geo,
        mock_mol_cls,
        mock_conformer_cls,
        mock_writer_cls,
        mock_mk_cls,
        mock_charges,
        mock_mmff_confs,
        mock_embed,
        mock_etkdg,
        mock_addhs,
        mock_molfrom,
        tmp_path,
    ):
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_molfrom.return_value = mock_mol
        mock_mol_copy = MagicMock()
        mock_mol_copy.GetNumAtoms.return_value = 3
        mock_mol_cls.return_value = mock_mol_copy
        mock_addhs.return_value = mock_mol
        mock_embed.return_value = [0]
        mock_mmff_confs.return_value = [(0, -10.0)]
        mock_conf = MagicMock()
        mock_mol.GetConformer.return_value = mock_conf
        mock_conf.GetAtomPosition.side_effect = lambda i: type(
            "pos", (), {"x": float(i) * 1.0, "y": 0.0, "z": 0.0}
        )()

        mock_mk = MagicMock()
        mock_mk_cls.return_value = mock_mk
        mock_setup = MagicMock()
        mock_mk.prepare.return_value = mock_setup

        mock_writer_cls.write_string.return_value = ("ATOM 1 C LIG\n", True, "")

        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand(
            "CCO", str(out), name="LIG", seed=42, molscrub_states=False, enumerate_stereo=False
        )
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


class TestPrepareLigandObabelFallback:
    @patch("autodock.preparation.obabel_convert")
    @patch("meeko.MoleculePreparation")
    @patch("rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges")
    def test_meeko_charge_error_fallback_to_obabel(
        self,
        mock_charges,
        mock_mk_cls,
        mock_obabel,
        tmp_path,
    ):
        """If Meeko raises a charge error, prepare_ligand falls back to Open Babel."""
        mock_mk = MagicMock()
        mock_mk_cls.return_value = mock_mk
        mock_mk.prepare.side_effect = Exception("atom number 0 has non finite charge, charge: nan")

        def obabel_side_effect(smi, out_pdbqt, **kwargs):
            Path(out_pdbqt).write_text("REMARK  obabel\nATOM 1 C LIG\n")
            return True

        mock_obabel.side_effect = obabel_side_effect

        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand(
            "CCO", str(out), name="LIG", seed=42, molscrub_states=False, enumerate_stereo=False
        )
        assert out.exists()
        assert "obabel" in out.read_text()
        assert result == str(out.resolve())


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
        # write_pdb_atoms appends an END record
        assert len(lines) == 2
        assert "SER" in lines[0]
        assert lines[1] == "END"


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
        lig.write_text(
            "HETATM    1  C   LIG A   1      5.000   5.000   5.000  1.00  0.00           C\n"
        )

        pockets = prep.find_top_pockets(str(pdb), ligand_pdb=str(lig))
        assert len(pockets) >= 1
        assert pockets[0]["center"] == (5.0, 5.0, 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive / Multi-conformer Ligand Preparation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not __import__("autodock.core", fromlist=["_HAVE_RDKIT"])._HAVE_RDKIT,
    reason="RDKit not available",
)
class TestClassifyLigandComplexity:
    def test_ethanol_is_simple(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        assert prep._classify_ligand_complexity(mol) == "simple"

    def test_ibuprofen_is_simple(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
        assert prep._classify_ligand_complexity(mol) == "simple"

    def test_oseltamivir_is_medium(self):
        """2HU4 ligand G39 — multiple chiral centers."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCC(CC)O[C@@H]1CC(C(O)O)C[C@H](N)[C@H]1NC(C)O")
        assert prep._classify_ligand_complexity(mol) == "medium"

    def test_large_ppar_ligand_is_complex(self):
        """1GWX ligand 433 — large, flexible, many rings."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles(
            "CC(C)(OC1CCC(CCCN(CCC2C(Cl)CCC[C@@H]2F)[C@@H](O)NC2CCCC(Cl)C2Cl)CC1)C(O)O"
        )
        assert prep._classify_ligand_complexity(mol) == "complex"


@pytest.mark.skipif(
    not __import__("autodock.core", fromlist=["_HAVE_RDKIT"])._HAVE_RDKIT,
    reason="RDKit not available",
)
class TestPrepareLigandAdaptive:
    def test_simple_ligand_returns_single_path(self, tmp_path):
        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand_adaptive("CCO", str(out), strategy="simple", seed=42)
        assert isinstance(result, str)
        assert Path(result).exists()

    def test_medium_ligand_returns_multiple_paths(self, tmp_path):
        out_dir = tmp_path / "conformers"
        result = prep.prepare_ligand_adaptive(
            "CCC(CC)O[C@@H]1CC(C(O)O)C[C@H](N)[C@H]1NC(C)O",
            str(out_dir),
            strategy="medium",
            seed=42,
        )
        assert isinstance(result, list)
        assert len(result) >= 2
        for p in result:
            assert Path(p).exists()

    def test_auto_detects_simple(self, tmp_path):
        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand_adaptive("CCO", str(out), seed=42)
        assert isinstance(result, str)

    def test_auto_detects_medium(self, tmp_path):
        out_dir = tmp_path / "conformers"
        result = prep.prepare_ligand_adaptive(
            "CCC(CC)O[C@@H]1CC(C(O)O)C[C@H](N)[C@H]1NC(C)O",
            str(out_dir),
            seed=42,
        )
        assert isinstance(result, list)
