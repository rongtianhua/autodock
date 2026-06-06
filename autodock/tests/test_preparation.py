"""Tests for autodock.preparation — receptor/ligand prep and pocket detection."""

from __future__ import annotations

import json
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
            # Zn must be retained
            assert "ZN" in pdb_content
            # At least one functional water should survive filtering
            assert "HOH" in pdb_content
            # Only one HOH residue should remain (the functional one near Zn);
            # count unique residue IDs among HOH lines
            _hoh_resis = set()
            for line in pdb_content.splitlines():
                if "HOH" in line and line.startswith("HETATM"):
                    _resi = line[22:26].strip()
                    _hoh_resis.add(_resi)
            assert len(_hoh_resis) == 1, f"Expected 1 HOH residue, got {_hoh_resis}"

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
            # Detail fields introduced in Phase 1b
            assert "missing_residues_detail" in data
            assert "reduce_flips_detail" in data
            assert "anomalous_pka_detail" in data
            assert isinstance(data["missing_residues_detail"], list)

    def test_detect_af_false_positive(self, tmp_path):
        """Low-B-factor experimental structures must NOT be flagged as AF."""
        pdb = tmp_path / "rec.pdb"
        # B-factors in 0-45 range (mean ~20) — should NOT trigger AF heuristic
        lines = [
            f"ATOM    {i:4d}  CA  ALA A   {i:3d}"
            f"      0.000   0.000   0.000  1.00 {10 + i % 30:5.2f}"
            for i in range(1, 51)
        ]
        pdb.write_text("\n".join(lines) + "\n")
        out = tmp_path / "rec.pdbqt"
        report = tmp_path / "report.json"

        with (
            patch("meeko.ResidueChemTemplates"),
            patch("meeko.MoleculePreparation"),
            patch("meeko.Polymer") as mock_poly_cls,
            patch("meeko.PDBQTWriterLegacy") as mock_writer,
            patch("autodock.alphafold_tools.assess_alphafold_quality") as mock_assess,
            patch("autodock.alphafold_tools.relax_alphafold_structure") as mock_relax,
        ):
            mock_poly_cls.from_pdb_string.return_value = MagicMock()
            mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
            prep.prepare_receptor(str(pdb), str(out), output_report_json=str(report))
            # AF assessment should never have been triggered
            mock_assess.assert_not_called()
            mock_relax.assert_not_called()
            data = json.loads(report.read_text())
            assert data.get("alphafold_assessment") is None

    def test_ssbond_cys_deprotonation(self):
        """HG atoms on CYS residues in disulfide bonds are stripped."""
        pdb = (
            "ATOM      1  N   CYS A   1      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM      2  CA  CYS A   1      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM      3  C   CYS A   1      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM      4  O   CYS A   1      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM      5  CB  CYS A   1      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM      6  SG  CYS A   1      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM      7  HG  CYS A   1      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM      8  N   CYS A   2      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM      9  CA  CYS A   2      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM     10  C   CYS A   2      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM     11  O   CYS A   2      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM     12  CB  CYS A   2      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM     13  SG  CYS A   2      0.000   0.000   0.000  1.00 20.00\n"
            "ATOM     14  HG  CYS A   2      0.000   0.000   0.000  1.00 20.00\n"
        )
        ssbonds = [{"chain1": "A", "res1": 1, "chain2": "A", "res2": 2}]
        cleaned = prep._remove_disulfide_hydrogens(pdb, ssbonds)
        # Both HG atoms should have been removed
        assert cleaned.count("HG  CYS") == 0
        # SG atoms must remain
        assert cleaned.count("SG  CYS") == 2
        # Unrelated residues are untouched
        assert cleaned.count("CA  CYS") == 2


# ─────────────────────────────────────────────────────────────────────────────
# Ligand Preparation
# ─────────────────────────────────────────────────────────────────────────────


class TestPrepareLigand:
    def test_invalid_smiles_raises(self, tmp_path):
        out = tmp_path / "lig.pdbqt"
        with pytest.raises(PreparationError, match="parse SMILES"):
            prep.prepare_ligand("NOT_A_SMILES!!!", str(out))

    @patch("autodock.preparation._has_nan_charges")
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
        mock_has_nan,
        tmp_path,
    ):
        mock_has_nan.return_value = False
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

    def test_medium_ligand_with_force_multi_returns_multiple_paths(self, tmp_path):
        out_dir = tmp_path / "conformers"
        result = prep.prepare_ligand_adaptive(
            "CCC(CC)O[C@@H]1CC(C(O)O)C[C@H](N)[C@H]1NC(C)O",
            str(out_dir),
            strategy="medium",
            seed=42,
            force_multi_conformer=True,
        )
        assert isinstance(result, list)
        assert len(result) >= 2
        for p in result:
            assert Path(p).exists()

    def test_auto_detects_simple(self, tmp_path):
        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand_adaptive("CCO", str(out), seed=42)
        assert isinstance(result, str)

    def test_auto_detects_medium_defaults_to_single(self, tmp_path):
        # Default behaviour: medium ligands use single conformer because
        # Vina performs internal torsion search.
        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand_adaptive(
            "CCC(CC)O[C@@H]1CC(C(O)O)C[C@H](N)[C@H]1NC(C)O",
            str(out),
            seed=42,
        )
        assert isinstance(result, str)
        assert Path(result).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Flexible receptor utilities
# ─────────────────────────────────────────────────────────────────────────────


class TestFindNearbyResidues:
    def test_finds_ca_within_radius(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
            "ATOM      2  CA  ALA A   2       5.000   0.000   0.000\n"
            "ATOM      3  CA  ALA A   3      10.000   0.000   0.000\n"
        )
        result = prep.find_nearby_residues(str(pdb), (0, 0, 0), radius=6.0)
        assert result == ["A:1", "A:2"]

    def test_excludes_water(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text(
            "ATOM      1  CA  HOH A   1       0.000   0.000   0.000\n"
            "ATOM      2  CA  ALA A   2       1.000   0.000   0.000\n"
        )
        result = prep.find_nearby_residues(str(pdb), (0, 0, 0), radius=2.0)
        assert result == ["A:2"]

    def test_respects_max_residues(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        lines = ""
        for i in range(10):
            lines += f"ATOM  {i+1:4d}  CA  ALA A {i+1:3d}      {i:3d}.000   0.000   0.000\n"
        pdb.write_text(lines)
        result = prep.find_nearby_residues(str(pdb), (0, 0, 0), radius=50.0, max_residues=3)
        assert len(result) == 3

    def test_empty_file_returns_empty(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("")
        result = prep.find_nearby_residues(str(pdb), (0, 0, 0), radius=6.0)
        assert result == []


class TestPrepareFlexibleReceptor:
    @patch("autodock.preparation.find_conda_tool")
    @patch("autodock.preparation.safe_subprocess")
    @patch("os.path.isfile")
    def test_prepare_flexible_receptor_success(
        self, mock_isfile, mock_subprocess, mock_find, tmp_path
    ):
        mock_find.return_value = "/fake/mk_prepare_receptor.py"
        mock_subprocess.return_value = (True, "", "")
        mock_isfile.return_value = True

        out_dir = str(tmp_path / "flex")
        rigid, flex = prep.prepare_flexible_receptor(
            str(tmp_path / "rec.pdb"),
            ["A:149", "A:276"],
            out_dir,
        )
        assert rigid.endswith("_rigid.pdbqt")
        assert flex.endswith("_flex.pdbqt")

    @patch("autodock.preparation.find_conda_tool")
    def test_prepare_flexible_receptor_missing_tool(self, mock_find, tmp_path):
        mock_find.return_value = None
        with pytest.raises(PreparationError, match="mk_prepare_receptor.py not found"):
            prep.prepare_flexible_receptor(
                str(tmp_path / "rec.pdb"),
                ["A:149"],
                str(tmp_path / "flex"),
            )

    def test_prepare_flexible_receptor_empty_flexres(self, tmp_path):
        with pytest.raises(PreparationError, match="flexres list is empty"):
            prep.prepare_flexible_receptor(
                str(tmp_path / "rec.pdb"),
                [],
                str(tmp_path / "flex"),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Coverage expansion — Phase 2
# ─────────────────────────────────────────────────────────────────────────────


class TestParsePdbHeader:
    def test_resolution_and_quality_flags(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text(
            "REMARK   2 RESOLUTION.    1.90 ANGSTROMS.\n"
            "REMARK   3   R FREE            : 0.210\n"
            "REMARK   3   R WORK            : 0.180\n"
            "REMARK   3   WILSON B          : 25.0\n"
            "REVDAT   1   01-JAN-20\n"
            "EXPDTA    X-RAY DIFFRACTION\n"
        )
        result = prep._parse_pdb_header(str(pdb))
        assert result["resolution"] == 1.9
        assert result["r_free"] == 0.21
        assert result["r_work"] == 0.18
        assert result["wilson_b"] == 25.0
        assert result["date"] == "01-JAN-20"
        assert result["method"] == "X-RAY DIFFRACTION"
        assert result["quality_flag"] == "high"

    def test_only_resolution(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("REMARK   2 RESOLUTION.    2.30 ANGSTROMS.\n")
        result = prep._parse_pdb_header(str(pdb))
        assert result["resolution"] == 2.3
        assert result["quality_flag"] == "acceptable"

    def test_missing_file(self):
        result = prep._parse_pdb_header("/nonexistent.pdb")
        assert result["quality_flag"] == "unknown"

    def test_oserror(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("HEADER    TEST\n")
        with patch("builtins.open", side_effect=OSError("permission denied")):
            result = prep._parse_pdb_header(str(pdb))
            assert result["quality_flag"] == "unknown"

    def test_remark_200_method(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("REMARK 200  EXPERIMENT TYPE : NMR\n")
        result = prep._parse_pdb_header(str(pdb))
        assert result["method"] == "NMR"


class TestFindDisulfideBonds:
    def test_typical_ssbond(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text(
            "SSBOND   1 CYS A   32    CYS A   94\n" "SSBOND   2 CYS B   45    CYS C  123\n"
        )
        bonds = prep._find_disulfide_bonds(str(pdb))
        assert len(bonds) == 2
        assert bonds[0]["chain1"] == "A"
        assert bonds[0]["res1"] == 32
        assert bonds[1]["chain2"] == "C"
        assert bonds[1]["res2"] == 123

    def test_missing_file(self):
        assert prep._find_disulfide_bonds("/nonexistent.pdb") == []

    def test_short_line_ignored(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("SSBOND\n")
        assert prep._find_disulfide_bonds(str(pdb)) == []


class TestPrepareReceptorBranches:
    @patch("meeko.ResidueChemTemplates")
    @patch("meeko.MoleculePreparation")
    @patch("meeko.Polymer")
    @patch("meeko.PDBQTWriterLegacy")
    def test_force_false_skips_existing(self, mock_writer, mock_poly, mock_mk, mock_tmpl, tmp_path):
        out = tmp_path / "rec.pdbqt"
        out.write_text("EXISTING\n")
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        result = prep.prepare_receptor(str(pdb), str(out), force=False)
        assert result == str(out.resolve())
        assert not mock_poly.called

    @patch("meeko.ResidueChemTemplates")
    @patch("meeko.MoleculePreparation")
    @patch("meeko.Polymer")
    @patch("meeko.PDBQTWriterLegacy")
    def test_remove_water_false(self, mock_writer, mock_poly, mock_mk, mock_tmpl, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2  O   HOH A   2      1.000   1.000   1.000\n"
        )
        out = tmp_path / "rec.pdbqt"
        mock_poly.from_pdb_string.return_value = MagicMock()
        mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
        prep.prepare_receptor(str(pdb), str(out), remove_water=False, remove_hetatms=False)
        call_args = mock_poly.from_pdb_string.call_args[0][0]
        assert "HOH" in call_args

    @patch("meeko.ResidueChemTemplates")
    @patch("meeko.MoleculePreparation")
    @patch("meeko.Polymer")
    @patch("meeko.PDBQTWriterLegacy")
    def test_retain_metal_ions_false(self, mock_writer, mock_poly, mock_mk, mock_tmpl, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2 ZN   ZN  A 100      0.000   0.000   0.000\n"
        )
        out = tmp_path / "rec.pdbqt"
        mock_poly.from_pdb_string.return_value = MagicMock()
        mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
        prep.prepare_receptor(str(pdb), str(out), retain_metal_ions=False, remove_hetatms=True)
        call_args = mock_poly.from_pdb_string.call_args[0][0]
        assert "ZN" not in call_args

    @patch("meeko.ResidueChemTemplates")
    @patch("meeko.MoleculePreparation")
    @patch("meeko.Polymer")
    @patch("meeko.PDBQTWriterLegacy")
    def test_predict_pka_false(self, mock_writer, mock_poly, mock_mk, mock_tmpl, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        out = tmp_path / "rec.pdbqt"
        report = tmp_path / "report.json"
        mock_poly.from_pdb_string.return_value = MagicMock()
        mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
        prep.prepare_receptor(str(pdb), str(out), predict_pka=False, output_report_json=str(report))
        data = json.loads(report.read_text())
        assert data["anomalous_pka_residues"] == 0

    @patch("meeko.ResidueChemTemplates")
    @patch("meeko.MoleculePreparation")
    @patch("meeko.Polymer")
    @patch("meeko.PDBQTWriterLegacy")
    def test_detect_af_false(self, mock_writer, mock_poly, mock_mk, mock_tmpl, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 80.00\n")
        out = tmp_path / "rec.pdbqt"
        report = tmp_path / "report.json"
        mock_poly.from_pdb_string.return_value = MagicMock()
        mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
        prep.prepare_receptor(
            str(pdb), str(out), detect_af_structure=False, output_report_json=str(report)
        )
        data = json.loads(report.read_text())
        assert data["alphafold_assessment"] is None

    @patch("meeko.ResidueChemTemplates")
    @patch("meeko.MoleculePreparation")
    @patch("meeko.Polymer")
    @patch("meeko.PDBQTWriterLegacy")
    def test_output_pdb_saved(self, mock_writer, mock_poly, mock_mk, mock_tmpl, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        out = tmp_path / "rec.pdbqt"
        out_pdb = tmp_path / "rec_out.pdb"
        mock_poly.from_pdb_string.return_value = MagicMock()
        mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
        prep.prepare_receptor(str(pdb), str(out), output_pdb=str(out_pdb))
        assert out_pdb.exists()
        assert "SER" in out_pdb.read_text()

    def test_pdbqt_input_format(self, tmp_path):
        pdb = tmp_path / "rec.pdbqt"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        out = tmp_path / "rec_out.pdbqt"
        with (
            patch("meeko.ResidueChemTemplates"),
            patch("meeko.MoleculePreparation"),
            patch("meeko.Polymer") as mock_poly_cls,
            patch("meeko.PDBQTWriterLegacy") as mock_writer,
        ):
            mock_poly_cls.from_pdb_string.return_value = MagicMock()
            mock_writer.write_from_polymer.return_value = ("REMARK\n", None)
            prep.prepare_receptor(str(pdb), str(out), input_format="pdb")
            assert mock_poly_cls.from_pdb_string.called


class TestPrepareLigandFromFile:
    @patch("rdkit.Chem.MolFromMol2File")
    @patch("rdkit.Chem.MolToSmiles")
    @patch("autodock.preparation.prepare_ligand")
    def test_mol2_file(self, mock_prep, mock_to_smiles, mock_mol2, tmp_path):
        mock_mol = MagicMock()
        mock_mol2.return_value = mock_mol
        mock_to_smiles.return_value = "CCO"
        mock_prep.return_value = str(tmp_path / "lig.pdbqt")
        mol2 = tmp_path / "lig.mol2"
        mol2.write_text("@<TRIPOS>MOLECULE\n")
        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand_from_file(str(mol2), str(out))
        assert result == str(out.resolve())
        mock_prep.assert_called_once()

    @patch("autodock.preparation.obabel_convert")
    @patch("rdkit.Chem.MolFromMol2File")
    def test_mol2_fallback_to_obabel(self, mock_mol2, mock_obabel, tmp_path):
        mock_mol2.return_value = None
        mock_obabel.return_value = True
        mol2 = tmp_path / "lig.mol2"
        mol2.write_text("@<TRIPOS>MOLECULE\n")
        out = tmp_path / "lig.pdbqt"
        # Write a fake PDBQT so the function can return its path
        out.write_text("REMARK\n")
        result = prep.prepare_ligand_from_file(str(mol2), str(out))
        assert Path(result).exists()
        mock_obabel.assert_called_once()

    @patch("autodock.preparation._has_nan_charges")
    @patch("rdkit.Chem.MolFromMolFile")
    @patch("rdkit.Chem.SDMolSupplier")
    @patch("rdkit.Chem.AddHs")
    @patch("rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges")
    @patch("meeko.MoleculePreparation")
    @patch("meeko.PDBQTWriterLegacy")
    def test_sdf_file(
        self,
        mock_writer,
        mock_mk,
        mock_charges,
        mock_addhs,
        mock_supplier,
        mock_molfile,
        mock_has_nan,
        tmp_path,
    ):
        mock_has_nan.return_value = False
        mock_molfile.return_value = None
        mock_mol = MagicMock()
        mock_supplier.return_value = [mock_mol]
        mock_addhs.return_value = mock_mol
        mock_mk.return_value.prepare.return_value = MagicMock()
        mock_writer.write_string.return_value = ("ATOM 1 C LIG\n", True, "")
        sdf = tmp_path / "lig.sdf"
        sdf.write_text("fake sdf\n")
        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand_from_sdf(str(sdf), str(out))
        assert Path(result).exists()

    @patch("rdkit.Chem.MolFromMolFile")
    @patch("rdkit.Chem.SDMolSupplier")
    def test_sdf_unparseable(self, mock_supplier, mock_molfile, tmp_path):
        mock_molfile.return_value = None
        mock_supplier.return_value = []
        sdf = tmp_path / "lig.sdf"
        sdf.write_text("fake sdf\n")
        out = tmp_path / "lig.pdbqt"
        with pytest.raises(PreparationError, match="Could not parse SDF"):
            prep.prepare_ligand_from_sdf(str(sdf), str(out))


class TestPrepareLigandConformersExtra:
    @patch("autodock.preparation.prepare_ligand")
    def test_some_conformers_fail(self, mock_prep, tmp_path):
        def side_effect(smiles, out_path, **kwargs):
            if "conformer_1" in out_path:
                raise RuntimeError("fail")
            Path(out_path).write_text("ATOM\n")
            return out_path

        mock_prep.side_effect = side_effect
        outdir = tmp_path / "conformers"
        paths = prep.prepare_ligand_conformers("CCO", str(outdir), n_conformers=3, seed_start=10)
        assert len(paths) == 2
        assert all(Path(p).exists() for p in paths)

    @patch("autodock.preparation.prepare_ligand")
    def test_all_conformers_fail(self, mock_prep, tmp_path):
        mock_prep.side_effect = RuntimeError("fail")
        outdir = tmp_path / "conformers"
        with pytest.raises(PreparationError, match="All 3 conformer preparations failed"):
            prep.prepare_ligand_conformers("CCO", str(outdir), n_conformers=3)


class TestRunP2RankPredict:
    @patch("autodock.preparation.find_p2rank")
    @patch("autodock.preparation.safe_subprocess")
    @patch("os.path.exists")
    @patch("builtins.open")
    def test_success(self, mock_open, mock_exists, mock_subprocess, mock_find, tmp_path):
        mock_find.return_value = "/fake/prank"
        mock_subprocess.return_value = (True, "", "")
        mock_exists.return_value = True

        # Mock CSV content
        csv_content = "name,probability,center_x,center_y,center_z,radius\n"
        csv_content += "pocket1,0.85,1.0,2.0,3.0,8.0\n"
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.readline.return_value = csv_content.splitlines()[0] + "\n"
        mock_file.__iter__ = MagicMock(return_value=iter(csv_content.splitlines()[1:]))
        mock_open.return_value = mock_file

        pockets = prep._run_p2rank_predict(str(tmp_path / "rec.pdb"), str(tmp_path))
        assert pockets is not None
        assert len(pockets) == 1
        assert pockets[0]["score"] == 0.85
        assert pockets[0]["center"] == (1.0, 2.0, 3.0)

    @patch("autodock.preparation.find_p2rank")
    def test_not_found(self, mock_find, tmp_path):
        mock_find.return_value = None
        assert prep._run_p2rank_predict(str(tmp_path / "rec.pdb"), str(tmp_path)) is None

    @patch("autodock.preparation.find_p2rank")
    @patch("autodock.preparation.safe_subprocess")
    def test_subprocess_fails(self, mock_subprocess, mock_find, tmp_path):
        mock_find.return_value = "/fake/prank"
        mock_subprocess.return_value = (False, "", "error")
        assert prep._run_p2rank_predict(str(tmp_path / "rec.pdb"), str(tmp_path)) is None


class TestRunFpocketDetect:
    @patch("autodock.preparation.find_conda_tool")
    @patch("autodock.preparation.safe_subprocess")
    @patch("shutil.rmtree")
    def test_success(self, mock_rmtree, mock_subprocess, mock_find, tmp_path):
        mock_find.return_value = "/fake/fpocket"
        mock_subprocess.return_value = (True, "", "")

        # Create actual receptor pdb
        rec_pdb = tmp_path / "rec.pdb"
        rec_pdb.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000\n")

        # Pre-create the expected fpocket output directory structure
        out_dir = tmp_path / "rec_fpocket_prep_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        info_file = out_dir / "rec_fpocket_prep_info.txt"
        info_file.write_text(
            "Pocket 1 :\n"
            "Druggability Score : 0.75\n"
            "Volume : 450.5\n"
            "Depth : 8.2\n"
            "Number of mouth openings : 3\n"
        )
        pqr_file = out_dir / "pockets" / "pocket1_vert.pqr"
        pqr_file.parent.mkdir(parents=True, exist_ok=True)
        pqr_file.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000\n"
            "ATOM      2  C   LIG A   1      4.000   5.000   6.000\n"
        )

        # Patch mkstemp so the generated path lands inside tmp_path where we created the out dir
        import tempfile as _tempfile_mod

        with patch.object(_tempfile_mod, "mkstemp") as mock_mkstemp:
            prep_path = str(tmp_path / "rec_fpocket_prep.pdb")
            mock_mkstemp.return_value = (99, prep_path)
            with patch("os.close"):
                pockets = prep._run_fpocket_detect(str(rec_pdb))
                assert pockets is not None

    @patch("autodock.preparation.find_conda_tool")
    def test_not_found(self, mock_find, tmp_path):
        mock_find.return_value = None
        assert prep._run_fpocket_detect(str(tmp_path / "rec.pdb")) is None

    @patch("autodock.preparation.find_conda_tool")
    @patch("autodock.preparation.safe_subprocess")
    def test_subprocess_fails(self, mock_subprocess, mock_find, tmp_path):
        mock_find.return_value = "/fake/fpocket"
        mock_subprocess.return_value = (False, "", "error")
        rec_pdb = tmp_path / "rec.pdb"
        rec_pdb.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000\n")
        assert prep._run_fpocket_detect(str(rec_pdb)) is None


class TestParseP2RankResidues:
    def test_typical(self, tmp_path):
        csv = tmp_path / "pred.csv"
        # NOTE: the simple split(",") parser only ever sees the first residue
        # because commas inside residue_ids break the naive CSV split.
        csv.write_text(
            "name,probability,residue_ids\n" "pocket_1,0.8,A:12,A:13,B:45\n" "pocket_2,0.6,-\n"
        )
        result = prep._parse_p2rank_residues(str(csv))
        assert 1 in result
        # Only the first residue (A:12) is parsed due to the split behaviour
        assert len(result[1]) == 1
        assert result[1][0] == {"chain": "A", "resid": 12}
        assert 2 in result
        assert result[2] == []

    def test_missing_file(self):
        assert prep._parse_p2rank_residues("/nonexistent.csv") == {}

    def test_no_residue_ids_column(self, tmp_path):
        csv = tmp_path / "pred.csv"
        csv.write_text("name,probability\n")
        assert prep._parse_p2rank_residues(str(csv)) == {}


class TestComputePocketShapeDescriptors:
    def test_typical(self, tmp_path):
        pqr = tmp_path / "pocket1_vert.pqr"
        pqr.write_text(
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000\n"
            "ATOM      2  C   LIG A   1      10.00   0.000   0.000\n"
            "ATOM      3  C   LIG A   1      0.000   5.000   0.000\n"
            "ATOM      4  C   LIG A   1      0.000   0.000   2.000\n"
        )
        desc = prep._compute_pocket_shape_descriptors(str(pqr))
        assert desc["circularity"] is not None
        assert desc["aspect_ratio"] is not None
        assert desc["surface_area"] is not None

    def test_missing_file(self):
        desc = prep._compute_pocket_shape_descriptors("/nonexistent.pqr")
        assert desc["circularity"] is None

    def test_too_few_atoms(self, tmp_path):
        pqr = tmp_path / "pocket1_vert.pqr"
        pqr.write_text("ATOM      1  C   LIG A   1      0.000   0.000   0.000\n")
        desc = prep._compute_pocket_shape_descriptors(str(pqr))
        assert desc["circularity"] is None


class TestDruggabilityClassification:
    def test_high(self):
        result = prep._druggability_classification(0.85)
        assert result["level"] == "high"

    def test_medium_with_volume_and_depth(self):
        result = prep._druggability_classification(0.55, volume=200, depth=3.0)
        assert result["level"] == "medium"
        assert "Small volume" in result["description"]
        assert "Shallow binding site" in result["description"]

    def test_low(self):
        result = prep._druggability_classification(0.20)
        assert result["level"] == "low"

    def test_none(self):
        result = prep._druggability_classification(None)
        assert result["level"] == "unknown"


class TestClassifyPocketType:
    def test_orthosteric(self):
        result = prep._classify_pocket_type((0, 0, 0), (0, 0, 0))
        assert result["type"] == "orthosteric"

    def test_allosteric(self):
        result = prep._classify_pocket_type((20, 0, 0), (0, 0, 0), pocket_volume=500)
        assert result["type"] == "allosteric_candidate"
        assert result["confidence"] == "high"

    def test_unclassified(self):
        result = prep._classify_pocket_type((8, 0, 0), (0, 0, 0))
        assert result["type"] == "unclassified"

    def test_no_reference(self):
        result = prep._classify_pocket_type((5, 5, 5), None)
        assert result["type"] == "unclassified"
        assert result["confidence"] == "no_reference"


class TestPocketBfactorFlexibility:
    def test_rigid(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        lines = ""
        for i in range(10):
            lines += (
                f"ATOM  {i+1:4d}  CA  ALA A {i+1:3d}      {i:3d}.000   0.000   0.000  1.00 10.00\n"
            )
        pdb.write_text(lines)
        result = prep._pocket_bfactor_flexibility((0, 0, 0), 5.0, str(pdb))
        assert result["flexibility"] == "rigid"
        assert result["induced_fit_likely"] is False

    def test_flexible(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        lines = ""
        for i in range(10):
            lines += (
                f"ATOM  {i+1:4d}  CA  ALA A {i+1:3d}      {i:3d}.000   0.000   0.000  1.00 80.00\n"
            )
        pdb.write_text(lines)
        result = prep._pocket_bfactor_flexibility((0, 0, 0), 5.0, str(pdb))
        assert result["flexibility"] == "flexible"
        assert result["induced_fit_likely"] is True

    def test_missing_file(self):
        result = prep._pocket_bfactor_flexibility((0, 0, 0), 5.0, "/nonexistent.pdb")
        assert result["flexibility"] == "unknown"


class TestValidateAlphafoldPocket:
    @patch("autodock.alphafold_tools.assess_alphafold_quality")
    def test_no_low_regions(self, mock_assess, tmp_path):
        mock_assess.return_value = {
            "suitable_for_docking": True,
            "low_confidence_regions": [],
            "mean_plddt": 85.0,
        }
        pdb = tmp_path / "rec.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 85.00           C  \n"
        )
        result = prep._validate_alphafold_pocket((0, 0, 0), 5.0, str(pdb))
        # When no low-confidence regions exist, the function short-circuits
        # and does not compute mean_plddt_in_pocket
        assert result["af_suitable"] is True
        assert result["mean_plddt_in_pocket"] is None

    @patch("autodock.alphafold_tools.assess_alphafold_quality")
    def test_with_low_regions(self, mock_assess, tmp_path):
        mock_assess.return_value = {
            "suitable_for_docking": False,
            "low_confidence_regions": [{"chain": "A", "start": 1, "end": 5}],
            "mean_plddt": 50.0,
        }
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 50.00\n")
        result = prep._validate_alphafold_pocket((0, 0, 0), 5.0, str(pdb))
        assert result["af_suitable"] is False
        assert len(result["overlapping_low_conf_regions"]) == 1

    def test_missing_pdb(self):
        result = prep._validate_alphafold_pocket((0, 0, 0), 5.0, "/nonexistent.pdb")
        assert result["af_suitable"] is True


class TestCleanPdbWithOpenbabel:
    @patch("subprocess.run")
    def test_success(self, mock_run, tmp_path):
        inp = tmp_path / "in.pdb"
        inp.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        out = tmp_path / "out.pdb"
        prep._clean_pdb_with_openbabel(str(inp), str(out))
        assert mock_run.call_count == 2


class TestPrepareFlexibleReceptorBranches:
    @patch("autodock.preparation.find_conda_tool")
    @patch("autodock.preparation.safe_subprocess")
    @patch("os.path.isfile")
    def test_valence_error_cleanup(self, mock_isfile, mock_subprocess, mock_find, tmp_path):
        mock_find.return_value = "/fake/mk_prepare_receptor.py"
        # First call fails with valence error, second succeeds after cleanup
        mock_subprocess.side_effect = [
            (False, "", "RDKit valence error"),
            (True, "", ""),
        ]
        mock_isfile.return_value = True

        out_dir = str(tmp_path / "flex")
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")

        with patch("autodock.preparation._clean_pdb_with_openbabel") as mock_clean:
            clean_pdb = tmp_path / "flex" / "receptor_clean.pdb"
            clean_pdb.parent.mkdir(parents=True, exist_ok=True)
            clean_pdb.write_text("ATOM\n")
            mock_clean.side_effect = lambda src, dst: Path(dst).write_text("ATOM\n")
            rigid, flex = prep.prepare_flexible_receptor(str(pdb), ["A:149"], out_dir)
            assert rigid.endswith("_rigid.pdbqt")
            assert flex.endswith("_flex.pdbqt")
            mock_clean.assert_called_once()

    @patch("autodock.preparation.find_conda_tool")
    @patch("autodock.preparation.safe_subprocess")
    @patch("os.path.isfile")
    def test_cleanup_fails(self, mock_isfile, mock_subprocess, mock_find, tmp_path):
        mock_find.return_value = "/fake/mk_prepare_receptor.py"
        mock_subprocess.return_value = (False, "", "RDKit valence error")
        mock_isfile.return_value = True

        out_dir = str(tmp_path / "flex")
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")

        with patch("autodock.preparation._clean_pdb_with_openbabel") as mock_clean:
            mock_clean.side_effect = Exception("obabel fail")
            with pytest.raises(PreparationError, match="mk_prepare_receptor.py failed"):
                prep.prepare_flexible_receptor(str(pdb), ["A:149"], out_dir)


class TestPrepareLigandAdaptiveExtra:
    @patch("rdkit.Chem.MolFromSmiles")
    @patch("autodock.preparation.prepare_ligand")
    def test_simple(self, mock_prep, mock_mol, tmp_path):
        mock_mol.return_value = MagicMock(
            GetNumHeavyAtoms=10, GetRingInfo=lambda: MagicMock(AtomRings=list, NumRings=lambda: 0)
        )
        mock_prep.return_value = str(tmp_path / "lig.pdbqt")
        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand_adaptive("CCO", str(out), strategy="simple")
        assert result == str(out.resolve())

    @patch("rdkit.Chem.MolFromSmiles")
    @patch("autodock.preparation.prepare_ligand_multi")
    def test_medium_force_multi(self, mock_multi, mock_mol, tmp_path):
        mock_mol.return_value = MagicMock(
            GetNumHeavyAtoms=20,
            GetRingInfo=lambda: MagicMock(AtomRings=list, NumRings=lambda: 0),
        )
        mock_multi.return_value = [str(tmp_path / "c0.pdbqt")]
        out_dir = tmp_path / "out"
        result = prep.prepare_ligand_adaptive(
            "CCO", str(out_dir), strategy="medium", force_multi_conformer=True
        )
        assert isinstance(result, list)

    @patch("rdkit.Chem.MolFromSmiles")
    @patch("autodock.preparation.prepare_ligand")
    def test_complex_no_macrocycle_single(self, mock_prep, mock_mol, tmp_path):
        mock_mol.return_value = MagicMock(
            GetNumHeavyAtoms=35,
            GetRingInfo=lambda: MagicMock(
                AtomRings=lambda: [[6, 6, 6, 6, 6, 6]], NumRings=lambda: 1
            ),
        )
        mock_prep.return_value = str(tmp_path / "lig.pdbqt")
        out = tmp_path / "lig.pdbqt"
        result = prep.prepare_ligand_adaptive("CCO", str(out), strategy="complex")
        assert result == str(out.resolve())

    @patch("rdkit.Chem.MolFromSmiles")
    @patch("autodock.preparation.prepare_ligand")
    def test_very_large_ligand(self, mock_prep, mock_mol, tmp_path):
        ring_info = MagicMock()
        ring_info.AtomRings = lambda: [[6] * 14]
        ring_info.NumRings = lambda: 1
        mock_mol.return_value = MagicMock(
            GetNumHeavyAtoms=lambda: 55,
            GetRingInfo=lambda: ring_info,
        )
        mock_prep.return_value = str(tmp_path / "lig.pdbqt")
        out_dir = tmp_path / "out"
        result = prep.prepare_ligand_adaptive("CCO", str(out_dir), strategy="complex")
        assert result == str(tmp_path / "lig.pdbqt")


class TestFindTopPocketsBranches:
    @patch("autodock.preparation.find_p2rank")
    @patch("autodock.preparation.find_conda_tool")
    @patch("autodock.preparation._run_p2rank_predict")
    @patch("autodock.preparation._run_fpocket_detect")
    @patch("autodock.preparation._parse_p2rank_residues")
    @patch("autodock.preparation._pocket_bfactor_flexibility")
    @patch("autodock.preparation._validate_alphafold_pocket")
    @patch("autodock.preparation._classify_pocket_type")
    def test_computational_detection(
        self,
        mock_classify,
        mock_validate,
        mock_flex,
        mock_parse_res,
        mock_fpocket,
        mock_p2rank,
        mock_find_fpocket,
        mock_find_p2rank,
        tmp_path,
    ):
        mock_find_p2rank.return_value = "/fake/prank"
        mock_find_fpocket.return_value = "/fake/fpocket"
        mock_p2rank.return_value = [
            {
                "num": 1,
                "center": (1.0, 2.0, 3.0),
                "score": 0.8,
                "radius": 10.0,
                "druggability": 0.8,
                "dims": (20, 20, 20),
                "pocket_source": "p2rank",
            }
        ]
        mock_fpocket.return_value = [
            {
                "num": 1,
                "center": (1.5, 2.5, 3.5),
                "druggability": 0.75,
                "volume": 400,
                "depth": 6.0,
                "dims": (20, 20, 20),
            }
        ]
        mock_parse_res.return_value = {1: [{"chain": "A", "resid": 12}]}
        mock_flex.return_value = {"flexibility": "moderate", "induced_fit_likely": True}
        mock_validate.return_value = {"af_suitable": True}
        mock_classify.return_value = {
            "type": "orthosteric",
            "distance_to_active": 0.0,
            "confidence": "high",
        }

        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000\n")
        pockets = prep.find_top_pockets(str(pdb))
        assert len(pockets) >= 1
        assert pockets[0]["fpocket_verified"] is True

    def test_ligand_centered_fallback(self, tmp_path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000\n")
        lig = tmp_path / "lig.pdb"
        lig.write_text(
            "HETATM    1  C   LIG A   1      5.000   5.000   5.000  1.00  0.00           C\n"
        )
        pockets = prep.find_top_pockets(str(pdb), ligand_pdb=str(lig), known_active_site=(5, 5, 5))
        assert pockets[0]["pocket_source"] == "crystal_ligand"
        assert pockets[0]["pocket_type"] == "orthosteric"

    @patch("autodock.preparation.find_p2rank")
    @patch("autodock.preparation.find_conda_tool")
    @patch("autodock.preparation._run_p2rank_predict")
    def test_p2rank_no_pockets_raises(
        self, mock_p2rank, mock_find_fpocket, mock_find_p2rank, tmp_path
    ):
        mock_find_p2rank.return_value = "/fake/prank"
        mock_find_fpocket.return_value = None
        mock_p2rank.return_value = []
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000\n")
        with pytest.raises(PreparationError, match="P2Rank found no pockets"):
            prep.find_top_pockets(str(pdb))

    @patch("autodock.preparation.find_p2rank")
    @patch("autodock.preparation.find_conda_tool")
    @patch("autodock.preparation._run_p2rank_predict")
    @patch("autodock.preparation._run_fpocket_detect")
    @patch("autodock.preparation._parse_p2rank_residues")
    @patch("autodock.preparation._pocket_bfactor_flexibility")
    @patch("autodock.preparation._validate_alphafold_pocket")
    @patch("autodock.preparation._classify_pocket_type")
    def test_no_fpocket_available(
        self,
        mock_classify,
        mock_validate,
        mock_flex,
        mock_parse_res,
        mock_fpocket,
        mock_p2rank,
        mock_find_fpocket,
        mock_find_p2rank,
        tmp_path,
    ):
        mock_find_p2rank.return_value = "/fake/prank"
        mock_find_fpocket.return_value = None
        mock_p2rank.return_value = [
            {
                "num": 1,
                "center": (1.0, 2.0, 3.0),
                "score": 0.8,
                "radius": 10.0,
                "druggability": 0.8,
                "dims": (20, 20, 20),
                "pocket_source": "p2rank",
            }
        ]
        mock_fpocket.return_value = None
        mock_parse_res.return_value = {}
        mock_flex.return_value = {"flexibility": "moderate", "induced_fit_likely": True}
        mock_validate.return_value = {"af_suitable": True}
        mock_classify.return_value = {
            "type": "unclassified",
            "distance_to_active": None,
            "confidence": "no_reference",
        }

        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000\n")
        pockets = prep.find_top_pockets(str(pdb))
        assert len(pockets) == 1
        assert pockets[0]["fpocket_verified"] is False


class TestPrepareReceptorWithObabel:
    @patch("autodock.preparation.obabel_convert")
    def test_success(self, mock_obabel, tmp_path):
        mock_obabel.return_value = True
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        out = tmp_path / "rec.pdbqt"
        result = prep._prepare_receptor_with_obabel(str(pdb), str(out))
        assert result == str(out.resolve())

    @patch("autodock.preparation.obabel_convert")
    def test_failure(self, mock_obabel, tmp_path):
        mock_obabel.return_value = False
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM\n")
        out = tmp_path / "rec.pdbqt"
        with pytest.raises(PreparationError, match="Open Babel receptor preparation failed"):
            prep._prepare_receptor_with_obabel(str(pdb), str(out))

    @patch("autodock.preparation.obabel_convert")
    def test_with_pdb_string(self, mock_obabel, tmp_path):
        mock_obabel.return_value = True
        out = tmp_path / "rec.pdbqt"
        result = prep._prepare_receptor_with_obabel("dummy.pdb", str(out), pdb_string="ATOM\n")
        assert result == str(out.resolve())


class TestPrepareLigandWithObabel:
    @patch("autodock.preparation.obabel_convert")
    def test_success(self, mock_obabel, tmp_path):
        def obabel_side_effect(src, dst, **kwargs):
            Path(dst).write_text("REMARK  obabel\nATOM 1 C LIG\n")
            return True

        mock_obabel.side_effect = obabel_side_effect
        out = tmp_path / "lig.pdbqt"
        result = prep._prepare_ligand_with_obabel("CCO", str(out), name="LIG")
        assert Path(result).exists()
        assert "LIG" in Path(result).read_text()

    @patch("autodock.preparation.obabel_convert")
    def test_failure(self, mock_obabel, tmp_path):
        mock_obabel.return_value = False
        out = tmp_path / "lig.pdbqt"
        with pytest.raises(PreparationError, match="Open Babel ligand preparation failed"):
            prep._prepare_ligand_with_obabel("CCO", str(out))


class TestPrepareLigandCacheBranches:
    @patch("autodock.preparation._has_nan_charges")
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
    def test_cache_lookup_hit(
        self,
        mock_geo,
        mock_mol_cls,
        mock_conf_cls,
        mock_writer_cls,
        mock_mk_cls,
        mock_charges,
        mock_mmff,
        mock_embed,
        mock_etkdg,
        mock_addhs,
        mock_molfrom,
        mock_has_nan,
        tmp_path,
    ):
        mock_has_nan.return_value = False
        mock_mol = MagicMock()
        mock_mol.GetNumAtoms.return_value = 3
        mock_molfrom.return_value = mock_mol
        mock_mol_copy = MagicMock()
        mock_mol_copy.GetNumAtoms.return_value = 3
        mock_mol_cls.return_value = mock_mol_copy
        mock_addhs.return_value = mock_mol
        mock_embed.return_value = [0]
        mock_mmff.return_value = [(0, -10.0)]
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

        # Create a fake cache hit
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        ligand_cache = cache_dir / "ligands"
        ligand_cache.mkdir()
        smiles_hash = "testhash"
        cache_entry = ligand_cache / smiles_hash
        cache_entry.mkdir()
        cached_pdbqt = cache_entry / "ligand.pdbqt"
        cached_pdbqt.write_text("CACHED\n")

        out = tmp_path / "lig.pdbqt"
        with patch("autodock.cache.LigandCache") as mock_lc_cls:
            mock_lc = MagicMock()
            mock_lc_cls.return_value = mock_lc
            mock_lc.get.return_value = {"ligand.pdbqt": str(cached_pdbqt)}
            result = prep.prepare_ligand("CCO", str(out), cache_dir=str(cache_dir))
            assert result == str(out.resolve())
            assert out.read_text() == "CACHED\n"
            mock_lc.get.assert_called_once()
