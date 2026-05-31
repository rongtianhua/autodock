"""Extended tests for autodock.utils — additional coverage."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from autodock import utils

# ── write_temp_file ──────────────────────────────────────────────────────────


class TestWriteTempFile:
    def test_success_path(self):
        path = utils.write_temp_file("hello world", suffix=".txt")
        assert path.endswith(".txt")
        try:
            with open(path) as fh:
                assert fh.read() == "hello world"
        finally:
            os.unlink(path)

    def test_default_suffix(self):
        path = utils.write_temp_file("content")
        assert path.endswith(".tmp")
        try:
            with open(path) as fh:
                assert fh.read() == "content"
        finally:
            os.unlink(path)


# ── read_pdb_atoms ───────────────────────────────────────────────────────────


class TestReadPdbAtomsExtended:
    def test_parse_hetatm(self, tmp_path):
        f = tmp_path / "test.pdb"
        f.write_text(
            "HETATM    1  O   HOH A   1      10.000  20.000  30.000  1.00  0.00           O  \n"
        )
        atoms = utils.read_pdb_atoms(str(f))
        assert len(atoms) == 1
        assert atoms[0]["record"] == "HETATM"
        assert atoms[0]["atom_name"] == "O"
        assert atoms[0]["res_name"] == "HOH"

    def test_parse_multiple_atoms(self, tmp_path):
        f = tmp_path / "test.pdb"
        f.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00  0.00           N  \n"
            "ATOM      2  CA  SER A   1      1.000   0.000   0.000  1.00  0.00           C  \n"
            "HETATM    3  O   HOH A   2      2.000   0.000   0.000  1.00  0.00           O  \n"
        )
        atoms = utils.read_pdb_atoms(str(f))
        assert len(atoms) == 3
        assert atoms[1]["atom_name"] == "CA"
        assert atoms[2]["chain"] == "A"
        assert atoms[2]["res_seq"] == 2

    def test_malformed_lines_skipped(self, tmp_path):
        f = tmp_path / "test.pdb"
        f.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00  0.00           N  \n"
            "ATOM    BAD  N   SER A   1      xxxxx   0.000   0.000\n"
            "ATOM      3  CA  SER A   1      1.000   0.000   0.000  1.00  0.00           C  \n"
        )
        atoms = utils.read_pdb_atoms(str(f))
        assert len(atoms) == 2
        assert atoms[0]["atom_num"] == 1
        assert atoms[1]["atom_num"] == 3

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.pdb"
        f.write_text("")
        atoms = utils.read_pdb_atoms(str(f))
        assert atoms == []

    def test_element_fallback_from_atom_name(self, tmp_path):
        f = tmp_path / "test.pdb"
        f.write_text("ATOM      1  CA  SER A   1      0.000   0.000   0.000\n")
        atoms = utils.read_pdb_atoms(str(f))
        assert atoms[0]["element"] == "C"


# ── compute_bounding_box ─────────────────────────────────────────────────────


class TestComputeBoundingBoxExtended:
    def test_single_atom(self):
        atoms = [{"x": 5.0, "y": 5.0, "z": 5.0}]
        center, size = utils.compute_bounding_box(atoms)
        assert center == (5.0, 5.0, 5.0)
        assert size == (0.0, 0.0, 0.0)

    def test_many_atoms(self):
        atoms = [
            {"x": -1.0, "y": -2.0, "z": -3.0},
            {"x": 1.0, "y": 2.0, "z": 3.0},
            {"x": 0.0, "y": 0.0, "z": 0.0},
        ]
        center, size = utils.compute_bounding_box(atoms)
        assert center == (0.0, 0.0, 0.0)
        assert size == (2.0, 4.0, 6.0)


# ── _sanitize_pdbqt_for_rdkit ────────────────────────────────────────────────


class TestSanitizePdbqtForRdkitExtended:
    def test_two_letter_element(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM      1  Cl  UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 Cl\n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        line = block.strip()
        assert line[76:78].strip() == "Cl"

    def test_unknown_ad_type_passes_through(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM      1  X   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 X \n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        line = block.strip()
        assert line[76:78].strip() == "X"

    def test_skips_non_atom_lines(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "REMARK   1\n"
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n"
            "TER\n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        lines = block.strip().splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("ATOM")

    def test_hd_mapping(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM      1  H   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 HD\n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        line = block.strip()
        assert line[76:78].strip() == "H"

    def test_na_mapping(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM      1  N   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 NA\n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        line = block.strip()
        assert line[76:78].strip() == "N"

    def test_sa_mapping(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM      1  S   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 SA\n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        line = block.strip()
        assert line[76:78].strip() == "S"


# ── pdb_chain_to_smiles ──────────────────────────────────────────────────────


class TestPdbChainToSmiles:
    @patch("autodock.utils.find_conda_tool")
    @patch("autodock.utils.extract_chain_from_pdb")
    @patch("subprocess.run")
    def test_success(self, mock_run, mock_extract, mock_find_tool, tmp_path):
        mock_find_tool.return_value = "/fake/obabel"

        def fake_extract(pdb_path, chain_id, output_pdb):
            with open(output_pdb, "w") as fh:
                fh.write("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
            return output_pdb

        mock_extract.side_effect = fake_extract

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "C1CCCCC1\tchain.pdb\n"
        mock_run.return_value = mock_result

        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")

        smiles = utils.pdb_chain_to_smiles(str(pdb_file), "A")
        assert smiles == "C1CCCCC1"

    @patch("autodock.utils.find_conda_tool")
    def test_obabel_not_found(self, mock_find_tool, tmp_path):
        mock_find_tool.return_value = None
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
        assert utils.pdb_chain_to_smiles(str(pdb_file), "A") is None

    @patch("autodock.utils.find_conda_tool")
    @patch("autodock.utils.extract_chain_from_pdb")
    @patch("subprocess.run")
    def test_obabel_failure(self, mock_run, mock_extract, mock_find_tool, tmp_path):
        mock_find_tool.return_value = "/fake/obabel"

        def fake_extract(pdb_path, chain_id, output_pdb):
            with open(output_pdb, "w") as fh:
                fh.write("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
            return output_pdb

        mock_extract.side_effect = fake_extract

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")

        assert utils.pdb_chain_to_smiles(str(pdb_file), "A") is None

    @patch("autodock.utils.find_conda_tool")
    @patch("autodock.utils.extract_chain_from_pdb")
    @patch("subprocess.run")
    def test_obabel_exception(self, mock_run, mock_extract, mock_find_tool, tmp_path):
        mock_find_tool.return_value = "/fake/obabel"

        def fake_extract(pdb_path, chain_id, output_pdb):
            with open(output_pdb, "w") as fh:
                fh.write("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")
            return output_pdb

        mock_extract.side_effect = fake_extract
        mock_run.side_effect = OSError("subprocess failed")

        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\n")

        assert utils.pdb_chain_to_smiles(str(pdb_file), "A") is None


# ── rmsd_matrix ──────────────────────────────────────────────────────────────


class TestRmsdMatrixExtended:
    @pytest.mark.skipif(
        not __import__("autodock.core", fromlist=["_HAVE_RDKIT"])._HAVE_RDKIT,
        reason="RDKit not available",
    )
    def test_two_different_molecules(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol1 = Chem.MolFromSmiles("CC")
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        mol2 = Chem.MolFromSmiles("CCC")
        AllChem.EmbedMolecule(mol2, randomSeed=43)
        poses = [mol1, mol2]
        mat = utils.rmsd_matrix(poses)
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert mat[1, 1] == pytest.approx(0.0, abs=1e-6)
        assert mat[0, 1] > 0

    @pytest.mark.skipif(
        not __import__("autodock.core", fromlist=["_HAVE_RDKIT"])._HAVE_RDKIT,
        reason="RDKit not available",
    )
    def test_exception_handling(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol1 = Chem.MolFromSmiles("CC")
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        mol2 = Chem.MolFromSmiles("CC")
        poses = [mol1, mol2]
        mat = utils.rmsd_matrix(poses)
        assert mat.shape == (2, 2)
        assert mat[0, 1] == 999.0
        assert mat[1, 0] == 999.0

    def test_empty_list(self):
        mat = utils.rmsd_matrix([])
        assert mat.shape == (0, 0)


# ── StructureCache ───────────────────────────────────────────────────────────


class TestStructureCacheExtended:
    def test_put(self, tmp_path):
        cache = utils.StructureCache(cache_dir=str(tmp_path))
        src = tmp_path / "source.pdb"
        src.write_text("A" * 200)
        dest = cache.put("testkey", str(src))
        assert (tmp_path / "testkey.pdb").exists()
        assert dest == str(tmp_path / "testkey.pdb")

    def test_clear(self, tmp_path):
        cache = utils.StructureCache(cache_dir=str(tmp_path))
        (tmp_path / "a.pdb").write_text("A" * 200)
        (tmp_path / "b.sdf").write_text("B" * 200)
        count = cache.clear()
        assert count == 2
        assert not any(f.is_file() for f in tmp_path.iterdir())

    def test_info(self, tmp_path):
        cache = utils.StructureCache(cache_dir=str(tmp_path))
        (tmp_path / "a.pdb").write_text("A" * 100)
        (tmp_path / "b.pdb").write_text("B" * 200)
        info = cache.info()
        assert info["n_files"] == 2
        assert info["total_bytes"] == 300
        assert info["cache_dir"] == str(tmp_path)

    def test_get_returns_none_for_small_file(self, tmp_path):
        cache = utils.StructureCache(cache_dir=str(tmp_path))
        (tmp_path / "small.pdb").write_text("tiny")
        assert cache.get("small") is None

    def test_get_returns_path_for_large_file(self, tmp_path):
        cache = utils.StructureCache(cache_dir=str(tmp_path))
        (tmp_path / "large.pdb").write_text("A" * 200)
        assert cache.get("large") == str(tmp_path / "large.pdb")


# ── extract_chain_from_pdb ───────────────────────────────────────────────────


class TestExtractChainFromPdb:
    def test_extract_chain_a(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "ATOM      2  CA  SER A   1      1.000   0.000   0.000\n"
            "ATOM      3  N   SER B   1      5.000   0.000   0.000\n"
            "END\n"
        )
        out = tmp_path / "chain_a.pdb"
        utils.extract_chain_from_pdb(str(pdb), "A", str(out))
        lines = out.read_text().splitlines()
        atom_lines = [ln for ln in lines if ln.startswith("ATOM")]
        assert len(atom_lines) == 2
        assert all(" A " in ln for ln in atom_lines)

    def test_extract_chain_b(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "ATOM      2  CA  SER B   1      1.000   0.000   0.000\n"
            "END\n"
        )
        out = tmp_path / "chain_b.pdb"
        utils.extract_chain_from_pdb(str(pdb), "B", str(out))
        lines = out.read_text().splitlines()
        atom_lines = [ln for ln in lines if ln.startswith("ATOM")]
        assert len(atom_lines) == 1
        assert " B " in atom_lines[0]

    def test_returns_string_when_no_output(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        result = utils.extract_chain_from_pdb(str(pdb), "A")
        assert isinstance(result, str)
        assert "ATOM" in result
        assert "END" in result

    def test_preserves_header(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "HEADER    TEST\n"
            "REMARK   1\n"
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "END\n"
        )
        result = utils.extract_chain_from_pdb(str(pdb), "A")
        assert "HEADER" in result
        assert "REMARK" in result

    def test_conect_records_included(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "ATOM      2  CA  SER A   1      1.000   0.000   0.000\n"
            "CONECT    1    2\n"
            "END\n"
        )
        out = tmp_path / "chain_a.pdb"
        utils.extract_chain_from_pdb(str(pdb), "A", str(out))
        text = out.read_text()
        assert "CONECT" in text

    def test_conect_records_excluded_when_disabled(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "ATOM      2  CA  SER A   1      1.000   0.000   0.000\n"
            "CONECT    1    2\n"
            "END\n"
        )
        out = tmp_path / "chain_a.pdb"
        utils.extract_chain_from_pdb(str(pdb), "A", str(out), include_connect=False)
        text = out.read_text()
        assert "CONECT" not in text

    def test_hetatm_included(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2  O   HOH A   2      1.000   1.000   1.000\n"
            "END\n"
        )
        out = tmp_path / "chain_a.pdb"
        utils.extract_chain_from_pdb(str(pdb), "A", str(out))
        text = out.read_text()
        assert "HETATM" in text

    def test_empty_result(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        out = tmp_path / "chain_z.pdb"
        utils.extract_chain_from_pdb(str(pdb), "Z", str(out))
        text = out.read_text()
        # END line is always preserved; no ATOM/HETATM lines for chain Z
        assert "END" in text
        assert not any(ln.startswith("ATOM") or ln.startswith("HETATM") for ln in text.splitlines())


# ── safe_pdb_slice ───────────────────────────────────────────────────────────


class TestSafePdbSlice:
    def test_normal_slice(self):
        line = "ATOM      1  N   SER A   1      0.000   0.000   0.000"
        assert utils.safe_pdb_slice(line, 12, 16) == "N"
        assert utils.safe_pdb_slice(line, 17, 20) == "SER"

    def test_truncated_line(self):
        short = "ATOM      1  N   SER"
        assert utils.safe_pdb_slice(short, 17, 20) == "SER"
        assert utils.safe_pdb_slice(short, 17, 20, default="X") == "SER"

    def test_line_shorter_than_start(self):
        short = "ATOM"
        assert utils.safe_pdb_slice(short, 17, 20) == ""
        assert utils.safe_pdb_slice(short, 17, 20, default="X") == "X"

    def test_line_between_start_and_end(self):
        line = "ATOM      1  N   SER A"
        assert utils.safe_pdb_slice(line, 17, 20) == "SER"


# ── write_pdb_atoms / _atom_dict_to_pdb_line ─────────────────────────────────


class TestWritePdbAtoms:
    def test_roundtrip(self, tmp_path):
        atoms = [
            {
                "record": "ATOM",
                "atom_num": 1,
                "atom_name": "N",
                "res_name": "SER",
                "chain": "A",
                "res_seq": 1,
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "element": "N",
            },
            {
                "record": "ATOM",
                "atom_num": 2,
                "atom_name": "CA",
                "res_name": "SER",
                "chain": "A",
                "res_seq": 1,
                "x": 2.0,
                "y": 3.0,
                "z": 4.0,
                "element": "C",
            },
        ]
        out = tmp_path / "out.pdb"
        utils.write_pdb_atoms(atoms, str(out))
        text = out.read_text()
        assert text.startswith("ATOM")
        assert text.endswith("END\n")
        assert "SER" in text

    def test_with_header_lines(self, tmp_path):
        atoms = [
            {
                "record": "ATOM",
                "atom_num": 1,
                "atom_name": "N",
                "res_name": "ALA",
                "chain": "A",
                "res_seq": 1,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "element": "N",
            }
        ]
        out = tmp_path / "out.pdb"
        utils.write_pdb_atoms(atoms, str(out), header_lines=["REMARK   1", "HEADER    TEST\n"])
        text = out.read_text()
        lines = text.splitlines()
        assert lines[0] == "REMARK   1"
        assert lines[1] == "HEADER    TEST"
        assert lines[-1] == "END"

    def test_atom_dict_to_pdb_line_short_name(self):
        atom = {
            "record": "ATOM",
            "atom_num": 1,
            "atom_name": "N",
            "res_name": "ALA",
            "chain": "A",
            "res_seq": 1,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "element": "N",
        }
        line = utils._atom_dict_to_pdb_line(atom)
        assert line.startswith("ATOM")
        assert "ALA" in line
        assert line[12:16].strip() == "N"

    def test_atom_dict_to_pdb_line_long_name(self):
        atom = {
            "record": "HETATM",
            "atom_num": 1,
            "atom_name": "FE",
            "res_name": "HEM",
            "chain": "A",
            "res_seq": 1,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "element": "FE",
        }
        line = utils._atom_dict_to_pdb_line(atom)
        assert line.startswith("HETATM")


# ── compute_bounding_box_from_pdb / pdbqt ────────────────────────────────────


class TestComputeBoundingBoxFromPdb:
    def test_with_residue_filter(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000  1.00  0.00           N  \n"
            "ATOM      2  CA  SER A   1      1.000   0.000   0.000  1.00  0.00           C  \n"
            "ATOM      3  N   ALA A   2      5.000   0.000   0.000  1.00  0.00           N  \n"
        )
        center, size = utils.compute_bounding_box_from_pdb(str(pdb), residue_filter={"SER"})
        assert center == (0.5, 0.0, 0.0)
        assert size == (1.0, 0.0, 0.0)


class TestComputeBoundingBoxFromPdbqt:
    def test_basic(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM      1  N   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 NA\n"
            "ATOM      2  C   UNL     1       2.000   4.000   6.000  0.00  0.00    +0.000 C \n"
        )
        center, size = utils.compute_bounding_box_from_pdbqt(str(pdbqt))
        assert center == (1.0, 2.0, 3.0)
        assert size == (2.0, 4.0, 6.0)

    def test_skips_non_atom_lines(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "REMARK   1\n"
            "ATOM      1  N   UNL     1       1.000   2.000   3.000  0.00  0.00    +0.000 N \n"
            "TER\n"
        )
        center, size = utils.compute_bounding_box_from_pdbqt(str(pdbqt))
        assert center == (1.0, 2.0, 3.0)

    def test_malformed_coords_ignored(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM      1  N   UNL     1       1.000   2.000   3.000  0.00  0.00    +0.000 N \n"
            "ATOM      2  C   UNL     1       xxxxx   yyyyy   zzzzz  0.00  0.00    +0.000 C \n"
        )
        center, size = utils.compute_bounding_box_from_pdbqt(str(pdbqt))
        assert center == (1.0, 2.0, 3.0)


# ── filter_pdb_lines ─────────────────────────────────────────────────────────


class TestFilterPdbLinesExtended:
    def test_keep_residues(self, tmp_path):
        inp = tmp_path / "in.pdb"
        inp.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "ATOM      2  CA  ALA A   2      1.000   1.000   1.000\n"
            "HETATM    3  O   LIG A   3      2.000   2.000   2.000\n"
        )
        out = tmp_path / "out.pdb"
        utils.filter_pdb_lines(str(inp), str(out), keep_residues={"ALA", "LIG"})
        lines = out.read_text().splitlines()
        assert len(lines) == 2
        assert any("ALA" in ln for ln in lines)
        assert any("LIG" in ln for ln in lines)

    def test_remove_hetatms(self, tmp_path):
        inp = tmp_path / "in.pdb"
        inp.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2  O   HOH A   2      1.000   1.000   1.000\n"
        )
        out = tmp_path / "out.pdb"
        utils.filter_pdb_lines(str(inp), str(out), remove_hetatms=True)
        lines = out.read_text().splitlines()
        assert len(lines) == 1
        assert "SER" in lines[0]

    def test_remove_water_false_keeps_hoh(self, tmp_path):
        inp = tmp_path / "in.pdb"
        inp.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2  O   HOH A   2      1.000   1.000   1.000\n"
        )
        out = tmp_path / "out.pdb"
        utils.filter_pdb_lines(str(inp), str(out), remove_water=False)
        lines = out.read_text().splitlines()
        assert len(lines) == 2

    def test_non_atom_lines_preserved(self, tmp_path):
        inp = tmp_path / "in.pdb"
        inp.write_text(
            "HEADER    TEST\n"
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "REMARK   1\n"
        )
        out = tmp_path / "out.pdb"
        utils.filter_pdb_lines(str(inp), str(out))
        text = out.read_text()
        assert "HEADER" in text
        assert "REMARK" in text


# ── write_temp_file exception path ───────────────────────────────────────────


class TestWriteTempFileExceptions:
    def test_exception_cleanup(self, monkeypatch):
        import tempfile

        original_mkstemp = tempfile.mkstemp

        def bad_mkstemp(*args, **kwargs):
            fd, path = original_mkstemp(*args, **kwargs)
            os.close(fd)
            # Return a closed fd so os.fdopen raises OSError
            return fd, path

        monkeypatch.setattr(tempfile, "mkstemp", bad_mkstemp)
        with pytest.raises(OSError):
            utils.write_temp_file("hello")
