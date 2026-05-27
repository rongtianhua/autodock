"""Tests for autodock.utils — file helpers, PDB parsing, coordinate math."""

from __future__ import annotations

import os

import pytest

from autodock import utils


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        d = tmp_path / "sub" / "dir"
        result = utils.ensure_dir(d)
        assert os.path.isdir(d)
        assert str(result) == str(d)

    def test_existing_directory(self, tmp_path):
        d = tmp_path / "existing"
        d.mkdir()
        result = utils.ensure_dir(d)
        assert str(result) == str(d)


class TestReadPdbAtoms:
    def test_parse_atom(self, tmp_path):
        f = tmp_path / "test.pdb"
        f.write_text(
            "ATOM      1  N   SER A   1      10.000  20.000  30.000  1.00  0.00           N  \n"
        )
        atoms = utils.read_pdb_atoms(str(f))
        assert len(atoms) == 1
        assert atoms[0]["element"] == "N"
        assert atoms[0]["x"] == 10.0
        assert atoms[0]["y"] == 20.0
        assert atoms[0]["z"] == 30.0

    def test_skip_remark(self, tmp_path):
        f = tmp_path / "test.pdb"
        f.write_text("REMARK   1\n")
        atoms = utils.read_pdb_atoms(str(f))
        assert atoms == []


class TestComputeBoundingBox:
    def test_from_atom_list(self):
        atoms = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 10.0, "y": 5.0, "z": -3.0},
        ]
        center, size = utils.compute_bounding_box(atoms)
        assert center == (5.0, 2.5, -1.5)
        assert size == (10.0, 5.0, 3.0)

    def test_empty_atoms(self):
        center, size = utils.compute_bounding_box([])
        assert center == (0.0, 0.0, 0.0)
        assert size == (10.0, 10.0, 10.0)


class TestRmsdMatrix:
    @pytest.mark.skipif(
        not __import__("autodock.core", fromlist=["_HAVE_RDKIT"])._HAVE_RDKIT,
        reason="RDKit not available",
    )
    def test_identity(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        poses = [mol, mol]
        mat = utils.rmsd_matrix(poses)
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert mat[0, 1] == pytest.approx(0.0, abs=1e-6)


class TestFilterPdbLines:
    def test_remove_water(self, tmp_path):
        inp = tmp_path / "in.pdb"
        inp.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2  O   HOH A   2      1.000   1.000   1.000\n"
        )
        out = tmp_path / "out.pdb"
        utils.filter_pdb_lines(str(inp), str(out), remove_water=True)
        lines = out.read_text().splitlines()
        assert len(lines) == 1
        assert "SER" in lines[0]

    def test_keep_water(self, tmp_path):
        inp = tmp_path / "in.pdb"
        inp.write_text(
            "ATOM      1  N   SER A   1      0.000   0.000   0.000\n"
            "HETATM    2  O   HOH A   2      1.000   1.000   1.000\n"
        )
        out = tmp_path / "out.pdb"
        utils.filter_pdb_lines(str(inp), str(out), remove_water=False)
        lines = out.read_text().splitlines()
        assert len(lines) == 2


class TestSanitizePdbqtForRdkit:
    def test_replaces_ad4_atom_types(self, tmp_path):
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 A \n"
            "ATOM      2  O   UNL     1       1.000   1.000   1.000  1.00  0.00     0.000 OA\n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        lines = block.strip().splitlines()
        # Element is written at PDB cols 77-78 (0-based positions 76-77)
        assert lines[0][76] == " " and lines[0][77] == "C"
        assert lines[1][76] == " " and lines[1][77] == "O"

    def test_fixes_g_atom_name(self, tmp_path):
        """Atom name 'G' triggers RDKit 'Element G not found' — must be remapped."""
        pdbqt = tmp_path / "test.pdbqt"
        # Use type 'C' (not 'G0') so the atom is kept and renamed
        pdbqt.write_text(
            "ATOM      1  G   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        line = block.strip()
        # atom name should become ' C  '
        assert line[12:16] == " C  "
        # element column should be C at 0-based position 76-77
        assert line[76:78].strip() == "C"

    def test_skips_g0_ghost_atoms(self, tmp_path):
        """Ghost atoms (name 'G' + type 'G0') must be skipped entirely."""
        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM      1  G   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 G0\n"
            "ATOM      2  C   UNL     1       1.000   1.000   1.000  1.00  0.00     0.000 C\n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        lines = block.strip().splitlines()
        assert len(lines) == 1
        assert lines[0][12:16].strip() == "C"

    @pytest.mark.skipif(
        not __import__("autodock.core", fromlist=["_HAVE_RDKIT"])._HAVE_RDKIT,
        reason="RDKit not available",
    )
    def test_two_letter_element_preserved(self, tmp_path):
        """Chlorine (Cl) and other two-letter elements must survive sanitization."""
        from rdkit import Chem

        pdbqt = tmp_path / "test.pdbqt"
        pdbqt.write_text(
            "ATOM     28  Cl  UNL     1      -5.651   1.205   0.772  1.00  0.00    -0.123 Cl\n"
        )
        block = utils._sanitize_pdbqt_for_rdkit(str(pdbqt))
        mol = Chem.MolFromPDBBlock(block, removeHs=True)
        assert mol is not None
        assert mol.GetNumAtoms() == 1
        assert mol.GetAtomWithIdx(0).GetSymbol() == "Cl"


class TestExtractLigandFromPdb:
    @pytest.mark.skipif(
        not __import__("autodock.core", fromlist=["_HAVE_RDKIT"])._HAVE_RDKIT,
        reason="RDKit not available",
    )
    def test_multi_copy_asu_takes_largest(self, tmp_path):
        """PDBs with multiple copies of the same ligand (different chains/resi)
        should return only the largest copy."""
        pdb = tmp_path / "multi.pdb"
        pdb.write_text(
            "HETATM    1  C   LIG A   1       0.000   0.000   0.000\n"
            "HETATM    2  C   LIG A   1       1.000   0.000   0.000\n"
            "HETATM    3  O   LIG B   2       5.000   0.000   0.000\n"
        )
        mol, sdf = utils.extract_ligand_from_pdb(str(pdb), "LIG")
        assert mol is not None
        # Chain A has 2 atoms, chain B has 1 atom — should pick chain A
        assert mol.GetNumAtoms() >= 2

    @pytest.mark.skipif(
        not __import__("autodock.core", fromlist=["_HAVE_RDKIT"])._HAVE_RDKIT,
        reason="RDKit not available",
    )
    def test_multi_fragment_keeps_largest(self, tmp_path):
        """If a single ligand copy still has multiple fragments, keep the largest."""
        from rdkit import Chem

        pdb = tmp_path / "frags.pdb"
        # Two disconnected heavy atoms = two fragments
        pdb.write_text(
            "HETATM    1  C   LIG A   1       0.000   0.000   0.000\n"
            "HETATM    2  O   LIG A   1      10.000   0.000   0.000\n"
        )
        mol, sdf = utils.extract_ligand_from_pdb(str(pdb), "LIG")
        assert mol is not None
        frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        assert len(frags) == 1


class TestStructureCache:
    def test_cache_path(self, tmp_path):
        cache = utils.StructureCache(cache_dir=str(tmp_path))
        path = cache._cache_path("test", ".pdb")
        assert str(tmp_path) in str(path)
        assert path.name == "test.pdb"

    def test_cache_get(self, tmp_path):
        cache = utils.StructureCache(cache_dir=str(tmp_path))
        assert cache.get("test") is None
        # Write >100 bytes to satisfy size threshold
        (tmp_path / "test.pdb").write_text("A" * 200)
        assert cache.get("test") is not None
