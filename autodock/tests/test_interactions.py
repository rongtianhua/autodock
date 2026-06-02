"""Comprehensive tests for autodock.interactions — PLIP/ProLIF interaction detection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from autodock import interactions as intx
from autodock.core import VisualizationError

# ── Helpers ──────────────────────────────────────────────────────────────────


class _FakeSeries:
    def __init__(self, values):
        self._values = values

    def sum(self):
        return sum(self._values)


class _FakeDataFrame:
    def __init__(self, data):
        self._data = data

    @property
    def empty(self):
        return len(self._data) == 0

    @property
    def columns(self):
        return list(self._data.keys())

    def __getitem__(self, key):
        return _FakeSeries(self._data[key])


def _make_plip_ligand(hetid="LIG", chain="A", position="1"):
    lig = MagicMock()
    lig.hetid = hetid
    lig.chain = chain
    lig.position = position
    return lig


def _setup_interaction_set(mock_interaction_set, **kwargs):
    """Set all interaction category attributes on a mock interaction set."""
    defaults = {
        "hbonds_ldon": [],
        "hbonds_pdon": [],
        "hydrophobic_contacts": [],
        "pistacking": [],
        "pication_laro": [],
        "pication_paro": [],
        "saltbridge_lneg": [],
        "saltbridge_pneg": [],
        "halogen_bonds": [],
        "water_bridges": [],
        "metal_complexes": [],
    }
    defaults.update(kwargs)
    for attr, val in defaults.items():
        setattr(mock_interaction_set, attr, val)


# ── _build_complex_pdb ───────────────────────────────────────────────────────


class TestBuildComplexPdb:
    def test_basic_merge(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  0.00  0.00    +0.000 C\n"
        )
        out = tmp_path / "complex.pdb"
        intx._build_complex_pdb(str(rec), str(lig), str(out))
        lines = out.read_text().splitlines()
        assert any("SER" in line for line in lines)
        assert any("HETATM" in line and "LIG" in line for line in lines)
        assert lines[-1] == "END"

    def test_skips_non_atom_lines(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        rec.write_text(
            "REMARK   1\nATOM      1  N   SER A   1      0.000   0.000   0.000\nENDMDL\n"
        )
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("REMARK 1\nATOM      1  C   LIG A   1      1.000   2.000   3.000\n")
        out = tmp_path / "complex.pdb"
        intx._build_complex_pdb(str(rec), str(lig), str(out))
        text = out.read_text()
        assert "ENDMDL" not in text
        assert "ATOM" in text
        assert "HETATM" in text

    def test_maps_ad4_atom_types(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  O   LIG A   1      1.000   2.000   3.000  0.00  0.00    +0.000  OA\n"
        )
        out = tmp_path / "complex.pdb"
        intx._build_complex_pdb(str(rec), str(lig), str(out))
        lines = out.read_text().splitlines()
        hetatm_lines = [ln for ln in lines if ln.startswith("HETATM")]
        assert len(hetatm_lines) == 1
        assert hetatm_lines[0][76:78].strip() == "O"

    def test_empty_ligand(self, tmp_path):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("REMARK empty ligand\n")
        out = tmp_path / "complex.pdb"
        intx._build_complex_pdb(str(rec), str(lig), str(out))
        lines = out.read_text().splitlines()
        assert lines[-1] == "END"
        assert not any(ln.startswith("HETATM") for ln in lines)


# ── detect_interactions_plip ─────────────────────────────────────────────────


class TestDetectInteractionsPlip:
    @patch.dict(
        sys.modules,
        {
            "plip": MagicMock(),
            "plip.structure": MagicMock(),
            "plip.structure.preparation": MagicMock(),
        },
    )
    @patch("autodock.interactions._HAVE_PLIP", True)
    def test_plip_success(self, tmp_path):
        mock_pdbcomplex_cls = sys.modules["plip.structure.preparation"].PDBComplex
        mock_mol = MagicMock()
        mock_pdbcomplex_cls.return_value = mock_mol

        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   LIG A   1      1.000   2.000   3.000  0.00  0.00    +0.000 C\n"
        )

        mock_ligand = _make_plip_ligand("LIG", "A", "1")
        mock_mol.ligands = [mock_ligand]

        mock_interaction_set = MagicMock()
        mock_mol.interaction_sets = {"LIG:A:1": mock_interaction_set}

        mock_rec = MagicMock()
        mock_rec.restype = "SER"
        mock_rec.resnr = "1"
        mock_rec.reschain = "A"
        mock_rec.atype = "N"
        mock_rec.distance = 2.8
        mock_rec.d = MagicMock(coords=(1.0, 2.0, 3.0))

        _setup_interaction_set(mock_interaction_set, hbonds_ldon=[mock_rec])

        result = intx.detect_interactions_plip(str(rec), str(lig))
        assert len(result) == 1
        assert result[0]["type"] == "H-bond"
        assert result[0]["resn"] == "SER"
        assert result[0]["resi"] == 1
        assert result[0]["distance"] == 2.8
        assert result[0]["ligand_atoms"] == [{"coords": (1.0, 2.0, 3.0)}]

    @patch.dict(
        sys.modules,
        {
            "plip": MagicMock(),
            "plip.structure": MagicMock(),
            "plip.structure.preparation": MagicMock(),
        },
    )
    @patch("autodock.interactions._HAVE_PLIP", True)
    def test_plip_multiple_interaction_types(self, tmp_path):
        mock_pdbcomplex_cls = sys.modules["plip.structure.preparation"].PDBComplex
        mock_mol = MagicMock()
        mock_pdbcomplex_cls.return_value = mock_mol

        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

        mock_ligand = _make_plip_ligand("LIG", "A", "1")
        mock_mol.ligands = [mock_ligand]
        mock_interaction_set = MagicMock()
        mock_mol.interaction_sets = {"LIG:A:1": mock_interaction_set}

        hb_rec = MagicMock()
        hb_rec.restype = "SER"
        hb_rec.resnr = "1"
        hb_rec.reschain = "A"
        hb_rec.atype = "N"
        hb_rec.distance = 2.8
        hb_rec.d = MagicMock(coords=(1.0, 2.0, 3.0))

        hp_rec = MagicMock()
        hp_rec.restype = "ALA"
        hp_rec.resnr = "2"
        hp_rec.reschain = "A"
        hp_rec.atype = "CB"
        hp_rec.distance = 3.5
        hp_rec.ligatom = MagicMock(coords=(4.0, 5.0, 6.0))

        _setup_interaction_set(
            mock_interaction_set,
            hbonds_ldon=[hb_rec],
            hydrophobic_contacts=[hp_rec],
        )

        result = intx.detect_interactions_plip(str(rec), str(lig))
        assert len(result) == 2
        types = {r["type"] for r in result}
        assert types == {"H-bond", "Hydrophobic"}

    @patch("autodock.interactions._HAVE_PLIP", False)
    def test_plip_not_available_raises(self):
        with pytest.raises(VisualizationError, match="PLIP not available"):
            intx.detect_interactions_plip("rec.pdb", "lig.pdbqt")

    @patch.dict(
        sys.modules,
        {
            "plip": MagicMock(),
            "plip.structure": MagicMock(),
            "plip.structure.preparation": MagicMock(),
        },
    )
    @patch("autodock.interactions._HAVE_PLIP", True)
    def test_plip_analysis_failure_raises(self, tmp_path):
        mock_pdbcomplex_cls = sys.modules["plip.structure.preparation"].PDBComplex
        mock_mol = MagicMock()
        mock_pdbcomplex_cls.return_value = mock_mol
        mock_mol.load_pdb.side_effect = RuntimeError("corrupt PDB")

        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

        with pytest.raises(VisualizationError, match="PLIP analysis failed"):
            intx.detect_interactions_plip(str(rec), str(lig))

    @patch.dict(
        sys.modules,
        {
            "plip": MagicMock(),
            "plip.structure": MagicMock(),
            "plip.structure.preparation": MagicMock(),
        },
    )
    @patch("autodock.interactions._HAVE_PLIP", True)
    def test_plip_no_ligands_returns_empty(self, tmp_path):
        mock_pdbcomplex_cls = sys.modules["plip.structure.preparation"].PDBComplex
        mock_mol = MagicMock()
        mock_pdbcomplex_cls.return_value = mock_mol
        mock_mol.ligands = []

        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

        result = intx.detect_interactions_plip(str(rec), str(lig))
        assert result == []

    @patch.dict(
        sys.modules,
        {
            "plip": MagicMock(),
            "plip.structure": MagicMock(),
            "plip.structure.preparation": MagicMock(),
        },
    )
    @patch("autodock.interactions._HAVE_PLIP", True)
    def test_plip_ligand_not_LIG_skipped(self, tmp_path):
        mock_pdbcomplex_cls = sys.modules["plip.structure.preparation"].PDBComplex
        mock_mol = MagicMock()
        mock_pdbcomplex_cls.return_value = mock_mol
        mock_ligand = _make_plip_ligand("UNK", "A", "1")
        mock_mol.ligands = [mock_ligand]
        mock_mol.interaction_sets = {}

        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

        result = intx.detect_interactions_plip(str(rec), str(lig))
        assert result == []

    @patch.dict(
        sys.modules,
        {
            "plip": MagicMock(),
            "plip.structure": MagicMock(),
            "plip.structure.preparation": MagicMock(),
        },
    )
    @patch("autodock.interactions._HAVE_PLIP", True)
    def test_plip_malformed_record_skipped(self, tmp_path):
        mock_pdbcomplex_cls = sys.modules["plip.structure.preparation"].PDBComplex
        mock_mol = MagicMock()
        mock_pdbcomplex_cls.return_value = mock_mol
        mock_ligand = _make_plip_ligand("LIG", "A", "1")
        mock_mol.ligands = [mock_ligand]

        mock_interaction_set = MagicMock()
        mock_mol.interaction_sets = {"LIG:A:1": mock_interaction_set}

        bad_rec = MagicMock()
        bad_rec.restype = "SER"
        bad_rec.resnr = "not_a_number"
        bad_rec.reschain = "A"
        bad_rec.atype = "N"
        bad_rec.distance = 2.8

        _setup_interaction_set(mock_interaction_set, hbonds_ldon=[bad_rec])

        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000\nEND\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")

        result = intx.detect_interactions_plip(str(rec), str(lig))
        assert result == []


# ── detect_interactions_prolif ───────────────────────────────────────────────


class TestDetectInteractionsProlif:
    def test_prolif_integration(self):
        """End-to-end test with real PDB + PDBQT data."""
        rec_pdb = "mettl8_docking/METTL8_af_iso2.pdb"
        lig_pdbqt = "mettl8_docking/idebenone.pdbqt"

        result = intx.detect_interactions_prolif(rec_pdb, lig_pdbqt)
        assert isinstance(result, list)
        assert len(result) > 0

        # Verify output schema matches PLIP format
        for r in result:
            assert "type" in r
            assert "color" in r
            assert "resn" in r
            assert "resi" in r
            assert "chain" in r
            assert "distance" in r
            assert "description" in r
            assert "ligand_atoms" in r
            for atom in r["ligand_atoms"]:
                assert "coords" in atom
                assert len(atom["coords"]) == 3

    def test_parse_smiles_from_pdbqt(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("REMARK SMILES CCO\nATOM      1  C   LIG A   1      1.000   2.000   3.000\n")
        smiles = intx._parse_smiles_from_pdbqt(str(lig))
        assert smiles == "CCO"

    def test_parse_smiles_missing_returns_none(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")
        smiles = intx._parse_smiles_from_pdbqt(str(lig))
        assert smiles is None

    def test_build_ligand_mol_fallback_no_smiles(self, tmp_path):
        """PDBQT without SMILES should fall back to direct PDBQT read."""
        lig = tmp_path / "lig.pdbqt"
        lig.write_text(
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C \n"
            "ATOM      2  O   UNL     1       1.200   0.000   0.000  0.00  0.00    +0.000 OA\n"
        )
        mol = intx._build_ligand_mol_for_prolif(str(lig))
        assert mol.GetNumAtoms() >= 2

    @patch("autodock.interactions._HAVE_RDKIT", False)
    @patch("autodock.interactions._HAVE_PROLIF", True)
    def test_prolif_missing_rdkit_raises(self):
        with pytest.raises(VisualizationError, match="ProLIF requires rdkit"):
            intx.detect_interactions_prolif("rec.pdb", "lig.pdbqt")

    @patch("autodock.interactions._HAVE_RDKIT", True)
    @patch("autodock.interactions._HAVE_PROLIF", False)
    def test_prolif_missing_prolif_raises(self):
        with pytest.raises(VisualizationError, match="ProLIF requires rdkit"):
            intx.detect_interactions_prolif("rec.pdb", "lig.pdbqt")

    def test_prolif_invalid_receptor_raises(self, tmp_path):
        rec = tmp_path / "empty.pdb"
        rec.write_text("")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000\n")
        with pytest.raises(VisualizationError):
            intx.detect_interactions_prolif(str(rec), str(lig))


# ── Unified detect_interactions ──────────────────────────────────────────────


class TestDetectInteractionsUnified:
    @patch("autodock.interactions.detect_interactions_plip")
    def test_plip_backend_success(self, mock_plip, tmp_path):
        mock_plip.return_value = [{"type": "H-bond", "resn": "SER"}]
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="plip")
        assert len(result) == 1
        assert result[0]["type"] == "H-bond"

    @patch("autodock.interactions.detect_interactions_prolif")
    def test_prolif_backend_success(self, mock_prolif, tmp_path):
        mock_prolif.return_value = [{"type": "Hydrophobic", "resn": "ALA"}]
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="prolif")
        assert len(result) == 1
        assert result[0]["type"] == "Hydrophobic"

    @patch("autodock.interactions.detect_interactions_plip")
    @patch("autodock.interactions.detect_interactions_prolif")
    def test_both_backends_fail_graceful_fallback(self, mock_prolif, mock_plip):
        mock_plip.side_effect = VisualizationError("PLIP failed")
        mock_prolif.side_effect = VisualizationError("ProLIF failed")
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="both")
        assert result == []

    @patch("autodock.interactions.detect_interactions_plip")
    @patch("autodock.interactions.detect_interactions_prolif")
    def test_both_mode_prolif_also_falls_back(self, mock_prolif, mock_plip):
        mock_plip.side_effect = VisualizationError("PLIP failed")
        mock_prolif.side_effect = RuntimeError("ProLIF failed")
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="both")
        assert result == []

    @patch("autodock.interactions.detect_interactions_plip")
    def test_empty_inputs_handled(self, mock_plip, tmp_path):
        empty_rec = tmp_path / "empty.pdb"
        empty_rec.write_text("")
        empty_lig = tmp_path / "empty.pdbqt"
        empty_lig.write_text("")

        mock_plip.return_value = []
        result = intx.detect_interactions(str(empty_rec), str(empty_lig), method="plip")
        assert result == []

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid interaction method"):
            intx.detect_interactions("rec.pdb", "lig.pdbqt", method="invalid")

    @patch("autodock.interactions.detect_interactions_plip")
    def test_plip_method_raises_on_failure(self, mock_plip):
        mock_plip.side_effect = VisualizationError("PLIP failed")
        with pytest.raises(VisualizationError, match="PLIP failed"):
            intx.detect_interactions("rec.pdb", "lig.pdbqt", method="plip")

    @patch("autodock.interactions.detect_interactions_prolif")
    def test_prolif_method_raises_on_failure(self, mock_prolif):
        mock_prolif.side_effect = VisualizationError("ProLIF failed")
        with pytest.raises(VisualizationError, match="ProLIF failed"):
            intx.detect_interactions("rec.pdb", "lig.pdbqt", method="prolif")

    @patch("autodock.interactions.detect_interactions_plip")
    @patch("autodock.interactions.detect_interactions_prolif")
    def test_both_mode_plip_only(self, mock_prolif, mock_plip):
        mock_plip.return_value = [{"type": "H-bond", "resn": "SER", "resi": 1, "chain": "A"}]
        mock_prolif.return_value = []
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="both")
        assert len(result) == 1

    @patch("autodock.interactions.detect_interactions_plip")
    @patch("autodock.interactions.detect_interactions_prolif")
    def test_both_mode_prolif_only(self, mock_prolif, mock_plip):
        mock_plip.side_effect = VisualizationError("PLIP failed")
        mock_prolif.return_value = [{"type": "Hydrophobic", "resn": "ALA", "resi": 2, "chain": "A"}]
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="both")
        assert len(result) == 1
        assert result[0]["type"] == "Hydrophobic"


# ── Interaction categories ───────────────────────────────────────────────────


class TestInteractionCategories:
    def test_categories_cover_major_types(self):
        types = {cat[2] for cat in intx.INTERACTION_CATEGORIES}
        expected = {
            "H-bond",
            "Hydrophobic",
            "π-π",
            "π-cation",
            "Salt bridge",
            "Halogen bond",
            "Water bridge",
            "Metal complex",
        }
        assert expected.issubset(types)
