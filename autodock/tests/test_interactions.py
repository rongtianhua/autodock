"""Tests for autodock.interactions — PLIP/ProLIF interaction detection."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autodock import interactions as intx
from autodock.core import VisualizationError

# ─────────────────────────────────────────────────────────────────────────────
# Complex PDB builder
# ─────────────────────────────────────────────────────────────────────────────


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
        # Ligand side strips non-ATOM lines; receptor keeps everything except END/ENDMDL
        assert "ENDMDL" not in text
        assert "ATOM" in text
        assert "HETATM" in text


# ─────────────────────────────────────────────────────────────────────────────
# Unified detect_interactions
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectInteractions:
    @patch("autodock.interactions.detect_interactions_plip")
    def test_plip_mode(self, mock_plip, tmp_path):
        mock_plip.return_value = [{"type": "H-bond", "resn": "SER"}]
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="plip")
        assert len(result) == 1
        assert result[0]["type"] == "H-bond"

    @patch("autodock.interactions.detect_interactions_prolif")
    def test_prolif_mode(self, mock_prolif, tmp_path):
        mock_prolif.return_value = [{"type": "Hydrophobic", "resn": "ALA"}]
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="prolif")
        assert len(result) == 1

    @patch("autodock.interactions.detect_interactions_plip")
    @patch("autodock.interactions.detect_interactions_prolif")
    def test_both_mode_deduplicates(self, mock_prolif, mock_plip, tmp_path):
        mock_plip.return_value = [
            {"type": "H-bond", "resn": "SER", "resi": 1, "chain": "A"},
            {"type": "H-bond", "resn": "SER", "resi": 1, "chain": "A"},  # duplicate
        ]
        mock_prolif.return_value = [
            {"type": "H-bond", "resn": "SER", "resi": 1, "chain": "A"},  # same as plip
            {"type": "Hydrophobic", "resn": "ALA", "resi": 2, "chain": "A"},
        ]
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="both")
        assert len(result) == 2
        types = {r["type"] for r in result}
        assert types == {"H-bond", "Hydrophobic"}

    @patch("autodock.interactions.detect_interactions_plip")
    @patch("autodock.interactions.detect_interactions_prolif")
    def test_both_mode_plip_falls_back(self, mock_prolif, mock_plip, tmp_path):
        mock_plip.side_effect = VisualizationError("PLIP failed")
        mock_prolif.return_value = [{"type": "Hydrophobic", "resn": "ALA"}]
        result = intx.detect_interactions("rec.pdb", "lig.pdbqt", method="both")
        assert len(result) == 1

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid interaction method"):
            intx.detect_interactions("rec.pdb", "lig.pdbqt", method="invalid")


# ─────────────────────────────────────────────────────────────────────────────
# Interaction categories
# ─────────────────────────────────────────────────────────────────────────────


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
