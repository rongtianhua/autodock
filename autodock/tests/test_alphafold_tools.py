"""Tests for autodock.alphafold_tools."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autodock import alphafold_tools


def _pdb_atom(serial, name, resname, chain, resseq, x, y, z, occ, bfac):
    """Return a properly formatted PDB ATOM line (66 chars, B-factor cols 61-66)."""
    return (
        f"ATOM  {serial:5d} {name:<4s}{resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}"
    )


class TestPLDDTThresholds:
    """Tests for pLDDT threshold constants."""

    def test_threshold_values(self):
        assert alphafold_tools.PLDDTThresholds.VERY_HIGH == 90.0
        assert alphafold_tools.PLDDTThresholds.HIGH == 70.0
        assert alphafold_tools.PLDDTThresholds.LOW == 50.0


class TestParsePLDDTFromPDB:
    """Tests for _parse_plddt_from_pdb."""

    def test_extracts_plddt_from_bfactor(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        lines = [
            _pdb_atom(1, "N  ", "SER", "A", 1, 10.0, 10.0, 10.0, 1.0, 95.0),
            _pdb_atom(2, "CA ", "SER", "A", 1, 11.0, 10.0, 10.0, 1.0, 92.0),
            _pdb_atom(3, "C  ", "SER", "A", 1, 12.0, 10.0, 10.0, 1.0, 90.0),
            _pdb_atom(4, "N  ", "SER", "A", 2, 13.0, 10.0, 10.0, 1.0, 45.0),
            _pdb_atom(5, "CA ", "SER", "A", 2, 14.0, 10.0, 10.0, 1.0, 48.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        vals, residues = alphafold_tools._parse_plddt_from_pdb(str(pdb))
        # Two residues; CA pLDDT takes precedence when available
        assert vals == [92.0, 48.0]
        assert residues == [("A", 1), ("A", 2)]

    def test_skips_non_atom_lines(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        het_line = "HETATM    2  O   HOH A 100       1.000   1.000   1.000  1.00 10.00"
        lines = [
            "REMARK   1",
            _pdb_atom(1, "CA ", "ALA", "A", 1, 0.0, 0.0, 0.0, 1.0, 88.0),
            het_line,
            "END",
        ]
        pdb.write_text("\n".join(lines) + "\n")
        vals, residues = alphafold_tools._parse_plddt_from_pdb(str(pdb))
        assert vals == [88.0]
        assert residues == [("A", 1)]

    def test_empty_file_returns_empty(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        pdb.write_text("")
        vals, residues = alphafold_tools._parse_plddt_from_pdb(str(pdb))
        assert vals == []
        assert residues == []

    def test_invalid_bfactor_skipped(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        bad_line = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 xx.xx"
        pdb.write_text(bad_line + "\n")
        vals, residues = alphafold_tools._parse_plddt_from_pdb(str(pdb))
        assert vals == []


class TestParsePLDDTFromCIF:
    """Tests for _parse_plddt_from_cif."""

    def test_returns_empty_when_gemmi_unavailable(self, tmp_path):
        with patch.dict("sys.modules", {"gemmi": None}):
            cif = tmp_path / "af.cif"
            cif.write_text("")
            with pytest.raises(ImportError, match="gemmi"):
                alphafold_tools._parse_plddt_from_cif(str(cif))

    def test_returns_empty_on_bad_cif(self, tmp_path):
        mock_doc = MagicMock()
        mock_block = MagicMock()
        mock_block.find_mmcif_category.side_effect = ValueError("no category")
        mock_doc.sole_block.return_value = mock_block

        with patch.object(alphafold_tools, "gemmi", create=True):
            with patch("gemmi.cif.read", return_value=mock_doc):
                cif = tmp_path / "af.cif"
                cif.write_text("")
                vals, residues = alphafold_tools._parse_plddt_from_cif(str(cif))
                assert vals == []
                assert residues == []


class TestAssessAlphaFoldQuality:
    """Tests for assess_alphafold_quality."""

    def test_high_confidence_structure(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        lines = [
            _pdb_atom(1, "CA ", "SER", "A", 1, 0.0, 0.0, 0.0, 1.0, 95.0),
            _pdb_atom(2, "CA ", "SER", "A", 2, 1.0, 0.0, 0.0, 1.0, 92.0),
            _pdb_atom(3, "CA ", "SER", "A", 3, 2.0, 0.0, 0.0, 1.0, 88.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        result = alphafold_tools.assess_alphafold_quality(
            str(pdb),
            plddt_threshold_high=70.0,
            plddt_threshold_low=50.0,
        )
        assert result["mean_plddt"] == pytest.approx(91.67, abs=0.1)
        assert result["suitable_for_docking"] is True
        assert result["warning"] is None

    def test_low_confidence_structure(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        lines = [
            _pdb_atom(1, "CA ", "SER", "A", 1, 0.0, 0.0, 0.0, 1.0, 45.0),
            _pdb_atom(2, "CA ", "SER", "A", 2, 1.0, 0.0, 0.0, 1.0, 42.0),
            _pdb_atom(3, "CA ", "SER", "A", 3, 2.0, 0.0, 0.0, 1.0, 38.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        result = alphafold_tools.assess_alphafold_quality(
            str(pdb),
            plddt_threshold_high=70.0,
            plddt_threshold_low=50.0,
        )
        assert result["suitable_for_docking"] is False
        assert result["warning"] is not None
        assert "Low overall confidence" in result["warning"]

    def test_low_confidence_regions_detected(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        lines = [
            _pdb_atom(1, "CA ", "SER", "A", 1, 0.0, 0.0, 0.0, 1.0, 95.0),
            _pdb_atom(2, "CA ", "SER", "A", 2, 1.0, 0.0, 0.0, 1.0, 45.0),
            _pdb_atom(3, "CA ", "SER", "A", 3, 2.0, 0.0, 0.0, 1.0, 40.0),
            _pdb_atom(4, "CA ", "SER", "A", 4, 3.0, 0.0, 0.0, 1.0, 95.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        result = alphafold_tools.assess_alphafold_quality(str(pdb))
        regions = result["low_confidence_regions"]
        assert len(regions) == 1
        assert regions[0]["chain"] == "A"
        assert regions[0]["start"] == 2
        assert regions[0]["end"] == 4
        assert regions[0]["min_plddt"] == 40.0

    def test_empty_pdb_returns_error_dict(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        pdb.write_text("REMARK 1\n")
        result = alphafold_tools.assess_alphafold_quality(str(pdb))
        assert result["n_residues"] == 0
        assert result["suitable_for_docking"] is False
        assert "No ATOM/HETATM" in (result["warning"] or "")

    def test_cif_extension_routes_to_cif_parser(self, tmp_path):
        cif = tmp_path / "af.cif"
        cif.write_text("")
        with patch.object(
            alphafold_tools, "_parse_plddt_from_cif", return_value=([], [])
        ) as mock_cif:
            result = alphafold_tools.assess_alphafold_quality(str(cif))
            mock_cif.assert_called_once_with(str(cif))
        assert result["n_residues"] == 0

    def test_custom_thresholds(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        lines = [
            _pdb_atom(1, "CA ", "SER", "A", 1, 0.0, 0.0, 0.0, 1.0, 80.0),
            _pdb_atom(2, "CA ", "SER", "A", 2, 1.0, 0.0, 0.0, 1.0, 80.0),
        ]
        pdb.write_text("\n".join(lines) + "\n")
        result = alphafold_tools.assess_alphafold_quality(
            str(pdb),
            plddt_threshold_high=90.0,
            plddt_threshold_low=50.0,
        )
        assert result["suitable_for_docking"] is False
        assert result["mean_plddt"] == pytest.approx(80.0)


class TestKabschRMSD:
    """Tests for _kabsch_rmsd."""

    def test_identical_structures_zero_rmsd(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rmsd = alphafold_tools._kabsch_rmsd(coords, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_rotated_structure_same_rmsd(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = coords @ rot.T
        rmsd = alphafold_tools._kabsch_rmsd(rotated, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_translated_structure_zero_rmsd(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        translated = coords + np.array([10.0, -5.0, 3.0])
        rmsd = alphafold_tools._kabsch_rmsd(translated, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_different_structures_nonzero_rmsd(self):
        mobile = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ref = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        rmsd = alphafold_tools._kabsch_rmsd(mobile, ref)
        assert rmsd > 0.5


class TestRelaxAlphaFoldStructure:
    """Tests for relax_alphafold_structure — mocked OpenMM."""

    def test_raises_mderror_when_openmm_missing(self, tmp_path):
        with patch.dict("sys.modules", {"openmm": None, "openmm.app": None}):
            with pytest.raises(alphafold_tools.MDError, match="OpenMM"):
                alphafold_tools.relax_alphafold_structure(
                    str(tmp_path / "af.pdb"),
                    output_dir=str(tmp_path / "out"),
                )

    def test_nan_guard_returns_failure(self, tmp_path):
        """If coordinates become NaN during MD, return success=False."""
        pdb = tmp_path / "af.pdb"
        pdb.write_text(_pdb_atom(1, "CA ", "ALA", "A", 1, 0.0, 0.0, 0.0, 1.0, 90.0) + "\n")
        out_dir = tmp_path / "out"

        mock_sim = MagicMock()
        ref_arr = np.array([[0.0, 0.0, 0.0]])
        nan_arr = np.full((1, 3), np.nan)

        mock_state = MagicMock()
        mock_state.getPositions.return_value = ref_arr
        mock_sim.context.getState.return_value = mock_state

        nan_state = MagicMock()
        nan_state.getPositions.return_value = nan_arr

        # First call: after minimisation (reference)
        # Then production loop calls
        call_count = [0]

        def _get_state(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_state  # reference frame
            return nan_state  # NaN during production

        mock_sim.context.getState.side_effect = _get_state

        with patch("openmm.app.Simulation", return_value=mock_sim):
            with patch("openmm.app.PDBFile") as mock_pdb:
                mock_topo = MagicMock()
                mock_topo.residues.return_value = []
                mock_topo.atoms.return_value = []
                mock_pdb.return_value.topology = mock_topo
                mock_pdb.return_value.positions = []
                with patch("pdbfixer.PDBFixer") as mock_fixer:
                    mock_fixer.return_value.topology = mock_topo
                    mock_fixer.return_value.positions = []
                    with patch.object(
                        alphafold_tools, "_build_af_system", return_value=MagicMock()
                    ):
                        result = alphafold_tools.relax_alphafold_structure(
                            str(pdb),
                            output_dir=str(out_dir),
                            nvt_ns=0.001,
                            production_ns=0.001,
                        )
        assert result["success"] is False
        assert "NaN" in result["error"]
