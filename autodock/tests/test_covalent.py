"""Tests for covalent-warhead detection (autodock.covalent)."""

from __future__ import annotations

import os
import tempfile

import pytest

from autodock import covalent


class TestDetectCovalentWarheads:
    """SMARTS-based warhead detection."""

    @pytest.mark.parametrize(
        ("name", "smiles", "expected_warheads"),
        [
            ("acrylamide", "C=CC(=O)N", {"acrylamide"}),
            ("methacrylamide", "C=C(C)C(=O)N", {"acrylamide"}),
            ("crotonamide", "C/C=C/C(=O)N", {"acrylamide"}),
            ("chloroacetamide", "ClCC(=O)N", {"haloacetamide"}),
            ("bromoacetamide", "BrCC(=O)N", {"haloacetamide"}),
            ("vinyl_sulfone", "C=CS(=O)(=O)C", {"vinyl_sulfone"}),
            ("maleimide", "O=C1C=CC(=O)N1", {"maleimide"}),
            ("epoxide", "C1OC1", {"epoxide"}),
            ("benzonitrile", "N#Cc1ccccc1", {"nitrile"}),
            ("acetonitrile", "CC#N", {"nitrile"}),
            ("benzaldehyde", "O=Cc1ccccc1", {"aldehyde"}),
            ("formaldehyde", "C=O", {"aldehyde"}),
            ("phenylboronic_acid", "OB(O)c1ccccc1", {"boronic_acid"}),
            (
                "sulfonyl_fluoride",
                "CS(=O)(=O)F",
                {"sulfonyl_fluoride"},
            ),
            (
                "fluorosulfate",
                "OS(=O)(=O)F",
                {"fluorosulfate"},
            ),
        ],
    )
    def test_detects_known_warheads(self, name, smiles, expected_warheads):
        annotation = covalent.detect_covalent_warheads(smiles)
        assert annotation.has_warhead is True
        found = {m.name for m in annotation.warhead_matches}
        assert expected_warheads <= found, f"{name}: expected {expected_warheads}, got {found}"

    @pytest.mark.parametrize(
        "smiles",
        [
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
            "c1ccccc1",  # benzene
            "CCO",  # ethanol
            "CC(=O)N",  # acetamide (no Michael acceptor)
        ],
    )
    def test_non_covalent_negatives(self, smiles):
        annotation = covalent.detect_covalent_warheads(smiles)
        assert annotation.has_warhead is False
        assert annotation.warhead_matches == []

    def test_recommended_residues_aggregated(self):
        annotation = covalent.detect_covalent_warheads("ClCC(=O)N")
        assert "CYS" in annotation.recommended_residues
        assert "LYS" in annotation.recommended_residues
        assert "HIS" in annotation.recommended_residues

    def test_risk_level_high_for_haloacetamide(self):
        annotation = covalent.detect_covalent_warheads("ClCC(=O)N")
        assert annotation.risk_level == "high"

    def test_risk_level_medium_for_acrylamide(self):
        annotation = covalent.detect_covalent_warheads("C=CC(=O)N")
        assert annotation.risk_level == "medium"

    def test_message_non_empty_for_warhead(self):
        annotation = covalent.detect_covalent_warheads("C=CC(=O)N")
        assert "Covalent warhead" in annotation.message
        assert "AutoDock Vina" in annotation.message

    def test_invalid_smiles_returns_empty_annotation(self):
        annotation = covalent.detect_covalent_warheads("not_a_smiles")
        assert annotation.has_warhead is False


class TestFindReactiveResidues:
    """Parsing receptor PDBQT for nucleophilic residues."""

    def test_finds_cysteine_sulfur(self):
        pdbqt = (
            "ATOM      1  N   CYS A   1      10.000  10.000  10.000  0.00  0.00\n"
            "ATOM      2  CA  CYS A   1      11.000  10.000  10.000  0.00  0.00\n"
            "ATOM      3  SG  CYS A   1      12.000  10.000  10.000  0.00  0.00\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as fh:
            fh.write(pdbqt)
            path = fh.name
        try:
            residues = covalent.find_reactive_residues_in_receptor(path)
            assert len(residues) == 1
            assert residues[0]["resname"] == "CYS"
            assert residues[0]["atomname"] == "SG"
            assert residues[0]["resnum"] == "1"
        finally:
            os.unlink(path)

    def test_box_filter_excludes_distant_residues(self):
        pdbqt = (
            "ATOM      1  SG  CYS A   1      50.000  50.000  50.000  0.00  0.00\n"
            "ATOM      2  SG  CYS A   2       0.000   0.000   0.000  0.00  0.00\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as fh:
            fh.write(pdbqt)
            path = fh.name
        try:
            residues = covalent.find_reactive_residues_in_receptor(
                path, center=(0.0, 0.0, 0.0), box_size=(10.0, 10.0, 10.0)
            )
            assert len(residues) == 1
            assert residues[0]["resnum"] == "2"
        finally:
            os.unlink(path)

    def test_residue_type_filter(self):
        pdbqt = (
            "ATOM      1  SG  CYS A   1      10.000  10.000  10.000  0.00  0.00\n"
            "ATOM      2  OG  SER A   2      11.000  11.000  11.000  0.00  0.00\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as fh:
            fh.write(pdbqt)
            path = fh.name
        try:
            residues = covalent.find_reactive_residues_in_receptor(path, residue_types={"SER"})
            assert len(residues) == 1
            assert residues[0]["resname"] == "SER"
        finally:
            os.unlink(path)


class TestCheckReactiveGeometry:
    """Coarse distance gate between ligand reactive atoms and receptor residues."""

    def test_finds_close_pair_within_threshold(self):
        # Ligand with a warhead carbon at index 0, placed at origin
        ligand_pdbqt = (
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00\n"
            "ATOM      2  N   LIG A   1       1.000   0.000   0.000  0.00  0.00\n"
        )
        receptor_pdbqt = "ATOM      1  SG  CYS A   1       3.000   0.000   0.000  0.00  0.00\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as lf:
            lf.write(ligand_pdbqt)
            lig_path = lf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as rf:
            rf.write(receptor_pdbqt)
            rec_path = rf.name
        try:
            result = covalent.check_reactive_geometry(
                lig_path,
                rec_path,
                reactive_atom_indices=(0,),
                reactive_residue_atoms=covalent.find_reactive_residues_in_receptor(rec_path),
                max_dist=5.0,
            )
            assert result["feasible"] is True
            assert result["min_dist"] == pytest.approx(3.0, abs=0.01)
        finally:
            os.unlink(lig_path)
            os.unlink(rec_path)

    def test_rejects_distant_pair(self):
        ligand_pdbqt = (
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00\n"
        )
        receptor_pdbqt = "ATOM      1  SG  CYS A   1      20.000   0.000   0.000  0.00  0.00\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as lf:
            lf.write(ligand_pdbqt)
            lig_path = lf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as rf:
            rf.write(receptor_pdbqt)
            rec_path = rf.name
        try:
            result = covalent.check_reactive_geometry(
                lig_path,
                rec_path,
                reactive_atom_indices=(0,),
                reactive_residue_atoms=covalent.find_reactive_residues_in_receptor(rec_path),
                max_dist=5.0,
            )
            assert result["feasible"] is False
        finally:
            os.unlink(lig_path)
            os.unlink(rec_path)
