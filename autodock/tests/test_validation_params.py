"""Tests for autodock.validation_params — input validation layer."""
from __future__ import annotations

import pytest

from autodock.validation_params import (
    validate_file_exists,
    validate_pdbqt_file,
    validate_smiles,
    validate_pdb_id,
    validate_exhaustiveness,
    validate_n_poses,
    validate_energy_range,
    validate_timeout,
    validate_seed,
    validate_box_size,
    validate_center,
    validate_n_workers,
    validate_docking_params,
)
from autodock.core import ConfigurationError, DockingCalculationError, PreparationError


class TestFileValidators:
    def test_validate_file_exists(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert validate_file_exists(f) == str(f.resolve())

    def test_validate_file_exists_missing(self):
        with pytest.raises(DockingCalculationError):
            validate_file_exists("/nonexistent/file.txt")

    def test_validate_pdbqt_file_valid(self, tmp_path):
        f = tmp_path / "ligand.pdbqt"
        f.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000  0.00  0.00    +0.000 C\n")
        assert validate_pdbqt_file(f) == str(f.resolve())

    def test_validate_pdbqt_file_no_atoms(self, tmp_path):
        f = tmp_path / "empty.pdbqt"
        f.write_text("REMARK   1\n")
        with pytest.raises(PreparationError):
            validate_pdbqt_file(f)


class TestSmilesValidator:
    def test_valid_smiles(self):
        assert validate_smiles("CCO") == "CCO"

    def test_invalid_smiles(self):
        with pytest.raises(PreparationError):
            validate_smiles("NOT_A_SMILES!!!")


class TestPdbIdValidator:
    def test_valid(self):
        assert validate_pdb_id("6LU7") == "6LU7"
        assert validate_pdb_id("1abc") == "1ABC"

    def test_invalid_length(self):
        with pytest.raises(ConfigurationError):
            validate_pdb_id("123")

    def test_invalid_chars(self):
        with pytest.raises(ConfigurationError):
            validate_pdb_id("AB CD")


class TestNumericValidators:
    def test_exhaustiveness_bounds(self):
        assert validate_exhaustiveness(32) == 32
        assert validate_exhaustiveness(1) == 1
        assert validate_exhaustiveness(1024) == 1024

    def test_exhaustiveness_too_low(self):
        with pytest.raises(ConfigurationError):
            validate_exhaustiveness(0)

    def test_exhaustiveness_too_high(self):
        with pytest.raises(ConfigurationError):
            validate_exhaustiveness(1025)

    def test_n_poses_bounds(self):
        assert validate_n_poses(20) == 20
        assert validate_n_poses(1) == 1

    def test_n_poses_too_low(self):
        with pytest.raises(ConfigurationError):
            validate_n_poses(0)

    def test_n_poses_too_high(self):
        with pytest.raises(ConfigurationError):
            validate_n_poses(1001)

    def test_energy_range_bounds(self):
        assert validate_energy_range(3.0) == 3.0
        assert validate_energy_range(0.1) == 0.1

    def test_energy_range_non_positive(self):
        with pytest.raises(ConfigurationError):
            validate_energy_range(0.0)

    def test_energy_range_too_high(self):
        with pytest.raises(ConfigurationError):
            validate_energy_range(101.0)

    def test_timeout_bounds(self):
        assert validate_timeout(600) == 600
        assert validate_timeout(1) == 1

    def test_timeout_too_low(self):
        with pytest.raises(ConfigurationError):
            validate_timeout(0)

    def test_timeout_too_high(self):
        with pytest.raises(ConfigurationError):
            validate_timeout(86401)

    def test_seed_none(self):
        assert validate_seed(None) is None

    def test_seed_valid(self):
        assert validate_seed(42) == 42
        assert validate_seed(0) == 0

    def test_seed_negative(self):
        with pytest.raises(ConfigurationError):
            validate_seed(-1)

    def test_seed_too_large(self):
        with pytest.raises(ConfigurationError):
            validate_seed(2_147_483_648)

    def test_box_size_valid(self):
        assert validate_box_size((20.0, 20.0, 20.0)) == (20.0, 20.0, 20.0)

    def test_box_size_wrong_length(self):
        with pytest.raises(ConfigurationError):
            validate_box_size((20.0, 20.0))

    def test_box_size_too_small(self):
        with pytest.raises(ConfigurationError):
            validate_box_size((2.0, 20.0, 20.0))

    def test_box_size_too_large(self):
        with pytest.raises(ConfigurationError):
            validate_box_size((50.0, 20.0, 20.0))

    def test_center_valid(self):
        assert validate_center((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)

    def test_center_wrong_length(self):
        with pytest.raises(ConfigurationError):
            validate_center((1.0, 2.0))

    def test_n_workers_valid(self):
        assert validate_n_workers(4) == 4
        assert validate_n_workers(-1) == -1

    def test_n_workers_invalid(self):
        with pytest.raises(ConfigurationError):
            validate_n_workers(0)
        with pytest.raises(ConfigurationError):
            validate_n_workers(-2)


class TestValidateDockingParams:
    def test_valid_params(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000  0.00  0.00    +0.000 N\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000  0.00  0.00    +0.000 C\n")

        params = validate_docking_params(
            rec, lig, (10.0, 10.0, 10.0), (20.0, 20.0, 20.0),
            exhaustiveness=32, n_poses=20, seed=42, timeout=600,
        )
        assert params["exhaustiveness"] == 32
        assert params["seed"] == 42
        assert params["center"] == (10.0, 10.0, 10.0)

    def test_missing_receptor(self, tmp_path):
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000  0.00  0.00    +0.000 C\n")
        with pytest.raises(DockingCalculationError):
            validate_docking_params(
                "missing.pdbqt", lig, (0, 0, 0), (20, 20, 20)
            )

    def test_invalid_exhaustiveness(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text("ATOM      1  N   SER A   1      0.000   0.000   0.000  0.00  0.00    +0.000 N\n")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM      1  C   LIG A   1      1.000   2.000   3.000  0.00  0.00    +0.000 C\n")
        with pytest.raises(ConfigurationError):
            validate_docking_params(
                rec, lig, (0, 0, 0), (20, 20, 20), exhaustiveness=-5
            )
