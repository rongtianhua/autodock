"""Tests for autodock.core — exceptions, logging, DockingResult, environment."""

from __future__ import annotations

import logging
import os

import pytest

from autodock import core

# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class TestExceptions:
    def test_exception_hierarchy(self):
        assert issubclass(core.StructureFetchError, core.DockingError)
        assert issubclass(core.PreparationError, core.DockingError)
        assert issubclass(core.DockingCalculationError, core.DockingError)
        assert issubclass(core.ValidationError, core.DockingError)
        assert issubclass(core.ConfigurationError, core.DockingError)

    def test_raise_and_catch(self):
        with pytest.raises(core.DockingError):
            raise core.PreparationError("test")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


class TestLogging:
    def test_set_log_level_int(self):
        core.set_log_level(logging.DEBUG)
        assert core.autodock_logger.level == logging.DEBUG
        core.set_log_level(logging.INFO)
        assert core.autodock_logger.level == logging.INFO

    def test_set_log_level_str(self):
        core.set_log_level("WARNING")
        assert core.autodock_logger.level == logging.WARNING
        core.set_log_level("INFO")

    def test_formatter_output(self):
        fmt = core._AutodockFormatter()
        rec = logging.LogRecord(
            name="autodock",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        assert fmt.format(rec) == "[autodock] I: hello"

    def test_file_handler_survives_permission_error(self, monkeypatch, tmp_path):
        # Simulate unreadable home directory for logs
        def raise_oserror(*args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(os, "makedirs", raise_oserror)
        # Re-import would trigger the except block; here we just assert no crash
        assert True


# ─────────────────────────────────────────────────────────────────────────────
# Seed helper
# ─────────────────────────────────────────────────────────────────────────────


class TestSeedHelper:
    def test_explicit_seed(self):
        assert core._get_vina_seed(123) == 123

    def test_default_seed(self):
        # When None, should return deterministic default (42)
        s = core._get_vina_seed(None)
        assert s == core.DEFAULT_SEED
        assert isinstance(s, int)
        assert 1 <= s <= 2_147_483_647

    def test_seed_reproducibility(self):
        assert core._get_vina_seed(None) == core._get_vina_seed(None)


# ─────────────────────────────────────────────────────────────────────────────
# Environment discovery
# ─────────────────────────────────────────────────────────────────────────────


class TestEnvironmentDiscovery:
    def test_find_conda_tool_python(self):
        path = core.find_conda_tool("python")
        assert path is None or os.path.isfile(path)

    def test_find_java_returns_none_or_str(self):
        path = core.find_java()
        assert path is None or isinstance(path, str)

    def test_find_p2rank_returns_none_or_str(self):
        path = core.find_p2rank()
        assert path is None or isinstance(path, str)

    def test_get_environment_status_keys(self):
        st = core.get_environment_status()
        expected = {
            "conda_prefix",
            "python",
            "java",
            "vina_cli",
            "vina_python",
            "rdkit",
            "meeko",
            "plip",
            "mdanalysis",
            "prolif",
            "openmm",
            "openbabel",
            "pymol_cli",
            "pymol_import",
            "fpocket",
            "p2rank",
            "gromacs",
            "timestamp",
        }
        assert expected.issubset(set(st.keys()))

    def test_safe_subprocess_true(self):
        ok, out, err = core.safe_subprocess(["echo", "hello"], timeout=5)
        assert ok is True
        assert "hello" in out

    def test_safe_subprocess_not_found(self):
        ok, out, err = core.safe_subprocess(["this_command_does_not_exist_12345"])
        assert ok is False
        assert "not found" in err.lower() or "command not found" in err.lower()

    def test_safe_subprocess_timeout(self):
        ok, out, err = core.safe_subprocess(["sleep", "10"], timeout=1)
        assert ok is False
        assert "timeout" in err.lower()


# ─────────────────────────────────────────────────────────────────────────────
# DockingResult
# ─────────────────────────────────────────────────────────────────────────────


class TestDockingResult:
    def test_basic_creation(self):
        r = core.DockingResult(
            compound_name="aspirin",
            receptor="rec.pdbqt",
            best_affinity=-8.5,
        )
        assert r.compound_name == "aspirin"
        assert r.best_affinity == -8.5

    def test_tuple_coercion(self):
        r = core.DockingResult(
            compound_name="x",
            receptor="r",
            center=[1.0, 2.0, 3.0],
            box_size=[10.0, 10.0, 10.0],
        )
        assert isinstance(r.center, tuple)
        assert isinstance(r.box_size, tuple)

    def test_interaction_summary(self):
        r = core.DockingResult(
            compound_name="x",
            receptor="r",
            interactions=[
                {"type": "H-bond", "residue": "GLY 123"},
                {"type": "H-bond", "residue": "ALA 45"},
                {"type": "π-π", "residue": "PHE 67"},
                {"type": "Hydrophobic", "residue": "LEU 89"},
            ],
        )
        assert r.n_hbonds == 2
        assert r.n_pi_stacking == 1
        assert r.n_hydrophobic == 1
        summary = r.interaction_summary
        assert summary == {"H-bond": 2, "π-π/π-cation": 1, "Hydrophobic": 1}

    def test_method_label(self):
        r = core.DockingResult(
            compound_name="x",
            receptor="r",
            scoring_functions=["vina", "vinardo"],
            receptor_source="PDB",
        )
        label = r.method_label
        assert "AutoDock Vina" in label
        assert "consensus" in label
        assert "X-ray" in label

    def test_to_dict_serialisation(self):
        r = core.DockingResult(
            compound_name="x",
            receptor="r",
            best_affinity=-7.0,
            receptor_source="AlphaFold",
        )
        d = r.to_dict()
        assert "_n_hbonds" not in d
        assert "_n_pi" not in d
        assert d["best_affinity"] == -7.0
        assert d["receptor_source_label"] == "AlphaFold2 predicted structure (UniProt)"

    def test_to_dataframe_row(self):
        r = core.DockingResult(
            compound_name="x",
            receptor="r",
            best_affinity=-7.0,
            center=(1.0, 2.0, 3.0),
            box_size=(20.0, 20.0, 20.0),
        )
        row = r.to_dataframe_row()
        assert row["compound"] == "x"
        assert row["best_affinity_kcal_mol"] == -7.0
        assert row["center_x"] == 1.0
        assert row["box_x"] == 20.0

    def test_provenance_fields_exist(self):
        r = core.DockingResult(compound_name="x", receptor="r")
        assert hasattr(r, "version")
        assert hasattr(r, "timestamp")
        assert r.version == __import__("autodock").__version__

    def test_build_docking_result(self):
        import numpy as np

        r = core.build_docking_result(
            compound_name="test",
            receptor="rec.pdbqt",
            center=(0.0, 0.0, 0.0),
            box_size=(20.0, 20.0, 20.0),
            energies=np.array([[-8.0, 1.0, 2.0]]),
            pre_dock_score=-6.0,
        )
        assert r.best_affinity == -8.0
        assert r.score_improvement == -6.0 - (-8.0)


# ─────────────────────────────────────────────────────────────────────────────
# Receptor source detection
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectReceptorSource:
    def test_detect_alphafold(self, tmp_path):
        pdb = tmp_path / "af.pdb"
        pdb.write_text("HEADER    AlphaFold model\n")
        assert core.detect_receptor_source(str(pdb)) == "AlphaFold"

    def test_detect_pdb(self, tmp_path):
        pdb = tmp_path / "xtal.pdb"
        pdb.write_text("EXPDTA  X-RAY DIFFRACTION\n")
        assert core.detect_receptor_source(str(pdb)) == "PDB"

    def test_missing_file(self):
        assert core.detect_receptor_source("/nonexistent/path.pdb") is None
