"""Tests for autodock.workflow — mock-based unit tests for the flagship API."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from autodock import workflow as wf
from autodock.core import DockingCalculationError, DockingResult

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_docking_result(
    affinity: float | None = -7.5, best_pose_pdbqt: str = "best.pdbqt"
) -> DockingResult:
    """Return a minimal DockingResult for mocking."""
    return DockingResult(
        compound_name="ligand",
        receptor="rec.pdbqt",
        center=(0.0, 0.0, 0.0),
        box_size=(20, 20, 20),
        best_affinity=affinity,
        best_pose_pdbqt=best_pose_pdbqt,
        all_poses_pdbqt="all.pdbqt",
        all_scores={},
    )


def _make_pocket(idx: int = 0) -> dict:
    return {
        "center": (float(idx), float(idx), float(idx)),
        "box_size": (20, 20, 20),
        "score": 0.9,
        "pocket_source": "fpocket",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointHelpers:
    def test_load_state_missing(self, tmp_path):
        assert wf._load_state(str(tmp_path)) == {}

    def test_load_state_invalid_json(self, tmp_path):
        (tmp_path / "workflow_state.json").write_text("not json")
        assert wf._load_state(str(tmp_path)) == {}

    def test_load_state_valid(self, tmp_path):
        data = {"step_1_complete": True}
        (tmp_path / "workflow_state.json").write_text(json.dumps(data))
        assert wf._load_state(str(tmp_path)) == data

    def test_save_state_roundtrip(self, tmp_path):
        wf._save_state(str(tmp_path), {"key": "val"})
        assert wf._load_state(str(tmp_path)) == {"key": "val"}

    def test_step_done_no_state(self):
        assert wf._step_done({}, "s1") is False

    def test_step_done_with_state_no_files(self):
        assert wf._step_done({"s1": True}, "s1") is True

    def test_step_done_missing_files(self, tmp_path):
        assert wf._step_done({"s1": True}, "s1", [str(tmp_path / "missing")]) is False

    def test_step_done_existing_files(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("x")
        assert wf._step_done({"s1": True}, "s1", [str(f)]) is True


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveParamsFromConfig:
    def test_none_config(self):
        params = wf._resolve_params_from_config(None, receptor_id="6LU7")
        assert params == {"receptor_id": "6LU7"}

    @patch("autodock.config.load_config")
    def test_with_config(self, mock_load, tmp_path):
        cfg = {
            "project": {"output_dir": "/out"},
            "receptor": {"pdb_id": "1ABC", "ph": 6.5},
            "pocket": {"top_n": 3, "padding": 4.0},
            "docking": {"exhaustiveness": 16, "num_modes": 10, "seed": 123},
        }
        mock_load.return_value = cfg
        result = wf._resolve_params_from_config("dummy.yml")
        assert result["output_dir"] == "/out"
        assert result["receptor_id"] == "1ABC"
        assert result["ph"] == 6.5
        assert result["max_pockets"] == 3
        assert result["pocket_padding"] == 4.0
        assert result["exhaustiveness"] == 16
        assert result["n_poses"] == 10
        assert result["seed"] == 123

    @patch("autodock.config.load_config")
    def test_explicit_override_config(self, mock_load):
        mock_load.return_value = {"project": {"output_dir": "/cfg"}}
        result = wf._resolve_params_from_config("dummy.yml", output_dir="/explicit")
        assert result["output_dir"] == "/explicit"


# ─────────────────────────────────────────────────────────────────────────────
# Ligand metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeLigandMetrics:
    def test_valid_smiles(self):
        metrics = wf._compute_ligand_metrics("CCO")
        assert metrics is not None
        assert metrics["n_heavy_atoms"] == 3
        assert metrics["molecular_weight"] > 40

    def test_invalid_smiles(self):
        assert wf._compute_ligand_metrics("NOT_A_SMILES") is None


# ─────────────────────────────────────────────────────────────────────────────
# Hydration helper
# ─────────────────────────────────────────────────────────────────────────────


class TestHydratePostprocessResults:
    def test_hydrate(self, tmp_path):
        pair = tmp_path / "pair"
        figs = pair / "03_figures"
        figs.mkdir(parents=True)
        (figs / "3d_complex.png").write_text("")
        (figs / "2d_interactions.png").write_text("")
        (figs / "session.pse").write_text("")
        reps = pair / "04_reports"
        reps.mkdir(parents=True)
        (reps / "report.pdf").write_text("")
        (reps / "report.csv").write_text("")

        result = wf.DockingWorkflowResult("r", "l", "file")
        wf._hydrate_postprocess_results(result, str(pair))
        assert len(result.figures_3d) == 1
        assert len(result.figures_2d) == 1
        assert len(result.pymol_sessions) == 1
        assert result.report_pdf is not None
        assert result.report_csv is not None

    def test_missing_dir(self):
        result = wf.DockingWorkflowResult("r", "l", "file")
        wf._hydrate_postprocess_results(result, "/nonexistent")
        assert result.figures_3d == []


# ─────────────────────────────────────────────────────────────────────────────
# run_docking_workflow — full path (local file receptor)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunDockingWorkflow:
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={"vina": "1.2.7"})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_full_path_local_file(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        tmp_path,
    ):
        rec_file = tmp_path / "rec.pdb"
        rec_file.write_text("ATOM 1 N ALA A 1 0 0 0\n")
        lig_file = tmp_path / "lig.pdbqt"
        lig_file.write_text("REMARK\n")

        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-8.0, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id=str(rec_file),
            receptor_source="file",
            ligand_smiles="CCO",
            ligand_name="ETH",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )

        assert result.receptor_source == "file"
        assert result.best_result is not None
        assert result.best_result.best_affinity == -8.0
        assert result.best_result.clash_acceptable is True
        assert result.best_result.posebusters_pass is True
        assert "le_le" in result.best_result.all_scores
        assert result.runtime_seconds is not None

    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.fetchers.download_pdb")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_pdb_source(
        self,
        mock_le,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_dl_pdb,
        mock_env,
        mock_log,
        mock_perf,
        tmp_path,
    ):
        pdb_path = tmp_path / "6LU7.cif"
        pdb_path.write_text("data_6LU7\n")
        mock_dl_pdb.return_value = str(pdb_path)
        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_dock.return_value = _make_docking_result(-7.5)
        mock_clash.return_value = {"clash_score": 0.6, "is_acceptable": True, "n_clashes": 0}
        mock_pp.return_value = {"pdf": None, "csv": None, "figures": []}
        mock_le.return_value = {"le": 0.3, "le_rb": 0.15}

        result = wf.run_docking_workflow(
            receptor_id="6LU7",
            receptor_source="pdb",
            ligand_smiles="CCO",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.receptor_source == "PDB"
        mock_dl_pdb.assert_called_once()

    @patch("autodock.workflow._compute_ligand_metrics", return_value=None)
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.fetchers.download_alphafold")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_alphafold_source(
        self,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_dl_af,
        mock_env,
        mock_log,
        mock_perf,
        mock_metrics,
        tmp_path,
    ):
        af_path = tmp_path / "AF.cif"
        af_path.write_text("data_AF\n")
        mock_dl_af.return_value = str(af_path)
        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_dock.return_value = _make_docking_result(-6.5)
        mock_clash.return_value = {"clash_score": 0.7, "is_acceptable": True, "n_clashes": 0}
        mock_pp.return_value = {}

        result = wf.run_docking_workflow(
            receptor_id="P68871",
            receptor_source="alphafold",
            ligand_smiles="CCO",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.receptor_source == "AlphaFold"

    @patch("autodock.workflow._compute_ligand_metrics", return_value=None)
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_resume_skips_completed_steps(
        self,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        mock_metrics,
        tmp_path,
    ):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        rec_pdbqt = out_dir / "rec_ligand_receptor.pdbqt"
        rec_pdb = out_dir / "rec_ligand_receptor.pdb"
        rec_pdbqt.write_text("REMARK\n")
        rec_pdb.write_text("ATOM\n")
        lig_pdbqt = out_dir / "rec_ligand_ligand.pdbqt"
        lig_pdbqt.write_text("REMARK\n")

        state = {
            "receptor_file": str(tmp_path / "rec.pdb"),
            "receptor_source": "file",
            "step_1_complete": True,
            "step_2_complete": True,
            "step_3_complete": True,
            "pockets": [_make_pocket(0)],
            "step_4_complete": True,
            "step_5_complete": True,
        }
        (out_dir / "workflow_state.json").write_text(json.dumps(state))

        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_dock.return_value = _make_docking_result(-9.0)
        mock_clash.return_value = {"clash_score": 0.4, "is_acceptable": True, "n_clashes": 0}
        mock_pp.return_value = {}

        result = wf.run_docking_workflow(
            receptor_id=str(tmp_path / "rec.pdb"),
            receptor_source="file",
            ligand_smiles="CCO",
            output_dir=str(out_dir),
            resume=True,
        )
        assert result.best_result.best_affinity == -9.0

    def test_missing_receptor_id(self):
        with pytest.raises(ValueError, match="receptor_id is required"):
            wf.run_docking_workflow(ligand_smiles="CCO", resume=False)

    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    def test_no_pockets_detected(
        self, mock_find, mock_prep, mock_env, mock_log, mock_perf, tmp_path
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        mock_prep.return_value = None
        mock_find.return_value = []
        with pytest.raises(DockingCalculationError, match="No binding pockets"):
            wf.run_docking_workflow(
                receptor_id=str(rec),
                receptor_source="file",
                ligand_smiles="CCO",
                output_dir=str(tmp_path / "out"),
                resume=False,
            )

    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_all_pockets_fail(
        self,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        tmp_path,
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0), _make_pocket(1)]
        mock_dock.side_effect = DockingCalculationError("dock fail")
        mock_clash.return_value = {"clash_score": 0.0, "is_acceptable": True, "n_clashes": 0}
        mock_pp.return_value = {}

        result = wf.run_docking_workflow(
            receptor_id=str(rec),
            receptor_source="file",
            ligand_smiles="CCO",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.warnings
        assert "All pockets failed" in result.warnings[0]

    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_pubchem_ligand_source(
        self,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        tmp_path,
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        mock_prep_rec.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_prep_lig.return_value = None
        mock_dock.return_value = _make_docking_result(-7.0)
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pp.return_value = {}

        with patch("autodock.fetchers.fetch_pubchem_smiles", return_value="CCO"):
            result = wf.run_docking_workflow(
                receptor_id=str(rec),
                receptor_source="file",
                ligand_smiles="2244",
                ligand_source="pubchem",
                output_dir=str(tmp_path / "out"),
                resume=False,
            )
        assert result.best_result.best_affinity == -7.0

    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand_from_file")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_file_ligand_source(
        self,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_file,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        tmp_path,
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        lig_sdf = tmp_path / "lig.sdf"
        lig_sdf.write_text("mock sdf\n")
        mock_prep_rec.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_prep_file.return_value = None
        mock_dock.return_value = _make_docking_result(-6.0)
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pp.return_value = {}

        result = wf.run_docking_workflow(
            receptor_id=str(rec),
            receptor_source="file",
            ligand_smiles=str(lig_sdf),
            ligand_source="file",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.best_result.best_affinity == -6.0

    @patch("autodock.workflow._compute_ligand_metrics", return_value=None)
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.minimization.minimize_docked_pose")
    def test_minimize_pose(
        self,
        mock_min,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        mock_metrics,
        tmp_path,
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        mock_prep_rec.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_prep_lig.return_value = None
        mock_dock.return_value = _make_docking_result(-8.0)
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pp.return_value = {}
        mock_min.return_value = {
            "success": True,
            "initial_energy_kJ_mol": 100.0,
            "final_energy_kJ_mol": 50.0,
        }

        result = wf.run_docking_workflow(
            receptor_id=str(rec),
            receptor_source="file",
            ligand_smiles="CCO",
            minimize_pose=True,
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.best_result is not None
        mock_min.assert_called_once()

    @patch("autodock.workflow._compute_ligand_metrics", return_value=None)
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_multi_conformer(
        self,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        mock_metrics,
        tmp_path,
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        mock_prep_rec.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_prep_lig.return_value = None
        mock_dock.return_value = _make_docking_result(-8.5)
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pp.return_value = {}

        wf.run_docking_workflow(
            receptor_id=str(rec),
            receptor_source="file",
            ligand_smiles="CCO",
            multi_conformer=True,
            n_conformers=5,
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        call_kwargs = mock_dock.call_args.kwargs
        assert call_kwargs.get("multi_conformer") is True
        assert call_kwargs.get("n_conformers") == 5

    @patch("autodock.workflow._compute_ligand_metrics", return_value=None)
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_posebusters_exception(
        self,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        mock_metrics,
        tmp_path,
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        mock_prep_rec.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_prep_lig.return_value = None
        mock_dock.return_value = _make_docking_result(-7.0)
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.side_effect = RuntimeError("pb fail")
        mock_pp.return_value = {}

        result = wf.run_docking_workflow(
            receptor_id=str(rec),
            receptor_source="file",
            ligand_smiles="CCO",
            run_posebusters=True,
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert any("PoseBusters" in w for w in result.warnings)

    @patch("autodock.workflow._compute_ligand_metrics", return_value=None)
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_clash_exception(
        self,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        mock_metrics,
        tmp_path,
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_prep_rec.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_prep_lig.return_value = None
        mock_dock.return_value = _make_docking_result(-7.0, best_pose_pdbqt=str(pose_file))
        mock_clash.side_effect = RuntimeError("clash fail")
        mock_pp.return_value = {}

        result = wf.run_docking_workflow(
            receptor_id=str(rec),
            receptor_source="file",
            ligand_smiles="CCO",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert any("Clash" in w or "clash" in w for w in result.warnings)

    @patch("autodock.workflow._compute_ligand_metrics", return_value=None)
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_postprocess_exception(
        self,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        mock_metrics,
        tmp_path,
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        mock_prep_rec.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        mock_prep_lig.return_value = None
        mock_dock.return_value = _make_docking_result(-7.0)
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pp.side_effect = RuntimeError("pp fail")

        result = wf.run_docking_workflow(
            receptor_id=str(rec),
            receptor_source="file",
            ligand_smiles="CCO",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert any("Post-processing" in w for w in result.warnings)

    @patch("autodock.workflow._compute_ligand_metrics", return_value=None)
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    def test_max_postprocess_pockets(
        self,
        mock_pp,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_env,
        mock_log,
        mock_perf,
        mock_metrics,
        tmp_path,
    ):
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM\n")
        mock_prep_rec.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0), _make_pocket(1)]
        mock_prep_lig.return_value = None

        def _dock_side(*args, **kwargs):
            idx = kwargs.get("compound_name", "x")
            aff = -8.0 if "pocket1" in idx else -7.0
            return _make_docking_result(aff)

        mock_dock.side_effect = _dock_side
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pp.return_value = {}

        result = wf.run_docking_workflow(
            receptor_id=str(rec),
            receptor_source="file",
            ligand_smiles="CCO",
            max_postprocess_pockets=2,
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.best_result.best_affinity == -8.0
        assert mock_pp.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# _dock_single_pocket
# ─────────────────────────────────────────────────────────────────────────────


class TestDockSinglePocket:
    @patch("autodock.docking.dock_ligand")
    def test_success(self, mock_dock, tmp_path):
        mock_dock.return_value = _make_docking_result(-7.5)
        pocket_results = []
        wr = wf.DockingWorkflowResult("r", "l", "file")
        wf._dock_single_pocket(
            "rec.pdbqt",
            "lig.pdbqt",
            _make_pocket(0),
            str(tmp_path),
            "test",
            0,
            32,
            20,
            42,
            False,
            10,
            600,
            3.0,
            "vina",
            None,
            wr,
            pocket_results,
        )
        assert len(pocket_results) == 1
        assert pocket_results[0].best_affinity == -7.5
        assert pocket_results[0].binding_pocket == _make_pocket(0)

    @patch("autodock.docking.dock_ligand")
    def test_failure(self, mock_dock, tmp_path):
        mock_dock.side_effect = DockingCalculationError("fail")
        pocket_results = []
        wr = wf.DockingWorkflowResult("r", "l", "file")
        wf._dock_single_pocket(
            "rec.pdbqt",
            "lig.pdbqt",
            _make_pocket(0),
            str(tmp_path),
            "test",
            0,
            32,
            20,
            42,
            False,
            10,
            600,
            3.0,
            "vina",
            None,
            wr,
            pocket_results,
        )
        assert len(pocket_results) == 1
        assert pocket_results[0].best_affinity is None
        assert wr.errors


# ─────────────────────────────────────────────────────────────────────────────
# DockingWorkflowResult
# ─────────────────────────────────────────────────────────────────────────────


class TestDockingWorkflowResult:
    def test_to_dict(self):
        dr = _make_docking_result(-7.5)
        wr = wf.DockingWorkflowResult(
            receptor_name="6LU7",
            ligand_name="ETH",
            receptor_source="PDB",
            best_result=dr,
            pocket_results=[dr],
        )
        d = wr.to_dict()
        assert d["receptor_name"] == "6LU7"
        assert "best_result" in d
        assert "pocket_results" in d

    def test_to_dict_with_none(self):
        wr = wf.DockingWorkflowResult("r", "l", "file")
        d = wr.to_dict()
        assert d["best_result"] is None
        assert d["pocket_results"] == []


# ─────────────────────────────────────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────────────────────────────────────


class TestMain:
    @patch("autodock.workflow.run_docking_workflow")
    def test_main_success(self, mock_run):
        mock_run.return_value = wf.DockingWorkflowResult(
            receptor_name="6LU7",
            ligand_name="ETH",
            receptor_source="file",
            best_result=_make_docking_result(-8.0),
            summary_json="/tmp/summary.json",
            runtime_seconds=10.0,
        )
        with patch("sys.argv", ["workflow", "--receptor", "6LU7", "--ligand-smiles", "CCO"]):
            wf.main()

    @patch("autodock.workflow.run_docking_workflow")
    def test_main_failure(self, mock_run):
        mock_run.return_value = wf.DockingWorkflowResult(
            receptor_name="6LU7",
            ligand_name="ETH",
            receptor_source="file",
            best_result=None,
            runtime_seconds=5.0,
        )
        with patch("sys.argv", ["workflow", "--receptor", "6LU7", "--ligand-smiles", "CCO"]):
            wf.main()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-chain receptor strategy
# ─────────────────────────────────────────────────────────────────────────────


class TestMultichainStrategy:
    """Tests for receptor_multichain_strategy in run_docking_workflow."""

    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.fetchers.fetch_protein_structure")
    @patch("autodock.fetchers.get_pdb_assembly_info")
    @patch("autodock.fetchers.extract_single_chain_from_mmcif")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_strategy_extract_single(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_extract,
        mock_asm_info,
        mock_fetch,
        mock_perf,
        mock_log,
        mock_env,
        tmp_path,
    ):
        """Multi-chain monomer PDB with strategy='extract_single' should extract one chain."""
        cif_path = str(tmp_path / "4F9Z.cif")
        mock_fetch.return_value = cif_path
        mock_asm_info.return_value = {
            "is_monomeric": True,
            "asymmetric_chains": ["A", "B", "C", "D", "E"],
            "oligomeric_count": 1,
            "chains_per_assembly": [["A", "B", "C", "D", "E"]],
        }
        extracted_pdb = str(tmp_path / "4F9Z_chain_extracted.pdb")
        mock_extract.return_value = extracted_pdb

        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-7.5, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id="4F9Z",
            receptor_source="auto",
            receptor_multichain_strategy="extract_single",
            ligand_smiles="CCO",
            ligand_name="ethanol",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.receptor_source == "PDB_single_chain"
        mock_extract.assert_called_once()

    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.fetchers.fetch_protein_structure")
    @patch("autodock.fetchers.get_pdb_assembly_info")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_strategy_auto_dimer_uses_multichain(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_asm_info,
        mock_fetch,
        mock_perf,
        mock_log,
        mock_env,
        tmp_path,
    ):
        """Default 'auto' strategy keeps multichain for dimeric PDB."""
        mock_fetch.return_value = str(tmp_path / "1ABC.cif")
        mock_asm_info.return_value = {
            "is_monomeric": False,
            "asymmetric_chains": ["A", "B"],
            "oligomeric_count": 2,
            "oligomeric_details": "homodimeric",
        }
        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-7.5, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id="1ABC",
            receptor_source="auto",
            receptor_multichain_strategy="auto",
            ligand_smiles="CCO",
            ligand_name="ethanol",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.receptor_source == "PDB"

    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.fetchers.fetch_protein_structure")
    @patch("autodock.fetchers.get_pdb_assembly_info")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_strategy_auto_higher_oligomer_warns(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_asm_info,
        mock_fetch,
        mock_perf,
        mock_log,
        mock_env,
        tmp_path,
    ):
        """Default 'auto' strategy keeps multichain and warns for higher-order oligomers."""
        mock_fetch.return_value = str(tmp_path / "2HU4.cif")
        mock_asm_info.return_value = {
            "is_monomeric": False,
            "asymmetric_chains": ["A", "B", "C", "D"],
            "oligomeric_count": 4,
            "oligomeric_details": "tetrameric",
        }
        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-7.5, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id="2HU4",
            receptor_source="auto",
            receptor_multichain_strategy="auto",
            ligand_smiles="CCO",
            ligand_name="ethanol",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.receptor_source == "PDB"
        assert any("higher-order oligomer" in w for w in result.warnings)

    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.fetchers.fetch_protein_structure")
    @patch("autodock.fetchers.get_pdb_assembly_info")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_strategy_multichain_keeps_all(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_asm_info,
        mock_fetch,
        mock_perf,
        mock_log,
        mock_env,
        tmp_path,
    ):
        """Strategy='multichain' should keep original multi-chain CIF."""
        cif_path = str(tmp_path / "4F9Z.cif")
        mock_fetch.return_value = cif_path
        mock_asm_info.return_value = {
            "is_monomeric": True,
            "asymmetric_chains": ["A", "B", "C", "D", "E"],
            "oligomeric_count": 1,
        }
        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-7.5, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id="4F9Z",
            receptor_source="auto",
            receptor_multichain_strategy="multichain",
            ligand_smiles="CCO",
            ligand_name="ethanol",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.receptor_source == "PDB"

    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.fetchers.fetch_protein_structure")
    @patch("autodock.fetchers.get_pdb_assembly_info")
    @patch("autodock.fetchers.download_alphafold")
    @patch("autodock.fetchers._resolve_to_uniprot")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_strategy_alphafold_fallback(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_resolve_uni,
        mock_dl_af,
        mock_asm_info,
        mock_fetch,
        mock_perf,
        mock_log,
        mock_env,
        tmp_path,
    ):
        """Strategy='alphafold' should download AF monomer instead of using PDB."""
        mock_fetch.return_value = str(tmp_path / "4F9Z.cif")
        mock_asm_info.return_value = {
            "is_monomeric": True,
            "asymmetric_chains": ["A", "B"],
            "oligomeric_count": 1,
        }
        mock_resolve_uni.return_value = "P42785"
        af_cif = str(tmp_path / "AF-P42785.cif")
        mock_dl_af.return_value = af_cif

        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-7.5, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id="4F9Z",
            receptor_source="auto",
            receptor_multichain_strategy="alphafold",
            ligand_smiles="CCO",
            ligand_name="ethanol",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.receptor_source == "AlphaFold"
        mock_dl_af.assert_called_once()

    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.fetchers.fetch_protein_structure")
    @patch("autodock.fetchers.get_pdb_assembly_info")
    @patch("autodock.fetchers.extract_single_chain_from_mmcif")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_strategy_auto_monomer_extracts_single(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_extract,
        mock_asm_info,
        mock_fetch,
        mock_perf,
        mock_log,
        mock_env,
        tmp_path,
    ):
        """Default 'auto' strategy extracts single chain for monomeric multi-chain PDB."""
        mock_fetch.return_value = str(tmp_path / "4F9Z.cif")
        mock_asm_info.return_value = {
            "is_monomeric": True,
            "asymmetric_chains": ["A", "B", "C"],
            "oligomeric_count": 1,
        }
        extracted_pdb = str(tmp_path / "4F9Z_chain.pdb")
        mock_extract.return_value = extracted_pdb

        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-7.5, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id="4F9Z",
            receptor_source="auto",
            receptor_multichain_strategy="auto",
            ligand_smiles="CCO",
            ligand_name="ethanol",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.receptor_source == "PDB_single_chain"
        mock_extract.assert_called_once()

    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.fetchers.fetch_protein_structure")
    @patch("autodock.fetchers.get_pdb_assembly_info")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_single_chain_pdb_skips_strategy(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_asm_info,
        mock_fetch,
        mock_perf,
        mock_log,
        mock_env,
        tmp_path,
    ):
        """A PDB with only 1 chain in asymmetric unit should skip strategy logic."""
        mock_fetch.return_value = str(tmp_path / "1A30.cif")
        mock_asm_info.return_value = {
            "is_monomeric": True,
            "asymmetric_chains": ["A"],
            "oligomeric_count": 1,
        }
        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-7.5, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id="1A30",
            receptor_source="auto",
            receptor_multichain_strategy="auto",
            ligand_smiles="CCO",
            ligand_name="ethanol",
            output_dir=str(tmp_path / "out"),
            resume=False,
        )
        assert result.receptor_source == "PDB"


class TestCovalentCheck:
    """Tests for covalent warhead annotation in the workflow."""

    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.fetchers.fetch_protein_structure")
    @patch("autodock.fetchers.get_pdb_assembly_info")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_covalent_check_adds_warning(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_asm_info,
        mock_fetch,
        mock_perf,
        mock_log,
        mock_env,
        tmp_path,
    ):
        """covalent_check=True should add a warning for acrylamide warheads."""
        mock_fetch.return_value = str(tmp_path / "1A30.cif")
        mock_asm_info.return_value = {
            "is_monomeric": True,
            "asymmetric_chains": ["A"],
            "oligomeric_count": 1,
        }
        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-7.5, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id="1A30",
            receptor_source="auto",
            ligand_smiles="C=CC(=O)N",
            ligand_name="acrylamide",
            output_dir=str(tmp_path / "out"),
            resume=False,
            covalent_check=True,
        )
        assert any("Covalent warhead" in w for w in result.warnings)

    @patch("autodock.workflow.get_environment_status", return_value={})
    @patch("autodock.workflow.set_log_level")
    @patch("autodock.workflow.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("autodock.fetchers.fetch_protein_structure")
    @patch("autodock.fetchers.get_pdb_assembly_info")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.validation.compute_clash_score")
    @patch("autodock.validation.validate_pose_with_posebusters")
    @patch("autodock.post_dock_pipeline.post_process_docking")
    @patch("autodock.analysis.compute_ligand_efficiency")
    def test_covalent_check_false_no_warning(
        self,
        mock_le,
        mock_pp,
        mock_pb,
        mock_clash,
        mock_dock,
        mock_prep_lig,
        mock_find_pockets,
        mock_prep_rec,
        mock_asm_info,
        mock_fetch,
        mock_perf,
        mock_log,
        mock_env,
        tmp_path,
    ):
        """covalent_check=False should not add covalent warnings."""
        mock_fetch.return_value = str(tmp_path / "1A30.cif")
        mock_asm_info.return_value = {
            "is_monomeric": True,
            "asymmetric_chains": ["A"],
            "oligomeric_count": 1,
        }
        mock_prep_rec.return_value = None
        mock_prep_lig.return_value = None
        mock_find_pockets.return_value = [_make_pocket(0)]
        pose_file = tmp_path / "best_pose.pdbqt"
        pose_file.write_text("ATOM\n")
        mock_dock.return_value = _make_docking_result(-7.5, best_pose_pdbqt=str(pose_file))
        mock_clash.return_value = {"clash_score": 0.5, "is_acceptable": True, "n_clashes": 0}
        mock_pb.return_value = {"available": True, "pass": True}
        mock_pp.return_value = {
            "pdf": str(tmp_path / "report.pdf"),
            "csv": str(tmp_path / "report.csv"),
            "figures": [str(tmp_path / "fig.png")],
        }
        mock_le.return_value = {"le": 0.35, "le_rb": 0.18, "lle": 4.5, "lem": 0.12}

        result = wf.run_docking_workflow(
            receptor_id="1A30",
            receptor_source="auto",
            ligand_smiles="C=CC(=O)N",
            ligand_name="acrylamide",
            output_dir=str(tmp_path / "out"),
            resume=False,
            covalent_check=False,
        )
        assert not any("Covalent warhead" in w for w in result.warnings)
