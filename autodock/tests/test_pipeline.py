"""Tests for autodock.pipeline."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from autodock.core import DockingResult
from autodock import pipeline


class TestBuildPairDir:
    """Tests for build_pair_dir."""

    def test_creates_directory_structure(self, tmp_path):
        root = tmp_path / "pair_1"
        dirs = pipeline.build_pair_dir(str(root))
        assert (root / "01_structures").is_dir()
        assert (root / "02_interactions").is_dir()
        assert (root / "03_figures").is_dir()
        assert (root / "04_reports").is_dir()
        assert set(dirs.keys()) == {"structures", "interactions", "figures", "reports"}


class TestCopyFile:
    """Tests for _copy_file."""

    def test_copies_file(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "sub" / "dst.txt"
        result = pipeline._copy_file(str(src), str(dst))
        assert os.path.exists(dst)
        assert dst.read_text() == "hello"
        assert result == str(dst)

    def test_handles_missing_source_gracefully(self, tmp_path):
        src = tmp_path / "missing.txt"
        dst = tmp_path / "dst.txt"
        result = pipeline._copy_file(str(src), str(dst))
        assert not os.path.exists(dst)
        assert result == str(dst)


class TestPostProcessDocking:
    """Tests for post_process_docking."""

    def _make_result(self, **kwargs):
        defaults = {
            "compound_name": "aspirin",
            "receptor": "6LU7",
            "best_affinity": -8.0,
        }
        defaults.update(kwargs)
        return DockingResult(**defaults)

    def test_basic_run_creates_outputs(self, tmp_path):
        result = self._make_result()
        pair_root = tmp_path / "pair"
        out = pipeline.post_process_docking(
            result,
            str(pair_root),
            do_interactions=False,
            do_rendering=False,
            do_report=False,
            copy_structures=False,
        )
        assert out["pair_root"] == str(pair_root)
        assert "summary_txt" in out
        assert (pair_root / "summary.txt").exists()

    def test_copy_structures_copies_files(self, tmp_path):
        rec = tmp_path / "rec.pdbqt"
        rec.write_text("RECEPTOR")
        best = tmp_path / "best.pdbqt"
        best.write_text("BEST")
        all_poses = tmp_path / "all.pdbqt"
        all_poses.write_text("ALL")
        rec_pdb = tmp_path / "rec.pdb"
        rec_pdb.write_text("RECPDB")

        result = self._make_result(
            receptor=str(rec),
            best_pose_pdbqt=str(best),
            all_poses_pdbqt=str(all_poses),
        )
        pair_root = tmp_path / "pair"
        out = pipeline.post_process_docking(
            result,
            str(pair_root),
            receptor_pdb=str(rec_pdb),
            do_interactions=False,
            do_rendering=False,
            do_report=False,
            copy_structures=True,
        )
        struct_dir = pair_root / "01_structures"
        assert (struct_dir / "receptor.pdbqt").exists()
        assert (struct_dir / "docking_best.pdbqt").exists()
        assert (struct_dir / "docking_all_poses.pdbqt").exists()
        assert (struct_dir / "receptor.pdb").exists()

    def test_interactions_saved_when_provided(self, tmp_path):
        result = self._make_result(
            interactions=[{"type": "H-bond", "residue": "ALA:1"}]
        )
        pair_root = tmp_path / "pair"
        out = pipeline.post_process_docking(
            result,
            str(pair_root),
            do_interactions=False,
            do_rendering=False,
            do_report=False,
            copy_structures=False,
        )
        assert "interactions_csv" in out
        assert "interaction_summary_txt" in out
        assert (pair_root / "02_interactions" / "interactions.csv").exists()

    def test_report_generation(self, tmp_path):
        result = self._make_result()
        pair_root = tmp_path / "pair"
        out = pipeline.post_process_docking(
            result,
            str(pair_root),
            do_interactions=False,
            do_rendering=False,
            do_report=True,
            copy_structures=False,
        )
        assert "json" in out
        assert (pair_root / "04_reports" / "result.json").exists()
        # Verify JSON content
        with open(out["json"]) as fh:
            data = json.load(fh)
        assert data["compound_name"] == "aspirin"

    def test_summary_with_all_fields(self, tmp_path):
        result = self._make_result(
            rmsd_from_crystal=1.5,
            posebusters_pass=True,
            n_clusters=3,
        )
        pair_root = tmp_path / "pair"
        out = pipeline.post_process_docking(
            result,
            str(pair_root),
            do_interactions=False,
            do_rendering=False,
            do_report=False,
            copy_structures=False,
        )
        summary = (pair_root / "summary.txt").read_text()
        assert "aspirin" in summary
        assert "1.50" in summary
        assert "PASS" in summary
        assert "3" in summary


class TestReadDockingResults:
    """Tests for read_docking_results."""

    def test_reads_valid_result_json(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        data = {
            "compound_name": "aspirin",
            "receptor": "6LU7",
            "best_affinity": -7.5,
        }
        (subdir / "result.json").write_text(json.dumps(data))
        results = pipeline.read_docking_results(str(tmp_dir := tmp_path))
        assert len(results) == 1
        assert results[0].compound_name == "aspirin"

    def test_skips_non_dict_json(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "result.json").write_text("[1, 2, 3]")
        results = pipeline.read_docking_results(str(tmp_path))
        assert len(results) == 0

    def test_skips_missing_required_fields(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "result.json").write_text(json.dumps({"best_affinity": -7.5}))
        results = pipeline.read_docking_results(str(tmp_path))
        assert len(results) == 0

    def test_skips_invalid_json(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "result.json").write_text("not json")
        results = pipeline.read_docking_results(str(tmp_path))
        assert len(results) == 0

    def test_reads_multiple_results(self, tmp_path):
        for name in ("a", "b"):
            subdir = tmp_path / name
            subdir.mkdir()
            data = {
                "compound_name": name,
                "receptor": "6LU7",
                "best_affinity": -7.0,
            }
            (subdir / "result.json").write_text(json.dumps(data))
        results = pipeline.read_docking_results(str(tmp_path))
        assert len(results) == 2
        names = {r.compound_name for r in results}
        assert names == {"a", "b"}
