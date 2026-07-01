"""Comprehensive tests for autodock.cli — argument parser and command dispatch."""

from __future__ import annotations

import argparse
import logging
import os
from unittest.mock import MagicMock, patch

from autodock import cli
from autodock.core import autodock_logger
from autodock.docking import DockingResult


# ────────────────────────────────
# Helper: build a Namespace from parser defaults
# ────────────────────────────────
def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


# ────────────────────────────────
# 1. build_parser()
# ────────────────────────────────
class TestBuildParser:
    def test_parser_creation(self):
        parser = cli.build_parser()
        assert parser is not None
        assert parser.prog == "autodock"

    def test_all_subcommands_exist(self):
        parser = cli.build_parser()
        # argparse stores subparsers in a private attr; we can verify by parsing each
        subcommands = [
            "status",
            "init",
            "fetch",
            "prepare-receptor",
            "prepare-ligand",
            "find-pockets",
            "dock",
            "validate",
            "analyze",
            "report",
            "benchmark-redock",
            "posebusters-eval",
            "batch-dock",
            "ensemble-dock",
            "virtual-screen",
            "md",
            "run",
        ]
        for sub in subcommands:
            args = parser.parse_args([sub] + self._minimal_args(sub))
            assert args.command == sub
            assert hasattr(args, "func")

    def _minimal_args(self, sub: str) -> list[str]:
        extras = {
            "fetch": ["pdb", "6LU7"],
            "prepare-receptor": ["rec.pdb"],
            "prepare-ligand": ["CCO"],
            "find-pockets": ["rec.pdb"],
            "dock": ["rec.pdbqt", "lig.pdbqt"],
            "validate": ["holo.pdb"],
            "analyze": ["rec.pdb", "lig.pdbqt"],
            "report": ["./results"],
            "benchmark-redock": ["--method", "plip"],
            "posebusters-eval": ["ids.txt"],
            "batch-dock": [
                "--receptors",
                "r1.pdbqt",
                "--ligands",
                "l1.pdbqt",
                "--pockets",
                "pockets.json",
            ],
            "ensemble-dock": ["rec.pdbqt", "lig.pdbqt"],
            "virtual-screen": ["--receptor", "6LU7", "--library", "lib.txt"],
            "md": ["--receptor", "rec.pdb", "--ligand", "lig.pdbqt"],
            "run": ["--receptor", "6LU7", "--ligand", "aspirin"],
        }
        return extras.get(sub, [])

    def test_common_args(self):
        parser = cli.build_parser()
        args = parser.parse_args(["-q", "-v", "--log-file", "log.txt", "status"])
        assert args.quiet is True
        assert args.verbose is True
        assert args.log_file == "log.txt"

    def test_fetch_choices(self):
        parser = cli.build_parser()
        for ft in [
            "pdb",
            "cif",
            "ligand",
            "alphafold",
            "swissmodel",
            "uniprot",
            "pubchem",
            "chembl",
        ]:
            args = parser.parse_args(["fetch", ft, "XYZ"])
            assert args.type == ft

    def test_dock_arguments(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "dock",
                "rec.pdbqt",
                "lig.pdbqt",
                "--center",
                "1",
                "2",
                "3",
                "--box-size",
                "10",
                "10",
                "10",
                "--exhaustiveness",
                "64",
                "--n-poses",
                "50",
                "--seed",
                "123",
                "--output-dir",
                "./out",
                "--name",
                "LIG",
            ]
        )
        assert args.receptor == "rec.pdbqt"
        assert args.ligand == "lig.pdbqt"
        assert args.center == [1.0, 2.0, 3.0]
        assert args.box_size == [10.0, 10.0, 10.0]
        assert args.exhaustiveness == 64
        assert args.n_poses == 50
        assert args.seed == 123
        assert args.output_dir == "./out"
        assert args.name == "LIG"

    def test_validate_arguments(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "validate",
                "holo.pdb",
                "--ligand-resname",
                "LIG",
                "--chain-id",
                "C",
                "--ligand-smiles",
                "CCO",
                "--box-padding",
                "4.0",
            ]
        )
        assert args.holo_pdb == "holo.pdb"
        assert args.ligand_resname == "LIG"
        assert args.chain_id == "C"
        assert args.ligand_smiles == "CCO"
        assert args.box_padding == 4.0

    def test_virtual_screen_arguments(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "virtual-screen",
                "--receptor",
                "6LU7",
                "--library",
                "lib.sdf",
                "--library-format",
                "sdf",
                "--outdir",
                "./vs",
                "--workers",
                "4",
            ]
        )
        assert args.receptor == "6LU7"
        assert args.library == "lib.sdf"
        assert args.library_format == "sdf"
        assert args.outdir == "./vs"
        assert args.workers == 4

    def test_run_arguments(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--receptor",
                "6LU7",
                "--ligand",
                "aspirin",
                "--outdir",
                "./results",
                "--exhaustiveness",
                "16",
                "--n-poses",
                "9",
                "--seed",
                "99",
            ]
        )
        assert args.receptor == "6LU7"
        assert args.ligand == "aspirin"
        assert args.outdir == "./results"
        assert args.exhaustiveness == 16
        assert args.n_poses == 9
        assert args.seed == 99

    def test_md_arguments(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "md",
                "--receptor",
                "rec.pdb",
                "--ligand",
                "lig.pdbqt",
                "--steps",
                "1000",
                "--dt",
                "1.0",
                "--temperature",
                "310.0",
                "--solvent",
                "explicit",
                "--platform",
                "CPU",
            ]
        )
        assert args.receptor == "rec.pdb"
        assert args.ligand == "lig.pdbqt"
        assert args.steps == 1000
        assert args.dt == 1.0
        assert args.temperature == 310.0
        assert args.solvent == "explicit"
        assert args.platform == "CPU"


# ────────────────────────────────
# 2. main()
# ────────────────────────────────
class TestMain:
    def test_no_args_prints_help(self, capsys):
        rc = cli.main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "usage:" in (captured.out + captured.err)

    def test_keyboard_interrupt_returns_130(self, monkeypatch):
        def _raise_interrupt(args):
            raise KeyboardInterrupt()

        monkeypatch.setattr(cli, "cmd_status", _raise_interrupt)
        rc = cli.main(["status"])
        assert rc == 130

    def test_generic_exception_returns_1(self, monkeypatch, capsys):
        def _raise_exc(args):
            raise RuntimeError("boom")

        monkeypatch.setattr(cli, "cmd_status", _raise_exc)
        rc = cli.main(["status"])
        assert rc == 1

    def test_verbose_traceback_on_exception(self, monkeypatch, capsys):
        def _raise_exc(args):
            raise ValueError("detailed boom")

        monkeypatch.setattr(cli, "cmd_status", _raise_exc)
        rc = cli.main(["-v", "status"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "ValueError" in (captured.out + captured.err)

    def test_main_dispatches_to_command(self, capsys):
        rc = cli.main(["init", "--config", "/dev/null"])
        assert rc == 0


# ────────────────────────────────
# _setup_logging
# ────────────────────────────────
class TestSetupLogging:
    def test_quiet(self):
        args = cli.build_parser().parse_args(["-q", "status"])
        cli._setup_logging(args)
        for h in autodock_logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                assert h.level == logging.WARNING
        cli._setup_logging(cli.build_parser().parse_args(["status"]))

    def test_verbose(self):
        args = cli.build_parser().parse_args(["-v", "status"])
        cli._setup_logging(args)
        for h in autodock_logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                assert h.level == logging.DEBUG
        cli._setup_logging(cli.build_parser().parse_args(["status"]))

    def test_log_file(self, tmp_path):
        log_path = tmp_path / "test.log"
        args = cli.build_parser().parse_args(["--log-file", str(log_path), "status"])
        cli._setup_logging(args)
        autodock_logger.info("test_log_file_message")
        assert log_path.exists()
        content = log_path.read_text()
        assert "test_log_file_message" in content
        # Clean up added handler to avoid leakage
        for h in autodock_logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                autodock_logger.removeHandler(h)


# ────────────────────────────────
# 3. cmd_status
# ────────────────────────────────
class TestCmdStatus:
    def test_cmd_status_mocks(self, capsys):
        with patch("autodock.cli.print_environment_status") as mock_status:
            args = cli.build_parser().parse_args(["status"])
            rc = cli.cmd_status(args)
            assert rc == 0
            mock_status.assert_called_once()


# ────────────────────────────────
# 4. cmd_init
# ────────────────────────────────
class TestCmdInit:
    def test_creates_config(self, tmp_path, capsys):
        config_path = tmp_path / "docking_config.yaml"
        args = _ns(config=str(config_path), quiet=False, verbose=False, log_file=None)
        rc = cli.cmd_init(args)
        assert rc == 0
        assert config_path.exists()
        content = config_path.read_text()
        assert "project:" in content
        captured = capsys.readouterr()
        assert "Default config written" in captured.out


# ────────────────────────────────
# 5. cmd_fetch (8 types)
# ────────────────────────────────
class TestCmdFetch:
    def _fetch_args(
        self, ftype: str, fid: str = "XYZ", fmt=None, outdir: str = "."
    ) -> argparse.Namespace:
        return _ns(
            type=ftype,
            id=fid,
            format=fmt,
            outdir=outdir,
            quiet=False,
            verbose=False,
            log_file=None,
        )

    @patch("autodock.fetchers.download_pdb", return_value="./6LU7.pdb")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_pdb(self, mock_ensure, mock_dl, capsys):
        args = self._fetch_args("pdb", "6LU7")
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_dl.assert_called_once_with("6LU7", ".", format="pdb")
        assert "Downloaded PDB" in capsys.readouterr().out

    @patch("autodock.fetchers.download_pdb", return_value="./6LU7.cif")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_cif(self, mock_ensure, mock_dl, capsys):
        args = self._fetch_args("cif", "6LU7")
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_dl.assert_called_once_with("6LU7", ".", format="cif")

    @patch("autodock.fetchers.download_ligand_sdf_from_pdb", return_value="./lig.sdf")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_ligand(self, mock_ensure, mock_dl, capsys):
        args = self._fetch_args("ligand", "ABC")
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_dl.assert_called_once_with("ABC", ".")

    @patch("autodock.fetchers.download_alphafold", return_value="./AF.pdb")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_alphafold(self, mock_ensure, mock_dl, capsys):
        args = self._fetch_args("alphafold", "P12345")
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_dl.assert_called_once_with("P12345", ".", format="cif")

    @patch("autodock.fetchers.download_swissmodel", return_value="./SM.pdb")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_swissmodel(self, mock_ensure, mock_dl, capsys):
        args = self._fetch_args("swissmodel", "P12345")
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_dl.assert_called_once_with("P12345", ".")

    @patch("autodock.fetchers.fetch_uniprot_fasta")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_uniprot(self, mock_ensure, mock_fetch, capsys):
        args = self._fetch_args("uniprot", "P12345", outdir=".")
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_fetch.assert_called_once()

    @patch("autodock.fetchers.fetch_pubchem_smiles", return_value="CCO")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_pubchem_smiles_only(self, mock_ensure, mock_smiles, capsys):
        args = self._fetch_args("pubchem", "aspirin", fmt=None)
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_smiles.assert_called_once_with("aspirin")

    @patch("pubchempy.get_compounds")
    @patch("autodock.fetchers.fetch_pubchem_sdf")
    @patch("autodock.fetchers.fetch_pubchem_smiles", return_value="CCO")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_pubchem_sdf(self, mock_ensure, mock_smiles, mock_sdf, mock_pcp, capsys):
        mock_compound = MagicMock()
        mock_compound.cid = 1234
        mock_pcp.return_value = [mock_compound]
        args = self._fetch_args("pubchem", "aspirin", fmt="sdf")
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_sdf.assert_called_once_with(1234, os.path.join(".", "aspirin.sdf"))

    @patch("autodock.fetchers.fetch_chembl_smiles", return_value="CCO")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_chembl_smiles_only(self, mock_ensure, mock_smiles, capsys):
        args = self._fetch_args("chembl", "CHEMBL123", fmt=None)
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_smiles.assert_called_once_with("CHEMBL123")

    @patch("autodock.fetchers.fetch_chembl_sdf")
    @patch("autodock.fetchers.fetch_chembl_smiles", return_value="CCO")
    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_chembl_sdf(self, mock_ensure, mock_smiles, mock_sdf, capsys):
        args = self._fetch_args("chembl", "CHEMBL123", fmt="sdf")
        rc = cli.cmd_fetch(args)
        assert rc == 0
        mock_sdf.assert_called_once_with("CHEMBL123", os.path.join(".", "CHEMBL123.sdf"))

    @patch("autodock.utils.ensure_dir", return_value=".")
    def test_fetch_unknown_type(self, mock_ensure, capsys):
        args = self._fetch_args("unknown", "XYZ")
        rc = cli.cmd_fetch(args)
        assert rc == 1
        assert "Unknown fetch type" in capsys.readouterr().out


# ────────────────────────────────
# 6. cmd_prepare_receptor
# ────────────────────────────────
class TestCmdPrepareReceptor:
    @patch("autodock.preparation.prepare_receptor")
    def test_prepare_receptor(self, mock_prep, capsys):
        args = _ns(
            pdb="rec.pdb",
            output="rec.pdbqt",
            keep_waters=False,
            remove_hetatms=True,
            keep_waters_near_metal=True,
            detect_af_structure=True,
            report_json=None,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_prepare_receptor(args)
        assert rc == 0
        mock_prep.assert_called_once_with(
            "rec.pdb",
            "rec.pdbqt",
            remove_water=True,
            remove_hetatms=True,
            keep_waters_near_metal=True,
            detect_af_structure=True,
            output_report_json=None,
        )
        assert "Receptor prepared" in capsys.readouterr().out

    @patch("autodock.preparation.prepare_receptor")
    def test_prepare_receptor_default_output(self, mock_prep, capsys):
        args = _ns(
            pdb="rec.pdb",
            output=None,
            keep_waters=True,
            remove_hetatms=False,
            keep_waters_near_metal=True,
            detect_af_structure=True,
            report_json=None,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_prepare_receptor(args)
        assert rc == 0
        mock_prep.assert_called_once_with(
            "rec.pdb",
            "rec.pdbqt",
            remove_water=False,
            remove_hetatms=False,
            keep_waters_near_metal=True,
            detect_af_structure=True,
            output_report_json=None,
        )


# ────────────────────────────────
# 7. cmd_prepare_ligand
# ────────────────────────────────
class TestCmdPrepareLigand:
    @patch("autodock.preparation.prepare_ligand")
    def test_prepare_ligand(self, mock_prep, capsys):
        args = _ns(
            smiles="CCO",
            output="ligand.pdbqt",
            name="LIG",
            seed=42,
            covalent_check=False,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_prepare_ligand(args)
        assert rc == 0
        mock_prep.assert_called_once_with(
            "CCO", "ligand.pdbqt", name="LIG", seed=42, covalent_check=False
        )


# ────────────────────────────────
# 8. cmd_find_pockets
# ────────────────────────────────
class TestCmdFindPockets:
    @patch("autodock.preparation.find_top_pockets")
    def test_find_pockets(self, mock_find, capsys):
        mock_find.return_value = [
            {
                "pocket_num": 1,
                "pocket_source": "fpocket",
                "center": (1.0, 2.0, 3.0),
                "box_size": (20.0, 20.0, 20.0),
                "druggability": 0.85,
                "druggability_level": "high",
                "p2rank_prob": 0.92,
                "fpocket_verified": True,
                "fpocket_match_distance": 2.5,
                "volume": 500.0,
                "depth": 12.0,
                "openings": 2,
                "n_apolar": 45,
                "n_polar": 12,
                "residue_ids": [{"chain": "A", "resid": 100}, {"chain": "A", "resid": 120}],
                "shape_circularity": 0.45,
                "shape_aspect_ratio": 1.8,
                "flexibility": "rigid",
                "induced_fit_likely": False,
                "af_suitable": True,
                "af_mean_plddt": 92.5,
                "af_min_plddt": 85.0,
                "pocket_type": "orthosteric",
                "distance_to_active": 3.2,
            }
        ]
        args = _ns(
            receptor="rec.pdb",
            ligand=None,
            padding=5.0,
            max_pockets=3,
            known_active_site=None,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_find_pockets(args)
        assert rc == 0
        mock_find.assert_called_once_with(
            "rec.pdb",
            ligand_pdb=None,
            padding=5.0,
            max_pockets=3,
            known_active_site=None,
        )
        out = capsys.readouterr().out
        assert "Found 1 pocket(s)" in out
        assert "P2Rank prob:  0.920" in out
        assert "fpocket-verified" in out

    @patch("autodock.preparation.find_top_pockets")
    def test_find_pockets_no_p2rank_prob(self, mock_find, capsys):
        mock_find.return_value = [
            {
                "pocket_num": 1,
                "pocket_source": "p2rank",
                "center": (0.0, 0.0, 0.0),
                "box_size": (20.0, 20.0, 20.0),
                "druggability": 0.5,
                "druggability_level": "medium",
                "p2rank_prob": None,
                "fpocket_verified": False,
                "fpocket_match_distance": None,
                "volume": 300.0,
                "depth": 8.0,
                "openings": None,
                "n_apolar": None,
                "n_polar": None,
                "residue_ids": [],
                "shape_circularity": None,
                "shape_aspect_ratio": None,
                "flexibility": "moderate",
                "induced_fit_likely": True,
                "af_suitable": None,
                "af_mean_plddt": None,
                "af_min_plddt": None,
                "pocket_type": "unclassified",
                "distance_to_active": None,
            }
        ]
        args = _ns(
            receptor="rec.pdb",
            ligand="lig.pdb",
            padding=4.0,
            max_pockets=1,
            known_active_site=None,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_find_pockets(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Found 1 pocket(s)" in out
        assert "Druggability: 0.500 [medium]" in out
        assert "fpocket-unverified" in out


# ────────────────────────────────
# 9. cmd_dock
# ────────────────────────────────
class TestCmdDock:
    @patch("autodock.docking.dock_ligand")
    def test_dock_with_center(self, mock_dock, capsys):
        result = DockingResult(
            compound_name="LIG",
            receptor="rec.pdbqt",
            best_affinity=-9.5,
            best_pose_pdbqt="pose.pdbqt",
            output_dir="./out",
        )
        mock_dock.return_value = result
        args = _ns(
            receptor="rec.pdbqt",
            ligand="lig.pdbqt",
            center=[1.0, 2.0, 3.0],
            box_size=[20.0, 20.0, 20.0],
            exhaustiveness=32,
            n_poses=20,
            seed=42,
            output_dir="./out",
            name="LIG",
            method="plip",
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_dock(args)
        assert rc == 0
        mock_dock.assert_called_once_with(
            "rec.pdbqt",
            "lig.pdbqt",
            center=(1.0, 2.0, 3.0),
            box_size=(20.0, 20.0, 20.0),
            exhaustiveness=32,
            n_poses=20,
            seed=42,
            output_dir="./out",
            compound_name="LIG",
            receptor_pdb=None,
        )
        out = capsys.readouterr().out
        assert "Docking Complete" in out or "Full report" in out

    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.docking.dock_ligand")
    def test_dock_auto_detect_pocket(self, mock_dock, mock_pockets, capsys):
        mock_pockets.return_value = [
            {
                "center": (5.0, 5.0, 5.0),
                "box_size": (25.0, 25.0, 25.0),
            }
        ]
        result = DockingResult(
            compound_name="LIG",
            receptor="rec.pdbqt",
            best_affinity=-8.0,
            best_pose_pdbqt="pose.pdbqt",
            output_dir="./out",
        )
        mock_dock.return_value = result
        args = _ns(
            receptor="rec.pdbqt",
            ligand="lig.pdbqt",
            center=None,
            box_size=None,
            exhaustiveness=32,
            n_poses=20,
            seed=42,
            output_dir="./out",
            name=None,
            method="plip",
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_dock(args)
        assert rc == 0
        mock_pockets.assert_called_once_with("rec.pdb")
        assert "Auto-detected pocket" in capsys.readouterr().out


# ────────────────────────────────
# 10. cmd_validate
# ────────────────────────────────
class TestCmdValidate:
    @patch("autodock.validation.run_redocking_validation")
    def test_validate_pass(self, mock_val, capsys):
        mock_val.return_value = {
            "rmsd": 1.2,
            "success": True,
            "threshold": 2.0,
            "best_affinity": -9.0,
        }
        args = _ns(
            holo_pdb="holo.pdb",
            ligand_resname="LIG",
            chain_id="C",
            ligand_smiles="CCO",
            exhaustiveness=32,
            n_poses=20,
            seed=42,
            output_dir="./val",
            box_padding=5.0,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_validate(args)
        assert rc == 0
        mock_val.assert_called_once()
        assert "PASS" in capsys.readouterr().out

    @patch("autodock.validation.run_redocking_validation")
    def test_validate_fail(self, mock_val, capsys):
        mock_val.return_value = {
            "rmsd": 3.5,
            "success": False,
            "threshold": 2.0,
            "best_affinity": -7.0,
        }
        args = _ns(
            holo_pdb="holo.pdb",
            ligand_resname=None,
            chain_id=None,
            ligand_smiles=None,
            exhaustiveness=32,
            n_poses=20,
            seed=42,
            output_dir="./val",
            box_padding=5.0,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_validate(args)
        assert rc == 1
        assert "FAIL" in capsys.readouterr().out

    @patch("autodock.validation.run_redocking_validation")
    def test_validate_rmsd_none(self, mock_val, capsys):
        mock_val.return_value = {
            "rmsd": None,
            "success": True,
            "threshold": 2.0,
            "best_affinity": -7.5,
        }
        args = _ns(
            holo_pdb="holo.pdb",
            ligand_resname=None,
            chain_id=None,
            ligand_smiles=None,
            exhaustiveness=32,
            n_poses=20,
            seed=42,
            output_dir="./val",
            box_padding=5.0,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_validate(args)
        assert rc == 0
        assert "N/A" in capsys.readouterr().out


# ────────────────────────────────
# 11. cmd_analyze
# ────────────────────────────────
class TestCmdAnalyze:
    @patch("autodock.rendering.render_interactions_2d")
    @patch("autodock.rendering.render_scene_pymol")
    @patch("autodock.interactions.detect_interactions")
    def test_analyze_with_output_dir(self, mock_detect, mock_render, mock_2d, capsys, tmp_path):
        mock_detect.return_value = [
            {"description": "Hydrogen bond A-LIG"},
            {"description": "Pi-stacking B-LIG"},
        ]
        outdir = tmp_path / "figs"
        args = _ns(
            receptor="rec.pdb",
            ligand="lig.pdbqt",
            output_dir=str(outdir),
            method="plip",
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_analyze(args)
        assert rc == 0
        mock_detect.assert_called_once_with("rec.pdb", "lig.pdbqt", method="plip")
        mock_render.assert_called_once()
        mock_2d.assert_called_once()
        assert "Detected 2 interactions" in capsys.readouterr().out
        assert outdir.exists()

    @patch("autodock.interactions.detect_interactions")
    def test_analyze_no_output_dir(self, mock_detect, capsys):
        mock_detect.return_value = []
        args = _ns(
            receptor="rec.pdb",
            ligand="lig.pdbqt",
            output_dir=None,
            method="plip",
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_analyze(args)
        assert rc == 0
        mock_detect.assert_called_once()


# ────────────────────────────────
# 12. cmd_report
# ────────────────────────────────
class TestCmdReport:
    def test_cmd_report_no_results(self, capsys):
        args = _ns(
            result_dir="./results",
            outdir=None,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_report(args)
        assert rc == 1  # no results found
        out = capsys.readouterr().out
        assert "No docking results found" in out


# ────────────────────────────────
# 13. cmd_virtual_screen
# ────────────────────────────────
class TestCmdVirtualScreen:
    @patch("autodock.docking.virtual_screen")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.utils.download_pdb")
    @patch("autodock.utils.ensure_dir")
    def test_virtual_screen_tsv(
        self, mock_ensure, mock_dl, mock_prep, mock_pockets, mock_vs, tmp_path, capsys
    ):
        mock_ensure.return_value = str(tmp_path)
        mock_pockets.return_value = [{"center": (0.0, 0.0, 0.0), "box_size": (20.0, 20.0, 20.0)}]
        result = DockingResult(compound_name="cmp1", receptor="6LU7", best_affinity=-7.5)
        mock_vs.return_value = ([result], str(tmp_path / "results.csv"))

        lib = tmp_path / "lib.txt"
        lib.write_text("cmp1\tCCO\n# comment\n\ncmp2\tCCN\n")

        args = _ns(
            receptor="6LU7",
            library=str(lib),
            library_format="tsv",
            outdir=str(tmp_path),
            exhaustiveness=16,
            n_poses=3,
            seed=42,
            workers=1,
            covalent_check=False,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_virtual_screen(args)
        assert rc == 0
        assert mock_dl.call_count == 1
        assert mock_dl.call_args[0][0] == "6LU7"
        assert str(mock_dl.call_args[0][1]) == str(tmp_path)
        mock_prep.assert_called_once()
        mock_pockets.assert_called_once_with(
            os.path.join(str(tmp_path), "6LU7.pdb"),
            max_pockets=3,
            cache_dir=os.path.expanduser("~/.autodock/cache"),
        )
        mock_vs.assert_called_once()
        out = capsys.readouterr().out
        assert "Virtual Screening Results" in out
        assert "cmp1" in out

    @patch("autodock.docking.virtual_screen")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.utils.download_pdb")
    @patch("autodock.fetchers.read_sdf_library")
    @patch("autodock.utils.ensure_dir")
    def test_virtual_screen_sdf(
        self,
        mock_ensure,
        mock_read_sdf,
        mock_dl,
        mock_prep,
        mock_pockets,
        mock_vs,
        tmp_path,
        capsys,
    ):
        mock_ensure.return_value = str(tmp_path)
        mock_pockets.return_value = [{"center": (0.0, 0.0, 0.0), "box_size": (20.0, 20.0, 20.0)}]
        mock_read_sdf.return_value = {"cmp1": "CCO", "cmp2": "CCN"}
        result = DockingResult(compound_name="cmp1", receptor="6LU7", best_affinity=-7.5)
        mock_vs.return_value = ([result], str(tmp_path / "results.csv"))

        lib = tmp_path / "lib.sdf"
        lib.write_text("")

        args = _ns(
            receptor="6LU7",
            library=str(lib),
            library_format="sdf",
            outdir=str(tmp_path),
            exhaustiveness=16,
            n_poses=3,
            seed=42,
            workers=1,
            covalent_check=False,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_virtual_screen(args)
        assert rc == 0
        mock_read_sdf.assert_called_once_with(str(lib))
        assert "Loaded 2 compounds from SDF" in capsys.readouterr().out

    @patch("autodock.docking.virtual_screen")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.utils.download_pdb")
    @patch("autodock.utils.ensure_dir")
    def test_virtual_screen_cached_receptor(
        self, mock_ensure, mock_dl, mock_prep, mock_pockets, mock_vs, tmp_path, capsys
    ):
        mock_ensure.return_value = str(tmp_path)
        mock_pockets.return_value = [{"center": (0.0, 0.0, 0.0), "box_size": (20.0, 20.0, 20.0)}]
        mock_vs.return_value = ([], str(tmp_path / "results.csv"))

        # Pre-create receptor PDB so download is skipped
        (tmp_path / "6LU7.pdb").write_text("ATOM")

        lib = tmp_path / "lib.txt"
        lib.write_text("cmp1\tCCO\n")

        args = _ns(
            receptor="6LU7",
            library=str(lib),
            library_format="tsv",
            outdir=str(tmp_path),
            exhaustiveness=16,
            n_poses=3,
            seed=42,
            workers=1,
            covalent_check=False,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_virtual_screen(args)
        assert rc == 0
        mock_dl.assert_not_called()


# ────────────────────────────────
# 14. cmd_run (full pipeline)
# ────────────────────────────────
class TestCmdRun:
    def _make_args(self, tmp_path) -> argparse.Namespace:
        return _ns(
            receptor="6LU7",
            ligand="aspirin",
            outdir=str(tmp_path),
            exhaustiveness=32,
            n_poses=20,
            seed=42,
            method="plip",
            quiet=False,
            verbose=False,
            log_file=None,
        )

    @patch("autodock.reporting.generate_csv_report")
    @patch("autodock.reporting.generate_pdf_report")
    @patch("autodock.rendering.composite_summary")
    @patch("autodock.rendering.render_interactions_2d")
    @patch("autodock.rendering.render_scene_pymol")
    @patch("autodock.interactions.detect_interactions")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.utils.download_pdb")
    def test_run_full_pipeline(
        self,
        mock_dl,
        mock_prep_rec,
        mock_prep_lig,
        mock_pockets,
        mock_dock,
        mock_detect,
        mock_render,
        mock_2d,
        mock_composite,
        mock_pdf,
        mock_csv,
        tmp_path,
        capsys,
    ):
        mock_pockets.return_value = [{"center": (0.0, 0.0, 0.0), "box_size": (20.0, 20.0, 20.0)}]
        result = DockingResult(
            compound_name="aspirin",
            receptor="6LU7",
            best_affinity=-8.5,
            best_pose_pdbqt=str(tmp_path / "best.pdbqt"),
            output_dir=str(tmp_path),
            consensus_affinity=-8.2,
        )
        mock_dock.return_value = result
        mock_detect.return_value = [{"description": "H-bond"}]

        args = self._make_args(tmp_path)
        rc = cli.cmd_run(args)
        assert rc == 0

        assert mock_dl.call_count == 1
        assert mock_dl.call_args[0][0] == "6LU7"
        assert str(mock_dl.call_args[0][1]) == str(tmp_path)
        mock_prep_rec.assert_called_once()
        mock_prep_lig.assert_called_once()
        mock_pockets.assert_called_once()
        mock_dock.assert_called_once()
        mock_detect.assert_called_once()
        mock_render.assert_called()
        mock_2d.assert_called_once()
        mock_csv.assert_called_once()
        out = capsys.readouterr().out
        assert "Pipeline Complete" in out
        assert "Best affinity" in out

    @patch("autodock.reporting.generate_csv_report")
    @patch("autodock.reporting.generate_pdf_report")
    @patch("autodock.rendering.composite_summary")
    @patch("autodock.rendering.render_interactions_2d")
    @patch("autodock.rendering.render_scene_pymol")
    @patch("autodock.interactions.detect_interactions")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.utils.download_pdb")
    @patch("pubchempy.get_compounds")
    def test_run_with_pubchem_smiles(
        self,
        mock_pcp,
        mock_dl,
        mock_prep_rec,
        mock_prep_lig,
        mock_pockets,
        mock_dock,
        mock_detect,
        mock_render,
        mock_2d,
        mock_composite,
        mock_pdf,
        mock_csv,
        tmp_path,
        capsys,
    ):
        mock_compound = MagicMock()
        mock_compound.connectivity_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        mock_compound.canonical_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        mock_pcp.return_value = [mock_compound]

        mock_pockets.return_value = [{"center": (0.0, 0.0, 0.0), "box_size": (20.0, 20.0, 20.0)}]
        result = DockingResult(
            compound_name="aspirin",
            receptor="6LU7",
            best_affinity=-8.0,
            best_pose_pdbqt=str(tmp_path / "best.pdbqt"),
            output_dir=str(tmp_path),
        )
        mock_dock.return_value = result
        mock_detect.return_value = []

        args = self._make_args(tmp_path)
        rc = cli.cmd_run(args)
        assert rc == 0
        mock_pcp.assert_called_once_with("aspirin", "name")
        out = capsys.readouterr().out
        assert "Ligand SMILES (PubChem)" in out

    @patch("autodock.reporting.generate_csv_report")
    @patch("autodock.reporting.generate_pdf_report")
    @patch("autodock.rendering.composite_summary")
    @patch("autodock.rendering.render_interactions_2d")
    @patch("autodock.rendering.render_scene_pymol")
    @patch("autodock.interactions.detect_interactions")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.utils.download_pdb")
    def test_run_rendering_exception(
        self,
        mock_dl,
        mock_prep_rec,
        mock_prep_lig,
        mock_pockets,
        mock_dock,
        mock_detect,
        mock_render,
        mock_2d,
        mock_composite,
        mock_pdf,
        mock_csv,
        tmp_path,
        capsys,
    ):
        mock_render.side_effect = RuntimeError("PyMOL not found")
        mock_pockets.return_value = [{"center": (0.0, 0.0, 0.0), "box_size": (20.0, 20.0, 20.0)}]
        result = DockingResult(
            compound_name="aspirin",
            receptor="6LU7",
            best_affinity=-7.5,
            best_pose_pdbqt=str(tmp_path / "best.pdbqt"),
            output_dir=str(tmp_path),
        )
        mock_dock.return_value = result
        mock_detect.return_value = []

        args = self._make_args(tmp_path)
        rc = cli.cmd_run(args)
        assert rc == 0
        # Should continue despite rendering failure
        mock_csv.assert_called_once()

    @patch("autodock.reporting.generate_csv_report")
    @patch("autodock.reporting.generate_pdf_report")
    @patch("autodock.rendering.composite_summary")
    @patch("autodock.rendering.render_interactions_2d")
    @patch("autodock.rendering.render_scene_pymol")
    @patch("autodock.interactions.detect_interactions")
    @patch("autodock.docking.dock_ligand")
    @patch("autodock.preparation.find_top_pockets")
    @patch("autodock.preparation.prepare_ligand")
    @patch("autodock.preparation.prepare_receptor")
    @patch("autodock.utils.download_pdb")
    def test_run_cached_receptor(
        self,
        mock_dl,
        mock_prep_rec,
        mock_prep_lig,
        mock_pockets,
        mock_dock,
        mock_detect,
        mock_render,
        mock_2d,
        mock_composite,
        mock_pdf,
        mock_csv,
        tmp_path,
        capsys,
    ):
        # Pre-create receptor so download is skipped
        (tmp_path / "6LU7.pdb").write_text("ATOM")
        mock_pockets.return_value = [{"center": (0.0, 0.0, 0.0), "box_size": (20.0, 20.0, 20.0)}]
        result = DockingResult(
            compound_name="aspirin",
            receptor="6LU7",
            best_affinity=-7.0,
            best_pose_pdbqt=str(tmp_path / "best.pdbqt"),
            output_dir=str(tmp_path),
        )
        mock_dock.return_value = result
        mock_detect.return_value = []

        args = self._make_args(tmp_path)
        rc = cli.cmd_run(args)
        assert rc == 0
        mock_dl.assert_not_called()
        assert "Using cached" in capsys.readouterr().out


# ────────────────────────────────
# 15. cmd_md
# ────────────────────────────────
class TestCmdMd:
    @patch("autodock.md_simulation.run_md_stability")
    def test_md(self, mock_md, capsys):
        mock_md.return_value = {
            "trajectory": "traj.dcd",
            "final_structure": "final.pdb",
            "ligand_rmsd_mean": 1.5,
            "ligand_rmsd_std": 0.3,
            "receptor_ca_rmsd_mean": 0.8,
            "n_hbonds_mean": 3.2,
        }
        args = _ns(
            receptor="rec.pdb",
            ligand="lig.pdbqt",
            outdir="./md",
            steps=500_000,
            dt=2.0,
            temperature=300.0,
            solvent="implicit",
            platform=None,
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_md(args)
        assert rc == 0
        mock_md.assert_called_once_with(
            receptor_pdb="rec.pdb",
            ligand_pdbqt="lig.pdbqt",
            output_dir="./md",
            n_steps=500_000,
            dt_fs=2.0,
            temperature_k=300.0,
            solvent_model="implicit",
            platform_name=None,
        )
        out = capsys.readouterr().out
        assert "MD Simulation Results" in out
        assert "Ligand RMSD" in out

    @patch("autodock.md_simulation.run_md_stability")
    def test_md_minimal_result(self, mock_md, capsys):
        mock_md.return_value = {
            "trajectory": "traj.dcd",
            "final_structure": "final.pdb",
        }
        args = _ns(
            receptor="rec.pdb",
            ligand="lig.pdbqt",
            outdir="./md",
            steps=1000,
            dt=2.0,
            temperature=300.0,
            solvent="implicit",
            platform="CPU",
            quiet=False,
            verbose=False,
            log_file=None,
        )
        rc = cli.cmd_md(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "MD Simulation Results" in out
        assert "Ligand RMSD" not in out
