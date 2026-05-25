"""Tests for autodock.cli — argument parser and command dispatch."""
from __future__ import annotations

import pytest

from autodock import cli


class TestBuildParser:
    def test_parser_creation(self):
        parser = cli.build_parser()
        assert parser is not None

    def test_status_subcommand(self):
        parser = cli.build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_dock_subcommand(self):
        parser = cli.build_parser()
        args = parser.parse_args([
            "dock", "rec.pdbqt", "lig.pdbqt",
            "--center", "0", "0", "0",
            "--box-size", "20", "20", "20",
            "--seed", "42",
        ])
        assert args.command == "dock"
        assert args.receptor == "rec.pdbqt"
        assert args.seed == 42
        assert args.center == [0.0, 0.0, 0.0]

    def test_dock_default_seed(self):
        parser = cli.build_parser()
        args = parser.parse_args(["dock", "rec.pdbqt", "lig.pdbqt"])
        # Default seed should now be deterministic
        assert args.seed == 42

    def test_validate_subcommand(self):
        parser = cli.build_parser()
        args = parser.parse_args(["validate", "holo.pdb", "--seed", "123"])
        assert args.command == "validate"
        assert args.seed == 123

    def test_virtual_screen_subcommand(self):
        parser = cli.build_parser()
        args = parser.parse_args([
            "virtual-screen", "--receptor", "6LU7",
            "--library", "lib.txt", "--workers", "4",
        ])
        assert args.command == "virtual-screen"
        assert args.workers == 4

    def test_batch_dock_subcommand(self):
        parser = cli.build_parser()
        args = parser.parse_args([
            "batch-dock",
            "--receptors", "r1.pdbqt", "r2.pdbqt",
            "--ligands", "l1.pdbqt", "l2.pdbqt",
            "--pockets", "pockets.json",
            "--seed", "42",
        ])
        assert args.command == "batch-dock"
        assert args.receptors == ["r1.pdbqt", "r2.pdbqt"]
        assert args.ligands == ["l1.pdbqt", "l2.pdbqt"]
        assert args.pockets == "pockets.json"

    def test_run_subcommand(self):
        parser = cli.build_parser()
        args = parser.parse_args([
            "run", "--receptor", "6LU7", "--ligand", "aspirin",
        ])
        assert args.command == "run"
        assert args.receptor == "6LU7"


class TestSetupLogging:
    def test_quiet(self):
        import logging
        args = cli.build_parser().parse_args(["-q", "status"])
        cli._setup_logging(args)
        from autodock.core import autodock_logger
        # Stream handlers should be at WARNING
        for h in autodock_logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                assert h.level == logging.WARNING
        # Reset
        cli._setup_logging(cli.build_parser().parse_args(["status"]))

    def test_verbose(self):
        import logging
        args = cli.build_parser().parse_args(["-v", "status"])
        cli._setup_logging(args)
        from autodock.core import autodock_logger
        for h in autodock_logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                assert h.level == logging.DEBUG
        cli._setup_logging(cli.build_parser().parse_args(["status"]))


class TestCmdStatus:
    def test_cmd_status(self, capsys):
        args = cli.build_parser().parse_args(["status"])
        rc = cli.cmd_status(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Environment Status" in captured.out


class TestMain:
    def test_no_command_prints_help(self, capsys):
        rc = cli.main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out or "usage:" in captured.err

    def test_keyboard_interrupt(self, monkeypatch):
        def raise_interrupt(args):
            raise KeyboardInterrupt()
        monkeypatch.setattr(cli, "cmd_status", raise_interrupt)
        rc = cli.main(["status"])
        assert rc == 130
