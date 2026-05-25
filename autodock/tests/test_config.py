"""Tests for autodock.config — YAML loading, defaults, validation."""
from __future__ import annotations

import pytest

from autodock import config
from autodock.core import ConfigurationError


class TestLoadConfig:
    def test_load_valid_config(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("docking:\n  exhaustiveness: 64\n")
        cfg = config.load_config(f)
        assert cfg["docking"]["exhaustiveness"] == 64
        # Defaults merged
        assert cfg["project"]["name"] == "docking_run"

    def test_file_not_found(self):
        with pytest.raises(ConfigurationError):
            config.load_config("/nonexistent/config.yaml")

    def test_invalid_yaml(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("docking: [unclosed")
        with pytest.raises(ConfigurationError):
            config.load_config(f)

    def test_not_a_dict(self, tmp_path):
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigurationError):
            config.load_config(f)


class TestValidate:
    def test_exhaustiveness_warning(self, caplog):
        import logging
        cfg = {"docking": {"exhaustiveness": 4}}
        with caplog.at_level(logging.WARNING):
            config._validate(cfg)
        assert "very low" in caplog.text

    def test_num_modes_warning(self, caplog):
        import logging
        cfg = {"docking": {"num_modes": 5}}
        with caplog.at_level(logging.WARNING):
            config._validate(cfg)
        assert "low" in caplog.text

    def test_posebusters_disabled_warning(self, caplog):
        import logging
        cfg = {"validation": {"posebusters": False}}
        with caplog.at_level(logging.WARNING):
            config._validate(cfg)
        assert "strongly discouraged" in caplog.text


class TestWriteDefaultConfig:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "default.yaml"
        result = config.write_default_config(path)
        assert path.exists()
        text = path.read_text()
        assert "exhaustiveness: 32" in text
        assert result == str(path)
