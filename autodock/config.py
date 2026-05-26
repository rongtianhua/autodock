"""
autodock.config — YAML configuration parser and validator.
============================================================
Reads docking pipeline configs and validates parameters against
publication-grade defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from autodock.core import (
    VINA_DEFAULT_EXHAUSTIVENESS,
    VINA_DEFAULT_N_POSES,
    ConfigurationError,
    logger,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigurationError: If file not found, unreadable, or invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")

    try:
        import yaml
    except ImportError:
        raise ConfigurationError(
            "PyYAML is required for config parsing. Install: conda install pyyaml"
        )

    try:
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse YAML config: {exc}")

    if not isinstance(cfg, dict):
        raise ConfigurationError("Config file must contain a top-level mapping.")

    cfg = _apply_defaults(cfg)
    _validate(cfg)
    return cfg


def _apply_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge user config with publication-grade defaults."""
    defaults = {
        "project": {
            "name": "docking_run",
            "output_dir": "./docking_results",
            "log_level": "INFO",
        },
        "receptor": {
            "source": "pdb",
            "pdb_id": None,
            "chain": None,
            "remove_water": True,
            "remove_hetatms": True,
            "ph": 7.0,
            "minimize": True,
            "forcefield": "amber14-all",
        },
        "pocket": {
            "method": "p2rank",
            "reference_ligand": None,
            "top_n": 3,
            "min_druggability": 0.15,
            "padding": 5.0,
        },
        "ligands": {
            "source": "file",
            "file": None,
            "format": "sdf",
            "enumerate_tautomers": True,
            "enumerate_protonation": True,
            "ph_range": 1.5,
            "max_conformers": 10,
            "energy_minimize": True,
        },
        "docking": {
            "engine": "vina",
            "exhaustiveness": VINA_DEFAULT_EXHAUSTIVENESS,
            "num_modes": VINA_DEFAULT_N_POSES,
            "energy_range": 3.0,
            "scoring_function": "vina",
            "box_buffer": 5.0,
            "repeats": 3,
            "timeout": 600,
            "seed": None,
        },
        "validation": {
            "posebusters": True,
            "rmsd_clustering": True,
            "rmsd_cutoff": 2.0,
            "max_clash_distance": 1.2,
            "redocking": True,
        },
        "analysis": {
            "prolif": True,
            "plip": True,
            "generate_pymol_script": True,
        },
        "reporting": {
            "generate_pdf": True,
            "generate_excel": True,
            "generate_pymol_figures": True,
            "dpi": 300,
        },
    }

    def deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = deep_merge(result[key], val)
            else:
                result[key] = val
        return result

    return deep_merge(defaults, cfg)


def _validate(cfg: dict[str, Any]) -> None:
    """Validate configuration values and warn about non-publication settings."""
    dock = cfg.get("docking", {})
    exhaust = dock.get("exhaustiveness", VINA_DEFAULT_EXHAUSTIVENESS)
    if exhaust < 8:
        logger.warning(
            f"Config: exhaustiveness={exhaust} is very low. "
            f"Publication standard is >=32 for reliable pose prediction."
        )
    if exhaust < 32:
        logger.info(
            f"Config: exhaustiveness={exhaust}. " f"Consider >=32 for publication-grade docking."
        )

    num_modes = dock.get("num_modes", VINA_DEFAULT_N_POSES)
    if num_modes < 9:
        logger.warning(
            f"Config: num_modes={num_modes} is low. "
            f"Recommend >=9-20 for clustering and validation."
        )

    val = cfg.get("validation", {})
    if not val.get("posebusters", True):
        logger.warning(
            "Config: posebusters validation disabled. "
            "This is strongly discouraged for publication-grade work."
        )

    pocket = cfg.get("pocket", {})
    if pocket.get("method") == "fpocket" and pocket.get("use_p2rank", True) is False:
        logger.info("Config: P2Rank rescoring disabled. " "Pocket ranking may be less reliable.")

    out_dir = cfg.get("project", {}).get("output_dir", "./docking_results")
    os.makedirs(out_dir, exist_ok=True)


def write_default_config(path: str | Path = "docking_config.yaml") -> str:
    """Write a default configuration file to disk."""
    path = Path(path)
    cfg_text = """# Publication-grade molecular docking configuration
project:
  name: "docking_run"
  output_dir: "./docking_results"
  log_level: "INFO"

receptor:
  source: "pdb"          # pdb | alphafold | file
  pdb_id: "6LU7"
  chain: "A"
  remove_water: true
  remove_hetatms: true
  ph: 7.0
  minimize: true
  forcefield: "amber14-all"

pocket:
  method: "p2rank"       # p2rank | fpocket | reference
  reference_ligand: null
  top_n: 3
  min_druggability: 0.15
  padding: 5.0

ligands:
  source: "file"
  file: "./ligands.sdf"
  format: "sdf"
  enumerate_tautomers: true
  enumerate_protonation: true
  ph_range: 1.5
  max_conformers: 10
  energy_minimize: true

docking:
  engine: "vina"
  exhaustiveness: 32
  num_modes: 20
  energy_range: 3.0
  scoring_function: "vina"
  box_buffer: 5.0
  repeats: 3
  timeout: 600
  seed: null

validation:
  posebusters: true
  rmsd_clustering: true
  rmsd_cutoff: 2.0
  max_clash_distance: 1.2
  redocking: true

analysis:
  prolif: true
  plip: true
  generate_pymol_script: true

reporting:
  generate_pdf: true
  generate_excel: true
  generate_pymol_figures: true
  dpi: 300
"""
    with open(path, "w") as fh:
        fh.write(cfg_text)
    logger.info(f"Default config written to {path}")
    return str(path)
