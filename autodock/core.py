"""
autodock.core — Core infrastructure for publication-grade molecular docking.
============================================================================
Exception hierarchy, structured logging, DockingResult dataclass,
environment auto-discovery, and shared constants.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Exception Hierarchy
# ─────────────────────────────────────────────────────────────────────────────


class DockingError(Exception):
    """Base exception for all autodock errors."""

    pass


class StructureFetchError(DockingError):
    """Failed to fetch protein/ligand structure from remote source."""

    pass


class PreparationError(DockingError):
    """Failed to prepare receptor or ligand (PDBQT generation failed)."""

    pass


class DockingCalculationError(DockingError):
    """AutoDock Vina docking failed (no poses, timeout, etc.)."""

    pass


class VisualizationError(DockingError):
    """Rendering / interaction detection failed."""

    pass


class ValidationError(DockingError):
    """Redocking validation or clash / RMSD computation failed."""

    pass


class MDError(DockingError):
    """Molecular dynamics simulation or analysis failed."""

    pass


class DataSourceError(DockingError):
    """External database query failed (BindingDB, ZINC, etc.)."""

    pass


class ConfigurationError(DockingError):
    """Invalid configuration file or parameter."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Structured Logger
# ─────────────────────────────────────────────────────────────────────────────


class _AutodockFormatter(logging.Formatter):
    """Compact formatter: [autodock] LEVEL: message"""

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname[0] if record.levelname != "DEBUG" else "D"
        return f"[autodock] {level}: {record.getMessage()}"


autodock_logger = logging.getLogger("autodock")
autodock_logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers if module is reloaded
if not autodock_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(_AutodockFormatter())
    autodock_logger.addHandler(_handler)

    # Optional file logging
    _log_dir = os.path.expanduser("~/.autodock/logs")
    try:
        os.makedirs(_log_dir, exist_ok=True)
        _file_handler = logging.FileHandler(
            os.path.join(_log_dir, "autodock.log"),
            mode="a",
        )
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        autodock_logger.addHandler(_file_handler)
    except (OSError, PermissionError):
        pass  # Cannot write to home directory log path

logger = autodock_logger  # backward-compat alias


def set_log_level(level: int | str) -> None:
    """Set autodock logger level. Accepts int (logging.DEBUG) or str ('INFO')."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    autodock_logger.setLevel(level)
    # Only adjust StreamHandler levels; leave FileHandler at DEBUG for audit trail
    for h in autodock_logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(level)


# ─────────────────────────────────────────────────────────────────────────────
# Vina seed helper
# ─────────────────────────────────────────────────────────────────────────────


# Default deterministic seed for publication-grade reproducibility
DEFAULT_SEED: int = 42


def _get_vina_seed(seed: int | None = None) -> int:
    """
    Return a valid Vina seed integer.
    - If seed is given, use it directly (deterministic).
    - If seed is None, return the global default seed (42) for reproducibility.
    """
    if seed is not None:
        return int(seed)
    return DEFAULT_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Environment Auto-Discovery (zero hard-coded paths)
# ─────────────────────────────────────────────────────────────────────────────

CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")


def find_conda_tool(name: str) -> str | None:
    """
    Locate an executable in the current conda environment, then fall back
    to system PATH.  Returns the absolute path or None if not found.
    """
    if CONDA_PREFIX:
        candidate = os.path.join(CONDA_PREFIX, "bin", name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which(name)


def find_java() -> str | None:
    """
    Locate a Java runtime.  Priority:
      1. $CONDA_PREFIX/bin/java
      2. /usr/libexec/java_home (macOS)
      3. shutil.which('java')
    """
    if CONDA_PREFIX:
        candidate = os.path.join(CONDA_PREFIX, "bin", "java")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    try:
        result = subprocess.run(
            ["/usr/libexec/java_home", "--failfast"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            java_home = result.stdout.strip()
            candidate = os.path.join(java_home, "bin", "java")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    except Exception:
        pass
    return shutil.which("java")


def find_p2rank() -> str | None:
    """
    Locate P2Rank prank script.  Searches conda env opt/ and bin/ first.
    """
    candidates = []
    if CONDA_PREFIX:
        candidates.extend(
            [
                os.path.join(CONDA_PREFIX, "opt", "p2rank_2.5.1", "prank"),
                os.path.join(CONDA_PREFIX, "bin", "prank"),
            ]
        )
    candidates.append(shutil.which("prank"))
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def find_pymol() -> str | None:
    """Locate PyMOL executable (CLI)."""
    if CONDA_PREFIX:
        candidate = os.path.join(CONDA_PREFIX, "bin", "pymol")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    # macOS Schrodinger PyMOL
    schrodinger = "/Applications/PyMOL.app/Contents/MacOS/PyMOL"
    if os.path.isfile(schrodinger) and os.access(schrodinger, os.X_OK):
        return schrodinger
    return shutil.which("pymol")


def safe_subprocess(
    cmd: list[str],
    timeout: int = 120,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bool, str, str]:
    """
    Run a subprocess safely with consistent error handling.

    Returns:
        (success: bool, stdout: str, stderr: str)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        logger.warning(
            f"Command failed (rc={result.returncode}): {' '.join(cmd[:6])}...\n"
            f"  stderr: {result.stderr[:300]}"
        )
        return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout}s: {' '.join(cmd[:6])}...")
        return False, "", f"timeout after {timeout}s"
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        return False, "", f"command not found: {cmd[0]}"
    except Exception as exc:
        logger.error(f"Unexpected error running {' '.join(cmd[:6])}...: {exc}")
        return False, "", str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Optional-dependency feature flags (probed at import time)
# ─────────────────────────────────────────────────────────────────────────────


def _probe(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


_HAVE_VINA = _probe("vina")
_HAVE_RDKIT = _probe("rdkit")
_HAVE_MEEKO = _probe("meeko")
_HAVE_PLIP = _probe("plip.basic.config")
_HAVE_MDANALYSIS = _probe("MDAnalysis")
_HAVE_PROLIF = _probe("prolif")
_HAVE_OPENMM = _probe("openmm")
_HAVE_OPENBABEL = _probe("openbabel")
_HAVE_PYMOL_CLI = find_pymol() is not None
_HAVE_VINA_CLI = find_conda_tool("vina") is not None
_HAVE_FPOCKET = find_conda_tool("fpocket") is not None
_HAVE_P2RANK = find_p2rank() is not None
_HAVE_JAVA = find_java() is not None
_HAVE_PYMOL_IMPORT = _probe("pymol")


def get_environment_status() -> dict[str, Any]:
    """Return a dictionary describing the runtime environment."""
    return {
        "conda_prefix": CONDA_PREFIX,
        "python": shutil.which("python") or "unknown",
        "java": find_java(),
        "vina_cli": find_conda_tool("vina"),
        "vina_python": _HAVE_VINA,
        "rdkit": _HAVE_RDKIT,
        "meeko": _HAVE_MEEKO,
        "plip": _HAVE_PLIP,
        "mdanalysis": _HAVE_MDANALYSIS,
        "prolif": _HAVE_PROLIF,
        "openmm": _HAVE_OPENMM,
        "openbabel": _HAVE_OPENBABEL,
        "pymol_cli": find_pymol(),
        "pymol_import": _HAVE_PYMOL_IMPORT,
        "fpocket": find_conda_tool("fpocket"),
        "p2rank": find_p2rank(),
        "gromacs": find_conda_tool("gmx"),
        "timestamp": datetime.now().isoformat(),
    }


def print_environment_status() -> None:
    """Pretty-print environment status to the console."""
    st = get_environment_status()
    print("=" * 55)
    print("🧬  Autodock Environment Status")
    print("=" * 55)
    for key, val in st.items():
        if key == "timestamp":
            continue
        status = "✅" if val else "❌"
        label = key.replace("_", " ").title()
        print(f"  {status}  {label:<20s}  {val if val else 'NOT FOUND'}")
    print("=" * 55)

    core_ok = all(
        [
            st["vina_python"],
            st["rdkit"],
            st["meeko"],
            st["vina_cli"],
            st["openbabel"],
        ]
    )
    if core_ok:
        print("✅  Core dependencies ready — docking pipeline available.")
    else:
        print("⚠️   Some core dependencies missing — check conda environment.")


# ─────────────────────────────────────────────────────────────────────────────
# DockingResult — structured publication-ready result container
# ─────────────────────────────────────────────────────────────────────────────

_RECEPTOR_SOURCE_LABELS = {
    "PDB": "X-ray crystal structure (RCSB PDB)",
    "PDB-REDO": "PDB-REDO optimized crystal structure",
    "AlphaFold": "AlphaFold2 predicted structure (UniProt)",
    "SWISS-MODEL": "SWISS-MODEL homology model",
}


def detect_receptor_source(pdb_path: str) -> str | None:
    """
    Auto-detect receptor source from PDB file header.
    Returns one of: 'PDB', 'PDB-REDO', 'AlphaFold', 'SWISS-MODEL', or None.
    """
    if not os.path.isfile(pdb_path):
        return None
    try:
        with open(pdb_path) as fh:
            text = fh.read(5000)
    except Exception:
        return None
    text_upper = text.upper()
    if "ALPHAFOLD" in text_upper or "TITLE  ALPHAFOLD" in text_upper:
        return "AlphaFold"
    if "EXPDTA  THEORETICAL MODEL" in text_upper:
        return "SWISS-MODEL"
    if "PDB-REDO" in text:
        return "PDB-REDO"
    if "EXPDTA  X-RAY" in text_upper or "EXPDTA  SYNCHROTRON" in text_upper:
        return "PDB"
    return None


@dataclass(slots=True)
class DockingResult:
    """
    Structured, publication-ready result from a single docking run.

    Use `.to_dict()` for JSON serialization, `.to_dataframe_row()` for CSV.
    """

    # ── Identity ─────────────────────────────────────────────────────
    compound_name: str
    receptor: str
    method: str = "AutoDock Vina"
    receptor_source: str | None = None

    # ── Parameters (reproducibility) ─────────────────────────────────
    center: tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    box_size: tuple[float, float, float] = field(default_factory=lambda: (20.0, 20.0, 20.0))
    exhaustiveness: int = 32
    n_poses: int = 10
    seed: int | None = None

    # ── Scores ───────────────────────────────────────────────────────
    best_affinity: float | None = None  # kcal/mol (more negative = tighter)
    scoring_functions: list[str] = field(default_factory=lambda: ["vina"])
    all_scores: dict[str, float] = field(default_factory=dict)
    consensus_affinity: float | None = None  # median of all_scores
    pre_dock_score: float | None = None
    score_improvement: float | None = None

    # ── Validation ───────────────────────────────────────────────────
    rmsd_from_crystal: float | None = None  # Å
    protocol_valid: bool | None = None
    redocking_threshold: float | None = None

    # ── Pose quality ─────────────────────────────────────────────────
    posebusters_pass: bool | None = None
    clash_score: float | None = None  # Å overlap
    clash_acceptable: bool | None = None

    # ── Pose clustering ──────────────────────────────────────────────
    pose_clusters: list[dict] | None = None
    n_clusters: int | None = None
    rmsd_clustering_threshold: float | None = None

    # ── Interactions (raw list) ──────────────────────────────────────
    interactions: list[dict] = field(default_factory=list)

    # ── Pocket metadata ──────────────────────────────────────────────
    binding_pocket: dict | None = None
    pocket_source: str | None = None

    # ── Output files ─────────────────────────────────────────────────
    best_pose_pdbqt: str | None = None
    all_poses_pdbqt: str | None = None
    output_dir: str | None = None

    # ── Provenance / reproducibility ─────────────────────────────────
    version: str = field(default_factory=lambda: __import__("autodock").__version__)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    runtime_seconds: float | None = None

    # ── Internal cached aggregates ───────────────────────────────────
    _interactions_computed: bool = field(default=False, repr=False)
    _n_hbonds: int = field(default=0, repr=False)
    _n_pi_pi: int = field(default=0, repr=False)
    _n_pi_cation: int = field(default=0, repr=False)
    _n_hydrophobic: int = field(default=0, repr=False)
    _n_saltbridge: int = field(default=0, repr=False)
    _n_halogen: int = field(default=0, repr=False)
    _n_waterbridge: int = field(default=0, repr=False)
    _n_metal: int = field(default=0, repr=False)

    def __post_init__(self):
        for attr in ("center", "box_size"):
            val = getattr(self, attr)
            if isinstance(val, list) and len(val) == 3:
                setattr(self, attr, tuple(float(v) for v in val))

    # ── Properties ───────────────────────────────────────────────────
    @property
    def n_hbonds(self) -> int:
        if not self._interactions_computed:
            self._aggregate_interactions()
        return self._n_hbonds

    @property
    def n_pi_pi(self) -> int:
        if not self._interactions_computed:
            self._aggregate_interactions()
        return self._n_pi_pi

    @property
    def n_pi_cation(self) -> int:
        if not self._interactions_computed:
            self._aggregate_interactions()
        return self._n_pi_cation

    @property
    def n_hydrophobic(self) -> int:
        if not self._interactions_computed:
            self._aggregate_interactions()
        return self._n_hydrophobic

    @property
    def n_salt_bridges(self) -> int:
        if not self._interactions_computed:
            self._aggregate_interactions()
        return self._n_saltbridge

    @property
    def n_halogen_bonds(self) -> int:
        if not self._interactions_computed:
            self._aggregate_interactions()
        return self._n_halogen

    @property
    def n_water_bridges(self) -> int:
        if not self._interactions_computed:
            self._aggregate_interactions()
        return self._n_waterbridge

    @property
    def n_metal_complexes(self) -> int:
        if not self._interactions_computed:
            self._aggregate_interactions()
        return self._n_metal

    @property
    def method_label(self) -> str:
        parts = [self.method]
        if len(self.scoring_functions) > 1:
            parts.append(f"[consensus: {'/'.join(self.scoring_functions)}]")
        if self.receptor_source:
            parts.append(
                f"({_RECEPTOR_SOURCE_LABELS.get(self.receptor_source, self.receptor_source)})"
            )
        return " ".join(parts)

    @property
    def interaction_summary(self) -> dict[str, int]:
        return {
            "H-bond": self.n_hbonds,
            "π-π": self.n_pi_pi,
            "π-cation": self.n_pi_cation,
            "Hydrophobic": self.n_hydrophobic,
            "Salt bridge": self.n_salt_bridges,
            "Halogen bond": self.n_halogen_bonds,
            "Water bridge": self.n_water_bridges,
            "Metal complex": self.n_metal_complexes,
        }

    # ── Private helpers ──────────────────────────────────────────────
    def _aggregate_interactions(self) -> None:
        self._n_hbonds = sum(1 for i in self.interactions if i.get("type") == "H-bond")
        self._n_pi_pi = sum(1 for i in self.interactions if i.get("type") == "π-π")
        self._n_pi_cation = sum(1 for i in self.interactions if i.get("type") == "π-cation")
        self._n_hydrophobic = sum(1 for i in self.interactions if i.get("type") == "Hydrophobic")
        self._n_saltbridge = sum(1 for i in self.interactions if i.get("type") == "Salt bridge")
        self._n_halogen = sum(1 for i in self.interactions if i.get("type") == "Halogen bond")
        self._n_waterbridge = sum(1 for i in self.interactions if i.get("type") == "Water bridge")
        self._n_metal = sum(1 for i in self.interactions if i.get("type") == "Metal complex")
        self._interactions_computed = True

    # ── Serialisation ────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # strip private cached fields
        for key in (
            "_n_hbonds",
            "_n_pi_pi",
            "_n_pi_cation",
            "_n_hydrophobic",
            "_n_saltbridge",
            "_n_halogen",
            "_n_waterbridge",
            "_n_metal",
            "_interactions_computed",
        ):
            d.pop(key, None)
        if self.receptor_source:
            d["receptor_source_label"] = _RECEPTOR_SOURCE_LABELS.get(
                self.receptor_source, self.receptor_source
            )
        return d

    def to_dataframe_row(self) -> dict[str, Any]:
        pocket = self.binding_pocket or {}
        return {
            "compound": self.compound_name,
            "receptor": os.path.basename(self.receptor) if self.receptor else None,
            "receptor_source": self.receptor_source,
            "receptor_source_label": (
                _RECEPTOR_SOURCE_LABELS.get(self.receptor_source) if self.receptor_source else None
            ),
            "best_affinity_kcal_mol": self.best_affinity,
            "consensus_affinity": self.consensus_affinity,
            "scoring_functions": ",".join(self.scoring_functions),
            "pre_dock_score": self.pre_dock_score,
            "score_improvement": self.score_improvement,
            "n_hbonds": self.n_hbonds,
            "n_pi_pi": self.n_pi_pi,
            "n_pi_cation": self.n_pi_cation,
            "n_hydrophobic": self.n_hydrophobic,
            "n_salt_bridges": self.n_salt_bridges,
            "n_halogen_bonds": self.n_halogen_bonds,
            "n_water_bridges": self.n_water_bridges,
            "n_metal_complexes": self.n_metal_complexes,
            "posebusters_pass": self.posebusters_pass,
            "clash_score_A": self.clash_score,
            "clash_acceptable": self.clash_acceptable,
            "rmsd_from_crystal_A": self.rmsd_from_crystal,
            "protocol_valid": self.protocol_valid,
            "pocket_num": pocket.get("pocket_num"),
            "pocket_druggability": pocket.get("druggability"),
            "pocket_p2rank_prob": pocket.get("p2rank_prob"),
            "center_x": self.center[0] if self.center else None,
            "center_y": self.center[1] if self.center else None,
            "center_z": self.center[2] if self.center else None,
            "box_x": self.box_size[0] if self.box_size else None,
            "box_y": self.box_size[1] if self.box_size else None,
            "box_z": self.box_size[2] if self.box_size else None,
            "exhaustiveness": self.exhaustiveness,
            "n_poses": self.n_poses,
            "seed": self.seed,
            "best_pose_pdbqt": self.best_pose_pdbqt,
            "all_poses_pdbqt": self.all_poses_pdbqt,
            "method": self.method_label,
        }


def build_docking_result(
    compound_name: str,
    receptor: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    energies: np.ndarray | None = None,
    poses: list[str] | None = None,
    interactions: list[dict] | None = None,
    clash_result: dict | None = None,
    pre_dock_score: float | None = None,
    binding_pocket: dict | None = None,
    receptor_source: str | None = None,
    best_pose_path: str | None = None,
    rmsd_from_crystal: float | None = None,
    protocol_valid: bool | None = None,
    redocking_threshold: float | None = None,
    exhaustiveness: int = 32,
    n_poses: int = 10,
    seed: int | None = None,
    all_scores: dict[str, float] | None = None,
    consensus_affinity: float | None = None,
    pocket_source: str | None = None,
) -> DockingResult:
    """Build a DockingResult from raw docking outputs."""
    best_affinity = None
    if energies is not None and energies.size > 0:
        best_affinity = float(energies[0][0])
    score_improvement = None
    if pre_dock_score is not None and best_affinity is not None:
        score_improvement = pre_dock_score - best_affinity

    return DockingResult(
        compound_name=compound_name,
        receptor=receptor,
        center=tuple(center) if center else None,
        box_size=tuple(box_size) if box_size else None,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        seed=seed,
        best_affinity=best_affinity,
        pre_dock_score=pre_dock_score,
        score_improvement=score_improvement,
        rmsd_from_crystal=rmsd_from_crystal,
        protocol_valid=protocol_valid,
        redocking_threshold=redocking_threshold,
        interactions=interactions or [],
        clash_score=clash_result.get("clash_score") if clash_result else None,
        clash_acceptable=clash_result.get("is_acceptable") if clash_result else None,
        binding_pocket=binding_pocket,
        receptor_source=receptor_source,
        best_pose_pdbqt=best_pose_path,
        all_scores=all_scores or {},
        consensus_affinity=consensus_affinity,
        pocket_source=pocket_source,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Water / non-structural residue names to skip during receptor preparation
_SKIP_WATER: set[str] = {"HOH", "WAT", "H2O", "DOD", "TIP", "SOL"}

# Common crystallographic additives — only skip when they cause preparation failures
_SKIP_ADDITIVES: set[str] = {
    "PJE",
    "02J",
    "010",
    "03U",
    "03T",
    "02K",
    "02L",
    "SO4",
    "PO4",
    "GOL",
    "EDO",
    "ACT",
    "PEG",
    "MES",
    "NAG",
    "MAN",
    "FUC",
    "GAL",
    "SIA",
    "NGA",
    "GLC",
}

# Physiologically relevant metal ions — must be retained for metal-dependent
# targets (metalloproteases, zinc fingers, etc.) when remove_hetatms=True
_METAL_IONS: set[str] = {
    "NA",
    "K",
    "CA",
    "MG",
    "ZN",
    "FE",
    "MN",
    "CO",
    "CU",
    "NI",
    "CD",
    "LI",
    "SR",
    "BA",
    "CS",
    "RB",
    "AL",
    "GA",
    "IN",
    "PB",
    "EU",
    "GD",
    "YB",
    "HO",
    "ER",
    "SM",
    "TB",
    "DY",
    "PR",
    "ND",
    "CE",
    "AG",
    "AU",
    "PT",
    "PD",
    "RH",
    "RU",
    "OS",
    "IR",
}

# Common physiologically relevant cofactors — should be kept for functional context
_METAL_COFACTORS: set[str] = {
    "HEM",
    "FAD",
    "NAD",
    "NAP",
    "NDP",
    "FMN",
    "COA",
    "ATP",
    "ADP",
    "GTP",
    "GDP",
    "SAM",
    "SAH",
    "PLP",
    "THF",
    "MGD",
}

# Combined skip set for backward compat
_SKIP_RESIDUES: set[str] = _SKIP_WATER | _SKIP_ADDITIVES

# Pocket sanity bounds (Angstroms)
_POCKET_MIN_DIM = 5.0
_POCKET_MAX_DIM = 40.0
_POCKET_MAX_VOLUME = 2000.0
_POCKET_MIN_DEPTH = 3.0
_P2RANK_PROB_THRESHOLD = 0.15
_DRUGGABILITY_THRESHOLD = 0.15

# Vina / publication defaults
VINA_DEFAULT_EXHAUSTIVENESS = 32
VINA_DEFAULT_N_POSES = 20
VINA_DEFAULT_ENERGY_RANGE = 3.0
VINA_DEFAULT_TIMEOUT = 600

# RMSD thresholds
REDocking_RMSD_THRESHOLD = 2.0  # Å
CLASH_THRESHOLD_EXPLICIT_H = 1.2  # Å
CLASH_THRESHOLD_HEAVY = 0.5  # Å

# Rendering defaults
DEFAULT_DPI = 300
DEFAULT_RAY_WIDTH = 2400
DEFAULT_RAY_HEIGHT = 1800
