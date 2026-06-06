"""
autodock.workflow — High-level docking analysis workflow.
=========================================================
Orchestrates the complete end-to-end molecular docking pipeline:

  1. Receptor acquisition (PDB / AlphaFold / local file)
  2. Receptor preparation (PDBFixer + reduce + optional PDB2PQR)
  3. Binding-pocket detection (P2Rank + fpocket cross-validation)
  4. Ligand preparation (SMILES → RDKit ETKDG → Meeko PDBQT)
  5. Multi-conformer docking (Vina, memory-aware parallelism)
  6. Post-processing (PLIP interactions + PoseBusters + clash)
  7. Publication-ready output (PDF report + 3D/2D figures + PS)

All parameters are publication-grade defaults (exhaustiveness=32,
n_poses=20, seed=42).  Every run records its full parameter set
and software versions for reproducibility.

References:
  - Eberhardt et al. (2021) JCIM — Vina 1.2 exhaustive search
  - Krivák & Hoksza (2018) Bioinformatics — P2Rank pocket detection
  - Laskowski & Swindells (2011) JCIM — LigPlot+ interaction diagrams
  - Riniker & Landrum (2015) JCIM — ETKDG conformer generation
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from autodock.core import (
    DockingCalculationError,
    DockingResult,
    get_environment_status,
    logger,
    set_log_level,
)
from autodock.utils import ensure_dir

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / resume helpers
# ─────────────────────────────────────────────────────────────────────────────

_STATE_FILE = "workflow_state.json"


def _load_state(out_dir: str) -> dict:
    """Load checkpoint state if it exists."""
    path = os.path.join(out_dir, _STATE_FILE)
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _save_state(out_dir: str, state: dict) -> None:
    """Write checkpoint state atomically."""
    path = os.path.join(out_dir, _STATE_FILE)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def _step_done(state: dict, step: str, required_files: list[str] | None = None) -> bool:
    """Return True if *step* is recorded complete and required files exist."""
    if not state.get(step):
        return False
    if required_files:
        return all(os.path.isfile(f) for f in required_files)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Config integration helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_params_from_config(
    config_path: str | None,
    **kwargs,
) -> dict:
    """
    Merge explicit keyword arguments with an optional YAML config file.

    Explicit arguments always win over config-file values.
    Returns a dict of parameters ready to pass to ``run_docking_workflow``.
    """
    if config_path is None:
        return kwargs

    from autodock.config import load_config

    cfg = load_config(config_path)

    def _get(*keys, default=None):
        d = cfg
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return default
        return d if d is not None else default

    # Map config schema → workflow parameter names
    param_map = {
        "output_dir": _get("project", "output_dir", default=kwargs.get("output_dir")),
        "log_level": _get("project", "log_level", default=kwargs.get("log_level")),
        "receptor_source": _get("receptor", "source", default=kwargs.get("receptor_source")),
        "receptor_format": kwargs.get("receptor_format", "auto"),
        "receptor_chain": _get("receptor", "chain", default=kwargs.get("receptor_chain")),
        "ph": _get("receptor", "ph", default=kwargs.get("ph")),
        "fix_protonation": _get("receptor", "minimize", default=kwargs.get("fix_protonation")),
        "max_pockets": _get("pocket", "top_n", default=kwargs.get("max_pockets")),
        "pocket_padding": _get("pocket", "padding", default=kwargs.get("pocket_padding")),
        "exhaustiveness": _get("docking", "exhaustiveness", default=kwargs.get("exhaustiveness")),
        "n_poses": _get("docking", "num_modes", default=kwargs.get("n_poses")),
        "seed": _get("docking", "seed", default=kwargs.get("seed")),
        # ligand_source mapping is context-dependent; keep explicit if provided
        "ligand_name": kwargs.get("ligand_name"),
    }

    # Only override if the user did NOT provide an explicit value
    merged = dict(kwargs)
    for key, cfg_val in param_map.items():
        if (key not in merged or merged[key] is None) and cfg_val is not None:
            merged[key] = cfg_val

    # Handle receptor_id from config when not explicitly passed
    if not merged.get("receptor_id"):
        pdb_id = _get("receptor", "pdb_id")
        uniprot = _get("receptor", "uniprot_id")
        rec_file = _get("receptor", "file")
        if pdb_id:
            merged["receptor_id"] = pdb_id
        elif uniprot:
            merged["receptor_id"] = uniprot
        elif rec_file:
            merged["receptor_id"] = rec_file

    # Handle ligand source from config
    lig_src_cfg = _get("ligands", "source", default="smiles")
    if merged.get("ligand_source") is None:
        merged["ligand_source"] = lig_src_cfg

    # Handle ligand_smiles from config
    if not merged.get("ligand_smiles"):
        smiles_cfg = _get("ligands", "smiles")
        cid_cfg = _get("ligands", "compound_id")
        file_cfg = _get("ligands", "file")
        if smiles_cfg:
            merged["ligand_smiles"] = smiles_cfg
        elif cid_cfg:
            merged["ligand_smiles"] = cid_cfg
        elif file_cfg:
            merged["ligand_smiles"] = file_cfg

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Workflow result container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DockingWorkflowResult:
    """
    Structured result from a complete docking workflow run.

    Serialisable to JSON for downstream analysis and provenance tracking.
    """

    # Identity
    receptor_name: str
    ligand_name: str
    receptor_source: str  # "PDB", "AlphaFold", "file"

    # File paths
    receptor_pdbqt: str | None = None
    receptor_pdb: str | None = None
    receptor_pdb_holo: str | None = None  # original PDB with waters for PLIP water-bridge detection
    ligand_pdbqt: str | None = None
    output_dir: str | None = None

    # Pocket info
    pockets: list[dict] = field(default_factory=list)
    best_pocket_idx: int | None = None

    # Docking results per pocket
    pocket_results: list[DockingResult] = field(default_factory=list)
    best_result: DockingResult | None = None

    # Post-processing outputs
    report_pdf: str | None = None
    report_csv: str | None = None
    summary_json: str | None = None
    figures_3d: list[str] = field(default_factory=list)
    figures_2d: list[str] = field(default_factory=list)
    pymol_sessions: list[str] = field(default_factory=list)

    # Provenance
    parameters: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = field(default_factory=lambda: __import__("autodock").__version__)

    # Error tracking
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary (for JSON export)."""
        d = dict(self.__dict__)
        d["pocket_results"] = [
            r.to_dict() if hasattr(r, "to_dict") else str(r) for r in d["pocket_results"]
        ]
        d["best_result"] = (
            d["best_result"].to_dict() if hasattr(d["best_result"], "to_dict") else None
        )
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Main workflow function
# ─────────────────────────────────────────────────────────────────────────────


def run_docking_workflow(
    # ── Receptor ──────────────────────────────────────────────────────────
    receptor_id: str | None = None,
    receptor_source: str = "auto",
    receptor_format: str = "auto",
    receptor_chain: str | None = None,
    # ── Ligand ────────────────────────────────────────────────────────────
    ligand_smiles: str | None = None,
    ligand_source: str = "smiles",
    ligand_name: str | None = None,
    # ── Docking parameters ────────────────────────────────────────────────
    exhaustiveness: int = 32,
    n_poses: int = 20,
    seed: int = 42,
    multi_conformer: bool = False,
    n_conformers: int = 10,
    # ── Advanced docking ──────────────────────────────────────────────────
    timeout: int = 600,
    energy_range: float = 3.0,
    scoring_function: str = "vina",
    # ── Pocket detection ─────────────────────────────────────────────────
    max_pockets: int = 5,
    pocket_padding: float = 5.0,
    # ── Receptor preparation ─────────────────────────────────────────────
    ph: float = 7.4,
    fix_protonation: bool = False,
    detect_af: bool = True,
    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "./docking_results",
    report_name: str | None = None,
    log_level: str = "INFO",
    # ── Figure rendering ─────────────────────────────────────────────────
    do_3d_figures: bool = True,
    do_2d_figures: bool = True,
    do_report: bool = True,
    # ── Advanced ──────────────────────────────────────────────────────────
    config_path: str | None = None,
    resume: bool = True,
    run_posebusters: bool = True,
    interaction_method: str = "plip",
    max_postprocess_pockets: int = 2,
    minimize_pose: bool = False,
) -> DockingWorkflowResult:
    """
    Run an end-to-end publication-grade molecular docking analysis.

    This is the recommended API for all production docking studies.
    It orchestrates receptor acquisition, preparation, pocket detection,
    ligand preparation, multi-conformer docking, post-processing, and
    report generation in a single call with robust error handling.

    Args:
        receptor_id: PDB ID (e.g. ``"6LU7"``), UniProt ID (e.g. ``"Q9H825-2"``),
            or path to a local structure file (``.pdb`` / ``.cif`` / ``.pdbqt``).
        receptor_source: Source type:
            ``"auto"`` (default) — auto-detect: file path → local, 4-char → PDB,
            otherwise → AlphaFold.
            ``"pdb"`` — download from RCSB PDB.
            ``"alphafold"`` — download from AlphaFold DB (UniProt ID).
            ``"file"`` — use local file (``receptor_id`` is the path).
        receptor_format: Format for PDB downloads: ``"pdb"`` or ``"cif"`` (default ``"auto"``).
        receptor_chain: Chain ID to extract (default: keep all chains).
        ligand_smiles: SMILES string of the ligand (e.g. ``"CC1=C(C(=O)...)..."``).
            Required when ``ligand_source="smiles"``.
        ligand_source: ``"smiles"`` (default), ``"pubchem"`` (CID), or ``"file"`` (SDF path).
        ligand_name: Name for the ligand in output files.  Derived from SMILES
            or CID if not provided.
        exhaustiveness: Vina search thoroughness (default 32, publication grade).
        n_poses: Number of poses to generate per conformer (default 20).
        seed: Random seed for reproducibility (default 42).
        multi_conformer: If True, generate N conformers via ETKDG and dock each
            independently.  **Not recommended for most Vina docking** — Vina
            already searches torsion space internally.  Only useful for
            macrocycles or rigid ring systems with distinct conformers.
        n_conformers: Number of conformers for multi-conformer docking (default 10).
        timeout: Wall-clock timeout per pocket in seconds (default 600).
        energy_range: Energy range above best pose in kcal/mol (default 3.0).
        scoring_function: Scoring function — ``"vina"`` (default) or
            ``"vinardo"``.
        max_pockets: Maximum pockets to detect and dock into (default 5).
        pocket_padding: Box padding around pocket dimensions (Å, default 5.0).
        ph: Target pH for protonation (default 7.4).
        fix_protonation: If True, run PDB2PQR+PROPKA for active protonation
            correction (default False).
        detect_af: If True, auto-detect AlphaFold structures and run pLDDT
            assessment (default True).
        output_dir: Root output directory (default ``"./docking_results"``).
        report_name: Base name for report files (default: ``{receptor}_{ligand}``).
        log_level: Logging level (``"INFO"``, ``"DEBUG"``, ``"WARNING"``).
        do_3d_figures: Render PyMOL 3D figures (complex/pocket/interaction).
        do_2d_figures: Render RDKit Cairo 2D interaction diagram.
        do_report: Generate PDF + CSV reports.
        config_path: Optional path to a YAML config file.  Explicit keyword
            arguments always override config values.
        resume: If ``True`` (default), skip steps whose outputs already exist
            in the output directory from a previous run.
        run_posebusters: If ``True`` (default), run PoseBusters validation on
            the best docked pose when PoseBusters is installed.
        interaction_method: Interaction backend for post-processing —
            ``"plip"`` (default), ``"prolif"``, or ``"both"``.

    Returns:
        :class:`DockingWorkflowResult` with all output paths and metadata.

    Raises:
        ValueError: If required parameters are missing (e.g. ``ligand_smiles``).
        DockingCalculationError: If docking fails entirely.
    """
    # ── Config integration ──────────────────────────────────────────────────
    if config_path is not None:
        merged = _resolve_params_from_config(
            config_path,
            receptor_id=receptor_id,
            receptor_source=receptor_source,
            receptor_format=receptor_format,
            receptor_chain=receptor_chain,
            ligand_smiles=ligand_smiles,
            ligand_source=ligand_source,
            ligand_name=ligand_name,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            seed=seed,
            multi_conformer=multi_conformer,
            n_conformers=n_conformers,
            max_pockets=max_pockets,
            pocket_padding=pocket_padding,
            ph=ph,
            fix_protonation=fix_protonation,
            detect_af=detect_af,
            output_dir=output_dir,
            report_name=report_name,
            log_level=log_level,
            do_3d_figures=do_3d_figures,
            do_2d_figures=do_2d_figures,
            do_report=do_report,
            run_posebusters=run_posebusters,
        )
        # Unpack merged parameters back into local variables
        receptor_id = merged.get("receptor_id", receptor_id)
        receptor_source = merged.get("receptor_source", receptor_source)
        receptor_format = merged.get("receptor_format", receptor_format)
        receptor_chain = merged.get("receptor_chain", receptor_chain)
        ligand_smiles = merged.get("ligand_smiles", ligand_smiles)
        ligand_source = merged.get("ligand_source", ligand_source)
        ligand_name = merged.get("ligand_name", ligand_name)
        exhaustiveness = merged.get("exhaustiveness", exhaustiveness)
        n_poses = merged.get("n_poses", n_poses)
        seed = merged.get("seed", seed)
        multi_conformer = merged.get("multi_conformer", multi_conformer)
        n_conformers = merged.get("n_conformers", n_conformers)
        max_pockets = merged.get("max_pockets", max_pockets)
        pocket_padding = merged.get("pocket_padding", pocket_padding)
        ph = merged.get("ph", ph)
        fix_protonation = merged.get("fix_protonation", fix_protonation)
        detect_af = merged.get("detect_af", detect_af)
        output_dir = merged.get("output_dir", output_dir)
        report_name = merged.get("report_name", report_name)
        log_level = merged.get("log_level", log_level)
        do_3d_figures = merged.get("do_3d_figures", do_3d_figures)
        do_2d_figures = merged.get("do_2d_figures", do_2d_figures)
        do_report = merged.get("do_report", do_report)
        run_posebusters = merged.get("run_posebusters", run_posebusters)

    if receptor_id is None:
        raise ValueError(
            "receptor_id is required (or set receptor.pdb_id / receptor.uniprot_id in config)."
        )

    _t0 = time.perf_counter()
    set_log_level(log_level)

    # ── Resolve output directory & report name ────────────────────────────
    out_dir = os.path.abspath(output_dir)
    ensure_dir(out_dir)
    state = _load_state(out_dir) if resume else {}
    result = DockingWorkflowResult(
        receptor_name=receptor_id,
        ligand_name=ligand_name or "ligand",
        receptor_source=receptor_source,
        output_dir=out_dir,
        parameters={
            "exhaustiveness": exhaustiveness,
            "n_poses": n_poses,
            "seed": seed,
            "multi_conformer": multi_conformer,
            "n_conformers": n_conformers,
            "timeout": timeout,
            "energy_range": energy_range,
            "scoring_function": scoring_function,
            "max_pockets": max_pockets,
            "pocket_padding": pocket_padding,
            "ph": ph,
            "fix_protonation": fix_protonation,
            "config_path": config_path,
            "resume": resume,
            "run_posebusters": run_posebusters,
            "interaction_method": interaction_method,
            "max_postprocess_pockets": max_postprocess_pockets,
            "minimize_pose": minimize_pose,
        },
        environment=get_environment_status(),
    )

    _rep_name = report_name or f"{os.path.basename(receptor_id)}_{ligand_name or 'ligand'}"

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Receptor acquisition
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"Step 1/6: Receptor acquisition ({receptor_id}, source={receptor_source})")
    logger.info("=" * 60)

    receptor_file: str | None = state.get("receptor_file")
    if receptor_file and os.path.isfile(receptor_file):
        logger.info(f"  ⏩ Resumed — using existing receptor: {receptor_file}")
        result.receptor_source = state.get("receptor_source", result.receptor_source)
    else:
        receptor_file = None

    if receptor_file is None:
        if receptor_source == "file" or (receptor_source == "auto" and os.path.isfile(receptor_id)):
            receptor_file = receptor_id
            result.receptor_source = "file"
            logger.info(f"  Using local file: {receptor_file}")

        elif receptor_source in ("pdb", "auto") and len(receptor_id) == 4 and receptor_id.isalnum():
            # PDB ID (4 alphanumeric chars)
            from autodock.fetchers import download_pdb

            fmt = "cif" if receptor_format in ("auto", "cif") else "pdb"
            receptor_file = download_pdb(receptor_id, out_dir, format=fmt)
            result.receptor_source = "PDB"
            logger.info(f"  Downloaded from PDB: {receptor_file}")

        elif receptor_source in ("alphafold", "auto"):
            # AlphaFold / UniProt
            from autodock.fetchers import download_alphafold

            receptor_file = download_alphafold(receptor_id, out_dir, format="cif")
            result.receptor_source = "AlphaFold"
            logger.info(f"  Downloaded AlphaFold: {receptor_file}")

        else:
            raise ValueError(
                f"Cannot resolve receptor: {receptor_id} (source={receptor_source}). "
                "Try: PDB ID (4 chars), UniProt ID, or local file path."
            )

        state["receptor_file"] = receptor_file
        state["receptor_source"] = result.receptor_source
        state["step_1_complete"] = True
        _save_state(out_dir, state)

    result.receptor_name = os.path.basename(receptor_file)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Receptor preparation
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 2/6: Receptor preparation (PDBFixer + reduce + ...)")
    logger.info("=" * 60)

    receptor_pdbqt = os.path.join(out_dir, f"{_rep_name}_receptor.pdbqt")
    receptor_pdb_out = os.path.join(out_dir, f"{_rep_name}_receptor.pdb")

    if _step_done(state, "step_2_complete", [receptor_pdbqt, receptor_pdb_out]):
        logger.info("  ⏩ Resumed — receptor already prepared")
        result.receptor_pdbqt = receptor_pdbqt
        result.receptor_pdb = receptor_pdb_out
        result.receptor_pdb_holo = state.get("receptor_file")
    else:
        try:
            from autodock.preparation import prepare_receptor

            prepare_receptor(
                receptor_file,
                receptor_pdbqt,
                ph=ph,
                remove_water=True,
                remove_hetatms=True,
                retain_metal_ions=True,
                predict_pka=True,
                fix_protonation=fix_protonation,
                detect_af_structure=detect_af,
                force=True,
                output_pdb=receptor_pdb_out,
                cache_dir=os.path.expanduser("~/.autodock/cache"),
            )
            result.receptor_pdbqt = receptor_pdbqt
            result.receptor_pdb = receptor_pdb_out
            result.receptor_pdb_holo = receptor_file
            logger.info(f"  Receptor PDBQT: {receptor_pdbqt}")
            state["step_2_complete"] = True
            _save_state(out_dir, state)
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            result.errors.append(f"Receptor preparation failed: {exc}")
            logger.error(f"  Receptor preparation failed: {exc}")
            raise

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Pocket detection
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 3/6: Pocket detection (P2Rank + fpocket)")
    logger.info("=" * 60)

    pockets = state.get("pockets")
    if pockets and _step_done(state, "step_3_complete"):
        logger.info(f"  ⏩ Resumed — using {len(pockets)} cached pocket(s)")
        result.pockets = pockets
    else:
        try:
            from autodock.preparation import find_top_pockets

            pockets = find_top_pockets(
                receptor_file,
                max_pockets=max_pockets,
                padding=pocket_padding,
                cache_dir=os.path.expanduser("~/.autodock/cache"),
            )
            result.pockets = pockets
            logger.info(f"  Found {len(pockets)} pocket(s)")
            state["pockets"] = pockets
            state["step_3_complete"] = True
            _save_state(out_dir, state)
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            result.errors.append(f"Pocket detection failed: {exc}")
            logger.error(f"  Pocket detection failed: {exc}")
            pockets = []

    if not pockets:
        raise DockingCalculationError("No binding pockets detected. Cannot proceed with docking.")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Ligand preparation
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 4/6: Ligand preparation")
    logger.info("=" * 60)

    ligand_pdbqt = os.path.join(out_dir, f"{_rep_name}_ligand.pdbqt")

    if _step_done(state, "step_4_complete", [ligand_pdbqt]):
        logger.info("  ⏩ Resumed — ligand already prepared")
        result.ligand_pdbqt = ligand_pdbqt
    else:
        if ligand_source == "smiles" and ligand_smiles:
            from autodock.preparation import prepare_ligand

            ligand_name_resolved = ligand_name or "LIG"
            prepare_ligand(
                ligand_smiles,
                ligand_pdbqt,
                ph=ph,
                name=ligand_name_resolved[:3],
                molscrub_states=True,
                enumerate_stereo=True,
                cache_dir=os.path.expanduser("~/.autodock/cache"),
            )
            logger.info(f"  Ligand from SMILES: {ligand_pdbqt}")
        elif ligand_source == "pubchem":
            from autodock.fetchers import fetch_pubchem_smiles

            if not ligand_smiles:
                raise ValueError(
                    "ligand_smiles must be provided as a PubChem CID "
                    "when ligand_source='pubchem'"
                )
            cid = ligand_smiles
            smiles = fetch_pubchem_smiles(cid)
            from autodock.preparation import prepare_ligand

            prepare_ligand(
                smiles, ligand_pdbqt, ph=ph, cache_dir=os.path.expanduser("~/.autodock/cache")
            )
            logger.info(f"  Ligand from PubChem CID {cid}: {ligand_pdbqt}")
        elif ligand_source == "file" and os.path.isfile(ligand_smiles or ""):
            from autodock.preparation import prepare_ligand_from_file

            prepare_ligand_from_file(ligand_smiles, ligand_pdbqt)
            logger.info(f"  Ligand from file: {ligand_pdbqt}")
        else:
            raise ValueError(
                f"Cannot prepare ligand: source={ligand_source}, "
                f"smiles={'provided' if ligand_smiles else 'missing'}"
            )

        result.ligand_pdbqt = ligand_pdbqt
        state["step_4_complete"] = True
        _save_state(out_dir, state)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: Docking (multi-pocket)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"Step 5/6: Docking ({len(pockets)} pocket(s))")
    logger.info("=" * 60)

    pocket_results: list[DockingResult] = []
    if _step_done(state, "step_5_complete"):
        logger.info("  ⏩ Resumed — loading cached docking results")
        for i, pocket in enumerate(pockets):
            pocket_dir = os.path.join(out_dir, f"pocket_{i + 1}")
            result_json = os.path.join(pocket_dir, "docking_result.json")
            if os.path.isfile(result_json):
                try:
                    with open(result_json) as fh:
                        data = json.load(fh)
                    r = DockingResult(**data)
                    pocket_results.append(r)
                    if r.best_affinity is not None:
                        logger.info(
                            f"    Pocket #{i + 1} (resumed): {r.best_affinity:.2f} kcal/mol"
                        )
                    continue
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            # Fallback: re-dock this pocket
            logger.info(f"    Pocket #{i + 1}: re-docking (cache missing)")
            _dock_single_pocket(
                receptor_pdbqt,
                ligand_pdbqt,
                pocket,
                pocket_dir,
                _rep_name,
                i,
                exhaustiveness,
                n_poses,
                seed,
                multi_conformer,
                n_conformers,
                timeout,
                energy_range,
                scoring_function,
                ligand_smiles,
                result,
                pocket_results,
            )
    else:
        for i, pocket in enumerate(pockets):
            pocket_dir = os.path.join(out_dir, f"pocket_{i + 1}")
            _dock_single_pocket(
                receptor_pdbqt,
                ligand_pdbqt,
                pocket,
                pocket_dir,
                _rep_name,
                i,
                exhaustiveness,
                n_poses,
                seed,
                multi_conformer,
                n_conformers,
                timeout,
                energy_range,
                scoring_function,
                ligand_smiles,
                result,
                pocket_results,
            )
        state["step_5_complete"] = True
        _save_state(out_dir, state)

    result.pocket_results = pocket_results

    # Rank pockets by affinity
    ranked = sorted(
        [(r, i) for i, r in enumerate(pocket_results) if r.best_affinity is not None],
        key=lambda x: x[0].best_affinity,
    )
    if ranked:
        result.best_result = ranked[0][0]
        result.best_pocket_idx = ranked[0][1]
        logger.info(
            f"  Best pocket: #{result.best_pocket_idx + 1} "
            f"({result.best_result.best_affinity:.2f} kcal/mol)"
        )
    else:
        result.warnings.append("All pockets failed to dock")
        logger.warning("  No successful docking results")

    # ── Ligand-efficiency metrics (best pocket) ───────────────────────────
    if ligand_smiles and result.best_result and result.best_result.best_affinity is not None:
        lig_metrics = _compute_ligand_metrics(ligand_smiles)
        if lig_metrics:
            from autodock.analysis import compute_ligand_efficiency

            le_dict = compute_ligand_efficiency(
                affinity=result.best_result.best_affinity,
                n_heavy_atoms=lig_metrics["n_heavy_atoms"],
                n_rotatable_bonds=lig_metrics["n_rotatable_bonds"],
                molecular_weight=lig_metrics["molecular_weight"],
            )
            result.best_result.all_scores.update(
                {
                    f"le_{k}": v
                    for k, v in le_dict.items()
                    if v is not None and not k.startswith("n_")
                }
            )
            logger.info(
                f"  Ligand efficiency: LE={le_dict.get('le'):.3f}, "
                f"LE_RB={le_dict.get('le_rb'):.3f}, MW={lig_metrics['molecular_weight']:.1f}"
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5b: PoseBusters validation (best pose)
    # ═══════════════════════════════════════════════════════════════════════
    if run_posebusters and result.best_result and result.best_result.best_pose_pdbqt:
        logger.info("=" * 60)
        logger.info("Step 5b: PoseBusters validation")
        logger.info("=" * 60)
        try:
            from autodock.validation import validate_pose_with_posebusters

            pb = validate_pose_with_posebusters(
                result.best_result.best_pose_pdbqt,
                result.receptor_pdb or receptor_pdb_out,
            )
            result.best_result.posebusters_pass = pb.get("pass")
            status = "PASS" if pb.get("pass") else "FAIL"
            avail = "available" if pb.get("available") else "unavailable"
            logger.info(f"  PoseBusters {avail}: {status}")
        except (RuntimeError, OSError, ValueError, ImportError) as exc:
            result.warnings.append(f"PoseBusters validation skipped: {exc}")
            logger.warning(f"  PoseBusters validation skipped: {exc}")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5c: Clash scoring (all successful pockets)
    # ═══════════════════════════════════════════════════════════════════════
    _receptor_for_clash = result.receptor_pdb or receptor_pdb_out
    if _receptor_for_clash:
        logger.info("=" * 60)
        logger.info("Step 5c: Clash scoring")
        logger.info("=" * 60)
        for i, pr in enumerate(result.pocket_results):
            if pr.best_pose_pdbqt and os.path.isfile(pr.best_pose_pdbqt):
                try:
                    from autodock.validation import compute_clash_score

                    clash = compute_clash_score(pr.best_pose_pdbqt, _receptor_for_clash)
                    pr.clash_score = clash.get("clash_score")
                    pr.clash_acceptable = clash.get("is_acceptable")
                    status = "PASS" if pr.clash_acceptable else "WARN"
                    logger.info(
                        f"  Pocket #{i + 1} clash: {pr.clash_score} Å "
                        f"({clash.get('n_clashes', 0)} clashes) — {status}"
                    )
                except (RuntimeError, OSError, ValueError, ImportError) as exc:
                    result.warnings.append(f"Clash scoring pocket #{i + 1} skipped: {exc}")
                    logger.warning(f"  Pocket #{i + 1} clash scoring skipped: {exc}")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5d: Optional energy minimisation (best pose only)
    # ═══════════════════════════════════════════════════════════════════════
    if minimize_pose and result.best_result and result.best_result.best_pose_pdbqt:
        logger.info("=" * 60)
        logger.info("Step 5d: Energy minimisation (best pose)")
        logger.info("=" * 60)
        try:
            from autodock.minimization import minimize_docked_pose

            min_out = os.path.join(out_dir, "best_pose_minimized.pdb")
            min_result = minimize_docked_pose(
                receptor_pdb=result.receptor_pdb or receptor_pdb_out,
                ligand_pdbqt=result.best_result.best_pose_pdbqt,
                output_pdb=min_out,
                ligand_smiles=ligand_smiles,
                max_iterations=500,
            )
            if min_result.get("success"):
                result.best_result.best_pose_pdbqt = min_out
                logger.info(
                    f"  Minimised: {min_result['initial_energy_kJ_mol']:.1f} → "
                    f"{min_result['final_energy_kJ_mol']:.1f} kJ/mol"
                )
            else:
                result.warnings.append(
                    f"Pose minimisation skipped: {min_result.get('error', 'unknown')}"
                )
                logger.warning(f"  Minimisation skipped: {min_result.get('error', 'unknown')}")
        except (RuntimeError, OSError, ValueError, ImportError) as exc:
            result.warnings.append(f"Pose minimisation failed: {exc}")
            logger.warning(f"  Minimisation failed: {exc}")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 6: Post-processing & report (top-N pockets)
    # ═══════════════════════════════════════════════════════════════════════
    # Rank successful pockets by affinity and post-process the top N.
    ranked_pockets = sorted(
        [
            (i, pr)
            for i, pr in enumerate(result.pocket_results)
            if pr.best_affinity is not None and pr.best_pose_pdbqt
        ],
        key=lambda x: x[1].best_affinity,
    )
    pockets_to_process = ranked_pockets[:max_postprocess_pockets]

    if pockets_to_process and result.receptor_pdb:
        logger.info("=" * 60)
        logger.info("Step 6/6: Post-processing & report generation")
        logger.info("=" * 60)

        for rank, (pidx, pr) in enumerate(pockets_to_process):
            pair_root = (
                os.path.join(out_dir, "best_result")
                if rank == 0
                else os.path.join(out_dir, f"pocket_{pidx + 1}_analysis")
            )
            summary_marker = os.path.join(pair_root, "summary.txt")

            if resume and _step_done(state, f"step_6_pocket_{pidx}_complete", [summary_marker]):
                logger.info(f"  ⏩ Pocket #{pidx + 1} post-processing already complete")
                if rank == 0:
                    _hydrate_postprocess_results(result, pair_root)
                continue

            logger.info(f"  Post-processing pocket #{pidx + 1} (affinity: {pr.best_affinity:.2f})")
            try:
                from autodock.post_dock_pipeline import post_process_docking

                pp_out = post_process_docking(
                    pr,
                    pair_root,
                    receptor_pdb=result.receptor_pdb,
                    receptor_pdb_holo=result.receptor_pdb_holo,
                    do_interactions=True,
                    do_rendering=do_3d_figures,
                    do_report=do_report,
                    interaction_method=interaction_method,
                )
                if rank == 0:
                    # Primary results from the best pocket
                    result.report_pdf = pp_out.get("pdf")
                    result.report_csv = pp_out.get("csv")
                    result.figures_3d = pp_out.get("figures", [])
                    fig_2d_png = pp_out.get("fig_2d_png")
                    fig_2d_pdf = pp_out.get("fig_2d_pdf")
                    result.figures_2d = [p for p in (fig_2d_png, fig_2d_pdf) if p]
                    result.pymol_sessions = (
                        [
                            os.path.join(pair_root, "03_figures", f)
                            for f in os.listdir(os.path.join(pair_root, "03_figures"))
                            if f.endswith(".pse")
                        ]
                        if os.path.isdir(os.path.join(pair_root, "03_figures"))
                        else []
                    )

                state[f"step_6_pocket_{pidx}_complete"] = True
                _save_state(out_dir, state)

            except (RuntimeError, OSError, ValueError, ImportError, KeyError) as exc:
                result.warnings.append(f"Post-processing pocket #{pidx + 1} failed: {exc}")
                logger.warning(f"  Pocket #{pidx + 1} post-processing failed: {exc}")

    # ── Write summary JSON ───────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "workflow_summary.json")
    result.runtime_seconds = round(time.perf_counter() - _t0, 2)
    try:
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        result.summary_json = summary_path
        logger.info(f"  Summary: {summary_path}")
    except (OSError, TypeError) as exc:
        result.warnings.append(f"Summary JSON write failed: {exc}")

    logger.info("=" * 60)
    logger.info(f"✅ Workflow complete ({result.runtime_seconds:.1f}s)")
    if result.errors:
        logger.warning(f"   {len(result.errors)} error(s), {len(result.warnings)} warning(s)")
    logger.info("=" * 60)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _compute_ligand_metrics(smiles: str) -> dict[str, Any] | None:
    """Compute heavy-atom count, MW, and rotatable bonds from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
            "n_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "molecular_weight": Descriptors.MolWt(mol),
        }
    except Exception:
        return None


def _hydrate_postprocess_results(result: DockingWorkflowResult, pair_root: str) -> None:
    """Re-hydrate figure/report paths from a completed pair_root directory."""
    if not os.path.isdir(pair_root):
        return
    fig_dir = os.path.join(pair_root, "03_figures")
    if os.path.isdir(fig_dir):
        result.figures_3d = [
            os.path.join(fig_dir, f)
            for f in os.listdir(fig_dir)
            if f.endswith(".png") and f.startswith("3d_")
        ]
        result.pymol_sessions = [
            os.path.join(fig_dir, f) for f in os.listdir(fig_dir) if f.endswith(".pse")
        ]
        result.figures_2d = [
            os.path.join(fig_dir, f) for f in os.listdir(fig_dir) if f.startswith("2d_interactions")
        ]
    rep_dir = os.path.join(pair_root, "04_reports")
    if os.path.isdir(rep_dir):
        for f in os.listdir(rep_dir):
            if f.endswith(".pdf"):
                result.report_pdf = os.path.join(rep_dir, f)
            elif f.endswith(".csv"):
                result.report_csv = os.path.join(rep_dir, f)


def _dock_single_pocket(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    pocket: dict,
    pocket_dir: str,
    rep_name: str,
    pocket_idx: int,
    exhaustiveness: int,
    n_poses: int,
    seed: int,
    multi_conformer: bool,
    n_conformers: int,
    timeout: int,
    energy_range: float,
    scoring_function: str,
    ligand_smiles: str | None,
    workflow_result: DockingWorkflowResult,
    pocket_results: list[DockingResult],
) -> None:
    """Dock a single pocket and append the result to *pocket_results*."""
    from autodock.utils import ensure_dir

    ensure_dir(pocket_dir)
    logger.info(f"  Pocket #{pocket_idx + 1}: center={pocket['center']}, box={pocket['box_size']}")

    try:
        from autodock.docking import dock_ligand

        r = dock_ligand(
            receptor_pdbqt,
            ligand_pdbqt,
            center=pocket["center"],
            box_size=pocket["box_size"],
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            seed=seed,
            output_dir=pocket_dir,
            compound_name=f"{rep_name}_pocket{pocket_idx + 1}",
            min_rmsd=1.0,
            multi_conformer=multi_conformer,
            n_conformers=n_conformers,
            timeout=timeout,
            energy_range=energy_range,
            scoring_function=scoring_function,
            ligand_smiles=ligand_smiles if multi_conformer else None,
        )
        r.binding_pocket = pocket
        r.pocket_source = pocket.get("pocket_source", "fpocket")
        pocket_results.append(r)
        logger.info(f"    Affinity: {r.best_affinity:.2f} kcal/mol")
        # Cache result JSON for resume
        try:
            import json

            result_json = os.path.join(pocket_dir, "docking_result.json")
            with open(result_json, "w") as fh:
                json.dump(r.to_dict(), fh, indent=2, default=str)
        except (OSError, TypeError):
            pass
    except (DockingCalculationError, RuntimeError, OSError, ValueError) as exc:
        workflow_result.errors.append(f"Pocket #{pocket_idx + 1} docking failed: {exc}")
        logger.warning(f"    Pocket #{pocket_idx + 1} docking failed: {exc}")
        pocket_results.append(
            DockingResult(
                compound_name=f"{rep_name}_pocket{pocket_idx + 1}",
                receptor=receptor_pdbqt,
                center=pocket["center"],
                box_size=pocket["box_size"],
                best_affinity=None,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Command-line entry point for the docking workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="autodock — Publication-grade molecular docking workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\\n"
            "  # Single docking from PDB ID + SMILES\\n"
            "  python -m autodock.workflow --receptor 6LU7 \\n"
            "      --ligand-smiles 'CC(C)Cc1ccc(C(C)C(=O)O)cc1' \\n"
            "      --outdir ./my_docking\\n\\n"
            "  # AlphaFold receptor + multi-conformer docking\\n"
            "  python -m autodock.workflow --receptor Q9H825-2 \\n"
            "      --ligand 'Idebenone' --multi-conformer \\n"
            "      --outdir ./results\\n"
        ),
    )
    parser.add_argument("--receptor", required=True, help="PDB ID, UniProt ID, or file path")
    parser.add_argument("--ligand-smiles", help="Ligand SMILES string")
    parser.add_argument("--ligand-name", help="Ligand name (for output files)")
    parser.add_argument(
        "--receptor-source",
        default="auto",
        choices=["auto", "pdb", "alphafold", "file"],
        help="Receptor source type",
    )
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--n-poses", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-conformer", action="store_true", help="Multi-conformer docking")
    parser.add_argument("--n-conformers", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per pocket in seconds")
    parser.add_argument(
        "--energy-range", type=float, default=3.0, help="Energy range above best in kcal/mol"
    )
    parser.add_argument(
        "--scoring-function",
        choices=["vina", "vinardo"],
        default="vina",
        help="Scoring function (default: vina)",
    )
    parser.add_argument("--max-pockets", type=int, default=5)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--fix-protonation", action="store_true")
    parser.add_argument("--outdir", default="./docking_results", help="Output directory")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D figures")
    parser.add_argument("--no-2d", action="store_true", help="Skip 2D figures")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--no-resume", action="store_true", help="Re-run all steps from scratch")
    parser.add_argument("--no-posebusters", action="store_true", help="Skip PoseBusters validation")
    parser.add_argument(
        "--minimize-pose", action="store_true", help="Energy-minimize best pose with OpenMM"
    )
    parser.add_argument(
        "--method",
        choices=["plip", "prolif", "both"],
        default="plip",
        help="Interaction detection engine (default: plip)",
    )

    args = parser.parse_args()

    result = run_docking_workflow(
        receptor_id=args.receptor,
        ligand_smiles=args.ligand_smiles,
        ligand_name=args.ligand_name,
        receptor_source=args.receptor_source,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
        multi_conformer=args.multi_conformer,
        n_conformers=args.n_conformers,
        timeout=args.timeout,
        energy_range=args.energy_range,
        scoring_function=args.scoring_function,
        max_pockets=args.max_pockets,
        ph=args.ph,
        fix_protonation=args.fix_protonation,
        output_dir=args.outdir,
        log_level=args.log_level,
        do_3d_figures=not args.no_3d,
        do_2d_figures=not args.no_2d,
        do_report=not args.no_report,
        config_path=args.config,
        resume=not args.no_resume,
        run_posebusters=not args.no_posebusters,
        interaction_method=args.method,
        minimize_pose=args.minimize_pose,
    )

    # Print summary
    if result.best_result and result.best_result.best_affinity is not None:
        print(f"\\n✅ Best affinity: {result.best_result.best_affinity:.2f} kcal/mol")
        _pidx = result.best_pocket_idx
        _pocket_str = f"#{_pidx + 1}" if _pidx is not None else "?"
        print(f"   Pocket {_pocket_str}")
    else:
        print("\\n❌ Docking failed — see workflow_summary.json for details")
    if result.report_pdf:
        print(f"   Report: {result.report_pdf}")
    if result.summary_json:
        print(f"   Summary: {result.summary_json}")
    print(f"   Runtime: {result.runtime_seconds:.1f}s")


if __name__ == "__main__":
    main()
