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
    receptor_id: str,
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
            independently (recommended for ligands with >8 rotatable bonds).
        n_conformers: Number of conformers for multi-conformer docking (default 10).
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

    Returns:
        :class:`DockingWorkflowResult` with all output paths and metadata.

    Raises:
        ValueError: If required parameters are missing (e.g. ``ligand_smiles``).
        DockingCalculationError: If docking fails entirely.
    """
    _t0 = time.perf_counter()
    set_log_level(log_level)

    # ── Resolve output directory & report name ────────────────────────────
    out_dir = os.path.abspath(output_dir)
    ensure_dir(out_dir)
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
            "max_pockets": max_pockets,
            "pocket_padding": pocket_padding,
            "ph": ph,
            "fix_protonation": fix_protonation,
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

    receptor_file: str | None = None

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

    result.receptor_name = os.path.basename(receptor_file)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Receptor preparation
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 2/6: Receptor preparation (PDBFixer + reduce + ...)")
    logger.info("=" * 60)

    receptor_pdbqt = os.path.join(out_dir, f"{_rep_name}_receptor.pdbqt")
    receptor_pdb_out = os.path.join(out_dir, f"{_rep_name}_receptor.pdb")
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
        )
        result.receptor_pdbqt = receptor_pdbqt
        result.receptor_pdb = receptor_pdb_out
        logger.info(f"  Receptor PDBQT: {receptor_pdbqt}")
    except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
        result.errors.append(f"Receptor preparation failed: {exc}")
        logger.error(f"  Receptor preparation failed: {exc}")
        raise

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Pocket detection
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Step 3/6: Pocket detection (P2Rank + fpocket)")
    logger.info("=" * 60)

    try:
        from autodock.preparation import find_top_pockets

        pockets = find_top_pockets(
            receptor_file,
            max_pockets=max_pockets,
            padding=pocket_padding,
        )
        result.pockets = pockets
        logger.info(f"  Found {len(pockets)} pocket(s)")
    except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
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
        )
        logger.info(f"  Ligand from SMILES: {ligand_pdbqt}")
    elif ligand_source == "pubchem":
        from autodock.fetchers import fetch_pubchem_smiles

        cid = receptor_id if ligand_source == "pubchem" else ligand_smiles
        smiles = fetch_pubchem_smiles(cid)
        from autodock.preparation import prepare_ligand

        prepare_ligand(smiles, ligand_pdbqt, ph=ph)
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

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: Docking (multi-pocket)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"Step 5/6: Docking ({len(pockets)} pocket(s))")
    logger.info("=" * 60)

    pocket_results: list[DockingResult] = []
    for i, pocket in enumerate(pockets):
        pocket_dir = os.path.join(out_dir, f"pocket_{i + 1}")
        ensure_dir(pocket_dir)
        logger.info(f"  Pocket #{i + 1}: center={pocket['center']}, box={pocket['box_size']}")

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
                compound_name=f"{_rep_name}_pocket{i + 1}",
                skip_consensus=False,
                min_rmsd=1.0,
                multi_conformer=multi_conformer,
                ligand_smiles=ligand_smiles if multi_conformer else None,
            )
            pocket_results.append(r)
            logger.info(f"    Affinity: {r.best_affinity:.2f} kcal/mol")
        except (DockingCalculationError, RuntimeError, OSError, ValueError, TypeError) as exc:
            result.errors.append(f"Pocket #{i + 1} docking failed: {exc}")
            logger.warning(f"    Pocket #{i + 1} docking failed: {exc}")
            # Continue with other pockets
            pocket_results.append(
                DockingResult(
                    compound_name=f"{_rep_name}_pocket{i + 1}",
                    receptor=receptor_pdbqt,
                    center=pocket["center"],
                    box_size=pocket["box_size"],
                    best_affinity=None,
                )
            )

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

    # ═══════════════════════════════════════════════════════════════════════
    # Step 6: Post-processing & report
    # ═══════════════════════════════════════════════════════════════════════
    if result.best_result and result.receptor_pdb:
        logger.info("=" * 60)
        logger.info("Step 6/6: Post-processing & report generation")
        logger.info("=" * 60)

        try:
            from autodock.pipeline import post_process_docking

            pair_root = os.path.join(out_dir, "best_result")
            pp_out = post_process_docking(
                result.best_result,
                pair_root,
                receptor_pdb=result.receptor_pdb,
                do_interactions=True,
                do_rendering=do_3d_figures,
                do_report=do_report,
            )
            result.report_pdf = pp_out.get("pdf")
            result.report_csv = pp_out.get("csv")
            result.figures_3d = pp_out.get("figures", [])
            result.pymol_sessions = (
                [
                    os.path.join(pair_root, "03_figures", f)
                    for f in os.listdir(os.path.join(pair_root, "03_figures"))
                    if f.endswith(".pse")
                ]
                if os.path.isdir(os.path.join(pair_root, "03_figures"))
                else []
            )

            # 2D interaction diagram
            if do_2d_figures:
                try:
                    from autodock.rendering import render_interactions_2d

                    fig_dir = os.path.join(pair_root, "03_figures")
                    png_2d = os.path.join(fig_dir, "2d_interactions.png")
                    pdf_2d = os.path.join(fig_dir, "2d_interactions.pdf")
                    render_interactions_2d(
                        result.receptor_pdb,
                        result.best_result.best_pose_pdbqt,
                        result.best_result.interactions,
                        output_png=png_2d,
                        output_pdf=pdf_2d,
                    )
                    result.figures_2d = [png_2d, pdf_2d]
                except (RuntimeError, OSError, ValueError, TypeError, ImportError) as exc:
                    result.warnings.append(f"2D rendering skipped: {exc}")

        except (RuntimeError, OSError, ValueError, TypeError, ImportError, KeyError) as exc:
            result.errors.append(f"Post-processing failed: {exc}")
            logger.warning(f"  Post-processing failed: {exc}")

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
    parser.add_argument("--max-pockets", type=int, default=5)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--fix-protonation", action="store_true")
    parser.add_argument("--outdir", default="./docking_results", help="Output directory")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D figures")
    parser.add_argument("--no-2d", action="store_true", help="Skip 2D figures")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")

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
        max_pockets=args.max_pockets,
        ph=args.ph,
        fix_protonation=args.fix_protonation,
        output_dir=args.outdir,
        log_level=args.log_level,
        do_3d_figures=not args.no_3d,
        do_2d_figures=not args.no_2d,
        do_report=not args.no_report,
    )

    # Print summary
    if result.best_result and result.best_result.best_affinity is not None:
        print(f"\\n✅ Best affinity: {result.best_result.best_affinity:.2f} kcal/mol")
        print(
            f"   Pocket #{result.best_pocket_idx + 1 if result.best_pocket_idx is not None else '?'}"
        )
    else:
        print("\\n❌ Docking failed — see workflow_summary.json for details")
    if result.report_pdf:
        print(f"   Report: {result.report_pdf}")
    if result.summary_json:
        print(f"   Summary: {result.summary_json}")
    print(f"   Runtime: {result.runtime_seconds:.1f}s")


if __name__ == "__main__":
    main()
