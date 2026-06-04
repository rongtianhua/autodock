"""
autodock.post_dock_pipeline — Standardized per-pair output pipeline.
===========================================================
Takes a completed DockingResult and produces a structured output directory
with all analysis artifacts: structures, interactions, figures (PNG+PDF),
PyMOL sessions, and publication-ready reports.
"""

from __future__ import annotations

import json
import os
from typing import Any

from autodock.core import DockingResult, logger
from autodock.utils import ensure_dir

# ─────────────────────────────────────────────────────────────────────────────
# Standard output directory structure
# ─────────────────────────────────────────────────────────────────────────────
# pair_root/
#   01_structures/        original + intermediate structural files
#   02_interactions/      interaction data tables
#   03_figures/           PNG + PDF (300+ dpi) + .pse PyMOL session
#   04_reports/           PDF report, CSV data, JSON metadata
#   summary.txt           one-line overview


def build_pair_dir(pair_root: str) -> dict[str, str]:
    """Create the standardized directory tree for a single receptor-ligand pair.

    Returns:
        Dict mapping section name → path, e.g.
        ``{"structures": "...", "interactions": "...", "figures": "...", "reports": "..."}``
    """
    dirs = {
        "structures": os.path.join(pair_root, "01_structures"),
        "interactions": os.path.join(pair_root, "02_interactions"),
        "figures": os.path.join(pair_root, "03_figures"),
        "reports": os.path.join(pair_root, "04_reports"),
    }
    for d in dirs.values():
        ensure_dir(d)
    return dirs


def post_process_docking(
    result: DockingResult,
    pair_root: str,
    receptor_pdb: str | None = None,
    receptor_pdb_holo: str | None = None,
    do_interactions: bool = True,
    do_rendering: bool = True,
    do_report: bool = True,
    copy_structures: bool = True,
    interaction_method: str = "plip",
) -> dict[str, Any]:
    """Run full post-processing on a DockingResult and write all outputs
    into the standardized pair directory.

    .. note::
        Not idempotent — each call re-generates all outputs.  Callers
        should guard against duplicate calls at the call site.

    Args:
        result: Completed DockingResult from dock_ligand().
        pair_root: Root directory for this pair (will be created).
        receptor_pdb: Path to prepared (apo) receptor PDB for rendering.
        receptor_pdb_holo: Path to original (holo) receptor PDB with waters
            for PLIP water-bridge detection. Falls back to ``receptor_pdb``
            if not provided.
        do_interactions: Run interaction detection.
        do_rendering: Render 2D/3D figures.
        do_report: Generate PDF + CSV reports.
        copy_structures: Copy original/intermediate structure files.
        interaction_method: Interaction backend — ``"plip"`` (default),
            ``"prolif"``, or ``"both"``.

    Returns:
        Dict with all output paths produced.
    """
    # NOTE: idempotency is NOT implemented here because tests share output
    # directories and any skip-if-exists logic causes cross-test interference.
    # Callers that want to avoid re-processing should check for their own
    # completeness markers (e.g. 04_reports/result.json) before calling.
    dirs = build_pair_dir(pair_root)
    outputs: dict[str, Any] = {
        "pair_root": pair_root,
        "dirs": dirs,
        "compound_name": result.compound_name,
        "best_affinity": result.best_affinity,
    }

    compound = result.compound_name or "ligand"
    rec_basename = os.path.basename(result.receptor) if result.receptor else ""
    receptor_name = os.path.splitext(rec_basename)[0] if rec_basename else "receptor"

    # ── 1. Copy structure files ──────────────────────────────────────────────
    if copy_structures:
        struct_dir = dirs["structures"]
        _copy_file(result.receptor, os.path.join(struct_dir, "receptor.pdbqt"))
        if result.best_pose_pdbqt and os.path.isfile(result.best_pose_pdbqt):
            _copy_file(result.best_pose_pdbqt, os.path.join(struct_dir, "docking_best.pdbqt"))
        if result.all_poses_pdbqt and os.path.isfile(result.all_poses_pdbqt):
            _copy_file(result.all_poses_pdbqt, os.path.join(struct_dir, "docking_all_poses.pdbqt"))
        if receptor_pdb and os.path.isfile(receptor_pdb):
            _copy_file(receptor_pdb, os.path.join(struct_dir, "receptor.pdb"))
        # Copy cluster representatives
        if result.pose_clusters:
            for i, c in enumerate(result.pose_clusters[:5], 1):
                rep_path = c.get("representative_path")
                if rep_path and os.path.isfile(rep_path):
                    dst = os.path.join(struct_dir, f"cluster_{i}_representative.pdbqt")
                    _copy_file(rep_path, dst)
        # Copy ligand if found alongside best_pose directory
        if result.output_dir:
            lig_pdbqt = os.path.join(result.output_dir, "ligand.pdbqt")
            if os.path.isfile(lig_pdbqt):
                _copy_file(lig_pdbqt, os.path.join(struct_dir, "ligand.pdbqt"))
            # Crystal ligand if available
            for fname in ("crystal_ligand.pdb", "crystal_ligand.sdf"):
                src = os.path.join(result.output_dir, fname)
                if os.path.isfile(src):
                    _copy_file(src, os.path.join(struct_dir, fname))
            # Apo receptor if available
            for fname in ("apo_receptor.pdb", "apo_receptor.pdbqt"):
                src = os.path.join(result.output_dir, fname)
                if os.path.isfile(src):
                    _copy_file(src, os.path.join(struct_dir, fname))

    # ── 2. Interaction detection ────────────────────────────────────────────
    interactions: list[dict[str, Any]] = result.interactions or []
    # Prefer holo PDB (with crystallographic waters) for PLIP water-bridge detection;
    # fall back to the prepared apo PDB if holo is unavailable.
    _receptor_for_plip = receptor_pdb_holo or receptor_pdb
    if do_interactions and not interactions and _receptor_for_plip and result.best_pose_pdbqt:
        try:
            from autodock.interactions import detect_interactions

            interactions = detect_interactions(
                _receptor_for_plip,
                result.best_pose_pdbqt,
                method=interaction_method,
                output_dir=dirs.get("interactions"),
            )
            result.interactions = interactions
            logger.info(f"Detected {len(interactions)} interactions for {compound}")
        except (RuntimeError, OSError, ValueError, TypeError, ImportError) as exc:
            logger.warning(f"Interaction detection failed: {exc}")

    # Save interaction table
    if interactions:
        intx_path = os.path.join(dirs["interactions"], "interactions.csv")
        try:
            import csv

            with open(intx_path, "w", newline="") as fh:
                if interactions:
                    w = csv.DictWriter(fh, fieldnames=interactions[0].keys())
                    w.writeheader()
                    w.writerows(interactions)
            logger.info(f"Interactions saved: {intx_path}")
            outputs["interactions_csv"] = intx_path
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Interaction CSV failed: {exc}")

        # Summary text
        summary_lines = ["Interaction Summary", "=" * 40]
        from collections import Counter

        type_counts = Counter(i.get("type", "Unknown") for i in interactions)
        for itype, cnt in sorted(type_counts.items()):
            summary_lines.append(f"  {itype}: {cnt}")
        summary_path = os.path.join(dirs["interactions"], "interaction_summary.txt")
        with open(summary_path, "w") as fh:
            fh.write("\n".join(summary_lines) + "\n")
        outputs["interaction_summary_txt"] = summary_path

    # ── 3. Render figures (2D + 3D) ─────────────────────────────────────────
    if do_rendering and receptor_pdb and result.best_pose_pdbqt:
        fig_dir = dirs["figures"]
        fig_paths: list[str] = []

        # 3D scenes
        try:
            from autodock.rendering import render_scene_pymol

            for scene_name in ("complex", "pocket", "interaction"):
                png_path = os.path.join(fig_dir, f"3d_{scene_name}.png")
                pdf_path = os.path.join(fig_dir, f"3d_{scene_name}.pdf")
                pse_path = os.path.join(fig_dir, f"session_{scene_name}.pse")

                kw = {
                    "receptor_pdb": receptor_pdb,
                    "ligand_pdbqt": result.best_pose_pdbqt,
                    "output_png": png_path,
                    "output_pdf": pdf_path,
                    "scene": scene_name,
                    "save_pse": pse_path,
                }
                if scene_name in ("pocket", "interaction"):
                    kw["center"] = result.center
                if scene_name == "interaction" and interactions:
                    kw["interactions"] = interactions

                try:
                    render_scene_pymol(**kw)
                    fig_paths.append(png_path)
                except (RuntimeError, OSError, ValueError, TypeError, ImportError) as exc:
                    logger.warning(f"3D render '{scene_name}' skipped: {exc}")
        except (RuntimeError, OSError, ValueError, TypeError, ImportError) as exc:
            logger.warning(f"3D rendering unavailable: {exc}")

        # 2D interaction diagram — RDKit Cairo route (primary)
        try:
            from autodock.rendering import render_interactions_2d

            png_2d = os.path.join(fig_dir, "2d_interactions.png")
            pdf_2d = os.path.join(fig_dir, "2d_interactions.pdf")
            render_interactions_2d(
                receptor_pdb,
                result.best_pose_pdbqt,
                interactions,
                output_png=png_2d,
                output_pdf=pdf_2d,
            )
            fig_paths.append(png_2d)
            outputs["fig_2d_png"] = png_2d
            outputs["fig_2d_pdf"] = pdf_2d
        except (RuntimeError, OSError, ValueError, TypeError, ImportError) as exc:
            logger.warning(f"2D RDKit rendering skipped: {exc}")

        # Composite figure
        if len(fig_paths) >= 2:
            try:
                from autodock.rendering import composite_summary

                composite_png = os.path.join(fig_dir, "composite.png")
                composite_summary(
                    fig_paths[:4],
                    composite_png,
                    ncols=2,
                    panel_titles=["Complex", "Pocket", "Interactions", "2D"],
                    figure_title=f"Docking: {compound}",
                )
                outputs["composite_png"] = composite_png
            except (OSError, ValueError, TypeError, ImportError) as exc:
                logger.warning(f"Composite figure skipped: {exc}")

        outputs["figures"] = fig_paths

    # ── 4. Reports ──────────────────────────────────────────────────────────
    if do_report:
        report_dir = dirs["reports"]

        # JSON metadata
        json_path = os.path.join(report_dir, "result.json")
        try:
            with open(json_path, "w") as fh:
                json.dump(result.to_dict(), fh, indent=2, default=str)
            outputs["json"] = json_path
        except (TypeError, OSError) as exc:
            logger.warning(f"JSON report failed: {exc}")

        # CSV report
        csv_path = os.path.join(report_dir, "report.csv")
        try:
            from autodock.reporting import generate_csv_report

            generate_csv_report([result], csv_path)
            outputs["csv"] = csv_path
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"CSV report failed: {exc}")

        # PDF report
        pdf_path = os.path.join(report_dir, "report.pdf")
        try:
            from autodock.reporting import generate_pdf_report

            figs_for_pdf = outputs.get("figures", [])
            generate_pdf_report(result, pdf_path, figure_paths=figs_for_pdf)
            outputs["pdf"] = pdf_path
        except (OSError, TypeError, ValueError, ImportError) as exc:
            logger.warning(f"PDF report failed: {exc}")

    # ── 5. Summary text ────────────────────────────────────────────────────
    try:
        summary_txt = os.path.join(pair_root, "summary.txt")
        aff = result.best_affinity
        aff_str = f"{aff:.3f} kcal/mol" if aff is not None else "N/A"
        with open(summary_txt, "w") as fh:
            fh.write(f"Compound:    {compound}\n")
            fh.write(f"Receptor:    {receptor_name}\n")
            fh.write(f"Best affinity: {aff_str}\n")
            fh.write(f"Method:      {result.method_label}\n")
            fh.write(f"Timestamp:   {result.timestamp}\n")
            if result.rmsd_from_crystal is not None:
                fh.write(f"RMSD:        {result.rmsd_from_crystal:.2f} Å\n")
            if result.posebusters_pass is not None:
                fh.write(f"PoseBusters: {'PASS' if result.posebusters_pass else 'FAIL'}\n")
            fh.write(f"Pose clusters: {result.n_clusters or 0}\n")
            fh.write(f"Output:      {pair_root}\n")
        outputs["summary_txt"] = summary_txt
    except OSError as exc:
        logger.warning(f"Summary text failed: {exc}")

    logger.info(f"Post-processing complete: {pair_root}")
    return outputs


def _copy_file(src: str, dst: str) -> str:
    """Copy a file, creating parent directories."""
    import shutil

    ensure_dir(os.path.dirname(dst))
    try:
        shutil.copy2(src, dst)
    except (OSError, shutil.Error) as exc:
        logger.warning(f"File copy failed: {src} → {dst}: {exc}")
    return dst


def read_docking_results(result_dir: str) -> list[DockingResult]:
    """Read all DockingResult JSON files from a directory tree.

    Walks ``result_dir`` looking for ``04_reports/result.json`` files
    (produced by ``post_process_docking``) or plain ``result.json`` files.
    Returns a list of deserialised DockingResult objects (as dicts).
    """
    from autodock.core import DockingResult

    results: list[DockingResult] = []
    for root, _dirs, files in os.walk(result_dir):
        for fname in files:
            if fname == "result.json":
                path = os.path.join(root, fname)
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    if not isinstance(data, dict):
                        logger.warning(
                            f"Skipping {path}: expected JSON object, got {type(data).__name__}"
                        )
                        continue
                    # Schema validation for required fields
                    _required = {"compound_name", "receptor"}
                    _missing = _required - data.keys()
                    if _missing:
                        logger.warning(f"Skipping {path}: missing required fields {_missing}")
                        continue
                    results.append(DockingResult(**data))
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping {path}: invalid JSON ({exc})")
                except (OSError, UnicodeDecodeError) as exc:
                    logger.warning(f"Skipping {path}: read error ({exc})")
    return results
