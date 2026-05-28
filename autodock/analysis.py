"""
autodock.analysis — Post-benchmark analysis and scoring bias diagnostics.
==========================================================================
Publication-grade analyses that run on top of completed benchmark outputs:
  * Scoring-bias scatter plots (affinity vs RMSD for all poses)
  * Scoring vs sampling decoupling (top-1 vs best-rmsd gap)
"""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np

from autodock.core import logger
from autodock.validation import compute_rmsd_to_crystal


def _parse_all_poses(
    all_poses_pdbqt: str,
) -> list[tuple[float, str]]:
    """
    Parse a multi-MODEL PDBQT file and return (affinity, pose_block) pairs.

    Vina writes ``REMARK VINA RESULT:    -8.236      0.000      0.000``
    at the beginning of each MODEL section.
    """
    if not os.path.isfile(all_poses_pdbqt):
        return []

    with open(all_poses_pdbqt) as fh:
        content = fh.read()

    # Split on MODEL lines; first chunk is header, subsequent are numbered
    models = re.split(r"MODEL\s+\d+\n", content)
    if len(models) <= 1:
        # Single pose — try to read affinity from REMARK
        affinity = _extract_affinity(content)
        if affinity is not None:
            return [(affinity, content)]
        return []

    results: list[tuple[float, str]] = []
    for model_block in models[1:]:
        block = model_block.split("ENDMDL")[0]
        if not block.strip():
            continue
        affinity = _extract_affinity(block)
        if affinity is None:
            continue
        results.append((affinity, block.strip()))

    return results


def _extract_affinity(text: str) -> float | None:
    """Extract Vina affinity from a REMARK line."""
    m = re.search(r"REMARK VINA RESULT:\s+(-?\d+\.?\d*)", text)
    if m:
        return float(m.group(1))
    return None


def analyze_scoring_bias(
    output_dir: str,
    target_ids: list[str] | None = None,
    figure_dir: str | None = None,
) -> dict:
    """
    Generate scoring-bias scatter plots for one or more benchmark targets.

    For each target, reads ``docking_all_poses.pdbqt`` and the crystal
    ligand reference.  Plots affinity vs RMSD for every docked pose,
    with two markers:
      * Red ★ — top-1 pose (lowest Vina affinity)
      * Green ★ — best-RMSD pose (lowest RMSD to crystal)

    Args:
        output_dir: Root benchmark output directory containing per-target
            subdirectories (e.g. ``benchmark_20target_final/``).
        target_ids: Specific PDB IDs to analyse (default: all subdirs).
        figure_dir: Where to save scatter plots (default: ``output_dir/figures``).

    Returns:
        Dict mapping ``pdb_id`` → per-target data:
          - poses: list of (affinity, rmsd) tuples
          - top1_affinity, top1_rmsd
          - best_rmsd, best_rmsd_idx (1-based)
          - top1_vs_best: difference between top-1 and best RMSD
          - figure_path: path to saved scatter plot (if figure_dir given)
    """
    if target_ids is None:
        target_ids = sorted(
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and len(d) == 4
        )

    if figure_dir is None:
        figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    results: dict = {}

    for pdb_id in target_ids:
        target_dir = os.path.join(output_dir, pdb_id)

        # Auto-detect all_poses filename (varies across benchmark runs)
        all_poses_candidates = [
            os.path.join(target_dir, "docking_all_poses.pdbqt"),
            os.path.join(target_dir, "all_poses.pdbqt"),
        ]
        all_poses_path = None
        for candidate in all_poses_candidates:
            if os.path.isfile(candidate):
                all_poses_path = candidate
                break

        crystal_path = os.path.join(target_dir, "crystal_ligand.pdb")

        if all_poses_path is None or not os.path.isfile(crystal_path):
            logger.warning(f"Skipping {pdb_id}: missing all_poses or crystal_ligand")
            continue

        logger.info(f"Analyzing scoring bias: {pdb_id}")

        # Parse poses + affinities
        pose_entries = _parse_all_poses(all_poses_path)
        if not pose_entries:
            logger.warning(f"No poses found for {pdb_id}")
            continue

        # Compute RMSD for each pose
        pose_data: list[tuple[float, float]] = []  # (affinity, rmsd)
        for i, (affinity, block) in enumerate(pose_entries):
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False)
            tmp.write(f"MODEL {i + 1}\n")
            tmp.write(block + "\n")
            tmp.write("ENDMDL\n")
            tmp.close()

            try:
                rmsd = compute_rmsd_to_crystal(tmp.name, crystal_path)
            finally:
                os.unlink(tmp.name)

            if rmsd is not None:
                pose_data.append((affinity, float(rmsd)))

        if not pose_data:
            logger.warning(f"No valid RMSD values for {pdb_id}")
            continue

        # Sort by affinity (most negative = best)
        affinities = [p[0] for p in pose_data]
        rmsd_vals = [p[1] for p in pose_data]

        best_rmsd = min(rmsd_vals)
        best_rmsd_idx = rmsd_vals.index(best_rmsd) + 1  # 1-based
        best_rmsd_affinity = affinities[best_rmsd_idx - 1]

        # Top-1 pose = lowest (most negative) affinity
        min_aff_idx = np.argmin(affinities)
        top1_affinity = affinities[min_aff_idx]
        top1_rmsd = rmsd_vals[min_aff_idx]

        results[pdb_id] = {
            "poses": pose_data,
            "top1_affinity": top1_affinity,
            "top1_rmsd": top1_rmsd,
            "best_rmsd": best_rmsd,
            "best_rmsd_affinity": best_rmsd_affinity,
            "best_rmsd_idx": best_rmsd_idx,
            "top1_vs_best": top1_rmsd - best_rmsd,
        }

        # ── Generate scatter plot ──────────────────────────────────────────
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — skipping scatter plot")
            continue

        fig, ax = plt.subplots(figsize=(6, 5))

        # All poses
        ax.scatter(
            rmsd_vals,
            affinities,
            c="steelblue",
            alpha=0.5,
            s=36,
            label=f"All poses (n={len(pose_data)})",
            zorder=2,
        )

        # Top-1 pose
        ax.scatter(
            [top1_rmsd],
            [top1_affinity],
            c="red",
            marker="*",
            s=200,
            edgecolors="darkred",
            linewidths=0.8,
            label=f"Top-1 (RMSD {top1_rmsd:.2f} Å)",
            zorder=5,
        )

        # Best-RMSD pose
        ax.scatter(
            [best_rmsd],
            [best_rmsd_affinity],
            c="green",
            marker="*",
            s=200,
            edgecolors="darkgreen",
            linewidths=0.8,
            label=f"Best RMSD (rank {best_rmsd_idx}, RMSD {best_rmsd:.2f} Å)",
            zorder=5,
        )

        # 2.0 Å threshold
        ax.axvline(
            2.0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="2.0 Å threshold"
        )

        ax.set_xlabel("RMSD from crystal (Å)")
        ax.set_ylabel("Vina affinity (kcal/mol)")
        ax.set_title(f"{pdb_id} — Scoring Bias Analysis")
        ax.legend(loc="upper left", fontsize=8)
        ax.invert_yaxis()  # more negative = better
        fig.tight_layout()

        fig_path = os.path.join(figure_dir, f"scoring_bias_{pdb_id}.png")
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        results[pdb_id]["figure_path"] = fig_path
        logger.info(f"Scoring bias plot saved: {fig_path}")

    return results
