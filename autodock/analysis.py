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
from typing import Any

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


# ─────────────────────────────────────────────────────────────────────────────
# Spearman correlation — affinity vs RMSD
# ─────────────────────────────────────────────────────────────────────────────


def compute_spearman_correlation(
    affinities: list[float],
    rmsd_values: list[float],
) -> dict[str, float | None]:
    """
    Compute Spearman rank correlation between docking score and RMSD.

    A strong positive correlation means lower (better) affinity reliably
    predicts lower RMSD — i.e. the scoring function is well-calibrated.
    A weak or negative correlation indicates scoring bias.

    Args:
        affinities: Vina affinity values (kcal/mol), more negative = tighter.
        rmsd_values: RMSD from crystal reference (Å).

    Returns:
        Dict with ``rho``, ``pvalue``, and ``n``.
    """
    if len(affinities) < 3 or len(rmsd_values) < 3:
        return {"rho": None, "pvalue": None, "n": len(affinities)}

    try:
        from scipy.stats import spearmanr

        rho, pvalue = spearmanr(affinities, rmsd_values)
        return {"rho": float(rho), "pvalue": float(pvalue), "n": len(affinities)}
    except ImportError:
        logger.warning("scipy not available — cannot compute Spearman correlation")
        return {"rho": None, "pvalue": None, "n": len(affinities)}


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment factor — virtual screening metric
# ─────────────────────────────────────────────────────────────────────────────


def compute_enrichment_factor(
    scored_compounds: list[tuple[str, float]],
    active_ids: set[str],
    ef_percent: float = 1.0,
) -> dict[str, float | None]:
    """
    Compute enrichment factor (EF) at a given percentile.

    EF measures how much better than random a docking score is at
    enriching actives in the top-ranked fraction of a library.

    Args:
        scored_compounds: List of ``(compound_id, score)`` tuples.
            Lower (more negative) score = better binder.
        active_ids: Set of known active compound IDs.
        ef_percent: Percentile threshold (default 1.0 = top 1%).

    Returns:
        Dict with ``ef``, ``auc_roc``, ``n_total``, ``n_actives``,
        ``n_top``, ``n_top_actives``.
    """
    if not scored_compounds or not active_ids:
        return {
            "ef": None,
            "auc_roc": None,
            "n_total": len(scored_compounds),
            "n_actives": len(active_ids),
            "n_top": 0,
            "n_top_actives": 0,
        }

    # Sort by score (most negative = best)
    sorted_compounds = sorted(scored_compounds, key=lambda x: x[1])
    n_total = len(sorted_compounds)
    n_actives = len(active_ids)

    # Number of compounds in top percentile
    n_top = max(1, int(np.ceil(n_total * ef_percent / 100.0)))
    top_ids = {cid for cid, _ in sorted_compounds[:n_top]}
    n_top_actives = len(top_ids & active_ids)

    # Random expectation
    random_expectation = n_top * (n_actives / n_total) if n_total > 0 else 0.0
    ef = (n_top_actives / random_expectation) if random_expectation > 0 else None

    # AUC-ROC (trapezoidal rule)
    auc_roc = None
    try:
        from sklearn.metrics import roc_auc_score

        labels = [1 if cid in active_ids else 0 for cid, _ in sorted_compounds]
        # Invert scores so higher = better for roc_auc_score
        scores_inv = [-s for _, s in sorted_compounds]
        if len(set(labels)) > 1:
            auc_roc = float(roc_auc_score(labels, scores_inv))
    except ImportError:
        logger.warning("scikit-learn not available — skipping AUC-ROC")
    except ValueError:
        pass

    return {
        "ef": float(ef) if ef is not None else None,
        "auc_roc": auc_roc,
        "n_total": n_total,
        "n_actives": n_actives,
        "n_top": n_top,
        "n_top_actives": n_top_actives,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Interaction fingerprint — per-residue binary encoding
# ─────────────────────────────────────────────────────────────────────────────


def compute_interaction_fingerprint(
    interactions: list[dict],
    residue_order: list[str] | None = None,
    interaction_types: tuple[str, ...] = (
        "H-bond",
        "Hydrophobic",
        "Salt bridge",
        "π-π",
        "π-cation",
        "Halogen bond",
        "Water bridge",
        "Metal complex",
    ),
) -> dict[str, Any]:
    """
    Encode protein-ligand interactions as a binary fingerprint.

    Produces a per-residue bit vector (one bit per interaction type)
    that can be used for similarity comparisons across docking poses
    or compounds.

    Args:
        interactions: List of interaction dicts from PLIP / ProLIF.
        residue_order: Optional fixed ordering of residue strings
            (e.g. ``"A:42:GLU"``).  If None, residues are sorted
            alphabetically.
        interaction_types: Tuple of interaction-type strings to encode.

    Returns:
        Dict with:
          - ``fingerprint``: 2-D NumPy bool array (n_residues × n_types)
          - ``residues``: list of residue strings in row order
          - ``types``: list of interaction-type strings in column order
          - ``flat``: 1-D flattened bool array
          - ``n_interactions``: total interaction count
          - ``density``: fraction of possible bits that are set
    """
    if not interactions:
        empty = np.zeros((0, len(interaction_types)), dtype=bool)
        return {
            "fingerprint": empty,
            "residues": [],
            "types": list(interaction_types),
            "flat": empty.ravel(),
            "n_interactions": 0,
            "density": 0.0,
        }

    # Extract (residue, type) pairs
    pairs: set[tuple[str, str]] = set()
    for ixn in interactions:
        # Try multiple residue-key conventions
        res_key = None
        for key in ("residue", "restype_ligand", "resnr", "reschain"):
            if key in ixn:
                res_key = str(ixn[key])
                break
        if res_key is None:
            # Composite key from protisnr + protchain
            resnum = ixn.get("protisnr") or ixn.get("resnr")
            chain = ixn.get("protchain") or ixn.get("reschain", "")
            restype = ixn.get("restype") or ""
            if resnum is not None:
                res_key = f"{chain}:{resnum}:{restype}".strip(":")
        if res_key is None:
            continue

        itype = ixn.get("type", "Unknown")
        pairs.add((res_key, itype))

    residues = sorted({r for r, _ in pairs})
    if residue_order is not None:
        # Keep only residues that appear in interactions
        residues = [r for r in residue_order if r in residues]

    type_list = list(interaction_types)
    fp = np.zeros((len(residues), len(type_list)), dtype=bool)
    for res, itype in pairs:
        if res in residues and itype in type_list:
            fp[residues.index(res), type_list.index(itype)] = True

    n_interactions = len(pairs)
    density = fp.sum() / fp.size if fp.size > 0 else 0.0

    return {
        "fingerprint": fp,
        "residues": residues,
        "types": type_list,
        "flat": fp.ravel(),
        "n_interactions": n_interactions,
        "density": float(density),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ligand efficiency
# ─────────────────────────────────────────────────────────────────────────────


def compute_ligand_efficiency(
    affinity: float,
    n_heavy_atoms: int,
    n_rotatable_bonds: int | None = None,
    molecular_weight: float | None = None,
) -> dict[str, float | None]:
    """
    Compute ligand-efficiency metrics from binding affinity.

    LE (ligand efficiency) = |ΔG| / N_heavy  [kcal/mol per heavy atom]
    LLE (lipophilic ligand efficiency) = |ΔG| – cLogP  (requires cLogP)
    LEM (ligand efficiency per MW) = |ΔG| / MW  [kcal/mol per Da]
    LE_RB (ligand efficiency per rotatable bond) = |ΔG| / N_rot  [kcal/mol per RB]

    Args:
        affinity: Binding affinity in kcal/mol (more negative = tighter).
        n_heavy_atoms: Number of non-hydrogen atoms.
        n_rotatable_bonds: Optional number of rotatable bonds.
        molecular_weight: Optional molecular weight in Da.

    Returns:
        Dict with ``le``, ``lle``, ``lem``, ``le_rb``, ``n_heavy``,
        ``n_rotatable``, ``mw``.
    """
    if affinity is None or n_heavy_atoms is None or n_heavy_atoms <= 0:
        return {
            "le": None,
            "lle": None,
            "lem": None,
            "le_rb": None,
            "n_heavy": n_heavy_atoms,
            "n_rotatable": n_rotatable_bonds,
            "mw": molecular_weight,
        }

    delta_g = abs(float(affinity))
    le = delta_g / n_heavy_atoms

    lle = None
    lem = None
    le_rb = None

    if molecular_weight is not None and molecular_weight > 0:
        lem = delta_g / molecular_weight

    if n_rotatable_bonds is not None and n_rotatable_bonds > 0:
        le_rb = delta_g / n_rotatable_bonds

    return {
        "le": float(le),
        "lle": float(lle) if lle is not None else None,
        "lem": float(lem) if lem is not None else None,
        "le_rb": float(le_rb) if le_rb is not None else None,
        "n_heavy": n_heavy_atoms,
        "n_rotatable": n_rotatable_bonds,
        "mw": molecular_weight,
    }
