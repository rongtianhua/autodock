"""
autodock.heatmap — Binding-energy heatmap for batch-docking results.
=====================================================================
Publication-ready static heatmap in R ggplot2 style with top-journal
colour palettes (Nature Reviews / JACS / eLife aesthetic).

Input:  dict[receptor_name, list[DockingResult]]
Output: High-resolution PNG + PDF (600 dpi).
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from autodock.core import DockingResult, logger
from autodock.utils import ensure_dir


def _ggtheme(ax: Any) -> None:
    """Apply a clean ggplot2-style theme to a matplotlib Axes."""
    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Thin grey axis lines
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color("#4d4d4d")
    # Tick style
    ax.tick_params(axis="both", which="major", labelsize=9, colors="#4d4d4d", length=3, width=0.5)
    ax.tick_params(axis="both", which="minor", length=2, width=0.3)
    # Grid
    ax.grid(
        True, which="major", axis="both", linestyle="--", linewidth=0.3, alpha=0.4, color="#cccccc"
    )
    ax.set_axisbelow(True)


def plot_energy_heatmap(
    batch_results: dict[str, list[DockingResult]],
    output_dir: str = ".",
    output_prefix: str = "binding_energy_heatmap",
    dpi: int = 600,
    figsize: tuple[float, float] | None = None,
    palette: str = "nature",
    annotate: bool = True,
    vrange: tuple[float, float] | None = None,
) -> dict[str, str]:
    """Generate a publication-quality binding-energy heatmap.

    Args:
        batch_results: Mapping ``receptor_name → [DockingResult, ...]``
            as returned by :func:`autodock.docking.batch_dock`.
        output_dir: Output directory for generated figures.
        output_prefix: Base name for output files (``{prefix}.png``, ``{prefix}.pdf``).
        dpi: Output resolution (default 600 for print).
        figsize: Figure size in inches (auto-computed if None).
        palette: Colour palette name — ``"nature"`` (blue-white-red),
            ``"elife"`` (teal-magenta), or ``"viridis"``.
        annotate: If True, write the numerical affinity value (kcal/mol)
            on each receptor-ligand pair cell in the heatmap.
        vrange: (vmin, vmax) for heatmap colour scale; auto-scaled if None.

    Returns:
        Dict with keys ``png``, ``pdf`` → output file paths.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:
        raise RuntimeError(f"matplotlib required for heatmap: {exc}")

    ensure_dir(output_dir)

    # ── Build matrix ──────────────────────────────────────────────────────────
    receptor_names = sorted(batch_results.keys(), key=lambda n: n.lower())
    if not receptor_names:
        raise ValueError("batch_results is empty")

    # Collect all unique ligand names across all receptors
    all_ligands: set[str] = set()
    for results in batch_results.values():
        for r in results:
            all_ligands.add(r.compound_name)
    ligand_names = sorted(all_ligands)

    if not ligand_names:
        raise ValueError("No ligands found in batch_results")

    # Build energy matrix: rows=receptors, cols=ligands
    n_rec = len(receptor_names)
    n_lig = len(ligand_names)
    matrix = np.full((n_rec, n_lig), np.nan)
    annotations = [[""] * n_lig for _ in range(n_rec)]

    for i, rec_name in enumerate(receptor_names):
        results = batch_results.get(rec_name, [])
        result_map = {r.compound_name: r for r in results}
        for j, lig_name in enumerate(ligand_names):
            r = result_map.get(lig_name)
            if r is not None and r.best_affinity is not None:
                matrix[i, j] = r.best_affinity
                annotations[i][j] = f"{r.best_affinity:.1f}"

    # ── Colour scale ──────────────────────────────────────────────────────────
    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        raise ValueError("No valid affinity values in batch_results")
    if vrange is not None:
        vmin, vmax = vrange
    else:
        vmin = math.floor(valid.min())
        vmax = math.ceil(valid.max())
    # Centre diverging palette on zero for binding energies
    abs_max = max(abs(vmin), abs(vmax))
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    if palette == "nature":
        cmap = "RdBu_r"
    elif palette == "elife":
        cmap = "PiYG"
    else:
        cmap = "viridis"

    # ── Figure ────────────────────────────────────────────────────────────────
    if figsize is None:
        # Auto-size: 0.6 inches per ligand + 1.5 for labels
        w = max(4, n_lig * 0.55 + 3.0)
        h = max(3, n_rec * 0.55 + 2.5)
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    # Heatmap
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # Axis labels (ggplot2-style: horizontal x labels if few, rotated if many)
    ax.set_xticks(range(n_lig))
    ax.set_yticks(range(n_rec))
    ax.set_xticklabels(
        ligand_names, fontsize=8, rotation=45 if n_lig > 5 else 0, ha="right", va="top"
    )
    ax.set_yticklabels(receptor_names, fontsize=8)

    # Affinity-value annotations on each receptor-ligand pair cell
    if annotate:
        for i in range(n_rec):
            for j in range(n_lig):
                if not np.isnan(matrix[i, j]):
                    val = matrix[i, j]
                    # Choose text colour for contrast against heatmap colour
                    lightness = abs(val) / abs_max if abs_max > 0 else 0.5
                    text_color = "white" if lightness > 0.6 else "#333333"
                    ax.text(
                        j,
                        i,
                        annotations[i][j],
                        ha="center",
                        va="center",
                        fontsize=6,
                        fontweight="bold",
                        color=text_color,
                    )

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Binding affinity (kcal/mol)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Axes labels
    ax.set_xlabel("Ligand", fontsize=9, color="#333333")
    ax.set_ylabel("Receptor", fontsize=9, color="#333333")
    ax.set_title("Binding Energy Heatmap", fontsize=11, color="#222222", fontweight="bold", pad=12)

    # Apply ggplot2 theme
    _ggtheme(ax)

    fig.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    png_path = os.path.join(output_dir, f"{output_prefix}.png")
    pdf_path = os.path.join(output_dir, f"{output_prefix}.pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info(f"Heatmap PNG: {png_path}")

    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info(f"Heatmap PDF: {pdf_path}")

    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}
