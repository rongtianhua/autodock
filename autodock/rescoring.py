"""
autodock.rescoring — Auxiliary pose re-scoring beyond AutoDock Vina.
========================================================
Currently provides interaction-fingerprint (IFP) re-scoring only.
Additional physics-based or ML scoring methods may be added here
when dependencies (GNINA, OpenMM, etc.) are available.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def combined_rescoring(
    all_poses_pdbqt: str,
    reference_pdbqt: str | None = None,
    methods: list[str] | None = None,
    receptor_pdb: str | None = None,
) -> dict[str, list[tuple[int, float, float | None]]]:
    """Run auxiliary rescoring methods on a pose ensemble.

    Args:
        all_poses_pdbqt: Multi-MODEL PDBQT from Vina.
        reference_pdbqt: Reference ligand PDBQT (required for ``"ifp"``).
        methods: List of method names.  Currently supported:
            * ``"ifp"`` — interaction-fingerprint Tanimoto (requires *receptor_pdb*)
        receptor_pdb: Receptor PDB file (required for ``"ifp"``).

    Returns:
        Dict mapping method name to sorted score list.
    """
    if methods is None:
        methods = []

    results: dict[str, list[tuple[int, float, float | None]]] = {}
    for method in methods:
        if method == "ifp":
            if receptor_pdb is None or reference_pdbqt is None:
                logger.warning("IFP rescoring skipped: receptor_pdb and reference_pdbqt required")
                continue
            try:
                from autodock.interactions import ifp_similarity_scores

                results["ifp"] = ifp_similarity_scores(
                    receptor_pdb, all_poses_pdbqt, reference_pdbqt
                )
            except Exception as exc:
                logger.warning(f"IFP rescoring failed: {exc}")
        else:
            logger.warning(f"Unknown rescoring method: {method}")
    return results


def select_best_by_method(
    scores: list[tuple[int, float, float | None]],
    method: str = "max",
) -> tuple[int, float] | None:
    """Select the best pose from a sorted score list.

    Args:
        scores: Sorted list from any rescoring function.
        method: ``"max"`` for descending-optimal (similarity),
            ``"min"`` for ascending-optimal (energy).

    Returns:
        ``(pose_index, best_score)`` or ``None`` if empty.
    """
    if not scores:
        return None
    best = min(scores, key=lambda x: x[1]) if method == "min" else scores[0]
    return best[0], best[1]
