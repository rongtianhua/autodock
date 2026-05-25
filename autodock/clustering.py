"""
autodock.clustering — Pose clustering for docking result analysis.
==================================================================
Greedy RMSD-based clustering of docked poses to identify distinct
binding modes.  This is a publication-grade best practice: reporting
only the energy-lowest pose misses alternative binding modes that may
be only slightly higher in energy but biologically relevant.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from autodock.core import logger
from autodock.validation import compute_rmsd, compute_rmsd_coordinate_based


def _write_pose_to_temp(pose_str: str, suffix: str = ".pdbqt") -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as fh:
        fh.write(pose_str)
        return fh.name


def cluster_poses(
    poses: list[str],
    energies: np.ndarray,
    rmsd_threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Greedy clustering of docked poses by RMSD.

    Algorithm:
        1. Sort poses by energy (most negative = best).
        2. Initialise first cluster with the lowest-energy pose.
        3. For each remaining pose, compute RMSD to the representative
           (lowest-energy member) of each existing cluster.
        4. If RMSD < threshold to any cluster representative, join that
           cluster; otherwise start a new cluster.

    Args:
        poses: List of PDBQT pose strings (one per MODEL).
        energies: NxM array from Vina (first column is affinity).
        rmsd_threshold: RMSD cutoff in Å (default 2.0 Å).

    Returns:
        List of cluster dicts sorted by representative energy:
        {
            "representative_index": int,
            "size": int,
            "member_indices": list[int],
            "representative_energy": float,
            "member_energies": list[float],
        }
    """
    if not poses or energies.size == 0:
        return []

    n = len(poses)
    if n != energies.shape[0]:
        logger.warning(
            f"Pose/energy count mismatch: {n} poses vs {energies.shape[0]} energies"
        )
        n = min(n, energies.shape[0])

    # Write all poses to temp files once
    temp_files = [_write_pose_to_temp(p) for p in poses[:n]]
    try:
        # Sort by energy (ascending: most negative first)
        sorted_indices = np.argsort(energies[:n, 0]).tolist()

        clusters: list[dict[str, Any]] = []

        for idx in sorted_indices:
            pose_file = temp_files[idx]
            energy = float(energies[idx, 0])
            assigned = False

            for cluster in clusters:
                rep_idx = cluster["representative_index"]
                rep_file = temp_files[rep_idx]

                # Try topology-aware RMSD first; fall back to coordinate-based
                rmsd = compute_rmsd(pose_file, rep_file)
                if rmsd is None:
                    rmsd = compute_rmsd_coordinate_based(pose_file, rep_file)
                if rmsd is None:
                    # Cannot compute RMSD — skip clustering for this pair
                    continue

                if rmsd < rmsd_threshold:
                    cluster["member_indices"].append(idx)
                    cluster["member_energies"].append(energy)
                    cluster["size"] += 1
                    assigned = True
                    break

            if not assigned:
                clusters.append({
                    "representative_index": int(idx),
                    "size": 1,
                    "member_indices": [int(idx)],
                    "representative_energy": energy,
                    "member_energies": [energy],
                })

        # Sort clusters by representative energy (best first)
        clusters.sort(key=lambda c: c["representative_energy"])
        logger.info(
            f"Pose clustering: {n} poses → {len(clusters)} clusters "
            f"(threshold={rmsd_threshold} Å)"
        )
        return clusters

    finally:
        for path in temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
