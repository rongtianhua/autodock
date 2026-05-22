"""
autodock.docking — Molecular docking with AutoDock Vina.
========================================================
Single-ligand, multi-conformer, and virtual-screening workflows
with consensus scoring and structured result output.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from autodock.core import (
    logger,
    DockingCalculationError,
    DockingResult,
    build_docking_result,
    _HAVE_VINA,
    _get_vina_seed,
    VINA_DEFAULT_EXHAUSTIVENESS,
    VINA_DEFAULT_N_POSES,
    VINA_DEFAULT_ENERGY_RANGE,
    VINA_DEFAULT_TIMEOUT,
)
from autodock.utils import ensure_dir


# ─────────────────────────────────────────────────────────────────────────────
# Low-level Vina wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _run_vina_dock(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    exhaustiveness: int = VINA_DEFAULT_EXHAUSTIVENESS,
    n_poses: int = VINA_DEFAULT_N_POSES,
    energy_range: float = VINA_DEFAULT_ENERGY_RANGE,
    seed: int | None = None,
    timeout: int = VINA_DEFAULT_TIMEOUT,
) -> tuple[np.ndarray, list[str]]:
    """
    Run Vina docking and return (energies_array, pose_strings).

    Raises:
        DockingCalculationError: If docking fails or times out.
    """
    if not _HAVE_VINA:
        raise DockingCalculationError(
            "vina Python package not available. Install: conda install -c conda-forge vina"
        )

    from vina import Vina

    v = Vina(sf_name="vina", seed=_get_vina_seed(seed))
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=list(center), box_size=list(box_size))

    # Dock with wall-clock timeout (Vina C++ extension cannot be interrupted)
    result_state: dict[str, Any] = {}

    def _worker() -> None:
        try:
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses, min_rmsd=1.0)
            result_state["done"] = True
        except Exception as exc:
            result_state["error"] = str(exc)
            result_state["done"] = True

    t = threading.Thread(target=_worker, daemon=False)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        logger.error(f"Docking timed out after {timeout}s")
        raise DockingCalculationError(
            f"Docking timed out after {timeout}s. Try smaller search space or lower exhaustiveness."
        )
    if "error" in result_state:
        raise DockingCalculationError(f"Docking failed: {result_state['error']}")

    energies = v.energies(n_poses=n_poses, energy_range=energy_range)

    # Extract poses as individual PDBQT strings
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as tf:
        tmp_path = tf.name
    try:
        v.write_poses(tmp_path, n_poses=n_poses, energy_range=energy_range, overwrite=True)
        with open(tmp_path) as fh:
            pdbqt_str = fh.read()
    finally:
        os.unlink(tmp_path)

    parts = pdbqt_str.split("MODEL ")
    poses = []
    for i, part in enumerate(parts[1:], start=1):
        if part.strip():
            poses.append(f"MODEL {i}\n{part}")

    return energies, poses


def _score_pose_with_sf(
    receptor_pdbqt: str,
    pose_pdbqt: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    sf_name: str,
    seed: int | None = None,
) -> float | None:
    """Re-score a single pose with an alternative scoring function."""
    try:
        from vina import Vina
        v = Vina(sf_name=sf_name, seed=_get_vina_seed(seed))
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(pose_pdbqt)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        score = v.score()
        total = float(score[0]) if hasattr(score, "__getitem__") else float(score)
        return total
    except Exception as exc:
        logger.debug(f"Re-scoring with {sf_name} failed: {exc}")
        return None


def _consensus_score(
    receptor_pdbqt: str,
    pose_pdbqt: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    vina_score: float,
    seed: int | None = None,
) -> tuple[dict[str, float], float | None]:
    """
    Compute consensus affinity from multiple scoring functions.

    Returns:
        (all_scores_dict, consensus_affinity)
        consensus_affinity is the median of all successful scores.
    """
    all_scores: dict[str, float] = {"vina": vina_score}
    for sf in ("vinardo",):
        s = _score_pose_with_sf(receptor_pdbqt, pose_pdbqt, center, box_size, sf, seed)
        if s is not None:
            all_scores[sf] = s
            logger.info(f"  {sf} score: {s:.3f} kcal/mol")

    if len(all_scores) > 1:
        median_e = sorted(all_scores.values())[len(all_scores) // 2]
        logger.info(
            f"Consensus affinity: {median_e:.3f} kcal/mol "
            f"(median of {list(all_scores.keys())})"
        )
        return all_scores, median_e
    return all_scores, None


# ─────────────────────────────────────────────────────────────────────────────
# Public API: single-ligand docking
# ─────────────────────────────────────────────────────────────────────────────

def dock_ligand(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    exhaustiveness: int = VINA_DEFAULT_EXHAUSTIVENESS,
    n_poses: int = VINA_DEFAULT_N_POSES,
    energy_range: float = VINA_DEFAULT_ENERGY_RANGE,
    seed: int | None = None,
    timeout: int = VINA_DEFAULT_TIMEOUT,
    output_dir: str | None = None,
    compound_name: str | None = None,
    receptor_pdb: str | None = None,
) -> DockingResult:
    """
    Dock a single ligand into a protein binding site.

    Args:
        receptor_pdbqt: Prepared receptor PDBQT file.
        ligand_pdbqt: Prepared ligand PDBQT file.
        center: (x, y, z) binding box center.
        box_size: (sx, sy, sz) box dimensions (Å).
        exhaustiveness: Search thoroughness (publication standard: 32).
        n_poses: Number of poses to generate (publication standard: 20).
        energy_range: Energy range above best (kcal/mol).
        seed: Random seed for reproducibility (None = random).
        timeout: Wall-clock timeout in seconds.
        output_dir: If provided, persist pose files here.
        compound_name: Name for result tracking.
        receptor_pdb: Original receptor PDB (for provenance).

    Returns:
        DockingResult with scores, file paths, and metadata.
    """
    if not os.path.isfile(receptor_pdbqt):
        raise DockingCalculationError(f"Receptor PDBQT not found: {receptor_pdbqt}")
    if not os.path.isfile(ligand_pdbqt):
        raise DockingCalculationError(f"Ligand PDBQT not found: {ligand_pdbqt}")

    name = compound_name or Path(ligand_pdbqt).stem

    logger.info(
        f"Docking {name}: center={center}, box={box_size}, "
        f"exhaustiveness={exhaustiveness}, n_poses={n_poses}"
    )

    energies, poses = _run_vina_dock(
        receptor_pdbqt, ligand_pdbqt, center, box_size,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        energy_range=energy_range,
        seed=seed,
        timeout=timeout,
    )

    if energies.size == 0 or not poses:
        raise DockingCalculationError("Vina produced no poses.")

    best_affinity = float(energies[0][0])
    logger.info(f"Best affinity: {best_affinity:.3f} kcal/mol ({len(poses)} poses)")

    # Persist poses
    best_pose_path = None
    all_poses_path = None
    if output_dir:
        ensure_dir(output_dir)
        best_pose_path = os.path.join(output_dir, "docking_best.pdbqt")
        all_poses_path = os.path.join(output_dir, "docking_all_poses.pdbqt")
        with open(best_pose_path, "w") as fh:
            fh.write(poses[0])
        with open(all_poses_path, "w") as fh:
            fh.write("\n".join(poses))
        logger.info(f"Poses saved: {best_pose_path}, {all_poses_path}")
    else:
        # Temp files if no output_dir
        best_pose_path = tempfile.mktemp(suffix="_best.pdbqt")
        with open(best_pose_path, "w") as fh:
            fh.write(poses[0])

    # Consensus scoring
    all_scores, consensus = _consensus_score(
        receptor_pdbqt, best_pose_path, center, box_size, best_affinity, seed
    )

    # Receptor source detection
    receptor_source = None
    if receptor_pdb and os.path.isfile(receptor_pdb):
        from autodock.core import detect_receptor_source
        receptor_source = detect_receptor_source(receptor_pdb)

    result = DockingResult(
        compound_name=name,
        receptor=receptor_pdbqt,
        center=tuple(center),
        box_size=tuple(box_size),
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        seed=seed,
        best_affinity=best_affinity,
        scoring_functions=list(all_scores.keys()),
        all_scores=all_scores,
        consensus_affinity=consensus,
        best_pose_pdbqt=best_pose_path,
        all_poses_pdbqt=all_poses_path,
        output_dir=output_dir,
        receptor_source=receptor_source,
    )
    return result


def dock_ligand_multi_conformer(
    receptor_pdbqt: str,
    conformer_pdbqts: list[str],
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    exhaustiveness: int = VINA_DEFAULT_EXHAUSTIVENESS,
    n_poses: int = VINA_DEFAULT_N_POSES,
    energy_range: float = VINA_DEFAULT_ENERGY_RANGE,
    seed: int | None = None,
    timeout: int = VINA_DEFAULT_TIMEOUT,
    output_dir: str | None = None,
    compound_name: str | None = None,
) -> DockingResult:
    """
    Dock multiple ligand conformers and return the globally best pose.

    Each conformer is docked independently; all poses are pooled and ranked.
    This is the recommended protocol for publication-quality docking.

    Args:
        conformer_pdbqts: List of prepared ligand conformer PDBQT files.
        ... (other args same as dock_ligand)

    Returns:
        DockingResult with best pose from all conformers.
    """
    if not conformer_pdbqts:
        raise DockingCalculationError("No conformers provided.")

    all_poses_pool: list[tuple[float, str]] = []
    n_success = 0

    for conf_path in conformer_pdbqts:
        if not os.path.isfile(conf_path):
            logger.warning(f"Conformer file not found: {conf_path}")
            continue
        try:
            energies, poses = _run_vina_dock(
                receptor_pdbqt, conf_path, center, box_size,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                energy_range=energy_range,
                seed=seed,
                timeout=timeout,
            )
            for i, pose in enumerate(poses):
                if i < energies.shape[0]:
                    all_poses_pool.append((float(energies[i][0]), pose))
            n_success += 1
            logger.debug(
                f"Conformer {n_success}: {len(poses)} poses, best={energies[0][0]:.2f} kcal/mol"
            )
        except Exception as exc:
            logger.warning(f"Conformer {conf_path} failed: {exc}")
            continue

    if not all_poses_pool:
        raise DockingCalculationError("All conformers failed to dock.")

    # Sort by energy (most negative = best)
    all_poses_pool.sort(key=lambda x: x[0])
    best_energy, best_pose = all_poses_pool[0]

    logger.info(
        f"Multi-conformer docking: {n_success}/{len(conformer_pdbqts)} succeeded, "
        f"{len(all_poses_pool)} total poses, best={best_energy:.2f} kcal/mol"
    )

    # Persist
    out_dir = output_dir or os.path.join(os.path.dirname(conformer_pdbqts[0]), "multi_conformer_results")
    ensure_dir(out_dir)
    best_pose_path = os.path.join(out_dir, "best_pose.pdbqt")
    with open(best_pose_path, "w") as fh:
        fh.write(best_pose)

    # Consensus scoring on best pose
    all_scores, consensus = _consensus_score(
        receptor_pdbqt, best_pose_path, center, box_size, best_energy, seed
    )

    return DockingResult(
        compound_name=compound_name or Path(conformer_pdbqts[0]).stem,
        receptor=receptor_pdbqt,
        center=tuple(center),
        box_size=tuple(box_size),
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        seed=seed,
        best_affinity=best_energy,
        scoring_functions=list(all_scores.keys()),
        all_scores=all_scores,
        consensus_affinity=consensus,
        best_pose_pdbqt=best_pose_path,
        output_dir=out_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Virtual Screening
# ─────────────────────────────────────────────────────────────────────────────

def _dock_single_compound(
    args: tuple,
) -> DockingResult:
    """Worker function for parallel virtual screening."""
    name, smiles, receptor_pdbqt, center, box_size, output_dir, exhaustiveness, n_poses, compound_seed = args

    from autodock.preparation import prepare_ligand

    ligand_pdbqt = os.path.join(output_dir, f"{name}.pdbqt")
    try:
        prepare_ligand(smiles, ligand_pdbqt, name=name, seed=compound_seed)
        result = dock_ligand(
            receptor_pdbqt, ligand_pdbqt, center, box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            seed=compound_seed,
            output_dir=os.path.join(output_dir, name),
            compound_name=name,
        )
        return result
    except Exception as exc:
        logger.error(f"{name}: docking failed — {exc}")
        return DockingResult(
            compound_name=name,
            receptor=receptor_pdbqt,
            center=center,
            box_size=box_size,
            best_affinity=None,
        )


def virtual_screen(
    receptor_pdbqt: str,
    ligand_smiles_dict: dict[str, str],
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    output_dir: str = "./docking_results",
    exhaustiveness: int = 16,
    n_poses: int = 3,
    seed: int | None = None,
    n_workers: int = 1,
) -> tuple[list[DockingResult], str]:
    """
    Screen a compound library against a protein target.

    Args:
        receptor_pdbqt: Prepared receptor PDBQT.
        ligand_smiles_dict: {compound_name: smiles_string}.
        center: Binding box center.
        box_size: Binding box dimensions.
        output_dir: Results directory.
        exhaustiveness: Per-compound exhaustiveness (16 for screening).
        n_poses: Poses per compound.
        seed: Base random seed.
        n_workers: Number of parallel workers. -1 = use all CPU cores.

    Returns:
        (list_of_DockingResult, csv_path)
    """
    import pandas as pd

    ensure_dir(output_dir)

    base_seed = _get_vina_seed(seed)
    items = list(ligand_smiles_dict.items())

    if n_workers == 1:
        # Serial execution
        results: list[DockingResult] = []
        for idx, (name, smiles) in enumerate(items):
            compound_seed = (base_seed + idx) if seed is None else base_seed
            args = (name, smiles, receptor_pdbqt, center, box_size, output_dir,
                    exhaustiveness, n_poses, compound_seed)
            results.append(_dock_single_compound(args))
    else:
        # Parallel execution
        if n_workers == -1:
            import multiprocessing
            n_workers = multiprocessing.cpu_count()

        work_items = []
        for idx, (name, smiles) in enumerate(items):
            compound_seed = (base_seed + idx) if seed is None else base_seed
            work_items.append((name, smiles, receptor_pdbqt, center, box_size, output_dir,
                               exhaustiveness, n_poses, compound_seed))

        from concurrent.futures import ProcessPoolExecutor, as_completed

        logger.info(f"Starting parallel virtual screening with {n_workers} workers")
        results = [None] * len(work_items)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_dock_single_compound, item): i for i, item in enumerate(work_items)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    name = work_items[idx][0]
                    logger.error(f"{name}: worker crashed — {exc}")
                    results[idx] = DockingResult(
                        compound_name=name,
                        receptor=receptor_pdbqt,
                        center=center,
                        box_size=box_size,
                        best_affinity=None,
                    )

    # Export CSV
    csv_path = os.path.join(output_dir, "docking_results.csv")
    if results:
        df = pd.DataFrame([r.to_dataframe_row() for r in results])
        df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"Virtual screening complete: {len(results)} compounds, results: {csv_path}")
    return results, csv_path
