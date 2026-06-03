"""
autodock.docking — Molecular docking with AutoDock Vina.
========================================================
Single-ligand, multi-conformer, and virtual-screening workflows
with consensus scoring and structured result output.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import multiprocessing
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from autodock.core import (
    _HAVE_VINA,
    VINA_DEFAULT_ENERGY_RANGE,
    VINA_DEFAULT_EXHAUSTIVENESS,
    VINA_DEFAULT_N_POSES,
    VINA_DEFAULT_TIMEOUT,
    DockingCalculationError,
    DockingResult,
    _get_vina_seed,
    logger,
)
from autodock.utils import ensure_dir, strip_model_headers, write_temp_file

# ─────────────────────────────────────────────────────────────────────────────
# Low-level Vina wrappers
# ─────────────────────────────────────────────────────────────────────────────


def _count_pdbqt_atoms(pdbqt_path: str) -> int:
    """Count ATOM/HETATM lines in a PDBQT file."""
    if not os.path.isfile(pdbqt_path):
        raise DockingCalculationError(f"Cannot count atoms — ligand file not found: {pdbqt_path}")
    count = 0
    with open(pdbqt_path) as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                count += 1
    return count


def _auto_exhaustiveness(ligand_pdbqt: str, base_exhaustiveness: int) -> int:
    """Reduce exhaustiveness for very large ligands to keep runtime tractable.

    Vina internally scales search steps with ligand size/flexibility.
    Large ligands + high exhaustiveness = combinatorial explosion.

    Thresholds are empirically derived from heavy-atom counts in the
    DUD-E / PDBbind refined sets (Mysinger et al. 2012, JCIM; Liu et al.
    2017, J. Chem. Inf. Model.): ~90% of drug-like ligands have ≤55
    heavy atoms, ~75% ≤45, ~50% ≤35.  These cutoffs catch exceptionally
    large ligands (peptide-like, macrocycles) where runtime risk is
    highest.

    Minimum floor is 16 — below this, Vina's stochastic search produces
    unreproducible poses even for trivial cases (Eberhardt et al. 2021,
    JCIM; Vina 1.2 exhaustive search recommendation).
    """
    n_atoms = _count_pdbqt_atoms(ligand_pdbqt)
    # Heavy-atom thresholds based on PDBbind refined set size distribution
    if n_atoms > 55:
        return max(16, base_exhaustiveness // 8)
    if n_atoms > 45:
        return max(16, base_exhaustiveness // 4)
    if n_atoms > 35:
        return max(16, base_exhaustiveness // 2)
    return base_exhaustiveness


def _vina_dock_worker(
    args: tuple,
    result_queue,
) -> None:
    """Worker function that runs in a separate process for true timeout control."""
    (
        receptor_pdbqt,
        ligand_pdbqt,
        center,
        box_size,
        exhaustiveness,
        n_poses,
        energy_range,
        seed,
        flex_receptor_pdbqt,
        scoring_function,
        min_rmsd,
    ) = args

    try:
        from vina import Vina

        v = Vina(sf_name=scoring_function, seed=_get_vina_seed(seed))
        v.set_receptor(receptor_pdbqt)
        if flex_receptor_pdbqt and os.path.isfile(flex_receptor_pdbqt):
            v.set_flex(flex_receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses, min_rmsd=min_rmsd)

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

        try:
            result_queue.put(("ok", energies, poses))
        except (TypeError, ValueError) as exc:
            # Queue put can fail if payloads are not serialisable (spawn context)
            result_queue.put(("error", f"result_queue put failed: {exc}", []))
    except Exception as exc:  # noqa: BLE001
        # Worker safety net — any crash must be reported via queue
        with contextlib.suppress(TypeError, ValueError, OSError):
            result_queue.put(("error", str(exc), []))


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
    auto_exhaustiveness: bool = False,
    flex_receptor_pdbqt: str | None = None,
    scoring_function: str = "vina",
    min_rmsd: float = 1.0,
    _use_subprocess: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Run Vina docking in an isolated process with hard timeout via terminate/kill.

    Args:
        _use_subprocess: Internal flag. When False, run Vina in-thread (used by
            unit tests that mock the Vina class, since mocks don't cross process
            boundaries with spawn).

    Raises:
        DockingCalculationError: If docking fails or times out.
    """
    if not _HAVE_VINA:
        raise DockingCalculationError(
            "vina Python package not available. Install: conda install -c conda-forge vina"
        )

    if auto_exhaustiveness:
        effective_exhaustiveness = _auto_exhaustiveness(ligand_pdbqt, exhaustiveness)
        if effective_exhaustiveness != exhaustiveness:
            logger.warning(
                f"Auto-adjusted exhaustiveness: {exhaustiveness} → {effective_exhaustiveness} "
                f"({_count_pdbqt_atoms(ligand_pdbqt)} heavy atoms). "
                f"For redocking validation, pass auto_exhaustiveness=False to preserve "
                f"publication-grade sampling."
            )
            exhaustiveness = effective_exhaustiveness

    # If we're already inside a multiprocessing child, run Vina directly.
    # Nested subprocesses can deadlock or hang with Vina's C++ extension.
    in_subprocess = multiprocessing.current_process().name != "MainProcess"

    # In-thread fallback for mocked tests (mocks don't survive spawn)
    if not _use_subprocess or in_subprocess:
        from vina import Vina

        v = Vina(sf_name=scoring_function, seed=_get_vina_seed(seed))
        v.set_receptor(receptor_pdbqt)
        if flex_receptor_pdbqt and os.path.isfile(flex_receptor_pdbqt):
            v.set_flex(flex_receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))

        if not _use_subprocess:
            # Tests: use threading timeout for mocked Vina
            result_state: dict[str, Any] = {}

            def _worker() -> None:
                try:
                    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses, min_rmsd=min_rmsd)
                    result_state["done"] = True
                except (RuntimeError, ValueError, TypeError, OSError) as exc:
                    result_state["error"] = str(exc)
                    result_state["done"] = True

            t = threading.Thread(target=_worker, daemon=False)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                raise DockingCalculationError(
                    f"Docking timed out after {timeout}s."
                    " Try smaller search space or lower exhaustiveness."
                )
            if "error" in result_state:
                raise DockingCalculationError(f"Docking failed: {result_state['error']}") from None
        else:
            # Already in a subprocess: run directly, timeout handled by parent
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses, min_rmsd=min_rmsd)

        energies = v.energies(n_poses=n_poses, energy_range=energy_range)
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

    args = (
        receptor_pdbqt,
        ligand_pdbqt,
        center,
        box_size,
        exhaustiveness,
        n_poses,
        energy_range,
        seed,
        flex_receptor_pdbqt,
        scoring_function,
        min_rmsd,
    )

    # Use spawn context to avoid fork-safety issues with Vina C++ extension
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    p = ctx.Process(target=_vina_dock_worker, args=(args, result_queue))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        logger.error(f"Docking timed out after {timeout}s — terminating process {p.pid}")
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            logger.error(f"Process {p.pid} still alive — force killing")
            p.kill()
            p.join(timeout=5)
        raise DockingCalculationError(
            f"Docking timed out after {timeout}s. Try smaller search space or lower exhaustiveness."
        )

    # Read queue first — even if exitcode != 0 the worker may have posted a
    # detailed error message before crashing.
    try:
        status, payload1, payload2 = result_queue.get(timeout=30)
    except (queue.Empty, OSError, EOFError):
        if p.exitcode != 0:
            raise DockingCalculationError(
                f"Docking subprocess exited with code {p.exitcode}"
            ) from None
        raise DockingCalculationError(
            "Docking subprocess completed but result queue was empty"
        ) from None

    if status == "error":
        raise DockingCalculationError(f"Docking failed: {payload1}") from None

    return payload1, payload2


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
        # Vina rejects PDBQT files containing MODEL/ENDMDL tags.
        # Strip them to a plain single-model PDBQT.
        with open(pose_pdbqt) as fh:
            pdbqt_text = fh.read()
        clean_pdbqt = strip_model_headers(pdbqt_text)
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as tf:
            tf.write(clean_pdbqt)
            tmp_pose = tf.name
        try:
            v.set_ligand_from_file(tmp_pose)
            v.compute_vina_maps(center=list(center), box_size=list(box_size))
            score = v.score()
            total = float(score[0]) if hasattr(score, "__getitem__") else float(score)
            return total
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_pose)
    except (ImportError, RuntimeError, OSError) as exc:
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
    # Extensible scoring-function list.  Vina Python API currently supports
    # "vina" and "vinardo".  Additional SFs (e.g. "ad4" via CLI, GNINA CNN
    # scores, etc.) can be registered here or passed via config.
    all_scores: dict[str, float] = {"vina": vina_score}
    for sf in ("vinardo",):
        s = _score_pose_with_sf(receptor_pdbqt, pose_pdbqt, center, box_size, sf, seed)
        if s is not None:
            all_scores[sf] = s
            logger.info(f"  {sf} score: {s:.3f} kcal/mol")

    if len(all_scores) > 1:
        median_e = sorted(all_scores.values())[len(all_scores) // 2]
        logger.info(
            f"Consensus affinity: {median_e:.3f} kcal/mol (median of {list(all_scores.keys())})"
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
    skip_consensus: bool = False,
    auto_exhaustiveness: bool = False,
    min_rmsd: float = 1.0,
    scoring_function: str = "vina",
    ligand_smiles: str | None = None,
    multi_conformer: bool = False,
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
        skip_consensus: If True, skip the extra Vinardo consensus scoring step
            (useful for bulk benchmarks where speed matters).
        auto_exhaustiveness: If True, reduce exhaustiveness for very large
            ligands (>35 heavy atoms) to avoid Vina combinatorial explosion.
            Default False to preserve publication-grade sampling.
        min_rmsd: Minimum RMSD (Å) between Vina-generated poses.  Vina
            discards poses within this threshold (default 1.0 Å, typical
            range 0.5–1.5 Å; see Fischer et al. 2021, J. Chem. Inf. Model.).
        scoring_function: Vina scoring function name.  Supported: ``"vina"``
            (default), ``"vinardo"``, ``"ad4"`` (AutoDock4).  Available
            functions depend on the Vina Python package version.
        ligand_smiles: SMILES of the ligand.  Required when
            ``multi_conformer=True``.
        multi_conformer: If True, pre-generate multiple 3D conformers
            from ``ligand_smiles`` and dock each one independently.
            The globally best pose across all conformers is returned.
            **Not recommended for typical Vina docking** — Vina already
            performs internal torsion search, so this adds runtime without
            improving accuracy for most ligands.  Only useful for
            macrocycles or rigid ring systems where Vina cannot cross
            conformational barriers.  Requires ``ligand_smiles``.
            (default False)

    Returns:
        DockingResult with scores, file paths, and metadata.
    """
    # Input validation layer
    from autodock.validation_params import validate_docking_params

    _params = validate_docking_params(
        receptor_pdbqt,
        ligand_pdbqt,
        center,
        box_size,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        energy_range=energy_range,
        seed=seed,
        timeout=timeout,
    )
    # Unpack validated values
    receptor_pdbqt = _params["receptor_pdbqt"]
    ligand_pdbqt = _params["ligand_pdbqt"]
    center = _params["center"]
    box_size = _params["box_size"]
    exhaustiveness = _params["exhaustiveness"]
    n_poses = _params["n_poses"]
    energy_range = _params["energy_range"]
    seed = _params["seed"]
    timeout = _params["timeout"]

    name = compound_name or Path(ligand_pdbqt).stem

    logger.info(
        f"Docking {name}: center={center}, box={box_size}, "
        f"exhaustiveness={exhaustiveness}, n_poses={n_poses}, seed={seed}"
    )

    # ── Multi-conformer docking ───────────────────────────────────────────
    # Pre-generate diverse 3D conformers from SMILES, dock each independently.
    # Combines pre-generation + Vina flexible sampling (top-journal practice).
    if multi_conformer:
        if not ligand_smiles:
            raise ValueError("multi_conformer=True requires ligand_smiles to be provided")
        from autodock.preparation import prepare_ligand_conformers

        tmp_dir = tempfile.mkdtemp(prefix="autodock_multi_")
        try:
            conf_pdbqts = prepare_ligand_conformers(
                ligand_smiles,
                tmp_dir,
                n_conformers=10,
                name=name[:3] if name else "LIG",
                molscrub_states=True,
            )
            logger.info(
                f"Multi-conformer: {len(conf_pdbqts)} conformers generated, docking each one..."
            )
            # dock_ligand_multi_conformer is defined in this same module;
            # no import needed — direct call via closure is correct.
            result = dock_ligand_multi_conformer(
                receptor_pdbqt,
                conf_pdbqts,
                center,
                box_size,
                exhaustiveness=max(exhaustiveness, 8),
                n_poses=max(n_poses, 5),
                energy_range=energy_range,
                seed=seed,
                timeout=timeout * 2,
                output_dir=output_dir,
                compound_name=name,
                scoring_function=scoring_function,
                min_rmsd=min_rmsd,
                skip_consensus=skip_consensus,
            )
            return result
        finally:
            with contextlib.suppress(Exception):
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)

    energies, poses = _run_vina_dock(
        receptor_pdbqt,
        ligand_pdbqt,
        center,
        box_size,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        energy_range=energy_range,
        seed=seed,
        timeout=timeout,
        auto_exhaustiveness=auto_exhaustiveness,
        scoring_function=scoring_function,
        min_rmsd=min_rmsd,
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
        # Best pose: strip MODEL/ENDMDL so Vina can reload it for re-scoring
        best_clean = strip_model_headers(poses[0])
        with open(best_pose_path, "w") as fh:
            fh.write(best_clean)
        with open(all_poses_path, "w") as fh:
            fh.write("\n".join(poses))
        logger.info(f"Poses saved: {best_pose_path}, {all_poses_path}")
    else:
        # Temp files if no output_dir
        fd, best_pose_path = tempfile.mkstemp(suffix="_best.pdbqt")
        os.close(fd)
        best_clean = strip_model_headers(poses[0])
        with open(best_pose_path, "w") as fh:
            fh.write(best_clean)

    # Pose clustering (publication-grade best practice)
    from autodock.clustering import cluster_poses

    clusters = cluster_poses(poses, energies, rmsd_threshold=2.0)

    # Persist cluster representatives if output_dir provided
    if output_dir and clusters:
        for i, cluster in enumerate(clusters[:5], 1):
            rep_idx = cluster["representative_index"]
            rep_path = os.path.join(output_dir, f"cluster_{i}_representative.pdbqt")
            with open(rep_path, "w") as fh:
                fh.write(poses[rep_idx])
            cluster["representative_path"] = rep_path

    # Consensus scoring (optional — skip for speed in bulk benchmarks)
    all_scores = {"vina": best_affinity}
    consensus = None
    if not skip_consensus:
        # Score best pose with all available scoring functions
        all_scores, consensus = _consensus_score(
            receptor_pdbqt, best_pose_path, center, box_size, best_affinity, seed
        )

        # Multi-pose Vinardo scoring: detect scoring-rank disagreement
        # between Vina and Vinardo across all poses.
        if len(poses) > 1:
            vinardo_scores: list[tuple[float, int]] = []  # (score, pose_idx)
            for _pidx, _pose in enumerate(poses):
                _tmp_path = write_temp_file(strip_model_headers(_pose), suffix="_pose.pdbqt")
                try:
                    _vs = _score_pose_with_sf(
                        receptor_pdbqt, _tmp_path, center, box_size, "vinardo", seed
                    )
                    if _vs is not None:
                        vinardo_scores.append((_vs, _pidx))
                finally:
                    with contextlib.suppress(OSError):
                        os.unlink(_tmp_path)

            if vinardo_scores:
                vinardo_scores.sort(key=lambda x: x[0])  # best (lowest) first
                best_vinardo_score, best_vinardo_idx = vinardo_scores[0]
                vina_best_idx = 0  # Vina already sorts pose 0 = best
                if best_vinardo_idx != vina_best_idx:
                    logger.warning(
                        f"Scoring bias detected: Vinardo prefers pose #{best_vinardo_idx + 1} "
                        f"({best_vinardo_score:.3f} kcal/mol) over Vina's top pose #1 "
                        f"({best_affinity:.3f} kcal/mol). "
                        "Consensus considers all scores; inspect affinity-vs-RMSD plot."
                    )
                else:
                    logger.debug(
                        f"Vinardo all-poses check: best pose #{best_vinardo_idx + 1} "
                        f"matches Vina ranking."
                    )
                # Store all Vinardo scores in DockingResult metadata
                all_scores["vinardo_all_poses"] = best_vinardo_score
                all_scores["vinardo_best_pose_idx"] = float(best_vinardo_idx)

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
        pose_clusters=clusters,
        n_clusters=len(clusters),
        rmsd_clustering_threshold=2.0,
    )
    return result


def _dock_conformer_worker(
    args: tuple,
) -> tuple[list[tuple[float, str]], int]:
    """Worker for parallel multi-conformer docking (picklable top-level function)."""
    (
        receptor_pdbqt,
        conf_path,
        center,
        box_size,
        exhaustiveness,
        n_poses,
        energy_range,
        seed,
        timeout,
        auto_exhaustiveness,
        scoring_function,
        min_rmsd,
    ) = args
    return _dock_conformer_core(
        receptor_pdbqt,
        conf_path,
        center,
        box_size,
        exhaustiveness,
        n_poses,
        energy_range,
        seed,
        timeout,
        auto_exhaustiveness,
        scoring_function=scoring_function,
        min_rmsd=min_rmsd,
    )


def _dock_conformer_core(
    receptor_pdbqt: str,
    conf_path: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    exhaustiveness: int,
    n_poses: int,
    energy_range: float,
    seed: int | None,
    timeout: int,
    auto_exhaustiveness: bool = True,
    scoring_function: str = "vina",
    min_rmsd: float = 1.0,
) -> tuple[list[tuple[float, str]], int]:
    """Core docking logic for a single conformer."""
    if not os.path.isfile(conf_path):
        return [], 0
    try:
        energies, poses = _run_vina_dock(
            receptor_pdbqt,
            conf_path,
            center,
            box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            energy_range=energy_range,
            seed=seed,
            timeout=timeout,
            auto_exhaustiveness=auto_exhaustiveness,
            scoring_function=scoring_function,
            min_rmsd=min_rmsd,
        )
        pool = []
        for i, pose in enumerate(poses):
            if i < energies.shape[0]:
                pool.append((float(energies[i][0]), pose))
        return pool, 1
    except DockingCalculationError:
        return [], 0


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
    scoring_function: str = "vina",
    min_rmsd: float = 1.0,
    skip_consensus: bool = False,
    max_workers: int = -1,
    auto_exhaustiveness: bool = True,
) -> DockingResult:
    """
    Dock multiple ligand conformers and return the globally best pose.

    Each conformer is docked independently in parallel; all poses are pooled
    and ranked.

    .. warning::

        **Multi-conformer docking is usually unnecessary for AutoDock Vina.**
        Vina performs its own internal torsion-angle search, so a single
        conformer from ``prepare_ligand()`` is sufficient for the vast
        majority of ligands.  This function is intended for special cases
        where Vina cannot cross conformational barriers, such as macrocycles
        or rigid ring systems with distinct conformers (chair vs. boat
        cyclohexane).  For typical drug-like molecules, multi-conformer
        docking increases runtime linearly without improving pose accuracy.

    Args:
        conformer_pdbqts: List of prepared ligand conformer PDBQT files.
        skip_consensus: If True, skip the extra Vinardo consensus scoring step.
        max_workers: Parallel workers for conformer docking (-1 = auto: CPU
            count capped by available memory, ~1.5 GB/worker).  When
            ``psutil`` is installed, the estimate is accurate; without it,
            falls back to CPU count only (may OOM on memory-constrained
            machines).
        ... (other args same as dock_ligand)

    Returns:
        DockingResult with best pose from all conformers.
    """
    if not conformer_pdbqts:
        raise DockingCalculationError("No conformers provided.")

    all_poses_pool: list[tuple[float, str]] = []
    n_success = 0

    # Parallelize conformer docking using direct subprocesses.
    # Each conformer runs in its own top-level subprocess; _run_vina_dock
    # detects it's already in a child and runs Vina directly (no nesting).
    #
    # Memory-aware worker count: each spawn'd Vina process uses ~1.5 GB RAM
    # (Vina grid map + Python imports + RDKit/OpenMM libraries).  On machines
    # with <8 GB total RAM, force single-worker to avoid OOM kills.
    _ESTIMATED_GB_PER_WORKER = 1.5
    if max_workers == -1:
        max_workers = min(multiprocessing.cpu_count(), len(conformer_pdbqts))
        try:
            import psutil

            avail_gb = psutil.virtual_memory().available / 1e9
            mem_workers = max(1, int(avail_gb / _ESTIMATED_GB_PER_WORKER))
            if mem_workers < max_workers:
                logger.info(
                    f"Memory-aware worker limit: {max_workers} → {mem_workers} "
                    f"({avail_gb:.1f} GB available, ~{_ESTIMATED_GB_PER_WORKER:.0f} GB/worker)"
                )
                max_workers = mem_workers
        except ImportError:
            logger.debug("psutil not available — skipping memory-aware worker limit")
    else:
        max_workers = min(max_workers, len(conformer_pdbqts))
        try:
            import psutil

            avail_gb = psutil.virtual_memory().available / 1e9
            mem_limit = max(1, int(avail_gb / _ESTIMATED_GB_PER_WORKER))
            if max_workers > mem_limit:
                logger.warning(
                    f"Requested {max_workers} workers but only {avail_gb:.1f} GB available "
                    f"(~{_ESTIMATED_GB_PER_WORKER:.0f} GB/worker). "
                    f"Clamping to {mem_limit} to avoid OOM."
                )
                max_workers = mem_limit
        except ImportError:
            pass

    if max_workers < 1:
        max_workers = 1

    # Derive a unique seed per conformer so that parallel searches are
    # statistically independent while remaining fully reproducible.
    base_seed = _get_vina_seed(seed)
    work_items = [
        (
            receptor_pdbqt,
            conf_path,
            center,
            box_size,
            exhaustiveness,
            n_poses,
            energy_range,
            base_seed + i,
            timeout,
            auto_exhaustiveness,
            scoring_function,
            min_rmsd,
        )
        for i, conf_path in enumerate(conformer_pdbqts)
    ]

    if max_workers > 1:
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_ctx
        ) as executor:
            future_to_item = {
                executor.submit(_dock_conformer_worker, item): item for item in work_items
            }
            for future in concurrent.futures.as_completed(
                future_to_item, timeout=timeout * len(work_items) + 10
            ):
                try:
                    pool, ok = future.result(timeout=timeout + 5)
                except Exception as exc:
                    conf_path = future_to_item[future][1]
                    logger.warning(f"Conformer docking failed for {conf_path}: {exc}")
                    continue
                all_poses_pool.extend(pool)
                n_success += ok
                if ok:
                    best_e = min(e for e, _ in pool)
                    logger.debug(f"Conformer done: {len(pool)} poses, best={best_e:.2f} kcal/mol")
    else:
        # Sequential fallback
        for item in work_items:
            pool, ok = _dock_conformer_core(
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
                item[5],
                item[6],
                item[7],
                item[8],
                auto_exhaustiveness=item[9] if len(item) > 9 else True,
                scoring_function=item[10] if len(item) > 10 else scoring_function,
                min_rmsd=item[11] if len(item) > 11 else min_rmsd,
            )
            all_poses_pool.extend(pool)
            n_success += ok
            if ok:
                best_e = min(e for e, _ in pool)
                logger.debug(f"Conformer done: {len(pool)} poses, best={best_e:.2f} kcal/mol")

    if not all_poses_pool:
        raise DockingCalculationError("All conformers failed to dock.")

    # Sort by energy (most negative = best)
    all_poses_pool.sort(key=lambda x: x[0])
    best_energy, best_pose = all_poses_pool[0]

    logger.info(
        f"Multi-conformer docking: {n_success}/{len(conformer_pdbqts)} succeeded, "
        f"{len(all_poses_pool)} total poses, best={best_energy:.2f} kcal/mol"
    )

    # Pose clustering across all conformers
    from autodock.clustering import cluster_poses

    # Build a Vina-compatible N×5 energy array (cluster_poses only uses column 0)
    all_energies = np.array([[e, 0.0, 0.0, 0.0, 0.0] for e, _ in all_poses_pool])
    all_poses = [p for _, p in all_poses_pool]
    clusters = cluster_poses(all_poses, all_energies, rmsd_threshold=2.0)

    # Persist
    out_dir = output_dir or os.path.join(
        os.path.dirname(conformer_pdbqts[0]), "multi_conformer_results"
    )
    ensure_dir(out_dir)
    best_pose_path = os.path.join(out_dir, "best_pose.pdbqt")
    # Strip MODEL/ENDMDL/model-number so Vina can reload it for re-scoring
    best_clean = strip_model_headers(best_pose)
    with open(best_pose_path, "w") as fh:
        fh.write(best_clean)

    # Persist all poses for best-achievable RMSD analysis
    all_poses_path = os.path.join(out_dir, "all_poses.pdbqt")
    with open(all_poses_path, "w") as fh:
        for i, pose in enumerate(all_poses, start=1):
            fh.write(f"MODEL {i}\n")
            # Strip any existing MODEL/ENDMDL headers line-by-line to avoid
            # global substring replacement corrupting atom names.
            cleaned_lines = []
            for line in pose.splitlines():
                if line.startswith("MODEL ") or line.strip() == "ENDMDL":
                    continue
                cleaned_lines.append(line)
            fh.write("\n".join(cleaned_lines))
            fh.write("\nENDMDL\n")

    # Persist cluster representatives
    if clusters:
        for i, cluster in enumerate(clusters[:5], 1):
            rep_idx = cluster["representative_index"]
            rep_path = os.path.join(out_dir, f"cluster_{i}_representative.pdbqt")
            with open(rep_path, "w") as fh:
                fh.write(all_poses[rep_idx])
            cluster["representative_path"] = rep_path

    # Consensus scoring (optional)
    if skip_consensus:
        all_scores = {"vina": best_energy}
        consensus = None
    else:
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
        all_poses_pdbqt=all_poses_path,
        output_dir=out_dir,
        pose_clusters=clusters,
        n_clusters=len(clusters),
        rmsd_clustering_threshold=2.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Virtual Screening
# ─────────────────────────────────────────────────────────────────────────────


def _dock_single_compound(
    args: tuple,
) -> DockingResult:
    """Worker function for parallel virtual screening."""
    (
        name,
        smiles,
        receptor_pdbqt,
        center,
        box_size,
        output_dir,
        exhaustiveness,
        n_poses,
        compound_seed,
    ) = args

    from autodock.preparation import prepare_ligand

    ligand_pdbqt = os.path.join(output_dir, f"{name}.pdbqt")
    try:
        prepare_ligand(smiles, ligand_pdbqt, name=name, seed=compound_seed)
        result = dock_ligand(
            receptor_pdbqt,
            ligand_pdbqt,
            center,
            box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            seed=compound_seed,
            output_dir=os.path.join(output_dir, name),
            compound_name=name,
        )
        return result
    except (RuntimeError, OSError, ValueError, DockingCalculationError) as exc:
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
            args = (
                name,
                smiles,
                receptor_pdbqt,
                center,
                box_size,
                output_dir,
                exhaustiveness,
                n_poses,
                compound_seed,
            )
            results.append(_dock_single_compound(args))
    else:
        # Parallel execution
        if n_workers == -1:
            n_workers = multiprocessing.cpu_count()

        work_items = []
        for idx, (name, smiles) in enumerate(items):
            compound_seed = (base_seed + idx) if seed is None else base_seed
            work_items.append(
                (
                    name,
                    smiles,
                    receptor_pdbqt,
                    center,
                    box_size,
                    output_dir,
                    exhaustiveness,
                    n_poses,
                    compound_seed,
                )
            )

        from concurrent.futures import ProcessPoolExecutor, as_completed

        logger.info(f"Starting parallel virtual screening with {n_workers} workers")
        results = [None] * len(work_items)
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
            futures = {
                executor.submit(_dock_single_compound, item): i for i, item in enumerate(work_items)
            }
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


# ─────────────────────────────────────────────────────────────────────────────
# Batch docking: multiple receptors × multiple ligands
# ─────────────────────────────────────────────────────────────────────────────


def _batch_dock_one(
    item: tuple,
) -> tuple[str, DockingResult]:
    """Picklable worker for batch_dock (must stay at module level)."""
    (
        rec_name,
        rec_path,
        lig_name,
        lig_path,
        center,
        box_size,
        job_seed,
        exhaustiveness,
        n_poses,
        energy_range,
        timeout,
        job_out,
    ) = item
    from autodock.utils import ensure_dir

    ensure_dir(job_out)
    t0 = time.perf_counter()
    try:
        result = dock_ligand(
            rec_path,
            lig_path,
            center,
            box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            energy_range=energy_range,
            seed=job_seed,
            timeout=timeout,
            output_dir=job_out,
            compound_name=lig_name,
        )
        result.runtime_seconds = round(time.perf_counter() - t0, 2)
        logger.info(f"[{rec_name} × {lig_name}] {result.best_affinity:.2f} kcal/mol")
        return rec_name, result
    except DockingCalculationError as exc:
        logger.error(f"[{rec_name} × {lig_name}] failed: {exc}")
        fail_result = DockingResult(
            compound_name=lig_name,
            receptor=rec_path,
            center=center,
            box_size=box_size,
            seed=job_seed,
            best_affinity=None,
            output_dir=job_out,
        )
        fail_result.runtime_seconds = round(time.perf_counter() - t0, 2)
        return rec_name, fail_result


def batch_dock(
    receptors: dict[str, str],
    ligands: dict[str, str],
    pockets: dict[str, dict[str, tuple[float, float, float]]],
    exhaustiveness: int = VINA_DEFAULT_EXHAUSTIVENESS,
    n_poses: int = VINA_DEFAULT_N_POSES,
    energy_range: float = VINA_DEFAULT_ENERGY_RANGE,
    seed: int | None = None,
    timeout: int = VINA_DEFAULT_TIMEOUT,
    output_dir: str = "./batch_docking_results",
    n_workers: int = 1,
) -> dict[str, list[DockingResult]]:
    """
    Perform pairwise docking across multiple receptors and ligands.

    This is the recommended API for large-scale comparison studies,
    cross-docking validation, and structure-activity relationship (SAR)
    exploration across multiple protein conformations.

    Args:
        receptors: Mapping of receptor_name → receptor_pdbqt_path.
        ligands: Mapping of ligand_name → ligand_pdbqt_path.
        pockets: Mapping of receptor_name → {"center": (x,y,z), "box_size": (sx,sy,sz)}.
        exhaustiveness: Search thoroughness per docking job.
        n_poses: Poses to generate per job.
        energy_range: Energy range above best (kcal/mol).
        seed: Random seed for reproducibility.
        timeout: Wall-clock timeout per job (seconds).
        output_dir: Root directory for all results.
        n_workers: Parallel workers. -1 = all CPU cores.

    Returns:
        Dictionary mapping receptor_name → list of DockingResult (one per ligand).
        Failed dockings are represented by DockingResult with best_affinity=None.

    Raises:
        DockingCalculationError: If no receptor/ligand files are valid.
        ValueError: If pocket definitions are missing for any receptor.
    """
    if not receptors or not ligands:
        raise DockingCalculationError("At least one receptor and one ligand required.")

    # Validate files and pockets
    for name, path in receptors.items():
        if not os.path.isfile(path):
            raise DockingCalculationError(f"Receptor file not found: {path} ({name})")
        if name not in pockets:
            raise ValueError(f"Pocket definition missing for receptor: {name}")
        pocket = pockets[name]
        if "center" not in pocket or "box_size" not in pocket:
            raise ValueError(f"Pocket for {name} must contain 'center' and 'box_size'")

    for name, path in ligands.items():
        if not os.path.isfile(path):
            raise DockingCalculationError(f"Ligand file not found: {path} ({name})")

    ensure_dir(output_dir)

    # Build work list: one item per receptor-ligand pair
    base_seed = _get_vina_seed(seed)
    work_items = []
    pair_idx = 0
    for rec_name, rec_path in receptors.items():
        for lig_name, lig_path in ligands.items():
            pair_seed = base_seed + pair_idx if seed is None else base_seed
            center = pockets[rec_name]["center"]
            box_size = pockets[rec_name]["box_size"]
            job_out = os.path.join(output_dir, rec_name, lig_name)
            work_items.append(
                (
                    rec_name,
                    rec_path,
                    lig_name,
                    lig_path,
                    center,
                    box_size,
                    pair_seed,
                    exhaustiveness,
                    n_poses,
                    energy_range,
                    timeout,
                    job_out,
                )
            )
            pair_idx += 1

    logger.info(
        f"Batch docking: {len(receptors)} receptors × {len(ligands)} ligands = "
        f"{len(work_items)} jobs, seed={base_seed}"
    )

    # Execute
    if n_workers == 1:
        raw_results = [_batch_dock_one(item) for item in work_items]
    else:
        if n_workers == -1:
            import multiprocessing

            n_workers = multiprocessing.cpu_count()
        from concurrent.futures import ProcessPoolExecutor, as_completed

        raw_results = [None] * len(work_items)
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
            futures = {
                executor.submit(_batch_dock_one, item): i for i, item in enumerate(work_items)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    raw_results[idx] = future.result()
                except (
                    TimeoutError,
                    DockingCalculationError,
                    RuntimeError,
                    OSError,
                    ValueError,
                ) as exc:
                    rec_name = work_items[idx][0]
                    lig_name = work_items[idx][2]
                    logger.error(f"[{rec_name} × {lig_name}] worker crashed: {exc}")
                    raw_results[idx] = (
                        rec_name,
                        DockingResult(
                            compound_name=lig_name,
                            receptor=receptors[rec_name],
                            center=pockets[rec_name]["center"],
                            box_size=pockets[rec_name]["box_size"],
                            best_affinity=None,
                        ),
                    )

    # Organize by receptor
    results_by_receptor: dict[str, list[DockingResult]] = {name: [] for name in receptors}
    for rec_name, result in raw_results:
        results_by_receptor[rec_name].append(result)

    # Export master CSV
    try:
        import pandas as pd

        rows = []
        for rec_name, res_list in results_by_receptor.items():
            for r in res_list:
                row = r.to_dataframe_row()
                row["receptor_name"] = rec_name
                rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(output_dir, "batch_docking_results.csv")
            df.to_csv(csv_path, index=False, float_format="%.4f")
            logger.info(f"Batch results CSV: {csv_path}")
    except (OSError, TypeError) as exc:
        logger.warning(f"Failed to write batch CSV: {exc}")

    return results_by_receptor


def dock_ensemble(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    n_repeats: int = 10,
    exhaustiveness: int = VINA_DEFAULT_EXHAUSTIVENESS,
    n_poses: int = VINA_DEFAULT_N_POSES,
    energy_range: float = VINA_DEFAULT_ENERGY_RANGE,
    seed: int | None = None,
    timeout: int = VINA_DEFAULT_TIMEOUT,
    output_dir: str | None = None,
    compound_name: str | None = None,
    receptor_pdb: str | None = None,
) -> dict[str, Any]:
    """
    Run repeated docking with independent seeds for ensemble statistics.

    This is the publication-grade protocol for assessing docking
    reproducibility and pose stability.  N independent runs with
    different random seeds are executed; the resulting poses are
    clustered and statistical metrics (mean, std, CV, RMSD) are
    computed to assign a confidence level.

    Args:
        receptor_pdbqt, ligand_pdbqt, center, box_size: Standard docking params.
        n_repeats: Number of independent docking runs (default 10).
        exhaustiveness, n_poses, energy_range, timeout: Vina parameters.
        seed: Base random seed.  Each repeat uses seed + i.
        output_dir: Root directory for all repeat outputs.
        compound_name, receptor_pdb: Provenance.

    Returns:
        Dictionary with keys:
            - repeats: list[DockingResult]
            - ensemble_best_affinity_mean, _std, _min, _max, _cv
            - ensemble_consensus_affinity_mean
            - pose_stability_rmsd_mean, _std, _max
            - n_clusters: int
            - confidence: "high" | "moderate" | "low"
            - recommendation: str
    """
    if n_repeats < 2:
        raise ValueError("n_repeats must be >= 2 for ensemble statistics.")

    base_seed = _get_vina_seed(seed)
    name = compound_name or Path(ligand_pdbqt).stem
    logger.info(f"Ensemble docking: {name}, {n_repeats} repeats, base_seed={base_seed}")

    repeats: list[DockingResult] = []
    for i in range(n_repeats):
        repeat_seed = base_seed + i
        repeat_out = None
        if output_dir:
            repeat_out = os.path.join(output_dir, f"repeat_{i + 1}")
            ensure_dir(repeat_out)

        t0 = time.perf_counter()
        try:
            result = dock_ligand(
                receptor_pdbqt,
                ligand_pdbqt,
                center,
                box_size,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                energy_range=energy_range,
                seed=repeat_seed,
                timeout=timeout,
                output_dir=repeat_out,
                compound_name=f"{name}_repeat{i + 1}",
                receptor_pdb=receptor_pdb,
            )
            result.runtime_seconds = round((time.perf_counter() - t0), 2)
            repeats.append(result)
            logger.info(f"Repeat {i + 1}/{n_repeats}: affinity={result.best_affinity:.3f} kcal/mol")
        except DockingCalculationError as exc:
            logger.error(f"Repeat {i + 1}/{n_repeats} failed: {exc}")
            # Append a placeholder so statistics can still be computed
            repeats.append(
                DockingResult(
                    compound_name=f"{name}_repeat{i + 1}",
                    receptor=receptor_pdbqt,
                    center=center,
                    box_size=box_size,
                    seed=repeat_seed,
                    best_affinity=None,
                    output_dir=repeat_out,
                )
            )

    valid_repeats = [r for r in repeats if r.best_affinity is not None]
    if len(valid_repeats) < 2:
        raise DockingCalculationError(
            f"Fewer than 2 successful repeats ({len(valid_repeats)}/{n_repeats}). "
            "Cannot compute ensemble statistics."
        )

    # ── Energy statistics ───────────────────────────────────────────
    affinities = np.array([r.best_affinity for r in valid_repeats])
    consensus_affinities = np.array(
        [r.consensus_affinity for r in valid_repeats if r.consensus_affinity is not None]
    )

    # NaN/inf guard: a single NaN poisons all statistics
    affinities = affinities[np.isfinite(affinities)]
    if consensus_affinities.size > 0:
        consensus_affinities = consensus_affinities[np.isfinite(consensus_affinities)]

    if affinities.size == 0:
        logger.warning("Ensemble summary: no finite affinity values — all repeats failed")
        energy_mean = energy_std = energy_cv = None
    else:
        energy_mean = float(np.mean(affinities))
        energy_std = float(np.std(affinities, ddof=1))
        energy_cv = abs(energy_std / energy_mean) if energy_mean != 0 else 0.0

    # ── Pose stability: RMSD between best poses of each repeat ──────
    best_pose_paths = []
    for r in valid_repeats:
        if r.best_pose_pdbqt and os.path.isfile(r.best_pose_pdbqt):
            best_pose_paths.append(r.best_pose_pdbqt)

    from autodock.validation import compute_rmsd, compute_rmsd_coordinate_based

    rmsd_values: list[float] = []
    n_paths = len(best_pose_paths)
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            rmsd = compute_rmsd(best_pose_paths[i], best_pose_paths[j])
            if rmsd is None or rmsd == 0.0:
                rmsd = compute_rmsd_coordinate_based(best_pose_paths[i], best_pose_paths[j])
            if rmsd is not None:
                rmsd_values.append(rmsd)

    if rmsd_values:
        rmsd_mean = float(np.mean(rmsd_values))
        rmsd_std = float(np.std(rmsd_values, ddof=1))
        rmsd_max = float(np.max(rmsd_values))
    else:
        rmsd_mean = rmsd_std = rmsd_max = None

    # ── Clustering of all best poses ────────────────────────────────
    from autodock.clustering import cluster_poses

    pose_strings = []
    pose_energies_list = []
    for r in valid_repeats:
        if r.best_pose_pdbqt and os.path.isfile(r.best_pose_pdbqt):
            with open(r.best_pose_pdbqt) as fh:
                pose_strings.append(fh.read())
            pose_energies_list.append(r.best_affinity)
    pose_energies = (
        np.array(pose_energies_list).reshape(-1, 1) if pose_energies_list else np.array([])
    )
    cluster_summary = cluster_poses(
        pose_strings,
        pose_energies,
        rmsd_threshold=2.0,
    )
    n_clusters = len(cluster_summary)

    # ── Confidence assessment ───────────────────────────────────────
    if energy_cv is not None and energy_cv < 0.05 and (rmsd_mean is not None and rmsd_mean < 1.0):
        confidence = "high"
        recommendation = (
            "Docking result is highly reproducible. "
            "The reported affinity and pose can be trusted for publication."
        )
    elif energy_cv is not None and energy_cv < 0.10 and (rmsd_mean is not None and rmsd_mean < 2.0):
        confidence = "moderate"
        recommendation = (
            "Docking result is moderately reproducible. "
            "Consider increasing exhaustiveness or verifying with MD."
        )
    else:
        confidence = "low"
        recommendation = (
            "Docking result shows poor reproducibility. "
            "The binding mode may be ambiguous; inspect cluster representatives."
        )

    summary = {
        "repeats": repeats,
        "n_repeats": n_repeats,
        "n_successful": len(valid_repeats),
        "ensemble_best_affinity_mean": energy_mean,
        "ensemble_best_affinity_std": energy_std,
        "ensemble_best_affinity_min": (float(np.min(affinities)) if affinities.size > 0 else None),
        "ensemble_best_affinity_max": (float(np.max(affinities)) if affinities.size > 0 else None),
        "ensemble_best_affinity_cv": energy_cv,
        "ensemble_consensus_affinity_mean": (
            float(np.mean(consensus_affinities)) if consensus_affinities.size > 0 else None
        ),
        "pose_stability_rmsd_mean": rmsd_mean,
        "pose_stability_rmsd_std": rmsd_std,
        "pose_stability_rmsd_max": rmsd_max,
        "n_clusters": n_clusters,
        "cluster_summary": cluster_summary,
        "confidence": confidence,
        "recommendation": recommendation,
    }

    _cv_str = f"{energy_cv:.3f}" if energy_cv is not None else "N/A"
    _rmsd_str = f"{rmsd_mean:.2f}" if rmsd_mean is not None else "N/A"
    _mean_str = f"{energy_mean:.3f}" if energy_mean is not None else "N/A"
    _std_str = f"{energy_std:.3f}" if energy_std is not None else "N/A"
    logger.info(
        f"Ensemble summary: mean={_mean_str} ± {_std_str} kcal/mol, "
        f"CV={_cv_str}, RMSD={_rmsd_str} Å, "
        f"clusters={n_clusters}, confidence={confidence}"
    )
    return summary
