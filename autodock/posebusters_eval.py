"""
autodock.posebusters_eval — PoseBusters benchmark evaluation.
===========================================================
Run redocking + PoseBusters validation on the PoseBusters benchmark set.
Supports both the full 428-set (V1) and the curated 308-set (V2).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from autodock.benchmark import auto_detect_ligand_resname
from autodock.core import ValidationError, logger
from autodock.validation import run_redocking_validation, validate_pose_with_posebusters


def load_posebusters_ids(path: str) -> list[tuple[str, str]]:
    """
    Load PoseBusters ID list.

    Format per line: PDBID_CCD (e.g. '5SAK_ZRY')
    Returns list of (pdb_id, ccd_code) tuples.
    """
    ids: list[tuple[str, str]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("_")
            if len(parts) >= 2:
                ids.append((parts[0], parts[1]))
    return ids


def run_posebusters_evaluation(
    id_list_path: str,
    output_dir: str = "./posebusters_results",
    exhaustiveness: int = 32,
    n_poses: int = 20,
    seed: int = 42,
    n_workers: int = 1,
    max_targets: int | None = None,
) -> dict[str, Any]:
    """
    Run redocking + PoseBusters validation on a PoseBusters ID list.

    Args:
        id_list_path: Path to text file with 'PDBID_CCD' per line.
        output_dir: Root directory for all outputs.
        exhaustiveness, n_poses, seed: Vina parameters.
        n_workers: Parallel workers (-1 = all CPU cores).
        max_targets: If set, limit to first N targets (for quick tests).

    Returns:
        Summary dict with statistics and per-target results.
    """

    targets = load_posebusters_ids(id_list_path)
    if max_targets:
        targets = targets[:max_targets]

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"PoseBusters evaluation: {len(targets)} targets, seed={seed}")

    work_items = []
    for pdb_id, ccd in targets:
        work_items.append(
            {
                "pdb_id": pdb_id,
                "ccd": ccd,
                "output_dir": os.path.join(output_dir, pdb_id),
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "seed": seed,
            }
        )

    raw_results: list[dict[str, Any]] = []
    if n_workers == 1:
        for item in work_items:
            raw_results.append(_run_single_posebuster(item))
    else:
        if n_workers == -1:
            import multiprocessing

            n_workers = multiprocessing.cpu_count()
        from concurrent.futures import ProcessPoolExecutor, as_completed

        raw_results = [None] * len(work_items)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_single_posebuster, item): i
                for i, item in enumerate(work_items)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    raw_results[idx] = future.result()
                except Exception as exc:
                    target = work_items[idx]
                    logger.error(f"[{target['pdb_id']}] Worker crashed: {exc}")
                    raw_results[idx] = {
                        "pdb_id": target["pdb_id"],
                        "success": False,
                        "error": str(exc),
                    }

    # Compile stats
    successes = [r for r in raw_results if r.get("success")]
    rmsds = [r["rmsd"] for r in successes if r.get("rmsd") is not None]
    pb_passes = [r for r in successes if r.get("posebusters_pass")]

    summary = {
        "n_total": len(targets),
        "n_success": len(successes),
        "success_rate": len(successes) / len(targets) if targets else 0.0,
        "mean_rmsd": float(np.mean(rmsds)) if rmsds else None,
        "median_rmsd": float(np.median(rmsds)) if rmsds else None,
        "rmsd_std": float(np.std(rmsds)) if rmsds else None,
        "posebusters_pass_rate": len(pb_passes) / len(successes) if successes else 0.0,
        "posebusters_pass_count": len(pb_passes),
        "parameters": {
            "exhaustiveness": exhaustiveness,
            "n_poses": n_poses,
            "seed": seed,
        },
        "per_target": raw_results,
    }

    json_path = os.path.join(output_dir, "posebusters_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info(
        f"PoseBusters complete: {summary['n_success']}/{summary['n_total']} succeeded, "
        f"{len(pb_passes)}/{len(successes)} PoseBusters-pass"
    )
    return summary


def _run_single_posebuster(item: dict[str, Any]) -> dict[str, Any]:
    """Run redocking + PoseBusters for a single target."""
    from autodock.utils import download_pdb

    pdb_id = item["pdb_id"]
    ccd = item["ccd"]
    outdir = item["output_dir"]
    os.makedirs(outdir, exist_ok=True)

    # Download holo structure
    holo_pdb = os.path.join(outdir, f"{pdb_id}.pdb")
    if not os.path.exists(holo_pdb):
        try:
            download_pdb(pdb_id, outdir)
        except Exception as exc:
            logger.error(f"[{pdb_id}] Download failed: {exc}")
            return {"pdb_id": pdb_id, "success": False, "error": f"download: {exc}"}

    # Auto-detect ligand if CCD not found
    ligand_resname = ccd
    if not auto_detect_ligand_resname(holo_pdb):
        logger.warning(f"[{pdb_id}] No ligand detected; skipping")
        return {"pdb_id": pdb_id, "success": False, "error": "No ligand detected"}

    # Run redocking
    try:
        result = run_redocking_validation(
            holo_pdb,
            ligand_resname=ligand_resname,
            exhaustiveness=item["exhaustiveness"],
            n_poses=item["n_poses"],
            seed=item["seed"],
            output_dir=outdir,
        )
    except ValidationError as exc:
        logger.warning(f"[{pdb_id}] Redocking failed: {exc}")
        return {"pdb_id": pdb_id, "success": False, "error": str(exc)}
    except Exception as exc:
        logger.error(f"[{pdb_id}] Unexpected error: {exc}")
        return {"pdb_id": pdb_id, "success": False, "error": str(exc)}

    # PoseBusters validation on best pose
    pb_result = {"available": False, "pass": None}
    best_pose = result.get("best_pose")
    if best_pose and os.path.isfile(best_pose):
        try:
            pb_result = validate_pose_with_posebusters(best_pose, holo_pdb)
        except Exception as exc:
            logger.warning(f"[{pdb_id}] PoseBusters validation failed: {exc}")

    return {
        "pdb_id": pdb_id,
        "success": result.get("success", False),
        "rmsd": result.get("rmsd"),
        "best_affinity": result.get("best_affinity"),
        "posebusters_pass": pb_result.get("pass"),
        "posebusters_available": pb_result.get("available"),
    }
