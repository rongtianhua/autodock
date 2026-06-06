#!/usr/bin/env python3
"""Parallel deep-sampling test for 6 sampling-failure targets.

Each target runs in its own subprocess so they can execute in parallel.
Uses HARD_TARGET_OVERRIDES already defined in benchmark.py:
  - exhaustiveness=64
  - auto_exhaustiveness=False
  - n_poses=50
  - timeout=1800s
"""

import sys
import os
import json
import multiprocessing
sys.path.insert(0, "/Users/tianhuarong/Molecular_Docking")

from autodock.benchmark import DEFAULT_BENCHMARK_TARGETS, _run_single_benchmark

SAMPLING_FAILURE_IDS = ["1B9S", "2BR1", "2HU4", "1H1P", "3ELJ", "4AQC"]
TARGETS_BY_ID = {t["pdb_id"]: t for t in DEFAULT_BENCHMARK_TARGETS}


def run_one(pdb_id: str) -> dict:
    target = TARGETS_BY_ID[pdb_id].copy()
    outdir = f"./benchmark_results_deep_sampling/{pdb_id}"
    os.makedirs(outdir, exist_ok=True)
    item = {
        "target": target,
        "output_dir": outdir,
        "exhaustiveness": 32,
        "n_poses": 20,
        "seed": 42,
        "skip_consensus": True,
        "minimize": False,
        "pocket_method": "crystal",
        "interaction_method": "plip",
        "auto_exhaustiveness": True,
        "timeout": 1800,
        "top_n_check": 3,
    }
    result = _run_single_benchmark(item)
    # Write per-target result
    with open(f"{outdir}/result.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    rmsd = result.get("rmsd")
    best = result.get("best_rmsd")
    top3 = result.get("top_n_best_rmsd")
    print(f"[{pdb_id}] DONE: top-1={rmsd:.2f}Å best={best:.2f}Å top-3={top3:.2f}Å")
    return result


if __name__ == "__main__":
    n_workers = min(6, max(1, os.cpu_count() or 1))
    print(f"Running deep-sampling benchmark for {len(SAMPLING_FAILURE_IDS)} targets with {n_workers} workers")
    print("Overrides: exhaustiveness=64, n_poses=50, timeout=1800s (from HARD_TARGET_OVERRIDES)\n")

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(run_one, SAMPLING_FAILURE_IDS)

    print("\n" + "=" * 60)
    print("DEEP SAMPLING RESULTS")
    print("=" * 60)
    for r in results:
        pdb = r["pdb_id"]
        rmsd = r.get("rmsd")
        best = r.get("best_rmsd")
        top3 = r.get("top_n_best_rmsd")
        print(f"  {pdb}: top-1={rmsd:.2f}Å best={best:.2f}Å top-3={top3:.2f}Å")
    print("=" * 60)
