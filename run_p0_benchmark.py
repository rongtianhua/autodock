#!/usr/bin/env python3
"""P0: Re-run full 20-target benchmark with fixed best_rmsd fallback."""

import sys
sys.path.insert(0, "/Users/tianhuarong/Molecular_Docking")

from autodock.benchmark import run_redocking_benchmark

summary = run_redocking_benchmark(
    output_dir="./benchmark_results_p0_fixed",
    exhaustiveness=32,
    n_poses=20,
    seed=42,
    n_workers=1,  # sequential for stability
    auto_exhaustiveness=True,
    top_n_check=3,
)

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
for k, v in summary.items():
    if k not in ("per_target", "by_family"):
        print(f"  {k}: {v}")
print("=" * 60)
