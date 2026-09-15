---
name: run-autodock
description: Run the autodock molecular docking pipeline (AutoDock Vina based) for single-ligand docking, virtual screening, batch docking, ensemble docking, redocking benchmarks, and MD stability checks. Use when a task involves preparing receptors/ligands, detecting binding pockets, docking small molecules into proteins (PDB/AlphaFold), validating poses (PoseBusters/RMSD), analysing protein-ligand interactions (PLIP/ProLIF), or generating publication figures/reports from docking results. Triggers include "分子对接", "docking", "virtual screen", "对接", "Vina", "redocking", or any request to compute binding poses/affinities for a receptor-ligand pair.
---

# Run the autodock pipeline

End-to-end docking orchestrator at `/Users/tianhuarong/Molecular_Docking` (repo: `rongtianhua/autodock`). Maintainer docs: `AGENTS.md` (architecture/conventions), `METHODS.md` (scientific defaults, in Chinese). Read those only when modifying code — not needed for routine runs.

## Environment

```bash
conda activate autodock        # REQUIRED: vina, rdkit, meeko, openmm, openff, pymol, fpocket, p2rank
autodock status                # verify environment before any run
```

If `autodock` command is missing: `pip install -e ".[all]"` inside the repo. Env overrides: `P2RANK_HOME`, `PYMOL_EXE`.

## Quick start (recommended API)

```python
from autodock.workflow import run_docking_workflow

result = run_docking_workflow(
    receptor_id="6LU7",                    # PDB ID | UniProt ID | local file path
    ligand_smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    output_dir="./my_docking",
    minimize_pose=False,                   # see pitfalls before enabling
)
# Outputs under ./my_docking/: 01_structures/ 02_interactions/ 03_figures/ 04_reports/
```

CLI equivalent: `autodock run --receptor 6LU7 --ligand <SMILES> --outdir ./demo`

## Publication defaults (do not lower without stating it in the report)

| Parameter | Default | Note |
|---|---|---|
| exhaustiveness | 32 | |
| n_poses | 20 | |
| seed | 42 | deterministic; for statistical robustness run 8 fixed seeds e.g. `[42, 137, 271, 514, 1024, 1729, 31337, 99999]` and report trimmed mean ± SD |
| ph | 7.4 | |
| RMSD success threshold | 2.0 Å | `autodock.core.REDocking_RMSD_THRESHOLD` |

## Key behaviours an agent must know

- **Resume**: `resume=True` (default) skips completed steps via `workflow_state.json`. Delete the output dir (not the repo) to force a full rerun.
- **Cache**: receptor/ligand/pocket preparation is SHA-256 cached (content + params) under `~/.autodock/cache`. Cache is parameter-sensitive — changing ph/padding/seed invalidates automatically.
- **Pocket detection fallback chain**: P2Rank primary (top-10) → 5 Å fpocket cross-validation → Drug Score re-rank. If P2Rank finds nothing: DoGSite3 (network) → **fpocket-only offline fallback** (ranks by Drug Score; `pocket_source="fpocket"`, `p2rank_prob=None`). Raise only when all three fail — small/low-pLDDT/transmembrane receptors are the typical trigger.
- **Multichain receptors**: default extracts the first chain for multimeric PDBs. Functional sites at chain interfaces require `receptor_multichain_strategy="multichain"`.
- **Minimization** (`minimize_pose=True`): ligand-only by default (`include_receptor=False` — complex minimization needs a gap-free chain, AlphaFold transmembrane proteins WILL crash). Pass `ligand_sdf` for robustness; conjugated ligands (flavonoids, quinolines) are handled by an internal coordinate-assignment fallback. Minimization barely moves poses (restraint 10000 kJ/mol/nm²) — scores unchanged.
- **SDF ligands**: `ligand_source="file"` accepts an SDF/MOL path in `ligand_smiles`; it is auto-reused for minimization.
- **Covalent ligands**: `covalent_check=True` only *detects and warns* about warheads; docking treats them as reversible — do not trust results for covalent inhibitors.
- **Figures**: 3D scenes (complex/pocket/interaction) need PyMOL; 2D interaction diagram needs RDKit. All 3D PNGs have solid (non-transparent) backgrounds. `3d_complex.png` is the standalone whole-receptor view for user compositing. The interaction scene shows an 8 Å pocket window with side-chain sticks; dashed interaction lines are drawn per ligand atom and colored by type (cyan H-bond, orange hydrophobic, green π-π, purple π-cation, red salt bridge, …) with the distance labelled on the closest pair; interacting residues get prominent bold labels and surrounding pocket residues smaller dim ones. The 2D diagram clips connectors at label borders, reserves the bottom-right legend from label placement, and scales all legend/label geometry to the canvas; multi-ring ligands get light-grey fill on non-interacting aromatic rings. Verify figure quality visually before publication — do not ship unviewed PNGs.
- **Known weak spots** (from maintainer docs): MM-GBSA rescoring and OpenMM complex minimization degrade gracefully but are low-coverage; SDF stereochemistry is preserved as-is (no tautomer/stereo enumeration on input coords).

## Task routing

- Single pair: `run_docking_workflow` (above).
- Virtual screen (library file of SMILES): `autodock virtual-screen --receptor 6LU7 --library compounds.txt --workers -1`.
- Batch / ensemble / redocking benchmark: `python -m autodock.benchmark --outdir ./benchmark_results`; ensemble repeats with derived seeds via `dock_ensemble()`.
- Validation-only (PoseBusters, RMSD vs crystal): functions in `autodock.validation`.
- Interaction analysis only: `autodock.interactions` (PLIP primary, ProLIF cross-validation).

## Verifying a run succeeded

1. `result.errors` empty; `result.warnings` reviewed (minimization/clash/PoseBusters skips land here).
2. `04_reports/report.csv` exists and best affinity is not `None`.
3. `02_interactions/interactions.csv` non-empty for the top pocket (empty = pocket or mapping problem, not necessarily a failure — hydrophobic-only sites can yield few typed contacts).
4. For redocking: RMSD ≤ 2.0 Å vs crystal ligand via `compute_rmsd_to_crystal`.
5. Spot-check at least one 3D figure and the 2D diagram visually.

## Failure triage (in order)

1. `autodock status` — missing binary (vina/p2rank/fpocket/pymol) is the most common cause.
2. Read the ERROR/WARNING lines in the log — the pipeline logs full stderr from subprocesses.
3. P2Rank no-pocket raise → now has fpocket-only fallback (updated v1.1); if still raising, the structure likely lacks a detectable cavity — consider `ligand_pdb=` centering (gold standard) or a manual `--center/--box-size` rescue docking.
4. Vina timeout (600 s/pocket default) → raise `timeout`, or reduce pockets via `max_pockets`.
5. Never edit package source to work around a failure mid-project — record the issue and report it to the maintainer (see AGENTS.md §8 limitations).
