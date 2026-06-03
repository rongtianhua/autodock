# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `autodock/cache.py` — Parameter-sensitive disk cache for receptor/ligand/pocket preparation.
  Three cache classes (`ReceptorCache`, `LigandCache`, `PocketCache`) using SHA-256 content
  hashing + JSON-serialized parameters for cache keys. Atomic writes via temp-file + rename
  for concurrency safety. Integrated into `prepare_receptor()`, `prepare_ligand()`,
  `find_top_pockets()`, `workflow.py`, and all CLI commands. Default cache directory:
  `~/.autodock/cache/{receptors,ligands,pockets}/`.
- `autodock/workflow.py` — `run_docking_workflow()` single-call entry point.
  Orchestrates: receptor acquisition (PDB/AlphaFold/file) → preparation →
  pocket detection → ligand prep → multi-pocket docking → post-processing.
  CLI: ``python -m autodock.workflow``.
- `render_interactions_ligplot()` — pure Python LigPlot+ v4.0 compatible
  renderer.  EPSF-3.0 vector output + Ghostscript 300 DPI PNG.
  Green dotted H-bonds, brick-red spoked-arc hydrophobic contacts,
  energy-minimized residue layout (1000-iter, prm-compatible).
- `prepare_receptor(output_pdb=...)` — saves filtered PDB for downstream
  PLIP/PyMOL/PoseBusters (single CIF→PDB conversion point).
- `find_top_pockets()` disk cache — MD5-hash based, `~/.cache/autodock/pockets/`.
- Memory-aware worker scaling in `dock_ligand_multi_conformer()` —
  uses ``psutil`` to cap workers by RAM (~1.5 GB/worker).
- Multi-pose Vinardo consensus scoring — all 20 poses re-scored.
- NaN/inf guards in ensemble statistics + Kabsch RMSD.
- File corruption detection in `validate_pdbqt_file()`.
- MIT `LICENSE` file, `strip_model_headers()` utility.
- `fix_protonation` parameter to `prepare_receptor()` — PDB2PQR+PROPKA
  active protonation correction (Option B).  Inserted between `reduce`
  and OpenMM: runs PDB2PQR with PROPKA pKa prediction, applies corrected
  protonation states, re-adds hydrogens, outputs corrected PDB.  Falls
  back gracefully to `reduce` output if PDB2PQR is unavailable.
- OpenBabel PDB normalization step after PDB2PQR: normalises atom/residue
  naming from PDB2PQR's AMBER-style naming to standard PDB conventions,
  improving Meeko Polymer parse reliability.  Runs only when
  `fix_protonation=True` and `obabel` is available.
- `tmp_pdb2pqr` tracked in temp-file cleanup list for leak safety.
- NaN/inf guard in ensemble statistics (`dock_ligand()` repeat summary): now
  filters with `np.isfinite` before computing mean/std/CV.
- NaN/inf guard in Kabsch RMSD (`validation.py`): raises `ValueError` instead
  of silently propagating NaN.
- File corruption detection in `validate_pdbqt_file()`: checks minimum file
  size (50 B) and missing `END`/`ENDMDL` records with warning.
- `_pdb2pqr_protonate()` helper in `minimization.py` for future
  protonation-consistent minimisation.
- Multi-pose Vinardo consensus scoring in `dock_ligand()`: all 20 poses
  are re-scored with Vinardo; if Vinardo ranks a pose different from
  Vina's #1, a scoring-bias warning is logged with the specific
  pose index and score delta.

### Changed
- P2Rank pocket filter strategy: **removed hard probability cutoff**.
  All top-10 P2Rank pockets now enter fpocket cross-validation regardless
  of score.  The old threshold (`_P2RANK_PROB_THRESHOLD=0.3`) was redundant
  with the existing top-10 rank limit + fpocket verification, and silently
  discarded ~15% of valid pockets (Krivák & Hoksza 2018 Table 3: Top-10
  recall ~90% vs threshold-limited ~75%).  The constant is retained for
  optional ultra-conservative mode but no longer acts as a skip filter.
- `_auto_exhaustiveness()` minimum floor raised from 4 → 16 (Eberhardt et al.
  2021, JCIM): prevents unreliable docking for large ligands.
  Thresholds documented with PDBbind size-distribution references.
- PoseBusters `_EXCLUDED_FROM_PASS` items now annotated with full scientific
  justification and literature references for each exclusion.
- `set_log_level()` now raises `ValueError` for unrecognised level strings
  instead of silently falling back to INFO.

### Changed
- **Architecture boundary**: `pipeline.py` renamed to `post_dock_pipeline.py`.
  `workflow.py` is now a pure orchestrator — all 2D/3D rendering delegated to
  `post_process_docking()` in `post_dock_pipeline.py`. Eliminates duplicate
  rendering logic between workflow and post-processing pipeline.

### Fixed
- `test_cli.py` mock assertions updated for `cache_dir` kwarg compatibility:
  switched from exact call-signature matching to semantic assertions on
  positional args and key kwargs.
- Replaced `yourorg` placeholders in `README.md` and `pyproject.toml` with
  actual repository owner.
- Added missing type hints to `_minimize_ligand_only()` and
  `_minimize_complex()` in `minimization.py`.
- Circular import in `dock_ligand()`: `from autodock.docking import
  dock_ligand_multi_conformer` replaced with direct call (same module).
- 11 bare `except Exception` instances in `md_simulation.py` narrowed to
  specific exception types (`ValueError`, `RuntimeError`, `TypeError`, etc.)
  so `MemoryError` and `KeyboardInterrupt` propagate correctly.
- Duplicate PDBQT MODEL/ENDMDL stripping logic consolidated into
  `strip_model_headers()` utility.

## [1.0.0] — 2025-05-29

### Added
- End-to-end molecular docking pipeline with AutoDock Vina integration.
- Receptor preparation: PDBQT generation, pocket detection (P2Rank / fpocket),
  structure repair with PDBFixer.
- Ligand preparation: SMILES → 3D conformer, SDF handling, protonation state
  enumeration at configurable pH.
- Pose clustering with Kabsch alignment and fallback coordinate-based RMSD.
- Interaction analysis via PLIP and ProLIF (hydrogen bonds, π-stacking,
  hydrophobic contacts, salt bridges).
- 2-D interaction diagram generation (LigPlot-style).
- Pose validation with PoseBusters chemical-validity checks.
- OpenMM molecular dynamics stability simulation (implicit and explicit solvent).
- OpenFF energy minimization for docked poses.
- Redocking benchmark suite against 20 diverse PDB targets.
- Virtual screening and batch docking with parallel execution.
- Ensemble docking (multiple receptor conformations).
- AlphaFold structure fetching and preparation.
- PDF/Excel/CSV report generation.
- Command-line interface (`autodock`) with subcommands for docking, screening,
  benchmark, and analysis.
- Comprehensive test suite with pytest and 60%+ coverage.
- GitHub Actions CI for Python 3.10–3.12 on Ubuntu and macOS.

### Fixed
- Kabsch rotation matrix bug in `_rmsd_kabsch_mols()` (`R = Vt.T @ U.T` →
  `R = U @ Vt`).
- `ProcessPoolExecutor` spawn-context crash on macOS by enforcing
  `multiprocessing.get_context("spawn")`.
- `_has_nan_charges()` now checks for `inf` in addition to `NaN`.
- `run_redocking_benchmark()` correctly distinguishes `targets=[]` from
  `targets=None`.
- Flaky `test_fetchers.py::test_not_found` by mocking `pubchempy` module.
