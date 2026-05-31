# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `seed` parameter to `run_md_stability()` for reproducible MD trajectories.
- `CONTRIBUTING.md` and `CHANGELOG.md` for publication readiness.

### Fixed
- Replaced `yourorg` placeholders in `README.md` and `pyproject.toml` with
  actual repository owner.
- Added missing type hints to `_minimize_ligand_only()` and
  `_minimize_complex()` in `minimization.py`.

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
