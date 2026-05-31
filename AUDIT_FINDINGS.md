# Comprehensive Code & Scientific Quality Audit: `rongtianhua/autodock`

**Date:** 2026-05-30  
**Scope:** 8 core modules (~8,500 LOC), project metadata, documentation  
**Auditor:** Automated codebase exploration specialist  
**Methodology:** Static analysis, pattern search, manual line-by-line review of key functions, cross-reference with pyproject.toml and README.

---

## 1. Executive Summary

The `autodock` codebase is a **well-architected, publication-oriented** molecular-docking pipeline with deliberate robustness patterns (extensive fallback chains, `spawn`-based multiprocessing, timeout wrappers) and scientifically defensible defaults. However, it exhibits several **publication-readiness blockers** around metadata, documentation, dead references, and defensive-coding hygiene that should be addressed before a v1.0 release or academic submission.

| Severity | Count | Themes |
|----------|-------|--------|
| 🔴 **Critical** | 5 | Missing LICENSE, dead README links, unvalidated user input, unhandled temp-file leaks, bare `except Exception` in MD analysis |
| 🟠 **Major** | 14 | Missing docstrings, hardcoded paths/constants, duplicate code blocks, no type-stub marker, README badge/URL placeholders |
| 🟡 **Minor** | 12 | Inconsistent logging, minor performance hot paths, non-canonical constant usage, emoji in CLI output |

**Overall verdict:** Code quality is **B+** (engineering), **C+** (publication hygiene). Fix the Critical issues and ~50% of Major issues before tagging v1.0.0.

---

## 2. Critical Issues (🔴 Fix Before Release)

### C1. Missing `LICENSE` file
- **Location:** Repository root  
- **Detail:** `pyproject.toml` declares `license = {text = "MIT"}`, but **no `LICENSE` file exists on disk**. GitHub will not auto-detect the license, and downstream packagers (conda, PyPI) may flag this.  
- **Fix:** Add a standard MIT `LICENSE` file.

### C2. Dead documentation references in README
- **Location:** `README.md`  
- **Detail:** README mentions `METHODS.md` and `docs/tutorials/`. Neither exists (`METHODS.md` absent; `docs/tutorials/` directory absent).  
- **Fix:** Either create these files or remove the references.

### C3. Unvalidated log-level string → silent fallback
- **Location:** `autodock/core.py:124-132` (`set_log_level`)  
- **Detail:** `getattr(logging, level.upper(), logging.INFO)` silently falls back to `INFO` for garbage input (e.g. `"DEBBUG"`). User receives no feedback that their requested level was ignored.  
- **Fix:** Raise `ValueError` for unrecognized level strings.

### C4. Temp-file leaks on exception paths
- **Location:** `autodock/docking.py:101`, `212`, `304`, `459`, `538`; `autodock/minimization.py:134`, `435`; `autodock/preparation.py:2687`, `2969`; `autodock/validation.py:72`, `465`  
- **Detail:** Many functions create `tempfile.mkstemp` / `NamedTemporaryFile(delete=False)` but **do not wrap creation in a `try/finally` block** to guarantee `os.unlink` on failure. In long-running batch pipelines this will exhaust disk space.  
- **Fix:** Use `contextlib.ExitStack` or `try/finally` for every `delete=False` temp file.

### C5. Bare `except Exception` in MD trajectory analysis
- **Location:** `autodock/md_simulation.py:443-574` (11 instances)  
- **Detail:** Each analysis sub-step (RMSD, RMSF, PCA, clustering, H-bonds, contact map) is wrapped in `except Exception as exc: logger.warning(...)`. This **masks `MemoryError`, `KeyboardInterrupt`, and library API changes**, making silent data loss very likely.  
- **Fix:** Catch specific exceptions (`ImportError`, `ValueError`, `AnalysisError`) per sub-step. Let `MemoryError` and `KeyboardInterrupt` propagate.

---

## 3. Major Issues (🟠 Should Fix)

### M1. Missing public-function docstrings
- **Location:** `autodock/core.py` (`n_hbonds`, `n_pi_pi`, `method_label`, `interaction_summary`, `to_dict`, `to_dataframe_row`, `format`, `deep_merge`), `autodock/cli.py` (`build_parser`, `main`), `autodock/validation_params.py` (all validators)  
- **Detail:** ~15 public API surface functions lack docstrings. This directly hurts Sphinx/API docs and reviewer comprehension.  
- **Fix:** Add one-line Google-style docstrings.

### M2. Missing return-type annotations
- **Location:** `autodock/docking.py`, `preparation.py`, `validation.py` (internal worker functions)  
- **Detail:** Public APIs are well-annotated, but many internal helpers (e.g. `_count_pdbqt_atoms`, `_parse_model_pose`, `_classify_ligand_complexity`, `_generate_multi_conformers`) lack `->` annotations. `mypy` will not type-check these.  
- **Fix:** Run `mypy --strict` and add annotations.

### M3. Hardcoded macOS PyMOL path
- **Location:** `autodock/core.py:228`  
- **Detail:** `find_pymol()` hardcodes `/Applications/PyMOL.app/Contents/MacOS/PyMOL`. On Linux/Windows this is dead code, but it clutters cross-platform portability.  
- **Fix:** Move to a `PATHS` config dict or environment variable.

### M4. Magic numbers without named constants
- **Location:** `autodock/docking.py:59-64` (`_auto_exhaustiveness`)  
- **Detail:** Thresholds `55`, `45`, `35` and divisors `8`, `4`, `2` are undocumented magic numbers. The comment on line 399 explains the *intent* but the constants themselves are not extracted.  
- **Fix:**
  ```python
  HEAVY_ATOM_LARGE = 55
  HEAVY_ATOM_MEDIUM = 45
  HEAVY_ATOM_SMALL = 35
  ```

### M5. Duplicate MODEL/ENDMDL stripping logic
- **Location:** `autodock/docking.py:525-530`, `540-544`, `819-823`, and `preparation.py`  
- **Detail:** The same 5-line block for stripping multi-model PDBQT headers appears at least 3×.  
- **Fix:** Extract `_strip_model_headers(pdbqt_text: str) -> str` utility.

### M6. Duplicate residue-renaming logic
- **Location:** `autodock/preparation.py:1477`, `1621`, `1897` (and likely more)  
- **Detail:** HETATM→ATOM renaming and residue-name normalization is copy-pasted across pocket-detection, receptor prep, and ligand prep paths.  
- **Fix:** Centralize in `autodock/utils.py`.

### M7. Hardcoded RMSD threshold in benchmark report string
- **Location:** `autodock/benchmark.py:299`, `329`, `330`  
- **Detail:** The literal `2.0` is used instead of the imported `REDocking_RMSD_THRESHOLD`. If the constant is changed, the log output becomes a lie.  
- **Fix:** Use f-string with `{REDocking_RMSD_THRESHOLD}`.

### M8. Hardcoded default in multi-conformer docking
- **Location:** `autodock/docking.py:460` (`dock_ligand_multi_conformer`)  
- **Detail:** `n_conformers: int = 10` is hardcoded; no constant or config override.  
- **Fix:** Add `DEFAULT_MULTI_CONFORMERS = 10` to `core.py` constants.

### M9. Hardcoded exhaustiveness in virtual screening
- **Location:** `autodock/docking.py:388`  
- **Detail:** `exhaustiveness: int = 16` instead of `VINA_DEFAULT_EXHAUSTIVENESS` (32). The docstring says "publication standard: 32" but the default contradicts it.  
- **Fix:** Use the constant.

### M10. No `py.typed` marker
- **Location:** Repository root / `autodock/`  
- **Detail:** `pyproject.toml` enables `disallow_untyped_defs = true`, but without a `py.typed` file, downstream `mypy` users will **ignore all type hints** in this package.  
- **Fix:** `touch autodock/py.typed` and ensure it is included in `package-data`.

### M11. Missing `CHANGELOG.md` and `CONTRIBUTING.md`
- **Location:** Repository root  
- **Detail:** A v1.0.0 release without a changelog or contribution guide is unprofessional for an open-source scientific tool.  
- **Fix:** Add minimal `CHANGELOG.md` (even if just v1.0.0) and `CONTRIBUTING.md`.

### M12. README badge/URL placeholders
- **Location:** `README.md`  
- **Detail:** Badges point to `yourorg/autodock`; citation block uses `yourorg` URL. The actual org is `rongtianhua`.  
- **Fix:** Replace all `yourorg` with `rongtianhua`.

### M13. `_get_vina_seed` returns deterministic seed for `None`
- **Location:** `autodock/core.py:143-150`  
- **Detail:** When `seed=None`, the function returns `DEFAULT_SEED` (42). This is documented but **surprising**; most users expect `None` → random seed.  
- **Fix:** Either change behavior or add a loud warning log when deterministic fallback is used.

### M14. `safe_subprocess` truncates stderr to 300 chars
- **Location:** `autodock/core.py:685-690`  
- **Detail:** `stderr = stderr[-300:]` silently drops the head of long error messages. Vina often emits multi-line parameter warnings before the actual error.  
- **Fix:** Log full stderr to DEBUG before truncating, or raise the limit to 2000 for `DockingCalculationError`.

---

## 4. Minor Issues (🟡 Polish)

1. **Inconsistent `except` tuple width** — `benchmark.py:592` catches 6 exception types in one tuple; this is overly broad. Prefer wrapping the call site in smaller helpers.
2. **`_NON_LIGAND_HETS` casing inconsistency** — `benchmark.py` uses `"Fru"` (sentence case) alongside `"RIB"` (uppercase). Standardize to uppercase PDB residue names.
3. **Local imports inside hot loops** — `docking.py` imports `tempfile` inside `_score_pose_with_sf` (minor, but unnecessary).
4. **Emoji in CLI output** — `cli.py` uses `⚠️`, `🔍`, etc. These break on Windows CMD and some CI logs. Recommend ASCII fallbacks.
5. **VDW radii table lacks inline citation** — `validation.py` VdW dict (Bondi approximations) is scientifically sound but has no inline reference. Add a `# Bondi 1964` comment.
6. **`DockingResult.__post_init__` missing length validation** — `center` and `box_size` are coerced to tuples but not checked for length-3. A malformed input will propagate silently.
7. **No `__all__` in `__init__.py`** — Public API surface is not explicitly exported; `from autodock import *` imports everything.
8. **`__import__("autodock").__version__` in `core.py`** — Brittle circular-import risk. Use `importlib.metadata.version("autodock")` instead.
9. **Performance: `compute_clash_score` is O(N×M)** — No spatial indexing (kd-tree or grid). Acceptable for single poses but slow for large ensembles.
10. **`set_log_level` only adjusts StreamHandlers** — If a user adds a custom FileHandler, its level is untouched. Document this behavior.
11. **README mentions `ruff`, `black`, `mypy` but CI only runs pytest** — `.github/workflows/` should be checked to ensure linting is actually enforced in CI.
12. **`md_simulation.py` deprecated param conversion** — `n_steps` → `ns` conversion uses hardcoded `500000` steps = 10 ns at 2 fs. This should be a named constant.

---

## 5. Per-Module Health Score

| Module | Lines | Grade | Key Concern |
|--------|-------|-------|-------------|
| `core.py` | 815 | A- | Minor docstring gaps, `__import__` anti-pattern |
| `docking.py` | 1,424 | B+ | Duplicate code blocks, magic numbers, temp-file leaks |
| `preparation.py` | 3,202 | B | Too large; duplicated residue-renaming; needs refactor into sub-modules |
| `validation.py` | 827 | A- | VdW table citation, O(N×M) clash score |
| `benchmark.py` | 751 | B+ | Hardcoded threshold literals, broad except tuple |
| `cli.py` | 1,215 | B | Emoji encoding, dead `--help` text if docs missing |
| `minimization.py` | 505 | B+ | Temp-file leaks, Kabsch alignment is correct but fragile |
| `md_simulation.py` | 578 | C+ | **11 bare `except Exception` blocks** — highest risk in the codebase |

---

## 6. Positive Observations (Keep Doing)

- ✅ **Multiprocessing safety:** Explicit `get_context("spawn")` everywhere prevents RDKit/Vina fork-safety crashes.
- ✅ **Timeout architecture:** `Process` + `Queue` + `terminate/kill` in `_run_vina_dock` is a robust pattern for C++ tools that can hang.
- ✅ **Scientific constants are literature-backed:** Vina defaults (32/20/3.0), RMSD 2.0 Å, clash 1.2 Å, pocket consensus 5.0 Å all align with community standards.
- ✅ **Graceful degradation:** Extensive fallback chains (Meeko → Open Babel; MMFF94 → UFF; topology-aware → coordinate-based RMSD) make the pipeline resilient in heterogeneous environments.
- ✅ **AlphaFold detection heuristic:** The B-factor + mean > 45 check is a pragmatic, fast way to flag predicted models for relaxation.
- ✅ **Type-hint coverage on public APIs:** Modern `str | None`, `tuple[float, ...]` syntax; `from __future__ import annotations` used consistently.
- ✅ **`HARD_TARGET_OVERRIDES`:** The per-PDB override dict with inline `_note` fields is excellent scientific documentation and makes the benchmark reproducible.

---

## 7. Recommended Action Plan (Priority Order)

1. **Immediate (today):** Add `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`; fix README `yourorg` → `rongtianhua`; remove dead `METHODS.md` / `docs/tutorials` references.
2. **Week 1:** Fix all `tempfile` leaks with `try/finally` or `ExitStack`; replace 11 bare `except Exception` in `md_simulation.py` with specific exceptions.
3. **Week 2:** Extract duplicate MODEL stripping and residue-renaming into `utils.py`; add named constants for `_auto_exhaustiveness` thresholds; replace hardcoded `2.0` in benchmark with `REDocking_RMSD_THRESHOLD`.
4. **Week 3:** Add missing docstrings; run `mypy --strict` and fill return-type annotations; add `py.typed`.
5. **Backlog:** Refactor `preparation.py` into `receptor_prep.py`, `ligand_prep.py`, `pocket_detection.py` sub-modules; add spatial indexing to clash detection; validate `DockingResult` field lengths in `__post_init__`.

---

*End of Audit.*
