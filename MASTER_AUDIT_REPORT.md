# Master Audit Report — `autodock` Molecular Docking Pipeline

**Audit date**: 2026-05-30  
**Auditors**: 5 parallel sub-agents (Dependencies, Platform, Code Quality, Scientific Correctness, Tests & CI)  
**Codebase**: ~6,930 LOC across 24 modules  
**Test suite**: 349 passed, 5 skipped, 53.88% coverage (fail-under=34)

---

## Executive Summary

The `autodock` pipeline is **functionally sound but has critical gaps in production hardening**, **scientific edge cases**, and **test coverage**. One critical bug (Kabsch RMSD) was found and fixed. Remaining issues span: macOS multiprocessing safety, temp file leaks, pH inconsistency, missing infinity checks, zero-coverage scientific modules, and a minimal CI matrix.

**Risk heat map** (by count):
| Severity | Code Quality | Scientific | Tests/CI | Platform | Dependencies | Total |
|----------|-------------|------------|----------|----------|--------------|-------|
| CRITICAL | 4 | 3 | 5 | 6 | 7 | **25** |
| HIGH | 6 | 5 | 7 | 9 | 8 | **35** |
| MEDIUM | 8 | 4 | 5 | 7 | 7 | **31** |
| LOW | 3 | 4 | 2 | 5 | 5 | **19** |

---

## Already Fixed

### ✅ Kabsch RMSD Bug (`clustering.py`)
- **Finding**: `_rmsd_kabsch_mols()` used `R = Vt.T @ U.T` instead of `R = U @ Vt`
- **Impact**: RMSD computation was mathematically incorrect; 90° rotation test gave 1.89 Å instead of ~0 Å
- **Fix**: Changed to `R = U @ Vt`, matching `validation.py` and `minimization.py`
- **Verification**: 349 tests passing, numerical test confirms 90° case now returns ~0 Å

---

## Critical Issues Remaining (Ordered by Impact × Effort)

### 1. Missing `mp_context="spawn"` in `virtual_screen()` / `batch_dock()` [Platform/CRITICAL]
- **File**: `autodock/docking.py`
- **Issue**: `virtual_screen()` and `batch_dock()` use bare `ProcessPoolExecutor()` without `mp_context="spawn"`. Other paths (`dock_ligand_multi_conformer()`, `_run_vina_dock()`) explicitly request spawn.
- **Impact**: On macOS, if global context is fork (or if user overrides), the Vina C++ extension (which holds the GIL) can deadlock or crash child processes.
- **Fix**: Add `mp_context="spawn"` to both `ProcessPoolExecutor` calls.

### 2. `prepare_receptor()` Temp-File Chain Leaks [Code Quality/CRITICAL]
- **File**: `autodock/preparation.py`
- **Issue**: ~500-line function creates `tmp_raw`, `tmp_fixer_in`, `tmp_fixed`, `tmp_reduced`, `tmp_min`; only `tmp_raw` is cleaned on failure. Others leak as orphan files.
- **Impact**: Disk space exhaustion during batch processing; sensitive data (PDB structures) left in `/tmp`.
- **Fix**: Wrap entire function in `tempfile.TemporaryDirectory()` sandbox.

### 3. `tempfile.mktemp()` Usage (Deprecated) [Code Quality/CRITICAL]
- **Files**: `autodock/rendering.py`, `autodock/preparation.py`
- **Issue**: `tempfile.mktemp()` is deprecated since Python 2.3; race condition between name creation and file open.
- **Impact**: Security vulnerability (CVE-classic temp race); potential file corruption.
- **Fix**: Replace with `tempfile.NamedTemporaryFile(delete=False)` or `mkstemp()`.

### 4. pH Inconsistency: Prep 7.4 vs Minimization 7.0 [Scientific/CRITICAL]
- **Files**: `autodock/preparation.py` (default pH 7.4), `autodock/minimization.py` (hardcoded pH 7.0)
- **Issue**: Receptor prepared at pH 7.4 but minimized at pH 7.0. Protonation states differ.
- **Impact**: Charge states, hydrogen bond networks, and binding energies are inconsistent between preparation and refinement.
- **Fix**: Unify to single pH constant (config default 7.4), pass through to minimization.

### 5. Missing `inf` Check in Gasteiger Charge Validation [Scientific/HIGH]
- **File**: `autodock/preparation.py`
- **Issue**: `_has_nan_charges()` only checks `c != c` (NaN), missing `math.isinf(c)`.
- **Impact**: Infinite charges from malformed input propagate silently, causing OpenMM crashes downstream.
- **Fix**: Add `math.isinf()` check.

### 6. `StructureCache` Non-Atomic Writes [Code Quality/HIGH]
- **File**: `autodock/fetchers.py`
- **Issue**: Two parallel fetches of same PDB ID can interleave writes, corrupting cache file.
- **Impact**: Corrupted PDB/mmCIF files in cache; subsequent reads crash parser.
- **Fix**: Write to temp file + atomic rename (`os.replace`).

### 7. `read_docking_results` JSON Validation [Code Quality/HIGH]
- **File**: `autodock/docking.py`
- **Issue**: Reconstructs `DockingResult` from raw JSON without schema validation; crashes on malformed/old result files.
- **Impact**: Unhandled `KeyError`/`TypeError` when loading legacy results.
- **Fix**: Add schema validation with graceful degradation.

### 8. Broad Exception Tuples [Code Quality/MEDIUM]
- **Files**: >15 locations across `preparation.py`, `docking.py`, etc.
- **Issue**: `(OSError, RuntimeError, ValueError, TypeError, ImportError)` copy-pasted; catches programming errors (`TypeError` from wrong arguments) silently.
- **Impact**: Bugs masked as "fallback succeeded"; silent data corruption.
- **Fix**: Narrow to specific expected exceptions per call site.

---

## High-Priority Test & CI Issues

### 9. Zero-Coverage Scientific Modules [Tests/CRITICAL]
| Module | Coverage | Missing Tests |
|--------|----------|---------------|
| `minimization.py` | **0%** | No `test_minimization.py` |
| `alphafold_tools.py` | **6%** | No `test_alphafold_tools.py` |
| `heatmap.py` | **9%** | No `test_heatmap.py` |
| `pipeline.py` | **46%** | No `test_pipeline.py` |

### 10. CI Matrix Minimal [Tests/CRITICAL]
- **File**: `.github/workflows/ci.yml`
- **Issue**: Single OS (`ubuntu-latest`), single Python (`3.12`), no timeout, no macOS runner.
- **Impact**: macOS-specific bugs (spawn context, Metal fallback, binary paths) invisible in CI.
- **Fix**: Expand to `os: [ubuntu-latest, macos-latest]`, `python: ["3.10", "3.11", "3.12"]`, add `timeout-minutes: 30`.

### 11. No Integration Tests for Real Vina Docking [Tests/CRITICAL]
- **File**: `autodock/tests/test_docking.py`
- **Issue**: All docking tests mock `vina.Vina`. Real subprocess path untested.
- **Impact**: Vina subprocess invocation, pose parsing, timeout handling regressions go undetected.
- **Fix**: Add `requires_vina` integration test with tiny receptor/ligand pair.

---

## Scientific Findings (Medium/Low Impact)

| Finding | File | Impact | Fix Effort |
|---------|------|--------|------------|
| Multi-conformer energy array loses Vina's 5 components | `docking.py` | Energy decomposition unavailable | Low |
| Exhaustiveness scaling too aggressive (>55 atoms → base//8) | `docking.py` | Under-sampling for large ligands | Low |
| No covalent inhibitor handling | N/A | Missing feature | High |
| SDF input skips tautomer/stereo enumeration | `preparation.py` | Incorrect stereochemistry | Medium |
| `extract_ligand_from_pdb()` keeps only largest fragment | `preparation.py` | Missing cofactors/ions | Medium |
| MD integrator not seeded | `md_simulation.py` | Non-reproducible trajectories | Low |
| Consensus scoring is unweighted median | `analysis.py` | Suboptimal ranking | Medium |
| Pocket detection aborts if P2Rank fails | `preparation.py` | No fpocket fallback | Low |
| Missing exotic elements in AD4→element map | `preparation.py` | Si, B, As mis-assigned | Low |
| PDB element fallback misassigns two-letter elements | `preparation.py` | CL→C, BR→B | Low |

---

## Recommended Fix Order

### Phase 1: Safety & Stability (Today)
1. Fix `ProcessPoolExecutor` spawn context
2. Fix `prepare_receptor()` temp-file leaks
3. Replace `tempfile.mktemp()` calls
4. Fix `StructureCache` atomic writes
5. Add `inf` check to charge validation

### Phase 2: Scientific Correctness (This week)
6. Unify pH default across prep/minimization
7. Add JSON schema validation for docking results
8. Narrow broad exception tuples
9. Fix exhaustiveness scaling

### Phase 3: Test Hardening (Next sprint)
10. Create `test_minimization.py` (even mocked)
11. Create `test_alphafold_tools.py`
12. Add CI matrix expansion (macOS + Python 3.10–3.12)
13. Add one real Vina integration test
14. Raise coverage fail-under to 50%

---

*Generated by multi-agent audit. Individual reports: DEPENDENCIES_AUDIT_REPORT.md, PLATFORM_AUDIT_REPORT.md, CODE_QUALITY_AUDIT_REPORT.md, SCIENTIFIC_AUDIT_REPORT.md, TESTS_CI_AUDIT_REPORT.md*
