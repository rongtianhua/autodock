# autodock Pipeline — Scientific Audit Report

**Scope:** 10 core modules (~6,930 LOC)  
**Date:** 2026-05-30  
**Dimensions:** Algorithm Correctness · Numerical Stability · Physical Units · Chemical Correctness · Reproducibility · Edge Cases · File Format Robustness

---

## Executive Summary

The `autodock` pipeline is a well-engineered, production-grade molecular docking framework with robust error handling, comprehensive fallback chains, and good documentation. **Two issues warrant immediate attention:** (1) a **Kabsch RMSD bug** in `clustering.py` that can return inflated RMSD values when the RDKit `GetBestRMS` fallback is triggered, and (2) a **pH inconsistency** between receptor preparation (default 7.4) and complex minimization (hardcoded 7.0). All remaining findings are moderate-to-low severity and represent opportunities for refinement.

**Notable strengths:** OpenMM unit handling is correct throughout (Quantity objects auto-convert); Vina seeding is deterministic and comprehensive; PROPKA thresholds are literature-based; metal/cofactor retention is thorough; PDB/mmCIF parsing is robust.

---

## 1. Algorithm Correctness

### 🔴 HIGH — Kabsch RMSD rotation bug in clustering fallback (`clustering.py`)

**Location:** `clustering.py`, function `_rmsd_kabsch_mols()` (lines 129–133)  
**Issue:** The Kabsch rotation matrix uses the **pre-multiplication** solution (`R = V U^T`) for a **post-multiplication** objective (`||P @ R - Q||²`). The correct rotation for the post-multiplication objective is `R = U V^T`.

**Current (wrong):**
```python
H = Pc.T @ Qc          # covariance matrix
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T         # = V @ U^T  ← pre-multiplication solution
Pr = Pc @ R            # post-multiplication objective
```

**Correct (as verified numerically):**
```python
H = Pc.T @ Qc
U, S, Vt = np.linalg.svd(H)
R = U @ Vt             # = U @ V^T  ← post-multiplication solution
Pr = Pc @ R
```

**Numerical verification:** With a 90° z-rotation test case, the current code gives RMSD = **1.89 Å**; the corrected formula gives RMSD = **~0 Å**.

**Impact:** `_rmsd_kabsch_mols()` is only called when RDKit's `GetBestRMS()` fails (e.g., different atom orderings or topology mismatches). In those fallback cases, RMSD values are systematically **inflated**, causing:
- Poses that should cluster together to be split into separate clusters
- False negatives in redocking validation when `compute_rmsd_to_crystal()` falls back to coordinate-based matching

**Fix:** Change line 130 from `R = Vt.T @ U.T` to `R = U @ Vt`.

> **Note:** `validation.py`'s `_kabsch_rmsd()` (line 290) and `minimization.py`'s Kabsch alignment are **correct** — the initial suspicion of a bug in `validation.py` was incorrect. `validation.py` uses `R = U @ Vt` (correct for post-multiplication), and `minimization.py` computes `H = Q^T @ P` (transposed) which makes `R = V @ U^T` the correct solution.

---

### 🟡 MEDIUM — Multi-conformer energy array shape hack (`docking.py`)

**Location:** `docking.py`, `dock_ligand_multi_conformer()`  
**Issue:** Vina returns 5 energy components per pose. The multi-conformer path pools poses and synthesizes a fake array:
```python
all_energies = np.array([[e, 0.0, 0.0, 0.0, 0.0] for e, _ in all_poses_pool])
```
This loses decomposed energy terms (inter, intra, torsions, etc.).

**Impact:** Clustering still works (uses column 0), but downstream analysis expecting valid Vina energy breakdowns will see zeros.

**Fix:** Propagate real energy components or document the limitation.

---

### 🟡 MEDIUM — Exhaustiveness scaling may be too aggressive for large ligands

**Location:** `docking.py`, `_auto_exhaustiveness()`  
**Issue:** For >55 heavy atoms, exhaustiveness drops to `base // 8` (min 4). A base of 32 becomes 4 — an 8× search reduction.

**Impact:** Large ligands may receive insufficient sampling.

**Recommendation:** Raise floor to 8 or make configurable.

---

### 🟢 LOW — Consensus scoring only uses median

**Location:** `docking.py`, `_consensus_score()`  
**Issue:** Simple median aggregator with no weighting or outlier rejection.

---

### 🟢 LOW — Pocket detection aborts if P2Rank finds no pockets

**Location:** `preparation.py`, `find_top_pockets()`  
**Issue:** Raises `PreparationError` if P2Rank fails, even if fpocket might succeed.

---

## 2. Numerical Stability

### 🟡 MEDIUM — Missing `inf` check in Gasteiger charge validation

**Location:** `preparation.py`, `_has_nan_charges()`  
**Issue:** Only checks `c != c` (NaN). Gasteiger charges can also produce `inf` on pathological molecules.

**Fix:**
```python
import math
if not math.isfinite(c):
    return True
```

---

### 🟢 LOW — Eigenvalue clipping in pocket shape descriptors

**Location:** `preparation.py`, `_compute_pocket_shape_descriptors()`  
**Issue:** `eigvals = np.maximum(eigvals, 1e-12)` is arbitrary. Flat pockets get artificial sphericity.

**Impact:** Informational only — not used for scoring.

---

### 🟢 LOW — Ensemble CV near-zero energy_mean

**Location:** `docking.py`, `dock_ensemble()`  
**Observation:** `energy_cv = abs(energy_std / energy_mean) if energy_mean != 0 else 0.0` is guarded, but when `energy_mean ≈ 0` (not exactly), CV explodes.

---

## 3. Physical Units

### ✅ CORRECT — OpenMM unit conversions are handled properly

**Observation:** After detailed review, OpenMM's `unit.Quantity` objects correctly auto-convert when passed to `addParticle()`, `addGlobalParameter()`, and other API methods. All unit conversions (Å→nm, kcal→kJ) are handled by OpenMM's internal unit system.

- `preparation.py`: 10 kcal/mol/Å² → ~4184 kJ/mol/nm² ✅
- `minimization.py`: 10000 kJ/mol/nm² for ligand, 500 for receptor ✅
- `md_simulation.py`: `local_minimize_radius * 0.1` (Å→nm) ✅
- `alphafold_tools.py`: 5 kcal/mol/Å² Cα restraints ✅

**The "unit mismatch" claim from an earlier draft of this report was incorrect and is retracted.**

---

### 🟢 LOW — `box_size` rounds to nearest 0.5 Å

**Location:** `preparation.py`, `_compute_box_size()`  
**Observation:** `rounded = round(v * 2) / 2` may slightly alter exact pocket dimensions.

---

## 4. Chemical Correctness

### 🟡 MEDIUM — pH inconsistency between receptor prep and complex minimization

**Location:** `preparation.py` (`prepare_receptor`, default `ph=7.4`) vs. `minimization.py` (`_minimize_complex`, hardcoded `ph=7.0`)  
**Issue:** Receptor preparation protonates at pH 7.4, but complex minimization re-protonates at pH 7.0. Titratable residues near the binding pocket (HIS, ASP, GLU) may change protonation state during minimization, altering the optimized pose.

**Fix:** Pass `ph` through to `_minimize_complex()` or store the preparation pH in metadata.

---

### 🟡 MEDIUM — No covalent inhibitor handling

**Location:** Throughout ligand prep and docking  
**Issue:** Covalent warheads are treated as reversible binders. No reactive-group detection, no covalent scoring, no warhead-specific preparation.

**Recommendation:** Document limitation or add a `--covalent` flag.

---

### 🟡 MEDIUM — SDF input skips tautomer/protonation enumeration

**Location:** `preparation.py`, `prepare_ligand_from_sdf()`  
**Issue:** Hardcodes `molscrub_states=False` and `enumerate_stereo=False`. SDFs with suboptimal tautomers or unspecified stereochemistry are used as-is.

---

### 🟢 LOW — Gasteiger charge limitations

**Observation:** Fast but less accurate than QM/RESP charges. Minor impact — Vina scoring is relatively insensitive.

---

### 🟢 LOW — Missing AD4 atom types for exotic elements

**Location:** `utils.py`, `_AD4_ELEMENT_MAP`  
**Observation:** Missing Si, B, As, etc. Would pass through unchanged and may break RDKit parsing.

---

## 5. Reproducibility

### 🟢 LOW — MD trajectories are not seeded

**Location:** `md_simulation.py`, `run_md_stability()`  
**Issue:** The Langevin integrator doesn't set a random seed:
```python
integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
```
**Impact:** MD trajectories are non-reproducible even with identical inputs.

**Fix:** `integrator.setRandomNumberSeed(seed)`

---

### 🟢 LOW — OpenMM platform-dependent numerical differences

**Location:** `md_simulation.py`  
**Issue:** Auto-selects Metal → OpenCL → CUDA → CPU. Different platforms may give slightly different trajectories.

**Recommendation:** Log the selected platform.

---

### 🟢 LOW — Virtual screening seed overflow

**Location:** `docking.py`, `virtual_screen()`  
**Observation:** Per-compound seeds are `base_seed + idx`. For libraries >2B compounds, this overflows the 32-bit Vina seed limit. `validate_seed()` caps individual seeds but not the sum.

---

## 6. Edge Cases

### 🟡 MEDIUM — Single-conformer forced for very large ligands

**Location:** `preparation.py`, `prepare_ligand_adaptive()`  
**Issue:** >50 heavy atoms → forced single-conformer. May miss binding modes for flexible large molecules.

---

### 🟡 MEDIUM — `extract_ligand_from_pdb` discards multi-fragment ligands

**Location:** `utils.py`, `extract_ligand_from_pdb()`  
**Issue:** Keeps only the largest fragment. Silently discards covalent adducts, cofactor-inhibitor complexes, or salt fragments.

---

### 🟢 LOW — Meeko `allow_bad_res=True` silently strips unknown residues

**Location:** `preparation.py`, `prepare_receptor()`  
**Issue:** Could remove phosphorylated residues, glycosylation, or other PTMs without explicit logging.

---

### 🟢 LOW — PoseBusters eval uses CCD code instead of auto-detected resname

**Location:** `posebusters_eval.py`, `_run_single_posebuster()`  
**Issue:** Auto-detection is checked but the original CCD code is passed to redocking regardless.

---

## 7. File Format Robustness

### 🟢 LOW — PDB element fallback may misassign two-letter elements

**Location:** `utils.py`, `_read_pdb_atoms_impl()`  
**Code:**
```python
"element": safe_pdb_slice(line, 76, 78) or safe_pdb_slice(line, 12, 14, "C")[0],
```
Fallback takes first character of atom name. "CL" → "C", "BR" → "B", "FE" → "F".

**Fix:** Handle two-character names in fallback.

---

### 🟢 LOW — mmCIF loses biological assembly information

**Location:** `utils.py`, `cif_to_pdb_string()`  
**Observation:** `gemmi.make_pdb_string()` converts the asymmetric unit. Functional multimers may be lost.

---

### 🟢 LOW — MODEL/ENDMDL nesting not validated

**Location:** `clustering.py`, `_parse_pose_to_mol()`  
**Observation:** Strips MODEL/ENDMDL tags without validating balance.

---

## Appendix: Correctness Verification Checklist

| Component | Status | Notes |
|-----------|--------|-------|
| Vina parameter validation | ✅ | `validation_params.py` covers bounds |
| PDB parsing | ✅ | `safe_pdb_slice` handles truncation |
| mmCIF parsing | ✅ | gemmi-based, robust |
| PDBQT sanitization | ✅ | Comprehensive AD4→element mapping |
| Kabsch alignment (validation.py) | ✅ | Correct: `R = U @ Vt` |
| Kabsch alignment (clustering.py) | 🔴 | **Bug: `R = V @ U^T` should be `R = U @ V^T`** |
| Kabsch alignment (minimization.py) | ✅ | Correct via transposed H |
| OpenMM unit conversions | ✅ | Quantity objects auto-convert |
| PROPKA thresholds | ✅ | Literature-based |
| Disulfide handling | ✅ | Parses SSBOND, strips HG |
| Metal retention | ✅ | Comprehensive ion/cofactor lists |
| Conformer seeding | ✅ | Deterministic ETKDGv3 |
| Virtual screening seeding | ✅ | Deterministic per-compound |
| Ensemble docking seeding | ✅ | Deterministic per-repeat |
| Gasteiger NaN check | ⚠️ | Missing `inf` check |
| MD integrator seeding | ⚠️ | Not seeded |
| pH consistency | ⚠️ | Prep 7.4 vs. minimization 7.0 |
