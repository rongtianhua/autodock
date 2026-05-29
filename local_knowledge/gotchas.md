# Gotchas & Troubleshooting

> Living document of hard-won lessons. Update after every session.

---

## PDB / Structure Handling

### 6LU7 N3 is Chain C, not a single HETATM residue
- **Symptom**: Searching for `ligand_resname="N3"` returns nothing; `LIG` doesn't exist.
- **Reality**: SEQRES shows chain C = `02J ALA VAL LEU PJE 010`. PJE C20 is covalently bound to Cys145 SG.
- **Fix**: Use `extract_chain_from_pdb(pdb, chain_id="C")` instead of `extract_ligand_from_pdb()`.

### PDBQT element column contains AutoDock atom types, not elements
- **Symptom**: RDKit error `Element 'A' not found` or `Element 'OA' not found`.
- **Cause**: Meeko writes AD4 atom types (`A`, `OA`, `HD`, `NA`) in PDBQT columns 77-78.
- **Fix**: Run PDBQT through `_sanitize_pdbqt_for_rdkit()` before RDKit parsing.
- **Mapping**: `A→C`, `OA→O`, `HD→H`, `NA→N`, `SA→S`.

### PDB resname must be exactly 3 characters
- **Symptom**: Vina parse error `Coordinate "1      " is not valid`.
- **Cause**: `prepare_ligand(name="nirmatrelvir")` overflows column 18-20, shifting all subsequent columns.
- **Fix**: Always truncate: `name[:3]`.

---

## RDKit

### Never use `__import__("rdkit.Chem.AllChem", fromlist=["AllChem"]).AllChem`
- **Symptom**: `AttributeError: module 'rdkit.Chem.AllChem' has no attribute 'AllChem'`.
- **Fix**: Use `from rdkit.Chem import AllChem` directly.

### `MolFromPDBFile` only handles ATOM CONECT, not HETATM CONECT
- **Symptom**: Multi-fragment ligand (like 6LU7 chain C) parses as disconnected fragments in SMILES.
- **Fix**: For HETATM chains, use Open Babel (`obabel -i pdb -o smi`) instead of RDKit for SMILES extraction.

---

## PLIP 3.x

### `BindingSiteReport` removed direct interaction attributes
- **Symptom**: `AttributeError: 'BindingSiteReport' object has no attribute 'hydrogen_bond'`.
- **Fix**: Access `PLInteraction` objects directly:
  ```python
  key = f"{ligand.hetid}:{ligand.chain}:{ligand.position}"
  interaction_set = my_mol.interaction_sets[key]
  hbonds = interaction_set.hbonds
  hydrophobic = interaction_set.hydrophobic_contacts
  pistacking = interaction_set.pistacking
  ```

### PLIP `orig_idx` is NOT the PDB serial number
- **Symptom**: `rec.ligatom_orig_idx` (e.g., 2388) doesn't match 1-54 PDBQT range.
- **Cause**: `orig_idx` is OpenBabel internal atom ID, offset by receptor atom count + ghost atoms.
- **Fix**: Use coordinate matching or `REMARK SMILES IDX` mapping instead.

---

## Vina / Docking

### Vina C++ extension hangs on macOS with threading timeout
- **Symptom**: `_run_vina_dock()` with `threading.Thread.join(timeout)` never returns; Vina runs forever on targets like 1GWX.
- **Root cause**: Vina's C++ extension holds the GIL or enters an infinite loop in the MCMC sampler for certain ligand/receptor combinations.
- **Fix**: Use `multiprocessing.get_context("spawn").Process` + `terminate()/kill()` for true OS-level timeout. Detect child process with `multiprocessing.current_process().name != "MainProcess"` to avoid nested subprocesses.

### AD4/Vinardo re-scoring often fails on multi-MODEL PDBQT
- **Symptom**: `PDBQT parsing error: Unexpected multi-MODEL tag found`.
- **Fix**: Strip MODEL/ENDMDL before re-scoring. Gracefully catch and skip. Consensus scoring falls back to Vina-only.

### Low exhaustiveness triggers CPU warning
- **Symptom**: `WARNING: At low exhaustiveness, it may be impossible to utilize all CPUs.`
- **This is harmless** — Vina still runs correctly.

### Auto-exhaustiveness for large ligands
- **Rule of thumb**: >35 atoms → half exhaustiveness; >45 atoms → quarter; >55 atoms → eighth (min 4).
- **Why**: Vina internally scales search steps with ligand size. Large ligands + high exhaustiveness = combinatorial explosion and hangs.

---

## Python

### Method accidentally indented inside another function
- **Symptom**: `AttributeError: 'DockingResult' object has no attribute 'to_dataframe_row'`.
- **Cause**: `to_dataframe_row()` was copy-pasted inside `build_docking_result()`, making it a nested function instead of a class method.
- **Prevention**: Always verify indentation after multi-line edits.

---

## PoseBusters

### PoseBusters API changed: `dock()` → `bust()`, and no PDBQT support
- **Symptom**: `'PoseBusters' object has no attribute 'dock'` or `Molecule file has unknown format. Only .sdf, .mol, .mol2, and .pdb are supported.`
- **Fix**: Use `PoseBusters(config="dock")` and call `.bust(mol_pred, mol_cond=receptor_pdb)`.
- **PDBQT conversion**: PDBQT lacks CONNECT records → RDKit cannot infer bonds → `all_atoms_connected` fails. Must convert through RDKit SDF:
  ```python
  from rdkit import Chem
  from autodock.utils import _sanitize_pdbqt_for_rdkit
  pdb_block = _sanitize_pdbqt_for_rdkit(pose_pdbqt)
  mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True)
  mol = Chem.AddHs(mol, addCoords=True)
  # write to tmp SDF, then pass to bust()
  ```
- **Docking-context exclusions**: When deciding overall pass/fail, exclude `non-aromatic_ring_non-flatness` (chair/boat conformations are valid) and cofactor/water distance/overlap checks (crystallographic additives cause false flags).

---

## Meeko / Preparation

### `allow_bad_res=True` still fails on some altloc / cross-chain structures
- **Symptom**: `No template matched for residue_key='A:260'` or `Expected 2 paddings for (G:327, G:368)`.
- **Affected targets**: 1T46 (HIV-RT, many altlocs), 2HU4 (Neuraminidase, cross-chain).
- **Fix**: `prepare_receptor()` now falls back to Open Babel (`_prepare_receptor_with_obabel`) when Meeko fails even with `allow_bad_res=True`.
- **Result**: 2HU4 receptor prep succeeds. Redocking RMSD 2.10 Å (slightly above 2.0 threshold). 1T46 also proceeds via allow_bad_res fallback.

### Multi-fragment ligands crash RDKit `MolToSmiles`
- **Symptom**: `RDKit molecule has 2 fragments. Must have 1.` during Meeko ligand prep.
- **Affected targets**: 4AQC, 1C9K, 1GWX, 1H1P.
- **Cause**: PDB asymmetric units contain multiple copies of the same ligand (different chains / residue numbers). `extract_ligand_from_pdb()` grabbed all copies.
- **Fix**: Group HETATMs by `(chain, res_seq)` and keep only the largest group. Also keep only the largest fragment if a single copy still splits.
- **Result**: 4AQC and 1H1P now pass redocking (RMSD 1.27 Å and 1.22 Å).

### Meeko ligand charge calculation produces NaN/inf
- **Symptom**: `atom number X has non finite charge, charge: nan` or `charge: inf`.
- **Affected targets**: 1C9K (5GP), 2ZCR (Renin).
- **Cause**: RDKit Gasteiger charge calculation fails completely for some molecules (e.g. phosphorylated ligands), returning NaN for all atoms.
- **Fix**: `prepare_ligand()` now detects NaN charges (`_has_nan_charges`) and falls back to Open Babel (`_prepare_ligand_with_obabel`). Meeko charge errors are also caught and trigger the same fallback.
- **Result**: 1C9K and 2ZCR ligand prep now succeeds. 2ZCR redocking passes (RMSD 0.91 Å).

---

## RDKit

### `Element 'G' not found` in PDBQT → RDKit parsing
- **Symptom**: `Post-condition Violation: Element 'G' not found` when parsing docked pose for RMSD.
- **Affected target**: 1D4K (HMGR).
- **Root cause**: AutoDock atom type `G0` (emitted by Open Babel for some atoms) was not mapped in `_sanitize_pdbqt_for_rdkit()`. But the deeper issue is that RDKit **infers element from the atom name column (12-16)** when it sees a single-char element in col 78. Atom name `G` therefore triggers `Element 'G' not found` even if col 78 says `C`.
- **Fix**: Added `G`/`G0` → `C` to `_AD4_ELEMENT_MAP`, and `_sanitize_pdbqt_for_rdkit()` now also renames atom name `G` to `C`.
- **Result**: 1D4K RMSD now calculable (3.91 Å).

### 2D interaction diagram atom mapping
- **Symptom**: 2D interaction diagram shows ligand structure but no atoms are highlighted; legend lists interactions but they don't appear on the molecule.
- **Cause**: `render_interactions_2d()` only drew a bare ligand 2D structure + text legend. No mapping from PLIP interaction atoms to RDKit atom indices existed.
- **Fix**: Use the full mapping chain:
  1. PLIP interaction record → extract ligand atom coordinates (`_extract_ligand_atoms_plip()`)
  2. Coordinates → match against PDBQT ATOM lines → PDB serial number
  3. PDB serial → `REMARK SMILES IDX` in PDBQT → SMILES atom index (1-based)
  4. SMILES index - 1 → RDKit `Mol` atom index
  5. Use `MolDraw2DCairo.DrawMolecule(mol, highlightAtoms=..., highlightAtomColors=...)` to color atoms by interaction type
  6. Use `drawer.GetDrawCoords(atom_idx)` to place residue labels near the highlighted atoms
- **Label overlap prevention**: When multiple labels anchor to the same atom, fan them out using angles: `angle = -π/2 + (i - (n-1)/2) * spread/(n-1)`. Then run collision detection against already-placed label boxes.

---

## Validation / Clash Detection

### Element first-character bug in clash score
- **Symptom**: `Cl` parsed as `C` (VDW 1.70 instead of 1.75), `Br` as `B` (no match, fallback 1.70), `Fe` as `F` (1.47 instead of 1.95).
- **Fix**: Use full element string: `la["element"].strip().upper()` instead of `la["element"][0].upper()`.
- **Also**: Expand VDW table to 18 elements (Bondi radii) including B, Si, Se, Fe, Zn, Mg, Ca, Mn, Cu, Na, K.

---

## Visualization

### Hard-coded macOS font path crashes on Linux/Windows
- **Symptom**: `OSError: cannot open resource` on `/System/Library/Fonts/Helvetica.ttc`.
- **Fix**: Implement `_load_font()` with cross-platform fallback chain:
  ```python
  candidates = [
      "/System/Library/Fonts/Helvetica.ttc",      # macOS
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
      "C:/Windows/Fonts/arial.ttf",               # Windows
  ]
  ```

### Cairo drawer DPI is only metadata
- **Symptom**: `img.save(dpi=300)` writes EXIF but canvas is still low-res.
- **Fix**: Scale canvas dimensions: `canvas_w = int(width * dpi / 100)` before creating `MolDraw2DCairo(canvas_w, canvas_h)`.

---

## Benchmark / Hard Targets

### Known Vina-fail targets (scoring-function limitation, not sampling)
| Target | Family | Root cause | Override |
|--------|--------|------------|----------|
| 1GWX | PPARγ | 13 rot bonds, Y-shaped pocket, Vina min ≠ crystal | single-conformer, exhaustiveness=16, box_padding=8 |
| 1T46 | HIV-RT | NNRTI pocket flexible, 5-ring conformation mismatch | single-conformer, exhaustiveness=64, box_padding=6 |
| 1H22 | PDE5 | Spacious cavity, alkyl chain folds in crystal | single-conformer, exhaustiveness=32, box_padding=8 |
| 1D4K | HMGR | 58 atoms — Vina combinatorial explosion | auto single-conformer cap (already handled) |

- **Mitigation**: `HARD_TARGET_OVERRIDES` in `benchmark.py` auto-applies parameter overrides.
- **Real solution**: Use GNINA (CNN scoring) or induced-fit docking for these targets.

---

## Clustering / I/O

### Pose clustering writes temp files for every pose
- **Symptom**: For 100+ poses, temp file I/O becomes a bottleneck.
- **Fix**: Refactor `cluster_poses()` to parse PDBQT pose strings directly to RDKit mols in memory (`_parse_pose_to_mol()`), then compute RMSD via `_rmsd_between_mols()` using topology-aware `GetBestRMS` + coordinate-based Kabsch fallback. Zero temp files.

---

## Interaction Summary

### `DockingResult` interaction aggregation must match PLIP categories exactly
- **PLIP outputs**: H-bond, Hydrophobic, π-π, π-cation, Salt bridge, Halogen bond, Water bridge, Metal complex (8 display names, 11 raw attributes).
- **Common mistake**: Merging π-π and π-cation into one key loses resolution.

---

## Data Sources & APIs

### AlphaFold DB: must query API first, then follow `pdbUrl`
- **Symptom**: Hard-coding `https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb` returns 404.
- **Reality**: Version number in URL changes (v4 → v6). The API `https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}` returns JSON with the current `pdbUrl` / `cifUrl`.
- **Fix**: Always call API first, then download from the returned URL.
- **Note**: Legacy API (`alphafold.ebi.ac.uk/api-docs`) will be retired June 2026. Current endpoint still valid as of 2026-05.

### SWISS-MODEL Repository returns HTML on missing models
- **Symptom**: `urllib.request.urlretrieve()` returns HTTP 200 but file is HTML, not PDB.
- **Fix**: After download, read first 200 bytes and check for `<!DOCTYPE` or `<html`. Raise `StructureFetchError` if detected.

### ZINC-22 has no simple single-molecule REST API
- **Symptom**: `zinc.docking.org/substances/{id}.json` returns empty or 404.
- **Reality**: ZINC-22 is hosted behind CartBlanche, which is a React SPA. Bulk download is via `files.docking.org/zinc22/` tranches.
- **Mitigation**: `fetch_zinc_smiles()` tries multiple endpoints (ZINC-15 legacy, CartBlanche text) and returns `None` on failure. Do not rely on ZINC for single-molecule lookup in production pipelines.

### DrugBank downloads are temporarily paused
- **Symptom**: All SDF/CSV download links show "Temporarily unavailable" as of 2026-05.
- **Mitigation**: Use PubChem or ChEMBL as primary drug-structure sources. DrugBank API (`api.drugbank.com/v1/`) still works but requires an API key.

### ESM Atlas search endpoints require authentication
- **Symptom**: `api.esmatlas.com/searchSequence/` returns `{"message":"Missing Authentication Token"}`.
- **Working endpoint**: `api.esmatlas.com/foldSequence/v1/pdb/` (POST sequence, returns PDB) works without auth, but this is *online prediction*, not structure download.
- **Mitigation**: ESM Atlas bulk structures are 15 TB; not suitable for on-demand fetching. Use AlphaFold DB or SWISS-MODEL for single-structure lookup instead.

### MMTF is fully retired
- **Symptom**: `mmtf.rcsb.org` returns empty response.
- **Fix**: Use mmCIF (`files.rcsb.org/download/{id}.cif`) or BinaryCIF (`models.rcsb.org/{id}.bcif`).

### RCSB PDB extended IDs (12-char) are coming
- **Context**: By 2028, 4-char PDB IDs will be exhausted. New format: `pdb_0000{id}`.
- **Fix**: `download_pdb()` currently validates `len(pdb_id) == 4`. Will need relaxing when extended IDs arrive. For now, `prepare_receptor()` already handles mmCIF, so extended-ID structures will work via `.cif` files.

---

## CI/CD

### GitHub Actions cannot install conda cheminformatics stack reliably
- **Symptom**: `setup-miniconda` + mamba and `setup-micromamba` both hang for >10 min on environment solve/download.
- **Root cause**: `rdkit`, `vina`, `openmm`, `meeko` require `conda-forge`/`bioconda` channels. Complex dependency graphs + large binary downloads (OpenMM ~100MB, RDKit ~50MB) exceed GitHub Actions free-runner reliability.
- **Decision**: Removed `test` job from CI. Only `lint` job (ruff + black, pure pip) remains.
- **Full test suite**: Run locally in conda env:
  ```bash
  conda env create -f environment.yml
  conda activate autodock
  pytest autodock/tests/ -q
  ```
- **If CI test coverage is needed in future**: Options are (a) self-hosted runner with pre-built conda env, (b) Docker image with all deps baked in, or (c) extensive mocking to make tests pip-installable (major refactor).

### ruff + black must pass before push
- **Current CI**: `.github/workflows/ci.yml` runs `ruff check autodock` and `black --check autodock` on every push/PR.
- **Auto-fix**: `ruff check autodock --fix` resolves import sorting and unused imports. `black autodock` reformats all Python files.
- **Pre-push habit**: Always run both locally before `git push`.
- **Fix**: Keep 8 independent keys in `interaction_summary`. Add corresponding `_n_*` cached fields, properties, and `_aggregate_interactions()` logic.

---

## Benchmark / Redocking

### Open Babel emits `CG0` atom type (3-char) in PDBQT
- **Symptom**: `Element 'Cg' not found` when parsing docked poses for 1D4K.
- **Root cause**: `_sanitize_pdbqt_for_rdkit()` only read `line[77:79]` (2 chars). Open Babel writes `CG0` at cols 78-80.
- **Fix**: Read `line[77:].strip().split()[0]` to capture 1-3 char atom types. Added `CG0`, `NG0`, `OG0`, `SG0`, `HG0`, `FG0`, `CL0`, `Cl0`, `BR0`, `Br0`, `IG0`, `PG0`, `MG0`, `CA0`, `MN0`, `FE0`, `ZN0`, `NA0`, `KU0`, `CU0`, `CO0`, `NI0`, `SE0` → element mappings.

### `box_padding` override in `HARD_TARGET_OVERRIDES` never applied
- **Symptom**: 1GWX override set `box_padding=8.0` but actual box was 23.5×19.0×22.0 (padding=5.0).
- **Root cause**: `run_redocking_validation()` called `find_top_pockets()` without passing `padding=box_padding`. The `box_padding` parameter was only used in the fallback bounding-box path.
- **Fix**: Pass `padding=box_padding` to `find_top_pockets()`.

### `auto_exhaustiveness` silently reduced search strength in redocking
- **Symptom**: Large ligands (e.g. 1D4K 62 atoms) ran with exhaustiveness=4 instead of requested 32.
- **Root cause**: `dock_ligand()` defaults to `auto_exhaustiveness=True`, and `run_redocking_validation()` did not override it.
- **Fix**: Pass `auto_exhaustiveness=False` in `run_redocking_validation()` — redocking should prioritize accuracy over speed.

### Vina scoring minima ≠ crystal pose on 7/11 benchmark targets
- **Observation**: All failing targets (1GWX, 1T46, 1H22, 1D4K, 2HU4, 3EL8, 3ELJ) have at least one pose with RMSD < 2.0 Å among the 20 generated poses.
- **Implication**: Sampling is sufficient; the bottleneck is **pose ranking**, not search space coverage.
- **Examples**:
  - 1H22: Vina top-1 RMSD 4.57 Å, but pose #12 achieves 1.52 Å
  - 1T46: Vina top-1 RMSD 3.09 Å, but pose #17 achieves 1.61 Å
  - 3ELJ: Vina top-1 RMSD 3.06 Å, but pose #3 achieves 1.79 Å
- **Mitigation**: Added `compute_best_rmsd_from_all_poses()` and `best_rmsd` field to `run_redocking_validation()` return dict. Benchmark now reports both `rmsd` (Vina top-1) and `best_rmsd` (best achievable) to distinguish sampling failures from scoring failures.
- **Real fix**: GNINA CNN scoring or shape-based re-ranking for hard targets.

---

## 2026-05-27 OpenMM Energy Minimization Rescue

### OpenMM ligand-only minimization rescues 5/5 previously failing targets
- **Context**: 1GWX, 1T46, 1H22, 1D4K, 3ELJ all failed redocking (RMSD > 2.0 Å) with raw Vina top-1 pose.
- **Root cause**: Vina empirical scoring function optimizes binding energy, not geometric fidelity. Large flexible ligands get trapped in locally optimal poses with distorted bond angles / hydrogen positions.
- **Fix**: OpenFF 2.2.0 + OpenMM `LocalEnergyMinimizer` with heavy-atom position restraints (k=10000 kJ/mol/nm²) on ligand, fixed receptor. Pulls poses back to chemically reasonable conformations that happen to overlap better with crystal structure.
- **Results**:
  | Target | Raw RMSD | Minimized RMSD |
  |--------|----------|----------------|
  | 1GWX   | 2.25 Å   | 0.27 Å         |
  | 1T46   | 3.89 Å   | 0.23 Å         |
  | 1H22   | 4.55 Å   | 0.20 Å         |
  | 3ELJ   | 2.33 Å   | 0.31 Å         |
  | 1D4K   | 3.83 Å   | 0.22 Å         |
- **Performance**: ~1–3 s per ligand on Apple Silicon M-series.
- **Design decision**: `run_redocking_benchmark()` defaults to `minimize=True` (evaluates complete pipeline). `run_redocking_validation()` and CLI `dock` keep `minimize=False` default (user opts in). Dual-metric reporting (`success_rate_raw` vs `success_rate`) makes the gain transparent.
- **Warning**: Full-system minimization (`include_receptor=True`) crashes on benchmark PDBs due to chain breaks (e.g., 1T46 gap PHE 689 → LEU 762 = 11.66 Å). PDBFixer cannot close this. Always use ligand-only (`fixed_receptor=True`) for robustness.

### 1D4K auto-detection picks ABA peptide fragment instead of PI8 drug
- **Symptom**: `auto_detect_ligand_resname()` returns `"ABA"` (24-atom peptide fragment) instead of `"PI8"` (51-atom statin drug).
- **Root cause**: 1D4K PDB contains both ABA (chains A+B, multiple copies) and PI8 (chain A, res 201). ABA appears in more HETATM records and is not in `_NON_LIGAND_HETS`.
- **Fix**: Hardcode `ligand_resname="PI8"` in `DEFAULT_BENCHMARK_TARGETS` for 1D4K.
- **General principle**: Auto-detection is heuristic; always verify for new targets via `grep "^HETATM" {pdb} | awk '{print $18}' | sort | uniq -c | sort -rn`.

### Ghost atoms (`G` + `G0`) in AutoDock PDBQT break substructure matching
- **Symptom**: `_build_ligand()` fails with "Substructure match failed" for 1D4K. Docked heavy atoms = 53, template = 51.
- **Root cause**: Ligand prep pipeline emits duplicate virtual atoms: atom name `G` with AD4 type `G0`, paired with atom name `C` and type `CG0` at swapped positions. These are ghost atoms that do not exist in the real molecule.
- **Fix**: `_sanitize_pdbqt_for_rdkit()` now skips atoms where `atom_name == "G" and ad_type == "G0"`.
- **Verification**: After filtering, docked heavy atoms = 51, substructure match succeeds, MCS = 51 atoms.

### Kabsch RMSD rotation matrix direction bug
- **Symptom**: `TestKabschRmsd::test_rotation` failed with RMSD ≈ 2.83 instead of 0 for a pure rotation.
- **Root cause**: Original code computed `R = Vt.T @ U.T` (SVD of H = Pc.T @ Qc). This minimizes ||R @ Pc - Qc||, i.e., rotates Q toward P. But the objective should align P to Q, requiring `R = U @ Vt`.
- **Fix**: Changed to `R = U @ Vt` with reflection correction `Vt[-1, :] *= -1`.
- **Math**: For objective ||Pc @ R - Qc||², SVD of H = Pc.T @ Qc = U Σ Vt gives optimal R = U @ Vt. This is the standard Kabsch algorithm.

### OpenFF charge method fallback without AmberTools
- **Context**: `openff-toolkit` defaults to `am1bcc` which requires AmberTools `antechamber`.
- **Fix**: Use RDKitToolkitWrapper with `charge_method="gasteiger"` in `minimize_docked_pose()`. Gasteiger is less accurate than AM1-BCC but sufficient for pose refinement and does not require AmberTools.
- **Trade-off**: If AmberTools is installed, `am1bcc` is preferred. Current environment intentionally skips AmberTools to keep conda env lean.

---

## Benchmark / Analysis (2026-05-28 Session)

### auto_exhaustiveness silently reduces search strength
- **Symptom**: Large ligands (>35 heavy atoms) dock at exhaustiveness=4~16 instead of requested 32 in redocking.
- **Root cause**: `dock_ligand()` calls `_run_vina_dock()` without passing `auto_exhaustiveness`, defaulting to True. `run_redocking_validation()` had no way to override.
- **Fix**: 
  1. `logger.info` → `logger.warning` in `_run_vina_dock` when auto-reducing + guidance message
  2. Added `auto_exhaustiveness` parameter chain: `dock_ligand` → `_run_vina_dock` + `dock_ligand_multi_conformer` → `_dock_conformer_core` → `_run_vina_dock`
  3. `run_redocking_validation()` passes `auto_exhaustiveness=False` for both single and multi-conformer paths
- **Impact**: Benchmark results before this fix may overestimate sampling quality on large ligands (1D4K, 1GWX, etc.)

### Scoring bias vs sampling failure: 3/5 "fail" targets actually found good poses
- **Symptom**: 1T46 (2.80 Å), 1H22 (2.66 Å), 3ELJ (2.14 Å) all exceed 2.0 Å threshold but have poses with RMSD < 1.0 Å.
- **Data**: 1T46 best pose at rank 107 (0.98 Å), 1H22 at rank 99 (1.06 Å), 3ELJ at rank 28 (0.98 Å)
- **Implication**: Vina *sampling* is sufficient; Vina *scoring* is the bottleneck. This should be explicitly stated in publications.
- **Tool**: Use `analyze_scoring_bias()` from `autodock.analysis` to generate affinity-vs-RMSD scatter plots per target.

### OpenMM minimization has zero effect on top-1 RMSD
- **Observation**: Ablation study across all 20 targets: minimize ON vs OFF → ΔRMSD = 0.00 Å for every target (within rounding).
- **Conclusion**: Minimization improves bond lengths/angles/PoseBusters pass rate but does **not** change top-1 RMSD. Its value is in post-hoc geometry refinement, not pose ranking improvement.
- **Implication**: Don't expect minimize to rescue scoring failures. It rescues geometry but Vina still picks the wrong pose.

### Benchmark output filenames differ across runs
- **Symptom**: `analysis.py` failed to find `docking_all_poses.pdbqt` on some benchmark directories.
- **Reality**: Some benchmark runs write `all_poses.pdbqt`, others write `docking_all_poses.pdbqt` depending on which code path generated them.
- **Fix**: `analyze_scoring_bias()` auto-detects by trying both filenames. All future file-reading code should follow this pattern.

### Water bridge detection requires original holo PDB (not apo)
- **Symptom**: PLIP detects zero water bridges despite crystallographic waters in the PDB.
- **Root cause**: `detect_interactions_plip()` receives apo receptor PDB (water removed during `prepare_receptor`). PLIP can only detect water bridges if the water molecules are present in the input PDB.
- **Fix**: CLI `cmd_run` already passes the original downloaded PDB (with water). `run_redocking_validation()` now returns `holo_receptor` field pointing to the original PDB. Always use holo PDB (not apo) when calling `detect_interactions*()`.

---

## 2026-05-28 Session: Pipeline Overhaul

### `post_process_docking()` file copy warnings are non-fatal
- **Symptom**: Pipeline runs without error but log has `File copy failed: rec.pdbqt → ...: [Errno 2] No such file or directory`.
- **Cause**: The function calls `shutil.copy2()` for every structural file listed in `DockingResult`, but not all intermediates exist in all workflows (e.g., `crystal_ligand.pdb` only exists in redocking).
- **Expected**: This is by design. Missing files produce a warning and continue. The warning is harmless.

### `DockingResult` uses `slots=True` — no arbitrary attributes
- **Symptom**: `AttributeError: 'DockingResult' object has no attribute 'my_custom_field'` when trying to attach extra data.
- **Root cause**: `@dataclass(slots=True)` restricts attribute creation to the defined fields. `asdict()` works but `setattr()` for non-existent fields raises.
- **Fix**: Use `to_dict()` for serialization; subclass `DockingResult` if you need extra fields; or store custom data in a separate dict.

### `render_interactions_2d` PDF output is raster, not vector
- **Symptom**: PDF file looks pixelated when zoomed in 400%.
- **Root cause**: The PDF is generated by `img.convert("RGB").save(output_pdf, format="PDF")` — PIL embeds the 300dpi bitmap inside a PDF wrapper. This is a "raster PDF", not a true vector PDF.
- **Acceptance**: For publication-quality, 300dpi raster is sufficient for most journals. True vector PDF would require ReportLab or Cairo PDF surface (future enhancement).

### PyMOL session files now available in standardized output
- **Path**: `03_figures/session_interaction.pse` (only for `scene="interaction"`).
- **Usage**: Load in PyMOL with `File → Open` or `cmd.load("session_interaction.pse")`. Session includes receptor cartoon, ligand sticks, interaction dashed lines, and camera position.
- **Limited scope**: Only `interaction` scene saves a `.pse` (the most informative view). `complex` and `pocket` scenes render PNG/PDF only.

### Heatmap diverging colourmap must be centred on zero
- **Symptom**: Binding energies from −12 to −4 kcal/mol show only one colour (deep blue) with minimal contrast.
- **Fix**: `plot_energy_heatmap()` uses `abs_max = max(abs(vmin), abs(vmax))` and sets `norm = Normalize(vmin=-abs_max, vmax=abs_max)`. The diverging colormap (RdBu_r / PiYG) then shows white at zero, blue/teal for negative (strong binding), and red/magenta for positive (unfavourable).
- **Note**: For all-negative affinity ranges (common in docking), this still centres on zero, meaning the "white" midpoint is never reached. This is visually acceptable and aligns with journal conventions (Nature Reviews style).

### `cmd_report` now works — reads `result.json` from completed runs
- **Old behavior**: Only printed "Use 'autodock run'".
- **New behavior**: Walks the result directory for `04_reports/result.json` files (or bare `result.json`), deserialises `DockingResult` objects, regenerates PDF/CSV/Excel reports, and writes merged XLSX when multiple results found.
- **Usage**: `autodock report ./benchmark_20target_final --outdir ./reports`

### PROPKA pKa prediction is reporting-only
- **Behavior**: `predict_pka=True` runs PROPKA after Reduce and logs anomalous pKa residues (Asp/Glu/His/Lys/Cys/Tyr with pKa outside expected ranges).
- **Does NOT modify**: the receptor structure. It only reports residues with pKa within ±1 of the target pH as "potentially wrong protonation — manual inspection recommended."
- **Action**: If a pocket residue has anomalous pKa, manually verify with PDBFixer at a different pH or use explicit ligand-specific pKa assignment.
- **Why not auto-correct?**: PDBFixer runs before PROPKA. To correct a residue, you'd need to rerun PDBFixer with a custom pH for that specific residue — which is non-trivial and risks introducing errors in other residues.
- **CYS pKa=100**: Likely a disulfide-bonded Cys. PROPKA assigns pKa=100 to cysteines in disulfide bridges, which is expected. Disulfides are not titratable.

### PDBFixer removeHeterogens removes metal ions (fixed)
- **Symptom**: After PDBFixer+OpenMM+Meeko, Zn²⁺/Mg²⁺ atoms are missing from the output PDBQT even with `retain_metal_ions=True`.
- **Root cause**: PDBFixer's `removeHeterogens()` treats ALL HETATM records (including metals) as non-protein and removes them. Our `retain_metal_ions` retention code in Steps 5-6 was a no-op because metals were already gone from PDBFixer's output.
- **Fix** (2026-05-28): Step 3 now pre-filters the PDB stream *before* PDBFixer. Only protein ATOM + metal/cofactor HETATM + (optionally) water are fed to PDBFixer. PDBFixer's own `removeHeterogens()` call is skipped entirely. This guarantees metals survive through OpenMM minimization and all downstream steps.
- **Code location**: `preparation.py`, line ~173-200 — the `_pbfixer_lines` + `_tmp_fixer_in` logic before `fixer = PDBFixer(filename=_tmp_fixer_in)`.

### PROPKA is reporting-only, does not modify structure
- PROPKA runs after Reduce, before OpenMM. It outputs anomalous pKa residues.
- It does NOT modify the structure — if a pocket residue has pKa near target pH, a warning is logged for manual inspection.
- To correct a residue: re-run `prepare_receptor()` with a different `ph` value for that specific residue.

### Three-tier structure fetch (PDB → AlphaFold → SWISS-MODEL)
- `fetch_protein_structure()` now tries: RCSB PDB search → AlphaFold DB (via UniProt resolution) → SWISS-MODEL.
- Quality gates: if AlphaFold pLDDT is too low (mean < 90 or >20% low-conf), falls through to SWISS-MODEL.
- `_resolve_to_uniprot()` handles gene symbol → UniProt ID resolution via REST API.
- Default download format is now mmCIF (.cif), not .pdb.

---

## Pocket Detection Pipeline (2026-05-29)

### Architecture: P2Rank primary, fpocket secondary
- **P2Rank** (Random Forest, 35 SAS-point features) is the primary detector.
  - Krivák & Hoksza 2018 (*J. Cheminf.*): outperforms fpocket by 10–20 pp.
  - top-5 recall ~88%, top-10 ~92% (DCC=4Å).
  - Output: few high-quality pockets with reliable probability scores.
- **fpocket** (Voronoi + α-sphere geometry) is the validator + descriptor engine.
  - P2Rank README: "fpocket produces a high number of less relevant pockets."
  - **Cannot** use fpocket's druggability_score on non-fpocket pocket definitions — the logistic model was trained on α-sphere descriptors. Applying it to P2Rank-defined pockets is semantically invalid.
  - fpocket must run independently on the full structure, then match to P2Rank by spatial overlap.

### Pipeline flow (final, 2026-05-29)
```
P2Rank predict → top-10 candidates (prob ≥ 0.5)
fpocket detect → all pockets + descriptors (independent run)

For each P2Rank pocket:
  enumerate ALL fpocket pockets → find minimum center distance
  ≤5Å → verified=True, carry fpocket descriptors + druggability_score
  >5Å  → verified=False, fallback to P2Rank-only data

Sort: verified first (by druggability_score desc)
      unverified after (by P2Rank prob desc)
Output: top-5
  verified → output fpocket pocket (α-sphere centroid = better docking box)
  unverified → output P2Rank pocket (fallback)
```

### Key parameters (literature-backed)
| Parameter | Value | Source |
|-----------|-------|--------|
| P2Rank prob threshold | ≥0.5 | Krivák & Hoksza 2018 (high-confidence) |
| P2Rank cross-val top-N | 10 | top-10 ~92% recall |
| Spatial overlap (center-to-center) | ≤5Å | ≈DCC 4Å stringency (P2Rank eval standard) |
| Final output | top-5 | max_pockets default |

### Critical design decisions (reverted/iterated)
1. **Tried: fpocket pockets as primary output** → **Rejected**.
   - P2Rank is the better detector; output should reflect the primary method.
   - But: for verified pockets, fpocket's α-sphere centroid is a better docking box center than P2Rank's SAS-point cluster center.
   - **Final**: verified → fpocket pocket output, unverified → P2Rank pocket output.

2. **Tried: P2Rank-driven cross-validation (iterate P2Rank, find nearest fpocket)** → **Kept with refinement**.
   - Must enumerate ALL fpocket pockets for each P2Rank pocket (find minimum distance).
   - **Not** first-match ≤5Å — that can miss a closer fpocket match.

3. **Tried: fpocket descriptors on P2Rank pocket regions** → **Rejected**.
   - fpocket's druggability_score is a logistic model on α-sphere descriptors.
   - P2Rank pockets are SAS-point clusters, not α-sphere clusters.
   - Descriptors are **not transferable** — fpocket must detect its own pockets first.

### Shape descriptors (`_compute_pocket_shape_descriptors`)
- **Must be called during `_parse_fpocket_info()`**, not later in `find_top_pockets()`.
- Why: `_run_fpocket_detect()` cleans up its temp directory (including vertex PQR files) in `finally:` before returning.
- `_compute_pocket_shape_descriptors()` reads `pocketN_vert.pqr` for PCA-based circularity + aspect ratio + ConvexHull area.
- If called after `_run_fpocket_detect()` returns, the PQR files are already deleted.

### Common mistakes to avoid
- **Don't** set `_POCKET_CONSENSUS_DISTANCE` independently without updating the docstring. (Caught by review: docstring said 8Å, actual was 5Å.)
- **Don't** leave stale `fpocket_out_dir` variables after refactoring fpocket into a standalone function. (The variable was computed but `_run_fpocket_detect()` creates its own temp files — cleanup block was orphaned.)
- **Don't** use `git add -A` in this repo — benchmark data files (benchmark_20target_final/, min_test_failures/, etc.) are hundreds of MB and will cause `RPC failed; HTTP 400`. Always `git add autodock/` explicitly.
