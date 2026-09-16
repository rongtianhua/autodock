# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **2D interaction diagram: standard-layout vector base**
  (`autodock/rendering.py`). The 2D base structure is now drawn with CoordGen
  2D coordinates (ChemDraw-lineage template layout) in ACS 1996 style as an
  SVG vector graphic, rasterised at the exact target size with cairosvg. The
  previous pipeline rasterised RDKit's natural-size flexi canvas (~150 px for
  typical ligands) and upscaled 10–30× with LANCZOS, blurring bonds and
  labels; the vector route keeps every edge crisp at publication DPI, and the
  ChemDraw-style layout replaces the recognisably "RDKit-flavoured" drawing.
  cairosvg added to the `vis` extra; without it the code degrades to the old
  low-resolution flexi-Cairo path with a warning.
- **3D rendering: post-render content validation**
  (`autodock/rendering.py`). PyMOL exits 0 and writes a valid PNG even when a
  selection expression is malformed and the scene silently renders blank (the
  class of bug behind the never-rendered pocket surface). Every rendered scene
  is now validated before legend overlay/autocrop/PDF: blank canvases and
  content below per-scene coverage/span floors raise `VisualizationError`,
  and selector errors on stderr are surfaced as loud warnings.
- **3D scenes: white background by default**
  (`autodock/rendering.py`). All 3D figures (complex/pocket/interaction) now
  render on a solid white background with black labels (`publication_white`
  becomes the default colour scheme; `presentation_black` remains available).
  Scheme-dependent details: pocket side-chain sticks use grey50 carbons
  (white was invisible on white), the interaction-scene legend composites a
  translucent white box with black text instead of a black box with white
  text, and the pocket surface uses skyblue (lightblue washed out).
- **Pocket scene: surface never rendered**
  (`autodock/rendering.py`). The surface selection used the invalid PyMOL
  expression `br. receptor and center x,y,z around 5`, which silently
  selected nothing — no pocket surface appeared in any render. The script now
  anchors a pseudoatom at the pocket center and selects
  `byres (receptor within <d> of pocket_ctr)`.
- **2D interaction diagram: hydrophobic arc direction (restored geometry)**
  (`autodock/rendering.py`). Arc geometry is restored to the previous
  parameters (radius 0.38 × label distance, span π/2.5, 7 spokes), and the
  arc's convex side (middle spoke) now points along the reverse of the
  atom→ring-centre line — outward through the contacting atom, away from the
  aromatic ring — computed per-ring (falling back to the ligand centroid for
  non-ring contact atoms), so arcs no longer sweep across rings or labels.
- **Pocket detection: duplicate cavities docked twice**
  (`autodock/preparation.py`, `autodock/core.py`). P2Rank routinely splits
  one cavity into several overlapping predictions; each candidate
  independently matched the SAME nearest fpocket pocket, so duplicate output
  entries with identical center/box were docked separately. Cross-validation
  is now an exclusive greedy match in descending score order (a claimed
  fpocket pocket is removed from the pool), and a final 3 Å center-dedup
  pass backfills `max_pockets` with the next distinct site. The pocket cache
  key gains a `pocket_schema` version so pre-fix duplicate entries are never
  reused.
- **3D interaction scene: residue labels anchored to the contact atom**
  (`autodock/rendering.py`). Labels were placed at the Cα and pushed 4 Å
  along the direction away from the ligand centroid, so residues contacting
  the ligand through a side chain that points away from the Cα (e.g. the
  GAPT–quercetin GLU72 contact) got their label stranded far from both the
  residue and its dashed line. Labels are now anchored to the residue heavy
  atom closest to the ligand (the actual contact atom) with a 2.5 Å
  outward push, falling back to Cα only when no anchor atom is found.
- **3D complex scene: automatic whitespace cropping**
  (`autodock/rendering.py`). `cmd.orient()` fills the view to the bounding
  sphere, so for elongated/U-shaped receptors a smaller zoom buffer cannot
  remove the empty margins without clipping — they are inherent to the
  camera framing. Whole-complex scenes are now auto-cropped after ray
  tracing (`_autocrop_png`: background-colour border detection via PIL,
  3% margin kept), which removes the margins regardless of shape; the zoom
  buffer is tightened to 1.0 Å.
- **2D interaction diagram: hydrophobic arc direction**
  (`autodock/rendering.py`). Hydrophobic arcs were centred on the angle
  bisector toward the residue label, which could make the arc sweep across
  the aromatic ring it contacts. Arcs are now centred on the direction
  *away from the ligand centroid* (leader line from the arc tip to the
  label), so they open outward from the ring system.
- **Pocket cross-validation: no more silent top-10 truncation**
  (`autodock/preparation.py`). P2Rank candidates are now explicitly sorted by
  score before shortlisting (the old code relied on CSV row order), the
  cross-validation pool grows with the request
  (`max(10, 2 × max_pockets)`), and when fewer than `max_pockets` shortlisted
  pockets verify, the remaining P2Rank candidates are cross-validated in an
  extension pass instead of being silently discarded.
- **Pocket cross-validation: dual consensus criterion**
  (`autodock/preparation.py`, `autodock/core.py`). A P2Rank pocket used to be
  verified only by a 5 Å center-distance match, so real pockets whose two
  detector centers disagreed by > 5 Å (large/irregular cavities) were
  rejected. Verification now also passes when centers are ≤ 10 Å apart AND
  the P2Rank sphere overlaps the fpocket pocket bounding box
  (`_sphere_box_overlap`, `_POCKET_CONSENSUS_DISTANCE_LOOSE`).
- **Unverified pockets: traceable source, honest druggability**
  (`autodock/preparation.py`). Unverified pockets are now labelled
  `p2rank_unverified`/`dogsite3_unverified` in `pocket_source` instead of
  masquerading as verified-method pockets, and they keep their own detector
  score for `druggability` — previously the value was borrowed from the
  nearest fpocket pocket even when that pocket was a different cavity > 10 Å
  away, misleading ranking and reports.
- **3D complex scene: higher resolution, tighter camera**
  (`autodock/rendering.py`). Whole-complex scenes now render at 3200×2400 by
  default (only when the caller did not request an explicit size) and the
  camera runs `cmd.orient()` with a 1.5 Å zoom buffer instead of a plain
  zoom with 5 Å buffer, removing the wide empty margins around small
  receptors.
- **3D interaction scene: dashed-line endpoints, thickness, colour legend**
  (`autodock/rendering.py`). Interaction dashes now end at the atom class
  that actually mediates the interaction (H-bond → N/O, hydrophobic → C,
  π → C/N; falls back to all residue heavy atoms when the preferred class is
  absent), the dash radius is 0.06 (was 0.10), and a colour legend
  (swatch + label per present interaction type) is composited onto the
  bottom-left of the PNG after rendering, before PDF conversion.
- **2D interaction diagram: hydrophobic arc overlap**
  (`autodock/rendering.py`). Hydrophobic arcs now use a compact radius
  (22% of the label distance, min 45×scale), a narrower span (π/3.2) and 5
  spokes, so they hug the interacting atom instead of sweeping across
  aromatic rings and neighbouring labels. Drawing is now two-pass (all
  connectors/arcs first, all residue label boxes last) so no arc overpaints
  another group's label.
- **3D scenes: speckle pixels around the cartoon, transparent background,
  invisible labels** (`autodock/rendering.py`). PyMOL shows the `lines`
  representation by default on load, which survived as spectrum-coloured
  specks around the cartoon on dark backgrounds — now hidden explicitly.
  Ray-traced PNGs carried a transparent alpha channel, so viewers
  composited them onto white and white residue/distance labels became
  invisible; `ray_opaque_background=1` forces a solid background.
  `composite_summary()` also flattens panels to RGB before pasting.
- **3D interaction scene: per-type dashed lines and pocket residue labels**
  (`autodock/rendering.py`). Interaction dashed lines are now drawn per
  ligand atom (each to its closest residue atom) instead of a single
  closest-pair line that made all types of one residue overlap into one
  indistinguishable dash, so per-type colours (cyan H-bond, orange
  hydrophobic, …) are readable. Non-interacting pocket residues (8 Å
  window) now get small dim labels alongside the prominent interacting
  residue labels; the residue-label block is wrapped in an explicit
  `python`/`python end` pair and the generated blocks are compile-checked
  in tests.
- **2D interaction diagram: legend collapse and connector overlap**
  (`autodock/rendering.py`). Legend paddings/row heights were fixed pixels
  while fonts scaled with canvas size, so on publication canvases the text
  piled up outside the box; all legend geometry now scales with the canvas
  and is measured with the real font. Connector lines/arcs are clipped at
  the label border (Liang–Barsky) instead of crossing under the label,
  line widths/arc spokes/label borders scale with the canvas, and the
  legend rectangle is reserved during label placement so labels never
  overlap it.
- **Config default `pocket.top_n` synced to 5** (`autodock/config.py`). The
  config default was 3 while `run_docking_workflow(max_pockets=5)`, the CLI
  `--max-pockets` default, and `find_top_pockets()` all use 5, so config-file
  runs silently docked fewer pockets than every other entry point. The
  in-code default and the `write_default_config()` template now both use 5.
- **Silent `except` blocks now log at debug level**
  (`autodock/workflow.py`, `autodock/reporting.py`, `autodock/preparation.py`).
  `_compute_ligand_metrics()`, `reporting.py` figure-size probing, and the
  spiro/bridgehead detection fallback swallowed all exceptions silently,
  making failures invisible. Each now emits a `logger.debug` record with the
  exception details. (`covalent.py` capability probing is intentionally
  unchanged — silence is part of that probe's contract.)
- **Cache file hashing truncated files > 8 MB** (`autodock/cache.py`).
  `_hash_file()` previously read at most 8 MB, so two inputs whose first
  8 MB are identical but differ afterwards received the same cache key and
  silently returned stale results. The default is now full-file hashing
  (`limit_bytes=None`); the parameter remains for callers that explicitly
  want a bounded read.
- **Resume checkpoint ignored parameter changes** (`autodock/workflow.py`).
  `workflow_state.json` tracked completed steps but not the parameters they
  were computed with, so re-running with a different seed, ligand, or
  exhaustiveness in the same output directory silently returned stale cached
  results. The workflow now writes a `params_fingerprint` (receptor, ligand,
  seed, exhaustiveness, poses, scoring function, pH, protonation, pocket
  settings) into the checkpoint and resets it — with a logged warning —
  when the fingerprint changes.
- **3D interaction scene cartoon transparency** (`autodock/rendering.py`).
  The interaction scene used `cartoon_transparency=1.0` outside a 15 Å
  `pocket_vis` selection with `0.2` inside; on small proteins (< 5000 atoms)
  the 15 Å window covers most of the structure, so the entire cartoon rendered
  ~80 % transparent. Transparent cartoon z-fought with the ball-and-stick
  ligand, β-strands rendered as distorted double ribbons, and CGO dashed
  interaction lines were visually lost. The scene now hides the receptor
  cartoon and re-shows only an 8 Å pocket window (solid), and displays pocket
  side chains as sticks so interacting residues are actually visible.
  Safe because interaction lines are CGO objects, not `distance` objects, so
  `cmd.hide("cartoon", ...)` does not affect them.
- **2D interaction diagram label layout** (`autodock/rendering.py`).
  `_compute_label_positions()` forced labels onto evenly spaced angular
  sectors at a fixed 40 % of the canvas diagonal, which scattered residue
  labels far from the ligand on large publication canvases and clipped them
  at the canvas edge. Labels now follow the natural centroid direction of
  each residue with collision nudging, at a distance adapted to the ligand's
  actual 2D extent. A residue with several interaction types (e.g. H-bond +
  hydrophobic) now shares ONE label position instead of one label per type.
  Unplaceable labels are skipped rather than drawn overlapping.
- **2D interaction diagram aromatic ring fill** (`autodock/rendering.py`).
  Only interacting atoms received highlight colours; for multi-ring ligands
  where one ring contacts the pocket (e.g. flavonoids), the non-interacting
  ring rendered as bare line strokes and the molecule looked split in half.
  Non-interacting aromatic atoms and aromatic-aromatic bonds now get a light
  grey fill so the full structure reads continuously.
- **`minimize_pose` never succeeded for conjugated ligands**
  (`autodock/minimization.py`, `autodock/workflow.py`). The workflow only
  passed `ligand_smiles` to `minimize_docked_pose()`, forcing a substructure
  match between the SMILES template and the RDKit topology inferred from the
  docked PDBQT, which fails for conjugated systems (flavonoids, quinolines,
  indoles) where bond-order/aromaticity perception differs — every
  minimisation attempt returned "Failed to build ligand molecule".
  Two fixes: (1) `run_docking_workflow()` gains an optional `ligand_sdf`
  parameter, auto-derived when `ligand_source="file"` points to an SDF/MOL
  file; (2) `_match_heavy_atoms()` falls back to element-aware optimal
  coordinate assignment (Hungarian + ICP-style Kabsch refinement) when
  substructure matching fails, recovering a correct atom mapping for the
  same molecule. Also fixed a latent `GetAtomPosition()` range error when the
  docked mol carries explicit hydrogens.
- **Pocket detection hard-failed when P2Rank found nothing**
  (`autodock/preparation.py`). `find_top_pockets()` fell back to DoGSite3
  (network) and then raised `PreparationError` without ever trying pure
  fpocket detection, hard-failing the whole pipeline for small / low-pLDDT /
  transmembrane receptors (observed: 20/20 pairs failed on a pLDDT ≈ 63
  AlphaFold transmembrane target whose pocket fpocket scores 0.49). A new
  second offline fallback tier runs fpocket, normalises its pockets, and
  ranks them by Drug Score through the unchanged downstream pipeline.

### Added
- **Headless PyMOL high-resolution rendering fix** (`autodock/rendering.py`).
  `render_scene_pymol()` now passes `-W {width} -H {height}` to the PyMOL CLI,
  preventing the silent 640×480 fallback in `-c` mode. Rendered output size is
  validated against the requested resolution and a warning is emitted on mismatch.
- **Adaptive composite figure layout** (`autodock/rendering.py`).
  `composite_summary()` now uses per-row adaptive heights while keeping columns
  aligned, eliminating the excessive whitespace caused by forcing all panels into
  a single uniform cell size.
- **PDF figure embedding preserves aspect ratio** (`autodock/reporting.py`).
  Images are scaled to the full text width while maintaining their original
  aspect ratio, avoiding the previous forced 16×12 cm stretch.
- **2D interaction SVG vector output** (`autodock/rendering.py`,
  `autodock/post_dock_pipeline.py`). `render_interactions_2d()` now supports
  `output_svg` for a true vector molecule diagram, and the workflow emits
  `2d_interactions.svg` by default.
- **JSON NaN/Inf sanitization** (`autodock/benchmark.py`).
  `benchmark_summary.json` and `repeat_docking_summary.json` now serialize
  non-finite floats as `null`, ensuring strict JSON parsers can read the files.
- **Validation metric layering** (`autodock/validation.py`, `autodock/core.py`,
  `autodock/benchmark.py`). Redocking results now expose `success_raw`,
  `success_min`, `success_cascade`/`rmsd_cascade`/`rescued_by`,
  `success_consensus`/`rmsd_consensus`, and `success_rescored`/`rmsd_rescored`/
  `rescored_by`. `success`/`rmsd` remain the final reported value after all
  rescue tiers. Benchmark CSV and JSON summaries include the new fields and
  per-rescue-tier counts.
- **MM-GBSA non-finite filtering** (`autodock/rescoring.py`). Simplified MM-GBSA
  now drops poses that yield `NaN` or `Inf` binding energies before ranking,
  and aborts coordinate mapping when any heavy atom cannot be aligned.
- **Distribution-based clash metrics** (`autodock/validation.py`, `autodock/core.py`).
  `compute_clash_score()` now reports median overlap, 90th-percentile overlap,
  fraction of atom pairs over threshold, and auto-selects a heavy-atom-only
  (0.5 Å) or explicit-H (1.2 Å) threshold. Legacy `clash_score`, `n_clashes`,
  and `is_acceptable` fields are preserved.
- **3D interaction distance labels and journal presets** (`autodock/rendering.py`).
  Interaction scenes can now annotate each dashed line with its distance in Å
  (`show_distance_labels`). `color_scheme` accepts journal presets
  (`nature`, `cell`, `acs`, `science`) that map to the existing publication
  colour schemes. 2D interaction diagrams fall back to the canonical
  `INTERACTION_COLORS` palette when no colour is provided.

### Fixed
- `METHODS.md` updated to reflect the actual implementation: removed obsolete
  P2Rank `prob ≥ 0.5` hard cutoff, corrected clash-score definition, and
  clarified that `consensus_affinity` is not currently computed.

### Added
- **Three-tier cascade fallback rescoring** (`autodock/validation.py`). When Vina top-1 RMSD ≥ 2.0 Å, automatically triggers tier-2 (IFP re-ranking + re-dock with 50 poses, e=8) and tier-3 (MM-GBSA on top 5 poses). Improves top-1 success from 35% → 55% (+20 pp) on 20-target benchmark with zero degradation.
- **Flexible receptor docking** (`autodock/preparation.py`, `autodock/docking.py`, `autodock/validation.py`). Opt-in fallback (`use_flexible_receptor=True`) that detects nearby residues, prepares flexible receptor PDBQT via Meeko, and re-docks with reduced exhaustiveness. POC on 1B9S improved best RMSD from 2.11 Å → 1.05–1.21 Å @ rank #1. Disabled by default due to runtime cost (~5 min per target).
- **MM-GBSA rescoring** (`autodock/rescoring.py`). OpenMM + OpenFF-based MM-GBSA ΔG computation for pose re-ranking. Functional for 17/20 targets after NaN bug fix.
- `_perturb_zero_charges()` helper in `rescoring.py` — works around OpenFF `SMIRNOFFTemplateGenerator` rejecting all-zero formal charges as "not user-provided".
- `compute_top_n_best_rmsd_from_all_poses()` in `validation.py` — evaluates top-N poses (default N=3) for best RMSD, enabling top-N success rate reporting.
- `find_nearby_residues()` and `prepare_flexible_receptor()` in `preparation.py` — Cα distance search and Meeko-based flexible receptor preparation.
- `_strip_flexible_residues_from_pdbqt_block()` in `utils.py` — strips receptor side-chain atoms from Vina flex output before RMSD computation.
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
- **Removed shape similarity and strain energy rescoring** (`23fe287`). Simplified pipeline to Vina → IFP → MM-GBSA only. Shape/strain methods were unreliable on multi-MODEL PDBQT and contributed no rescues.
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
- **Baseline `best_rmsd` under-reporting bug** — pre-Jun-5 code lacked coordinate-based fallback in `compute_best_rmsd_from_all_poses()` when RDKit `GetBestRMS` failed on topology mismatch. Stored `benchmark_results_current/` JSON/CSV were stale; manually recomputed all 20 targets. Corrected metrics: top-1=7/20 (35%), top-3=10/20 (50%), best-achievable=14/20 (70%).
- **MM-GBSA NaN cascade bug** (4-layer fix in `rescoring.py` + `validation.py`):
  1. `validation.py`: `Chem.MolToSmiles(Chem.RemoveHs(crystal_mol))` prevents `[HH]` SMILES that break RDKit re-parsing.
  2. `rescoring.py`: Detect NaN Gasteiger charges → fallback to `formal_charge` + `_perturb_zero_charges()` (±1e-6 e) to bypass `SMIRNOFFTemplateGenerator._molecule_has_user_charges()` zero-charge rejection.
  3. `rescoring.py`: Pass `molecules=[offmol_base]` to `create_system()` so assigned charges are reused instead of triggering unavailable `am1bcc`.
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
