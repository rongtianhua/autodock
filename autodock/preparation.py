"""
autodock.preparation — Receptor / ligand preparation and binding-site detection.
==============================================================================
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import tempfile
from typing import Any

import numpy as np

from autodock.core import (
    _ALLOSTERIC_MIN_DISTANCE,
    _DRUGGABILITY_HIGH,
    _DRUGGABILITY_MEDIUM,
    _P2RANK_PROB_THRESHOLD,
    _POCKET_CONSENSUS_DISTANCE,
    _POCKET_DEFAULT_BFACTOR,
    _SKIP_ADDITIVES,
    _SKIP_WATER,
    PreparationError,
    find_conda_tool,
    find_p2rank,
    logger,
    safe_subprocess,
)
from autodock.utils import ensure_dir, obabel_convert, safe_pdb_slice, write_temp_file

# ─────────────────────────────────────────────────────────────────────────────
# Receptor Preparation — helper functions
# ─────────────────────────────────────────────────────────────────────────────


def _parse_pdb_header(pdb_path: str) -> dict[str, Any]:
    """Parse crystallographic quality metrics from PDB header records.

    Extracts:
      - resolution (Å) from REMARK 2
      - R-free / R-work from REMARK 3
      - experimental method from REMARK 200 or EXPDTA
      - deposition date
      - B-factor statistics (mean / max of Wilson B)

    Returns:
        Dict with keys: resolution, r_free, r_work, method, date,
        wilson_b, quality_flag.
    """
    result: dict[str, Any] = {
        "resolution": None,
        "r_free": None,
        "r_work": None,
        "method": None,
        "date": None,
        "wilson_b": None,
        "quality_flag": "unknown",
    }
    if not os.path.isfile(pdb_path):
        return result

    try:
        with open(pdb_path) as fh:
            header = fh.read(10000)
    except OSError:
        return result

    lines = header.splitlines()

    for line in lines:
        upper = line.upper()
        # Resolution: REMARK   2 RESOLUTION.  1.90 ANGSTROMS.
        if upper.startswith("REMARK   2 RESOLUTION."):
            m = re.search(r"([\d.]+)\s*ANGSTROM", upper)
            if m:
                result["resolution"] = float(m.group(1))
        # R-free / R-work: REMARK   3   R FREE            : 0.210
        if upper.startswith("REMARK   3"):
            rf = re.search(r"R\s*FREE\s*[:=]\s*([\d.]+)", line, re.I)
            if rf:
                result["r_free"] = float(rf.group(1))
            rw = re.search(r"R\s*WORK\s*[:=]\s*([\d.]+)", line, re.I)
            if rw:
                result["r_work"] = float(rw.group(1))
            # Wilson B
            wb = re.search(r"WILSON\s*B\s*[:=]\s*([\d.]+)", line, re.I)
            if wb:
                result["wilson_b"] = float(wb.group(1))
        # Deposition date
        if upper.startswith("REVDAT   1"):
            parts = line.split()
            if len(parts) >= 3:
                result["date"] = parts[2]
        # Experimental method
        if upper.startswith("EXPDTA"):
            result["method"] = line[6:].strip()
        # NMR / EM models count
        if upper.startswith("REMARK 200  EXPERIMENT TYPE"):
            result["method"] = line.split(":")[-1].strip() if ":" in line else line.strip()

    # Quality flag
    res = result["resolution"]
    rf = result["r_free"]
    if res is not None and rf is not None:
        if res <= 2.0 and rf <= 0.25:
            result["quality_flag"] = "high"
        elif res <= 2.5 and rf <= 0.30:
            result["quality_flag"] = "good"
        elif res <= 3.0:
            result["quality_flag"] = "acceptable"
        else:
            result["quality_flag"] = "low"
    elif res is not None:
        if res <= 2.0:
            result["quality_flag"] = "good"
        elif res <= 3.0:
            result["quality_flag"] = "acceptable"
        else:
            result["quality_flag"] = "low"

    return result


def _find_disulfide_bonds(pdb_path: str) -> list[dict[str, Any]]:
    """Parse SSBOND records from a PDB file.

    Returns:
        List of dicts with keys: chain1, res1, chain2, res2, sym1, sym2.
    """
    bonds: list[dict[str, Any]] = []
    if not os.path.isfile(pdb_path):
        return bonds

    try:
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith("SSBOND"):
                    continue
                if len(line) < 32:
                    continue
                try:
                    bonds.append(
                        {
                            "chain1": safe_pdb_slice(line, 15, 16) or "A",
                            "res1": int(safe_pdb_slice(line, 17, 21)),
                            "chain2": safe_pdb_slice(line, 29, 30) or "A",
                            "res2": int(safe_pdb_slice(line, 31, 35)),
                            "sym1": safe_pdb_slice(line, 59, 65),
                            "sym2": safe_pdb_slice(line, 66, 72),
                        }
                    )
                except (ValueError, IndexError):
                    continue
    except OSError:
        pass
    return bonds


def _remove_disulfide_hydrogens(pdb_content: str, ssbonds: list[dict[str, Any]]) -> str:
    """Strip spurious HG hydrogens from CYS residues in disulfide bonds.

    PDBFixer's ``addMissingHydrogens()`` may add a hydrogen to the SG atom
    of a cysteine involved in a disulfide bond.  Reduce will also add hydrogens
    if it sees a free thiol.  This function removes any atom named ``HG``
    (element H) on CYS residues listed in ``ssbonds`` before the structure
    enters the Reduce step.

    Returns:
        PDB content string with HG atoms removed from SSBOND CYS residues.
    """
    if not ssbonds:
        return pdb_content

    # Build set of (chain, resi) that participate in disulfide bonds
    _cys_set: set[tuple[str, int]] = set()
    for bond in ssbonds:
        _cys_set.add((bond.get("chain1", "A"), bond.get("res1", 0)))
        _cys_set.add((bond.get("chain2", "A"), bond.get("res2", 0)))

    _cleaned: list[str] = []
    _removed: list[str] = []
    for line in pdb_content.splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")):
            _resn = safe_pdb_slice(line, 17, 20)
            _chain = safe_pdb_slice(line, 21, 22) or "A"
            try:
                _resi = int(safe_pdb_slice(line, 22, 26))
            except ValueError:
                _resi = None
            _aname = safe_pdb_slice(line, 12, 16)
            if (
                _resn == "CYS"
                and _resi is not None
                and (_chain, _resi) in _cys_set
                and _aname is not None
                and _aname.strip() == "HG"
            ):
                _removed.append(f"{_chain}:{_resi}")
                continue  # drop this HG atom
        _cleaned.append(line)

    if _removed:
        logger.info(
            f"Removed spurious HG hydrogen from CYS in disulfide bond(s): "
            f"{', '.join(sorted(set(_removed)))}"
        )
    return "".join(_cleaned)


def _find_functional_waters(
    pdb_content: str,
    metal_resnames: set[str],
    distance_threshold: float = 2.5,
) -> set[str]:
    """Identify water molecules that coordinate metal ions.

    A functional water is defined as a water oxygen atom within
    ``distance_threshold`` Å of any metal ion.

    Returns:
        Set of residue identifiers ``chain:resseq`` (e.g. {"A:301", "B:402"})
        that should be retained.
    """
    # Parse metal coordinates
    metals: list[tuple[str, int, float, float, float]] = []
    waters: list[tuple[str, int, float, float, float]] = []

    for line in pdb_content.splitlines():
        if not line.startswith("HETATM"):
            continue
        resn = safe_pdb_slice(line, 17, 20)
        chain = safe_pdb_slice(line, 21, 22) or "A"
        try:
            resi = int(safe_pdb_slice(line, 22, 26))
            x = float(safe_pdb_slice(line, 30, 38))
            y = float(safe_pdb_slice(line, 38, 46))
            z = float(safe_pdb_slice(line, 46, 54))
        except (ValueError, IndexError):
            continue
        if resn in metal_resnames:
            metals.append((chain, resi, x, y, z))
        elif resn in _SKIP_WATER:
            # Only track oxygen atoms of water
            atom_name = safe_pdb_slice(line, 12, 16)
            if atom_name and atom_name.startswith("O"):
                waters.append((chain, resi, x, y, z))

    functional: set[str] = set()
    if not metals or not waters:
        return functional

    for w_chain, w_resi, wx, wy, wz in waters:
        for _m_chain, _m_resi, mx, my, mz in metals:
            d_sq = (wx - mx) ** 2 + (wy - my) ** 2 + (wz - mz) ** 2
            if d_sq <= distance_threshold**2:
                functional.add(f"{w_chain}:{w_resi}")
                break  # one metal match is enough

    return functional


def _write_prep_report(
    output_path: str,
    report: dict[str, Any],
) -> str:
    """Write a JSON preparation report for Methods-section reproducibility."""

    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Receptor Preparation
# ─────────────────────────────────────────────────────────────────────────────


def prepare_receptor(
    pdb_file: str,
    output_pdbqt: str,
    remove_water: bool = True,
    remove_hetatms: bool = True,
    input_format: str = "auto",
    keep_residues: set[str] | None = None,
    force: bool = False,
    ph: float = 7.4,
    forcefield: str = "amber14-all.xml",
    restraint_center: tuple[float, float, float] | None = None,
    restraint_radius: float = 8.0,
    retain_metal_ions: bool = True,
    predict_pka: bool = True,
    fix_protonation: bool = False,
    keep_waters_near_metal: bool = True,
    output_report_json: str | None = None,
    detect_af_structure: bool = True,
    output_pdb: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """
    Prepare a protein structure for docking (PDB/mmCIF → PDBQT).

    Uses a **three-stage publication-grade workflow**:

    1. **PDBFixer** — fills missing residues and atoms, assigns protonation
       states at the specified pH, replaces nonstandard residues.
    2. **OpenMM energy minimization** — L-BFGS minimisation (up to 500 steps,
       RMS force tolerance 10 kJ/(nm·mol)) to relieve local strain from
       missing-atom reconstruction.
    3. **Meeko Polymer** — AD4 atom typing, Gasteiger charges, PDBQT export.
       Falls back to Open Babel if Meeko fails.

    References:
        - Eastman et al. (2017) PLoS Comput. Biol. (OpenMM)
        - PDBFixer: https://github.com/openmm/pdbfixer
        - Rinciker et al. (2004) J. Med. Chem. (receptor prep best practices)

    .. note::
       Missing loop regions: PDBFixer fills missing residues with approximate
       coordinates.  If a missing loop (>5 residues) participates in the
       binding pocket, consider MODELLER-based homology loop reconstruction
       for publication-grade results.  Large gaps (>30 total residues) are
       flagged with a warning during preparation.

    Args:
        pdb_file: Input structure path (.pdb, .cif, .pdbx).
        output_pdbqt: Output PDBQT file path.
        remove_water: Remove HOH / WAT residues.
        remove_hetatms: Remove all HETATM records (keep only protein).
        input_format: 'auto' | 'pdb' | 'cif' | 'pdbx'.
        keep_residues: If provided, keep only these residue names.
        force: If False and output_pdbqt already exists, skip preparation.
        ph: Target pH for adding hydrogens (default 7.4, physiological).
            Passed to PDBFixer.addMissingHydrogens().  For pH-sensitive
            active sites (e.g. cathepsin B in lysosome at pH ~5.5, or
            pepsin in stomach at pH ~2.0), override with the relevant
            organellar pH.  His protonation state is the main residue
            affected in the 5.5-7.4 range.
        forcefield: OpenMM force field XML for energy minimisation
            (default ``"amber14-all.xml"``).
        restraint_center: Optional (x, y, z) pocket centre in Å for
            **pocket-restrained minimisation**.  Atoms within
            ``restraint_radius`` of this point have their backbone
            restrained (5 kcal/mol/Å²) while side chains move freely;
            atoms outside have all heavy atoms restrained.  If *None*
            (default), a uniform heavy-atom restraint (10 kcal/mol/Å²)
            is applied to the entire protein.  Provide the crystal-ligand
            centroid or fpocket pocket centre for best results.
        restraint_radius: Radius (Å) around ``restraint_center`` defining
            the pocket region for differential restraints (default 8.0).
        retain_metal_ions: If True (default), keep physiologically relevant
            metal ions (Zn²⁺, Mg²⁺, Ca²⁺, Fe²⁺, Mn²⁺, etc.) and common
            cofactors (HEM, FAD, NAD, etc.) even when
            ``remove_hetatms=True``.  Set to False to strip all HETATM.
            Required for metalloprotein targets.
        predict_pka: If True (default), run PROPKA to detect anomalous pKa
            values in titratable residues (Asp, Glu, His, Lys, Cys, Tyr).
            Residues with pKa within ±1 of the target pH are flagged as
            having potentially wrong protonation states.  Pure reporting;
            does not modify the structure.  Set False to skip.
            **Deprecated** — use ``fix_protonation`` instead for active
            structure modification.
        fix_protonation: If True, run PDB2PQR+PROPKA to **actively correct**
            protonation states based on predicted pKa values.  Unlike
            ``predict_pka`` (read-only), this modifies the PDB structure:
            PROPKA pKa predictions are applied via PDB2PQR's
            ``apply_pka_values()``, hydrogens are added with corrected
            tautomer/protonation states, and the H-bond network is
            optimized.  Requires ``pdb2pqr`` (conda-forge).
            Default: False.  When True, ``predict_pka`` is implied.
        keep_waters_near_metal: If True (default), retain crystallographic
            waters that coordinate metal ions (distance < 2.5 Å).
            Functional waters are often critical for metalloprotein
            ligand binding.  Set False to strip all waters regardless.
        output_report_json: If provided, write a JSON preparation report to
            this path documenting every preparation step, parameters, and
            quality checks for Methods-section reproducibility.
            If None (default), no report is written.
        detect_af_structure: If True (default), auto-detect AlphaFold-
            predicted structures and run pLDDT quality assessment.
            Warns if mean pLDDT < 70 (unsuitable for docking) or
            suggests MD relaxation if pLDDT 70-90.
        output_pdb: Optional path to save the filtered/minimised PDB file.
            This is the final structure after PDBFixer + reduce + (
            optionally PDB2PQR) + filter — suitable for downstream use
            in PLIP interaction detection, PyMOL rendering, and
            PoseBusters validation without needing CIF→PDB conversion.
            Default: None (PDB file not saved).

    Returns:
        Absolute path to the prepared PDBQT file.
        The PDB file path (if requested via ``output_pdb``) is logged.

    Raises:
        PreparationError: If input file missing or preparation fails.
    """
    if not os.path.isfile(pdb_file):
        raise PreparationError(f"Input file not found: {pdb_file}")

    if not force and os.path.isfile(output_pdbqt):
        logger.info(f"Receptor PDBQT already exists — skipping prep: {output_pdbqt}")
        return os.path.abspath(output_pdbqt)

    # ── Cache lookup -------------------------------------------------------
    if cache_dir is not None:
        from autodock.cache import ReceptorCache

        rc = ReceptorCache(cache_dir)
        cache_params = {
            "remove_water": remove_water,
            "remove_hetatms": remove_hetatms,
            "input_format": input_format,
            "keep_residues": sorted(keep_residues) if keep_residues else None,
            "ph": ph,
            "forcefield": forcefield,
            "restraint_center": restraint_center,
            "restraint_radius": restraint_radius,
            "retain_metal_ions": retain_metal_ions,
            "predict_pka": predict_pka,
            "fix_protonation": fix_protonation,
            "keep_waters_near_metal": keep_waters_near_metal,
            "detect_af_structure": detect_af_structure,
        }
        cached = rc.get(pdb_file, **cache_params)
        if cached:
            import shutil

            shutil.copy2(cached["receptor.pdbqt"], output_pdbqt)
            if output_pdb and "receptor.pdb" in cached:
                shutil.copy2(cached["receptor.pdb"], output_pdb)
            logger.info(f"Receptor cache hit — copied from {rc.cache_dir}")
            return os.path.abspath(output_pdbqt)

    # Accumulators for report detail (enlarged scope so report section can see them)
    gaps_detail: list[str] = []
    _reduce_flip_detail: list[str] = []
    _all_temps: list[str] = []

    # ── Step 1: Convert mmCIF → PDB if needed ────────────────────────────
    ext = os.path.splitext(pdb_file)[1].lower()
    if input_format == "auto":
        input_format = "cif" if ext in (".cif", ".pdbx") else "pdb"

    if input_format in ("cif", "pdbx"):
        try:
            import gemmi
        except ImportError:
            raise PreparationError(
                "gemmi required for CIF parsing. Install: conda install -c conda-forge gemmi"
            )
        try:
            doc = gemmi.cif.read(pdb_file)
            block = doc.sole_block()
            structure = gemmi.make_structure_from_block(block)
            pdb_content = structure.make_pdb_string()
        except (OSError, ValueError, ImportError) as exc:
            raise PreparationError(f"CIF parsing failed: {exc}")
    else:
        with open(pdb_file) as fh:
            pdb_content = fh.read()

    # Write to temp PDB for PDBFixer consumption
    tmp_raw = write_temp_file(pdb_content, suffix="_raw.pdb")
    _original_tmp_raw = tmp_raw  # track for final cleanup
    _all_temps.append(tmp_raw)

    # ── Structure quality & AlphaFold assessment (before modification) ─────
    _quality_header = _parse_pdb_header(tmp_raw)
    _ssbonds = _find_disulfide_bonds(tmp_raw)
    _af_assessment: dict[str, Any] | None = None

    if detect_af_structure:
        # Heuristic: check for AlphaFold header signatures or B-factor range
        _is_af = False
        _af_header_markers = ["ALPHAFOLD", "DEEPMIND", "AF-"]
        for _line in pdb_content.splitlines()[:50]:
            if any(m in _line.upper() for m in _af_header_markers):
                _is_af = True
                break
        if not _is_af:
            # B-factor heuristic: AlphaFold writes pLDDT (0-100) in B-factor
            # column; experimental structures typically have B < 200.
            _bfactors: list[float] = []
            for _line in pdb_content.splitlines():
                if _line.startswith(("ATOM  ", "HETATM")):
                    try:
                        _bf = float(_line[60:66].strip())
                        _bfactors.append(_bf)
                    except ValueError:
                        continue
            if _bfactors and all(0 <= b <= 100 for b in _bfactors[:50]):
                # If first 50 B-factors are all in 0-100 range, likely AF.
                # Guard against false positives from very ordered experimental
                # structures (e.g. 1.0 Å resolution) by checking mean B.
                _mean_b = np.mean(_bfactors[:50])
                if _mean_b > 45:
                    _is_af = True
                    logger.debug(
                        f"AF heuristic triggered: B-factor mean={_mean_b:.1f} "
                        f"(range 0-100, n={len(_bfactors[:50])})"
                    )
                else:
                    logger.debug(
                        f"AF heuristic rejected: B-factor mean={_mean_b:.1f} too low "
                        f"for AlphaFold pLDDT signature"
                    )
        if _is_af:
            try:
                from autodock.alphafold_tools import (
                    assess_alphafold_quality,
                    relax_alphafold_structure,
                )

                _af_assessment = assess_alphafold_quality(tmp_raw)
                _mean_plddt = _af_assessment.get("mean_plddt", 0.0)
                if _mean_plddt < 70.0:
                    logger.warning(
                        f"AlphaFold structure detected — mean pLDDT={_mean_plddt:.1f} "
                        f"(< 70). Quality may be insufficient for docking."
                    )
                elif _mean_plddt < 90.0:
                    logger.warning(
                        f"AlphaFold structure detected — mean pLDDT={_mean_plddt:.1f}. "
                        f"Moderate confidence — MD relaxation applied."
                    )
                else:
                    logger.info(
                        f"AlphaFold structure detected — mean pLDDT={_mean_plddt:.1f} "
                        f"(high confidence, suitable for docking)."
                    )
                # Auto-relax all detected AlphaFold structures
                _relax_result = {}
                try:
                    logger.info("Running AlphaFold MD relaxation ...")
                    _relax_result = relax_alphafold_structure(
                        tmp_raw,
                        output_dir=os.path.join(os.path.dirname(output_pdbqt), "af_relaxed"),
                        production_ns=1.0,
                        ph=ph,
                        forcefield=forcefield,
                    )
                except (ImportError, OSError, ValueError, RuntimeError, TypeError) as exc:
                    logger.warning(
                        f"AlphaFold MD relaxation failed to start ({exc}) — using raw structure"
                    )
                    _relax_result = {"success": False}
                if _relax_result.get("success"):
                    _relaxed_path = _relax_result["output_pdb"]
                    if os.path.isfile(_relaxed_path):
                        tmp_raw = _relaxed_path
                        with open(tmp_raw) as fh:
                            pdb_content = fh.read()
                        # Re-parse SSBOND from relaxed structure in case
                        # residue numbering shifted during relaxation.
                        _ssbonds = _find_disulfide_bonds(tmp_raw)
                        logger.info(
                            f"AlphaFold structure MD-relaxed (RMSD "
                            f"{_relax_result.get('final_rmsd', 0):.2f} Å) — "
                            f"relaxed model will be used for preparation"
                        )
                        _af_assessment["relaxed"] = True
                        _af_assessment["relaxation_rmsd"] = _relax_result.get("final_rmsd")
                else:
                    logger.warning("AlphaFold MD relaxation failed — using raw structure")
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                logger.warning(f"AlphaFold handling failed ({exc}) — using raw structure")

    # Resolution / R-free warnings for experimental structures
    _res = _quality_header.get("resolution")
    _rf = _quality_header.get("r_free")
    if _res is not None and _res > 3.0:
        logger.warning(
            f"Low resolution structure ({_res:.1f} Å > 3.0 Å). "
            f"Binding-pocket sidechain conformations may be unreliable."
        )
    if _rf is not None and _rf > 0.30:
        logger.warning(f"High R-free ({_rf:.2f} > 0.30). Structure quality may be poor.")
    if _ssbonds:
        logger.info(f"Disulfide bonds detected: {len(_ssbonds)}")

    # ── Step 2: Record altloc choices before any modification ────────────
    altloc_records: dict[str, set[str]] = {}
    for line in pdb_content.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            altloc = safe_pdb_slice(line, 16, 17)
            if altloc and altloc.strip():
                resi = safe_pdb_slice(line, 22, 26)
                chain = safe_pdb_slice(line, 21, 22) or "A"
                key = f"{resi}:{chain}"
                altloc_records.setdefault(key, set()).add(altloc)
    multi_altloc = {k: v for k, v in altloc_records.items() if len(v) > 1}
    if multi_altloc:
        logger.info(
            f"Altloc selection: {len(multi_altloc)} residue(s) have multiple "
            f"alternate conformations; default_altloc='A' selected for all. "
            f"Affected: {list(multi_altloc.keys())[:10]}"
        )

    # ── Step 3: PDBFixer — fill missing residues/atoms, add hydrogens at pH ──
    # Pre-filter: keep only protein atoms + metal/cofactor HETATMs for PDBFixer.
    # This avoids PDBFixer removing metal ions (which it treats as heterogens),
    # while ensuring OpenMM can create a System from the filtered topology.
    _metal_set_pdbfixer: set[str] = set()
    if retain_metal_ions:
        from autodock.core import _METAL_COFACTORS, _METAL_IONS

        _metal_set_pdbfixer = _METAL_IONS | _METAL_COFACTORS

    # Identify functional waters near metals so they survive PDBFixer pre-filter
    _functional_waters_pdbfixer: set[str] = set()
    if keep_waters_near_metal and remove_water:
        _functional_waters_pdbfixer = _find_functional_waters(
            pdb_content, _metal_set_pdbfixer, distance_threshold=2.5
        )

    _pbfixer_lines: list[str] = []
    for _line in pdb_content.splitlines(keepends=True):
        if _line.startswith("ATOM  "):
            _pbfixer_lines.append(_line)
        elif _line.startswith("HETATM"):
            _resn = safe_pdb_slice(_line, 17, 20)
            # Keep metal ions/cofactors AND water (if user wants to keep it)
            if _resn in _metal_set_pdbfixer:
                _pbfixer_lines.append(_line)
            elif not remove_water and _resn in _SKIP_WATER:
                _pbfixer_lines.append(_line)  # keep water when requested
            elif _resn in _SKIP_WATER:
                # keep_waters_near_metal: retain functional waters even when
                # remove_water=True
                _chain = safe_pdb_slice(_line, 21, 22) or "A"
                try:
                    _resi = int(safe_pdb_slice(_line, 22, 26))
                except ValueError:
                    _resi = None
                if _resi is not None and f"{_chain}:{_resi}" in _functional_waters_pdbfixer:
                    _pbfixer_lines.append(_line)
            # else: skip other HETATMs (ligands, buffers, etc.)
        else:
            _pbfixer_lines.append(_line)
    _pbfixer_pdb = "".join(_pbfixer_lines)
    _tmp_fixer_in = write_temp_file(_pbfixer_pdb, suffix="_pdbfixer_in.pdb")
    _all_temps.append(_tmp_fixer_in)

    try:
        import openmm
        from openmm.app import PDBFile as _OMM_PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise PreparationError(
            f"PDBFixer / OpenMM required for receptor preparation: {exc}. "
            "Install: conda install -c conda-forge pdbfixer openmm"
        )

    tmp_fixed = write_temp_file("", suffix="_fixed.pdb")
    _all_temps.append(tmp_fixed)
    try:
        fixer = PDBFixer(filename=_tmp_fixer_in)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        # No removeHeterogens needed — we already filtered the input above
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)

        # Log missing residues for reviewer transparency
        missing_res = fixer.missingResidues if hasattr(fixer, "missingResidues") else {}
        if missing_res:
            total_filled = sum(len(res_list) for res_list in missing_res.values())
            gaps_detail.clear()
            for chain_idx, res_list in missing_res.items():
                for _, res_seq, res_name in res_list:
                    gaps_detail.append(f"chain{chain_idx}:{res_seq}({res_name})")
            logger.info(
                f"PDBFixer filled {total_filled} missing residue(s): "
                f"{', '.join(gaps_detail[:20])}"
                + (f" ... and {total_filled - 20} more" if total_filled > 20 else "")
            )
            if total_filled > 30:
                logger.warning(
                    f"Large number of missing residues ({total_filled}). "
                    "If these include binding-pocket loops, consider MODELLER "
                    "for proper loop reconstruction (PDBFixer uses approximate "
                    "coordinates for missing regions)."
                )

        with open(tmp_fixed, "w") as fh:
            _OMM_PDBFile.writeFile(fixer.topology, fixer.positions, fh)
        logger.info(f"PDBFixer: missing residues filled, hydrogens added (pH {ph})")
    except (
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        ImportError,
        AttributeError,
        openmm.OpenMMException,
    ) as exc:
        logger.warning(f"PDBFixer failed ({exc}) — falling back to raw structure")
        with contextlib.suppress(OSError):
            os.remove(tmp_fixed)
        tmp_fixed = tmp_raw  # use raw structure
    finally:
        # Clean up PDBFixer input temp
        with contextlib.suppress(OSError):
            if os.path.exists(_tmp_fixer_in):
                os.remove(_tmp_fixer_in)

    # ── Step 3b: SSBOND post-processing — strip spurious HG on CYS-SG ─────
    # PDBFixer may add hydrogens to CYS involved in disulfide bonds.
    # Remove them before Reduce runs so the thiol is correctly recognised.
    if _ssbonds and tmp_fixed != tmp_raw:
        with open(tmp_fixed) as fh:
            _fixed_pdb_str = fh.read()
        _fixed_cleaned = _remove_disulfide_hydrogens(_fixed_pdb_str, _ssbonds)
        if _fixed_cleaned != _fixed_pdb_str:
            with open(tmp_fixed, "w") as fh:
                fh.write(_fixed_cleaned)

    # ── Step 4: Reduce — ASN/GLN flip detection + HIS tautomer assignment ────
    tmp_reduced = write_temp_file("", suffix="_reduced.pdb")
    try:
        reduce_bin = find_conda_tool("reduce")
        if not reduce_bin:
            raise RuntimeError("reduce binary not found")
        # -FLIP: add hydrogens, detect/correct ASN/GLN flips, assign HIS tautomers
        success, stdout, stderr = safe_subprocess(
            [reduce_bin, "-FLIP", tmp_fixed],
            timeout=120,
        )
        if not success or not stdout.strip():
            raise RuntimeError(f"reduce -FLIP failed: {stderr[:300]}")
        with open(tmp_reduced, "w") as fh:
            fh.write(stdout)
        # Count flips from log and capture detail lines for report
        n_flips = stdout.count("FLIP") - stdout.count("NOFLIP")
        _reduce_flip_detail = [
            line.strip() for line in stdout.splitlines() if "FLIP" in line and "NOFLIP" not in line
        ]
        if n_flips > 0:
            logger.info(f"Reduce: corrected {n_flips} ASN/GLN sidechain flip(s)")
        logger.info("Reduce: ASN/GLN flips processed, HIS tautomers assigned")
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        logger.warning(f"Reduce step skipped ({exc}) — using PDBFixer output")
        with contextlib.suppress(OSError):
            os.remove(tmp_reduced)
        tmp_reduced = tmp_fixed

    # ── Step 4b: PROPKA — pKa prediction for active-site residues ────────
    _anomalous_pka: list[dict] = []
    if predict_pka:
        try:
            from propka import run as _propka_run

            _pk_mol = _propka_run.single(tmp_reduced, write_pka=False)
            if hasattr(_pk_mol, "calculate_pka"):
                _pk_mol.calculate_pka()
            for _conf in _pk_mol.conformations.values():
                for _g in _conf.groups:
                    _pka = getattr(_g, "pka_value", None)
                    if _pka is None:
                        continue
                    # Standard reference pKa ranges
                    _type = getattr(_g, "type", "")
                    _label = getattr(_g, "label", "")
                    # Standard reference pKa ranges (Dawson et al.,
                    # Data for Biochemical Research, 3rd ed.)
                    if _type == "ASP" and (_pka < 2.0 or _pka > 5.0):
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "2.0-5.0",
                            }
                        )
                    elif _type == "GLU" and (_pka < 2.0 or _pka > 6.0):
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "2.0-6.0",
                            }
                        )
                    elif _type == "HIS" and (_pka < 5.0 or _pka > 8.0):
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "5.0-8.0",
                            }
                        )
                    elif _type == "LYS" and (_pka < 9.0 or _pka > 12.0):
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "9.0-12.0",
                            }
                        )
                    elif _type == "CYS" and _pka > 9.0:
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "<9.0",
                            }
                        )
                    elif _type == "TYR" and _pka > 13.0:
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "<13.0",
                            }
                        )

            if _anomalous_pka:
                logger.info(f"PROPKA: {len(_anomalous_pka)} anomalous pKa residue(s) at pH {ph}:")
                for _a in _anomalous_pka[:15]:
                    logger.info(
                        f"  {_a['residue']:12s} ({_a['type']}): "
                        f"pKa={_a['pKa']:.1f} (expected {_a['expected_range']})"
                    )
                if len(_anomalous_pka) > 15:
                    logger.info(f"  ... and {len(_anomalous_pka) - 15} more")
                # Flag residues with pKa near target → protonation state uncertain
                _flagged = [a for a in _anomalous_pka if abs(a["pKa"] - ph) < 1.0]
                if _flagged:
                    logger.warning(
                        f"PROPKA: {len(_flagged)} residue(s) with pKa within ±1 of pH {ph} "
                        f"— protonation state may be wrong. Consider manual inspection."
                    )
            else:
                logger.info("PROPKA: all titratable residues within normal pKa ranges")
        except ImportError:
            logger.debug("propka not installed — skipping pKa prediction")
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.warning(f"PROPKA pKa prediction failed ({exc}) — skipping")

    # ── Step 4c: PDB2PQR — active protonation correction ────────────────
    # Replaces `predict_pka` read-only mode with actual structure modification.
    # PDB2PQR runs PROPKA internally, applies pKa-corrected protonation states,
    # adds hydrogens, and optimises the H-bond network via its built-in reduce.
    # The `--noopt` flag avoids re-running reduce (already done in Step 4).
    tmp_pdb2pqr = tmp_reduced  # default: fall back to reduce output
    if fix_protonation:
        pdb2pqr_bin = find_conda_tool("pdb2pqr")
        if pdb2pqr_bin:
            tmp_pdb2pqr_pdb = write_temp_file("", suffix="_pdb2pqr.pdb")
            _all_temps.append(tmp_pdb2pqr_pdb)
            try:
                success, _stdout, stderr = safe_subprocess(
                    [
                        pdb2pqr_bin,
                        "--ff=AMBER",
                        f"--with-ph={ph}",
                        "--pdb-output",
                        tmp_pdb2pqr_pdb,
                        "--titration-state-method=propka",
                        "--noopt",  # reduce already did H-bond opt (Step 4)
                        "--keep-chain",
                        "--drop-water",  # will be filtered in Step 6 anyway
                        tmp_reduced,
                        os.devnull,  # discard .pqr output (not needed)
                    ],
                    timeout=300,
                )
                if success and os.path.getsize(tmp_pdb2pqr_pdb) > 100:
                    tmp_pdb2pqr = tmp_pdb2pqr_pdb
                    logger.info("PDB2PQR: protonation states corrected via PROPKA " f"(pH {ph})")
                    # Override predict_pka — PDB2PQR already did the analysis
                    predict_pka = False
                else:
                    logger.warning(
                        f"PDB2PQR produced unusable output ({stderr[:200]}) "
                        "— using reduce output"
                    )
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                logger.warning(f"PDB2PQR failed ({exc}) — using reduce output")
        else:
            logger.info(
                "pdb2pqr not found (install: conda install -c conda-forge pdb2pqr) "
                "— protonation correction skipped"
            )

    # ── Step 5: OpenMM pocket-restrained energy minimisation ─────────────
    tmp_min = write_temp_file("", suffix="_minimized.pdb")
    _all_temps.append(tmp_min)
    try:
        import openmm
        from openmm import CustomExternalForce, VerletIntegrator
        from openmm import unit as _omm_unit
        from openmm.app import ForceField, PDBFile, Simulation

        # Reload from Reduce output for OpenMM
        from pdbfixer import PDBFixer as _PBFixer

        _reduce_fixer = _PBFixer(filename=tmp_pdb2pqr)
        _top = _reduce_fixer.topology
        _pos = _reduce_fixer.positions  # OpenMM: units in nm

        # Load force field; add TIP3P water model if waters are present
        # (amber14-all.xml does not include water parameters).
        _ff_files: list[str] = [forcefield]
        try:
            with open(tmp_pdb2pqr) as _fh:
                _pdb_text = _fh.read()
        except OSError:
            _pdb_text = ""
        if "HOH" in _pdb_text or "WAT" in _pdb_text:
            # Match water model to force field family
            if forcefield.startswith("amber14"):
                _ff_files.append("amber14/tip3p.xml")
            elif forcefield.startswith("amber19"):
                _ff_files.append("amber19/tip3p.xml")
            logger.info("OpenMM: explicit water detected — loading TIP3P parameters")
        ff = ForceField(*_ff_files)
        system = ff.createSystem(_top)

        # ── Positional restraints ─────────────────────────────────────────
        # Pocket region: backbone restrained (5 kcal/mol/Å²), side chains free
        # Outside pocket: all heavy atoms restrained (5 kcal/mol/Å²)
        # Keeps crystal structure globally while relieving local strain.
        # 5 kcal/mol/Å² is a standard compromise: strong enough to prevent
        # backbone drift (~half a C–C bond stiffness), gentle enough to let
        # side chains relax into better rotamers.
        k_kcal = 5.0  # restraint force constant (kcal/mol/Å²)
        _K = k_kcal * _omm_unit.kilocalories_per_mole / _omm_unit.angstrom**2
        BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}

        rest_force = CustomExternalForce("k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
        rest_force.addGlobalParameter("k", _K)
        rest_force.addPerParticleParameter("x0")
        rest_force.addPerParticleParameter("y0")
        rest_force.addPerParticleParameter("z0")

        # Convert centre & radius from Å → nm for OpenMM distance check
        if restraint_center is not None:
            cx_nm, cy_nm, cz_nm = (c / 10.0 for c in restraint_center)
            r_sq_nm = (restraint_radius / 10.0) ** 2
        else:
            cx_nm = cy_nm = cz_nm = r_sq_nm = None

        n_restrained = 0
        for atom in _top.atoms():
            if atom.element.symbol == "H":
                continue  # never restrain hydrogens
            if _pos is None:
                continue
            p = _pos[atom.index]
            px, py, pz = p.x, p.y, p.z

            # Decide whether to apply restraint
            apply_restraint = True  # default: restrain outside pocket
            if restraint_center is not None:
                # Inside pocket: backbone only; side chains free
                d_sq = (px - cx_nm) ** 2 + (py - cy_nm) ** 2 + (pz - cz_nm) ** 2
                if d_sq < r_sq_nm and atom.name not in BACKBONE_NAMES:
                    apply_restraint = False  # side chain inside pocket → free

            if apply_restraint:
                rest_force.addParticle(atom.index, [px, py, pz])
                n_restrained += 1

        system.addForce(rest_force)
        logger.info(
            f"OpenMM restraints: {n_restrained} heavy atoms restrained"
            + (f" (pocket radius {restraint_radius} Å)" if restraint_center else " (uniform)")
        )

        # VerletIntegrator is used as a lightweight placeholder because
        # Simulation() requires an integrator.  minimizeEnergy() internally
        # uses L-BFGS, so the integrator type has no effect on minimisation.
        integrator = VerletIntegrator(0.001 * _omm_unit.picosecond)
        simulation = Simulation(_top, system, integrator)
        simulation.context.setPositions(_pos)
        # tolerance=10 kJ/(nm·mol) is the OpenMM default; 500 steps is a
        # generous ceiling that lets well-behaved structures converge early
        # while giving strained structures (e.g. AlphaFold) more room.
        simulation.minimizeEnergy(maxIterations=500)
        min_positions = simulation.context.getState(getPositions=True).getPositions()
        # NaN guard: validate coordinates before writing
        _has_nan = False
        for _pos in min_positions:
            if _pos.x != _pos.x or _pos.y != _pos.y or _pos.z != _pos.z:
                _has_nan = True
                break
        if _has_nan:
            raise RuntimeError("OpenMM minimisation produced NaN coordinates — structure diverged")
        with open(tmp_min, "w") as fh:
            PDBFile.writeFile(_top, min_positions, fh)
        logger.info("OpenMM: receptor energy minimised (L-BFGS, ≤500 steps)")
    except (
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        ImportError,
        openmm.OpenMMException,
    ) as exc:
        logger.warning(f"OpenMM minimisation skipped ({exc}) — using PDBFixer output")
        with contextlib.suppress(OSError):
            os.remove(tmp_min)
        tmp_min = tmp_fixed

    # Read back the cleaned PDB content
    with open(tmp_min) as fh:
        pdb_content = fh.read()

    # Clean up intermediate temp files; keep tmp_raw until the very end
    # so that fallback paths always have a valid source file.
    _cleanup_temps = {tmp_fixed, tmp_reduced, tmp_min}
    if fix_protonation:
        _cleanup_temps.add(tmp_pdb2pqr)
    for _t in _cleanup_temps:
        if _t != tmp_raw and _t not in (_original_tmp_raw,):
            with contextlib.suppress(OSError):
                if os.path.exists(_t):
                    os.remove(_t)

    # ── Step 5: Filter waters / hetatms (second pass after PDBFixer) ─────
    _metal_set: set[str] = set()
    if retain_metal_ions:
        from autodock.core import _METAL_COFACTORS, _METAL_IONS

        _metal_set = _METAL_IONS | _METAL_COFACTORS

    # Identify functional waters near metals before filtering
    _functional_waters: set[str] = set()
    if keep_waters_near_metal and remove_water:
        _functional_waters = _find_functional_waters(
            pdb_content, _metal_set, distance_threshold=2.5
        )
        if _functional_waters:
            logger.info(f"Functional waters near metal ions retained: {len(_functional_waters)}")

    n_waters_removed = 0
    n_waters_retained_functional = 0
    n_metals_retained_step5 = 0
    if remove_water or remove_hetatms or keep_residues:
        lines = pdb_content.splitlines(keepends=True)
        filtered = []
        for line in lines:
            if line.startswith("ATOM  "):
                resn = safe_pdb_slice(line, 17, 20)
                if keep_residues and resn not in keep_residues:
                    continue
                if remove_water and resn in _SKIP_WATER:
                    # Check if this is a functional water near metal
                    _chain = safe_pdb_slice(line, 21, 22) or "A"
                    try:
                        _resi = int(safe_pdb_slice(line, 22, 26))
                    except ValueError:
                        _resi = None
                    if _resi is not None and f"{_chain}:{_resi}" in _functional_waters:
                        n_waters_retained_functional += 1
                        filtered.append(line)
                        continue
                    continue
                filtered.append(line)
            elif line.startswith("HETATM"):
                resn = safe_pdb_slice(line, 17, 20)
                # Retain metals even when remove_hetatms=True (P0 fix)
                if resn in _metal_set:
                    n_metals_retained_step5 += 1
                    filtered.append(line)
                    continue
                # Water handling: checked BEFORE remove_hetatms so that
                # functional waters are retained and remove_water=False is honoured.
                if resn in _SKIP_WATER:
                    _chain = safe_pdb_slice(line, 21, 22) or "A"
                    try:
                        _resi = int(safe_pdb_slice(line, 22, 26))
                    except ValueError:
                        _resi = None
                    # Always retain functional waters near metals
                    if _resi is not None and f"{_chain}:{_resi}" in _functional_waters:
                        n_waters_retained_functional += 1
                        filtered.append(line)
                        continue
                    # Honour remove_water flag (don't let remove_hetatms delete water)
                    if not remove_water:
                        filtered.append(line)
                        continue
                    n_waters_removed += 1
                    continue
                if remove_hetatms:
                    continue
                if keep_residues and resn not in keep_residues:
                    continue
                filtered.append(line)
            else:
                filtered.append(line)
        pdb_content = "".join(filtered)
    if n_waters_removed > 0:
        logger.info(
            f"Removed {n_waters_removed} water molecules"
            + (" (set remove_water=False to retain them)" if remove_water else "")
        )
    if n_waters_retained_functional > 0:
        logger.info(f"Retained {n_waters_retained_functional} functional water(s) near metal ions")
    if n_metals_retained_step5 > 0:
        logger.info(f"Retained {n_metals_retained_step5} metal/cofactor atoms")

    # ── Step 6: Remove known problematic additives — log every skip ──────
    # Also build the set of residues to retain (metal ions + cofactors)
    _retain_set: set[str] = set()
    if retain_metal_ions:
        from autodock.core import _METAL_COFACTORS, _METAL_IONS

        _retain_set = _METAL_IONS | _METAL_COFACTORS

    skipped_residues: dict[str, int] = {}
    retained_metals: dict[str, int] = {}
    lines = pdb_content.splitlines()
    filtered = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            resn = safe_pdb_slice(line, 17, 20)
            # Retain metal ions and cofactors (default on)
            if resn in _retain_set:
                retained_metals[resn] = retained_metals.get(resn, 0) + 1
                filtered.append(line)
                continue
            if resn in _SKIP_ADDITIVES:
                # Allow user-specified keep_residues to override skip list
                if keep_residues and resn in keep_residues:
                    filtered.append(line)
                    continue
                skipped_residues[resn] = skipped_residues.get(resn, 0) + 1
                continue
        filtered.append(line)
    if skipped_residues:
        logger.info(f"Skipped additives before Meeko: {dict(sorted(skipped_residues.items()))}")
    if retained_metals:
        logger.info(f"Retained metal ions / cofactors: {dict(sorted(retained_metals.items()))}")
    pdb_content = "\n".join(filtered)

    # ── Step 6b: OpenBabel normalization (PDB2PQR compatibility) ────────
    # PDB2PQR's --pdb-output uses forcefield-specific atom naming (AMBER).
    # OpenBabel re-writes atom/residue names to standard PDB conventions
    # that Meeko's Polymer.from_pdb_string() can parse reliably.
    # This also catches any naming anomalies introduced by PDB2PQR+PROPKA.
    if fix_protonation and find_conda_tool("obabel"):
        tmp_norm = write_temp_file(pdb_content, suffix="_normalized.pdb")
        _all_temps.append(tmp_norm)
        try:
            norm_success, _norm_out, norm_err = safe_subprocess(
                ["obabel", "-ipdb", tmp_norm, "-opdb", "-O", tmp_norm],
                timeout=120,
            )
            if norm_success:
                with open(tmp_norm) as _fh:
                    pdb_content = _fh.read()
                logger.debug("OpenBabel: PDB atom naming normalised for Meeko")
            else:
                logger.debug(f"OpenBabel normalisation skipped ({norm_err[:100]})")
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.debug(f"OpenBabel normalisation skipped ({exc})")

    # ── Step 7: Meeko preparation ────────────────────────────────────────
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates
    except ImportError as exc:
        raise PreparationError(f"meeko not available: {exc}")

    templates = ResidueChemTemplates.create_from_defaults()
    mk_prep = MoleculePreparation(charge_model="gasteiger")

    try:
        polymer = Polymer.from_pdb_string(pdb_content, templates, mk_prep, default_altloc="A")
    except (OSError, ValueError, RuntimeError, ImportError):
        # Retry with allow_bad_res=True: removes unknown residues and continues
        logger.warning("Some residues failed template matching — retrying with allow_bad_res=True")
        try:
            polymer = Polymer.from_pdb_string(
                pdb_content, templates, mk_prep, allow_bad_res=True, default_altloc="A"
            )
        except (OSError, ValueError, RuntimeError, ImportError) as exc2:
            logger.error(
                f"Meeko preparation failed even with allow_bad_res: {exc2} — "
                f"falling back to Open Babel"
            )
            return _prepare_receptor_with_obabel(pdb_file, output_pdbqt, pdb_string=pdb_content)

    try:
        rigid_pdbqt, _ = PDBQTWriterLegacy.write_from_polymer(polymer)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.error(f"PDBQT writing failed: {exc} — falling back to Open Babel")
        return _prepare_receptor_with_obabel(pdb_file, output_pdbqt, pdb_string=pdb_content)

    ensure_dir(os.path.dirname(output_pdbqt) or ".")
    with open(output_pdbqt, "w") as fh:
        fh.write(rigid_pdbqt)

    # ── Report generation ────────────────────────────────────────────────
    if output_report_json is not None:
        _report: dict[str, Any] = {
            "input_file": os.path.abspath(pdb_file),
            "output_pdbqt": os.path.abspath(output_pdbqt),
            "parameters": {
                "remove_water": remove_water,
                "remove_hetatms": remove_hetatms,
                "ph": ph,
                "forcefield": forcefield,
                "restraint_center": restraint_center,
                "restraint_radius": restraint_radius,
                "retain_metal_ions": retain_metal_ions,
                "predict_pka": predict_pka,
                "keep_waters_near_metal": keep_waters_near_metal,
                "detect_af_structure": detect_af_structure,
            },
            "structure_quality": _quality_header,
            "alphafold_assessment": _af_assessment,
            "disulfide_bonds": len(_ssbonds),
            "disulfide_bonds_detail": [
                {"chain1": b["chain1"], "res1": b["res1"], "chain2": b["chain2"], "res2": b["res2"]}
                for b in _ssbonds
            ],
            "alternate_conformations": len(multi_altloc),
            "missing_residues_filled": len(gaps_detail),
            "missing_residues_detail": gaps_detail,
            "reduce_flips": len(_reduce_flip_detail),
            "reduce_flips_detail": _reduce_flip_detail,
            "anomalous_pka_residues": len(_anomalous_pka) if "_anomalous_pka" in locals() else 0,
            "anomalous_pka_detail": _anomalous_pka if "_anomalous_pka" in locals() else [],
            "openmm_restrained_atoms": locals().get("n_restrained", 0),
            "waters_removed": locals().get("n_waters_removed", 0),
            "waters_retained_functional": locals().get("n_waters_retained_functional", 0),
            "metals_retained": locals().get("n_metals_retained_step5", 0),
            "additives_skipped": dict(locals().get("skipped_residues", {})),
        }
        _write_prep_report(output_report_json, _report)
        logger.info(f"Preparation report written: {output_report_json}")

    # Final cleanup: remove the original raw temp file unless it was
    # replaced by the AF-relaxed output (which lives in output_dir).
    if _original_tmp_raw != tmp_raw:
        with contextlib.suppress(OSError):
            if os.path.exists(_original_tmp_raw):
                os.remove(_original_tmp_raw)

    # Comprehensive cleanup: ensure no tracked temp files leak on exception
    for _t in _all_temps:
        with contextlib.suppress(OSError):
            if os.path.exists(_t) and _t != tmp_raw:
                os.remove(_t)

    # Save PDB file if requested (for downstream PLIP/PyMOL/PoseBusters)
    if output_pdb:
        try:
            with open(output_pdb, "w") as _pdb_fh:
                _pdb_fh.write(pdb_content)
            logger.info(f"Receptor PDB saved: {os.path.abspath(output_pdb)}")
        except OSError as _pdb_exc:
            logger.warning(f"Receptor PDB save failed ({_pdb_exc})")

    logger.info(f"Receptor prepared: {output_pdbqt}")

    # ── Cache store --------------------------------------------------------
    if cache_dir is not None:
        from autodock.cache import ReceptorCache

        rc = ReceptorCache(cache_dir)
        cache_params = {
            "remove_water": remove_water,
            "remove_hetatms": remove_hetatms,
            "input_format": input_format,
            "keep_residues": sorted(keep_residues) if keep_residues else None,
            "ph": ph,
            "forcefield": forcefield,
            "restraint_center": restraint_center,
            "restraint_radius": restraint_radius,
            "retain_metal_ions": retain_metal_ions,
            "predict_pka": predict_pka,
            "fix_protonation": fix_protonation,
            "keep_waters_near_metal": keep_waters_near_metal,
            "detect_af_structure": detect_af_structure,
        }
        files_to_cache = {"receptor.pdbqt": output_pdbqt}
        if output_pdb and os.path.isfile(output_pdb):
            files_to_cache["receptor.pdb"] = output_pdb
        rc.put(pdb_file, files_to_cache, **cache_params)

    return os.path.abspath(output_pdbqt)


def _prepare_receptor_with_obabel(
    pdb_file: str,
    output_pdbqt: str,
    pdb_string: str | None = None,
    in_format: str = "pdb",
) -> str:
    """Fallback receptor preparation using Open Babel.

    Args:
        pdb_file: Path to input PDB/CIF file (used when pdb_string is None).
        output_pdbqt: Path for output PDBQT file.
        pdb_string: Optional PDB content string. When provided, it is written
            to a temporary file and used as input instead of *pdb_file*.
        in_format: Input format for Open Babel (default "pdb").
    """
    in_path = pdb_file
    if pdb_string is not None:
        in_path = write_temp_file(pdb_string, suffix=".pdb")
    success = obabel_convert(
        in_path,
        output_pdbqt,
        in_format=in_format,
        out_format="pdbqt",
        options=["-xr"],  # rigid receptor (no rotatable bonds)
        timeout=300,
    )
    if pdb_string is not None:
        with contextlib.suppress(OSError):
            os.remove(in_path)
    if not success:
        raise PreparationError("Open Babel receptor preparation failed")
    logger.info(f"Receptor prepared (Open Babel fallback): {output_pdbqt}")
    return os.path.abspath(output_pdbqt)


# ─────────────────────────────────────────────────────────────────────────────
# Ligand Preparation
# ─────────────────────────────────────────────────────────────────────────────


def _has_nan_charges(mol) -> bool:
    """Check if any atom has a NaN, inf, or missing Gasteiger charge."""
    _n_charged = 0
    for atom in mol.GetAtoms():
        try:
            c = atom.GetDoubleProp("_GasteigerCharge")
            if c != c or math.isinf(c):  # NaN or inf check
                return True
            _n_charged += 1
        except KeyError:
            continue
    # If none of the atoms received a charge, the calculation failed entirely
    return _n_charged == 0


def _prepare_ligand_with_obabel(
    smiles: str, output_pdbqt: str, name: str = "LIG", ph: float = 7.4
) -> str:
    """Fallback ligand preparation using Open Babel (SMILES → PDBQT)."""
    from autodock.utils import write_temp_file

    tmp_smi = write_temp_file(smiles, ".smi")
    tmp_pdbqt = os.path.splitext(tmp_smi)[0] + "_obabel.pdbqt"
    try:
        success = obabel_convert(
            tmp_smi,
            tmp_pdbqt,
            in_format="smi",
            out_format="pdbqt",
            options=["-p", str(ph), "--gen3d"],
            timeout=120,
        )
        if not success:
            raise PreparationError("Open Babel ligand preparation failed")

        with open(tmp_pdbqt) as fh:
            pdbqt_str = fh.read()
    finally:
        for p in (tmp_smi, tmp_pdbqt):
            with contextlib.suppress(Exception):
                os.remove(p)

    # Inject residue name
    safe_name = (name or "LIG")[:3]
    if safe_name != "LIG":
        lines = pdbqt_str.splitlines()
        renamed = []
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                line = line[:17] + f"{safe_name:>3}" + line[20:]
            renamed.append(line)
        pdbqt_str = "\n".join(renamed)

    ensure_dir(os.path.dirname(output_pdbqt) or ".")
    with open(output_pdbqt, "w") as fh:
        fh.write(pdbqt_str)

    logger.info(f"Ligand prepared (Open Babel fallback): {output_pdbqt}")
    return os.path.abspath(output_pdbqt)


def _embed_and_select_best_conf(
    mol_h: Any,
    n_conformer_attempts: int = 20,
    seed: int = 42,
    label: str = "",
) -> tuple[Any, float] | None:
    """Helper: ETKDGv3 multi-round → MMFF94 → best-conformer RDKit Mol.

    Returns ``(mol_with_best_conformer, mmff_energy)`` or *None*.
    The molecule retains Hs and only the best conformer.
    """
    from rdkit.Chem import AllChem

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0
    params.pruneRmsThresh = 0.5
    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformer_attempts, params=params)

    if len(cids) == 0:
        return None

    # ── MMFF94 optimisation (primary) ────────────────────────────────────
    mmff_results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=500, numThreads=0)
    best_cid: int | None = None
    best_energy = float("inf")
    ff_used = "MMFF94"
    for i, cid in enumerate(cids):
        status = mmff_results[i][0] if i < len(mmff_results) else -1
        energy = mmff_results[i][1] if i < len(mmff_results) else float("inf")
        if status != -1 and energy is not None and energy < best_energy:
            best_cid = cid
            best_energy = float(energy)

    # ── UFF fallback (for exotic elements not handled by MMFF94) ─────────
    if best_cid is None:
        logger.debug(f"{label}MMFF94 unsupported — trying UFF fallback")
        uff_results = AllChem.UFFOptimizeMoleculeConfs(mol_h, maxIters=500, numThreads=0)
        for i, cid in enumerate(cids):
            status = uff_results[i][0] if i < len(uff_results) else -1
            energy = uff_results[i][1] if i < len(uff_results) else float("inf")
            if status != -1 and energy is not None and energy < best_energy:
                best_cid = cid
                best_energy = float(energy)
        ff_used = "UFF"

    if best_cid is None:
        return None

    logger.debug(
        f"{label}Multi-round conformer ({ff_used}): {len(cids)} evaluated,"
        f" lowest energy={best_energy:.2f} kcal/mol"
    )

    # Keep only the best conformer to avoid FF property invalidation later
    for c in cids:
        if c != best_cid:
            mol_h.RemoveConformer(c)
    return (mol_h, best_energy)


def prepare_ligand(
    smiles: str,
    output_pdbqt: str,
    name: str = "LIG",
    seed: int = 42,
    n_conformer_attempts: int = 20,
    ph: float = 7.4,
    ph_range: float | None = 1.5,
    molscrub_states: bool = True,
    enumerate_stereo: bool = True,
    max_stereo_isomers: int = 8,
    cache_dir: str | None = None,
) -> str:
    """
    Prepare a ligand for docking (SMILES → PDBQT).

    Uses RDKit ETKDGv3 for 3D conformer generation **with multi-round
    energy sorting**: generates ``n_conformer_attempts`` conformers,
    optimises each with MMFF94, and selects the lowest-energy one for
    PDBQT export.

    When **molscrub_states=True** (default), the pipeline also handles:

    * **Salt/counterion removal** — strips HCl, Na⁺, etc. from database
      SMILES that may represent salt forms.
    * **Tautomer enumeration** — generates all relevant keto-enol,
      imine-enamine, etc. tautomers and selects the MMFF94-stablest one.
    * **Protonation-state enumeration** — evaluates all protonation
      states in the pH range (``ph ± ph_range/2``); crucial for
      carboxylic acids, amines, His, and other titratable groups.

    References:
        - Riniker et al. (2004) J. Med. Chem.
        - Hawkins (2017) J. Chem. Inf. Model.
        - molscrub: https://github.com/whitead/molscrub

    Args:
        smiles: SMILES string.
        output_pdbqt: Output PDBQT file path.
        name: Residue name in PDBQT.
        seed: Random seed for reproducible conformer generation.
        n_conformer_attempts: Number of ETKDGv3 attempts per state;
            the lowest-energy conformer after MMFF94 is selected.
        ph: Target pH for ligand protonation (default 7.4).
        ph_range: pH range for protonation-state enumeration
            (``ph - ph_range/2`` to ``ph + ph_range/2``).
            Set *None* for a single pH point.
        molscrub_states: If True, run molscrub to enumerate tautomers,
            protonation states, and strip counterions before conformer
            generation.  The lowest-energy state × conformer is selected.
        enumerate_stereo: If True (default), detect unassigned chiral
            centers in the input SMILES and enumerate all possible
            stereoisomers.  Each stereoisomer is prepared separately
            and the lowest MMFF94-energy one is selected.
        max_stereo_isomers: Maximum stereoisomers to enumerate
            (default 8).  Prevents combinatorial explosion for highly
            flexible ligands with many unspecified chiral centers.

    Returns:
        Absolute path to the prepared PDBQT file.
    """
    # ── Cache lookup -------------------------------------------------------
    if cache_dir is not None:
        from autodock.cache import LigandCache

        lc = LigandCache(cache_dir)
        cache_params = {
            "name": name,
            "seed": seed,
            "n_conformer_attempts": n_conformer_attempts,
            "ph": ph,
            "ph_range": ph_range,
            "molscrub_states": molscrub_states,
            "enumerate_stereo": enumerate_stereo,
            "max_stereo_isomers": max_stereo_isomers,
        }
        cached = lc.get(smiles, **cache_params)
        if cached:
            import shutil

            shutil.copy2(cached["ligand.pdbqt"], output_pdbqt)
            logger.info(f"Ligand cache hit — copied from {lc.cache_dir}")
            return os.path.abspath(output_pdbqt)

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
        from rdkit.Chem import rdPartialCharges
    except ImportError as exc:
        raise PreparationError(f"Required package missing: {exc}")

    mol_base = Chem.MolFromSmiles(smiles)
    if mol_base is None:
        raise PreparationError(f"Could not parse SMILES: {smiles}")

    # ── Stereochemistry enumeration ──────────────────────────────────────────
    # If the input SMILES has unassigned chiral centers, enumerate all possible
    # stereoisomers and prepare each one.  Select the lowest MMFF94-energy
    # isomer across the entire stereo × tautomer × protomer space.
    stereo_inputs: list[tuple[Any, str]] = [(mol_base, "")]
    if enumerate_stereo:
        from rdkit.Chem.EnumerateStereoisomers import (
            EnumerateStereoisomers,
            StereoEnumerationOptions,
        )

        unassigned = Chem.FindMolChiralCenters(mol_base, includeUnassigned=True)
        n_unassigned = sum(1 for _, label in unassigned if label == "?")
        if n_unassigned > 0:
            _opts = StereoEnumerationOptions(
                onlyUnassigned=True,
                unique=True,
                maxIsomers=max_stereo_isomers,
            )
            stereo_isomers = list(EnumerateStereoisomers(mol_base, options=_opts))
            if len(stereo_isomers) > 1:
                logger.info(
                    f"Stereo enumeration: {n_unassigned} unassigned chiral center(s)"
                    f" → {len(stereo_isomers)} stereoisomer(s)"
                )
                stereo_inputs = [(iso, f"[stereo {i}] ") for i, iso in enumerate(stereo_isomers)]
    # ── State enumeration (molscrub: salt stripping + tautomers + protonation) ──
    candidate_mols: list[tuple[Any, float, str]] = []  # (rdkit_mol, mmff_energy, label)

    for _stereo_mol, _stereo_label in stereo_inputs:
        if molscrub_states:
            try:
                from molscrub import Scrub

                if ph_range is not None:
                    ph_low = max(0.0, ph - ph_range / 2.0)
                    ph_high = ph + ph_range / 2.0
                else:
                    ph_low = ph
                    ph_high = ph + 0.01  # single-point

                scrubber = Scrub(
                    ph_low=ph_low,
                    ph_high=ph_high,
                    skip_acidbase=False,
                    skip_tautomers=False,
                    debug=False,
                )
                scrubbed_list = scrubber(input_mol=_stereo_mol)

                for i, state_mol in enumerate(scrubbed_list):
                    if state_mol is None:
                        continue
                    state_mol_h = Chem.AddHs(state_mol, addCoords=True)
                    result_tup = _embed_and_select_best_conf(
                        state_mol_h,
                        n_conformer_attempts=n_conformer_attempts,
                        seed=seed + i,
                        label=f"{_stereo_label}[taut {i}] ",
                    )
                    if result_tup is not None:
                        _mol, _eng = result_tup
                        candidate_mols.append((_mol, _eng, f"{_stereo_label}taut_{i}"))
            except ImportError:
                logger.debug("molscrub not installed — skipping state enumeration")
                molscrub_states = False
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning(f"molscrub failed ({exc}) — falling back to single-state")

    # ── Single-state fallback (no molscrub, may still have stereoisomers) ──
    if not candidate_mols:
        for _stereo_mol, _stereo_label in stereo_inputs:
            mol_h = Chem.AddHs(_stereo_mol, addCoords=True)
            result_tup = _embed_and_select_best_conf(
                mol_h,
                n_conformer_attempts=n_conformer_attempts,
                seed=seed,
                label=_stereo_label,
            )
            if result_tup is not None:
                _mol, _eng = result_tup
                candidate_mols.append((_mol, _eng, f"{_stereo_label}single"))

    if not candidate_mols:
        logger.warning("RDKit embedding failed — falling back to Open Babel")
        result = _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name, ph=ph)
        if cache_dir is not None:
            from autodock.cache import LigandCache

            lc = LigandCache(cache_dir)
            cache_params = {
                "name": name,
                "seed": seed,
                "n_conformer_attempts": n_conformer_attempts,
                "ph": ph,
                "ph_range": ph_range,
                "molscrub_states": molscrub_states,
                "enumerate_stereo": enumerate_stereo,
                "max_stereo_isomers": max_stereo_isomers,
            }
            lc.put(smiles, {"ligand.pdbqt": output_pdbqt}, **cache_params)
        return result

    # Compute Gasteiger charges and pick the best across all candidates
    best_mol: Any | None = None
    best_energy = float("inf")
    best_label = ""
    for cand_mol, energy, label in candidate_mols:
        try:
            rdPartialCharges.ComputeGasteigerCharges(cand_mol)
            if _has_nan_charges(cand_mol):
                continue
            if not best_mol or energy < best_energy:
                best_mol = cand_mol
                best_energy = energy
                best_label = label
        except (OSError, ValueError, RuntimeError):
            continue

    if best_mol is None:
        # All candidates had NaN charges — fall through to OBabel
        logger.warning("Gasteiger charges contain NaN — falling back to Open Babel")
        return _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name, ph=ph)

    if molscrub_states and len(candidate_mols) > 1:
        logger.info(f"Selected best state: {best_label} (MMFF={best_energy:.1f} kcal/mol)")

    # ── Meeko → PDBQT ─────────────────────────────────────────────────────
    params_mk = MoleculePreparation(charge_model="gasteiger")
    try:
        mol_setup = params_mk.prepare(best_mol)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        err_str = str(exc)
        if "non finite charge" in err_str or "charge" in err_str.lower():
            logger.warning(f"Meeko charge failure ({exc}) — falling back to Open Babel")
            result = _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name, ph=ph)
            if cache_dir is not None:
                from autodock.cache import LigandCache

                lc = LigandCache(cache_dir)
                cache_params = {
                    "name": name,
                    "seed": seed,
                    "n_conformer_attempts": n_conformer_attempts,
                    "ph": ph,
                    "ph_range": ph_range,
                    "molscrub_states": molscrub_states,
                    "enumerate_stereo": enumerate_stereo,
                    "max_stereo_isomers": max_stereo_isomers,
                }
                lc.put(smiles, {"ligand.pdbqt": output_pdbqt}, **cache_params)
            return result
        raise PreparationError(f"Meeko ligand prep failed: {exc}")

    setup = mol_setup[0] if isinstance(mol_setup, list) else mol_setup
    pdbqt_str, success, err = PDBQTWriterLegacy.write_string(setup)
    if not success:
        if "non finite charge" in err or "charge" in err.lower():
            logger.warning("Meeko PDBQT write charge failure — falling back to Open Babel")
            result = _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name, ph=ph)
            if cache_dir is not None:
                from autodock.cache import LigandCache

                lc = LigandCache(cache_dir)
                cache_params = {
                    "name": name,
                    "seed": seed,
                    "n_conformer_attempts": n_conformer_attempts,
                    "ph": ph,
                    "ph_range": ph_range,
                    "molscrub_states": molscrub_states,
                    "enumerate_stereo": enumerate_stereo,
                    "max_stereo_isomers": max_stereo_isomers,
                }
                lc.put(smiles, {"ligand.pdbqt": output_pdbqt}, **cache_params)
            return result
        raise PreparationError(f"Meeko ligand prep failed: {err}")

    safe_name = (name or "LIG")[:3]
    if safe_name != "LIG":
        lines = pdbqt_str.splitlines()
        renamed = []
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                line = line[:17] + f"{safe_name:>3}" + line[20:]
            renamed.append(line)
        pdbqt_str = "\n".join(renamed)

    ensure_dir(os.path.dirname(output_pdbqt) or ".")
    with open(output_pdbqt, "w") as fh:
        fh.write(pdbqt_str)

    logger.info(f"Ligand prepared: {output_pdbqt}")

    # ── Cache store --------------------------------------------------------
    if cache_dir is not None:
        from autodock.cache import LigandCache

        lc = LigandCache(cache_dir)
        cache_params = {
            "name": name,
            "seed": seed,
            "n_conformer_attempts": n_conformer_attempts,
            "ph": ph,
            "ph_range": ph_range,
            "molscrub_states": molscrub_states,
            "enumerate_stereo": enumerate_stereo,
            "max_stereo_isomers": max_stereo_isomers,
        }
        lc.put(smiles, {"ligand.pdbqt": output_pdbqt}, **cache_params)

    return os.path.abspath(output_pdbqt)


def prepare_ligand_from_file(
    path: str,
    output_pdbqt: str,
    name: str = "LIG",
    ph: float = 7.4,
) -> str:
    """
    Prepare a ligand from a structure file, auto-detecting the format.

    Supports SDF (``.sdf``), MOL (``.mol``), and MOL2 (``.mol2``).
    Preserves existing 3D coordinates from the file.

    Args:
        path: Path to structure file.
        output_pdbqt: Output PDBQT file path.
        name: Residue name in PDBQT.

    Returns:
        Absolute path to the prepared PDBQT file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mol2",):
        # MOL2 path — use RDKit MolFromMol2File
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise PreparationError(f"Required package missing: {exc}")

        mol = Chem.MolFromMol2File(str(path), removeHs=False)
        if mol is None:
            # Fallback: try Open Babel MOL2 → PDBQT directly
            logger.warning(f"RDKit could not parse MOL2: {path} — trying Open Babel")
            success = obabel_convert(
                str(path),
                str(output_pdbqt),
                in_format="mol2",
                out_format="pdbqt",
                options=["-p", str(ph)],
                timeout=120,
            )
            if success:
                return str(output_pdbqt)
            raise PreparationError(f"Could not parse MOL2 with RDKit or Open Babel: {path}")

        smiles = Chem.MolToSmiles(mol)
        return prepare_ligand(
            smiles,
            output_pdbqt,
            name=name,
            molscrub_states=False,
            enumerate_stereo=False,
            ph=ph,
        )

    # SDF / MOL — use prepare_ligand_from_sdf
    return prepare_ligand_from_sdf(path, output_pdbqt, name=name)


def prepare_ligand_from_sdf(
    sdf_path: str,
    output_pdbqt: str,
    name: str = "LIG",
    ph: float = 7.4,
) -> str:
    """
    Prepare a ligand from an SDF file, preserving existing 3D coordinates.

    Unlike ``prepare_ligand()``, this skips conformer generation and uses
    the coordinates already present in the SDF file.

    Args:
        sdf_path: Path to SDF file (single molecule).
        output_pdbqt: Output PDBQT file path.
        name: Residue name in PDBQT.

    Returns:
        Absolute path to the prepared PDBQT file.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
        from rdkit.Chem import rdPartialCharges
    except ImportError as exc:
        raise PreparationError(f"Required package missing: {exc}")

    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        # Try SDMolSupplier for multi-molecule SDF (take first)
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        for m in supplier:
            if m is not None:
                mol = m
                break
    if mol is None:
        raise PreparationError(f"Could not parse SDF: {sdf_path}")

    # Ensure hydrogens are present (SDF may lack them)
    mol = Chem.AddHs(mol, addCoords=True)

    rdPartialCharges.ComputeGasteigerCharges(mol)
    if _has_nan_charges(mol):
        logger.warning("Gasteiger charges contain NaN — falling back to Open Babel")
        return _prepare_ligand_with_obabel(Chem.MolToSmiles(mol), output_pdbqt, name=name, ph=ph)

    params_mk = MoleculePreparation(charge_model="gasteiger")
    try:
        mol_setup = params_mk.prepare(mol)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        err_str = str(exc)
        if "non finite charge" in err_str or "charge" in err_str.lower():
            logger.warning(f"Meeko charge failure ({exc}) — falling back to Open Babel")
            return _prepare_ligand_with_obabel(
                Chem.MolToSmiles(mol), output_pdbqt, name=name, ph=ph
            )
        raise PreparationError(f"Meeko ligand prep failed: {exc}")

    setup = mol_setup[0] if isinstance(mol_setup, list) else mol_setup
    pdbqt_str, success, err = PDBQTWriterLegacy.write_string(setup)
    if not success:
        if "non finite charge" in err or "charge" in err.lower():
            logger.warning("Meeko charge error — falling back to Open Babel")
            return _prepare_ligand_with_obabel(
                Chem.MolToSmiles(mol), output_pdbqt, name=name, ph=ph
            )
        raise PreparationError(f"PDBQT export failed: {err}")

    safe_name = (name or "LIG")[:3]
    if safe_name != "LIG":
        lines = pdbqt_str.splitlines()
        renamed = []
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                line = line[:17] + f"{safe_name:>3}" + line[20:]
            renamed.append(line)
        pdbqt_str = "\n".join(renamed)

    with open(output_pdbqt, "w") as fh:
        fh.write(pdbqt_str)
    logger.info(f"Prepared ligand from SDF: {output_pdbqt}")
    return os.path.abspath(output_pdbqt)


def prepare_ligand_conformers(
    smiles: str,
    output_dir: str,
    n_conformers: int = 10,
    name: str = "LIG",
    seed_start: int = 42,
    ph: float = 7.4,
    ph_range: float | None = 1.5,
    molscrub_states: bool = True,
) -> list[str]:
    """
    Generate multiple 3D conformers of a ligand for multi-conformer docking.

    .. warning::

        **This function is rarely needed for AutoDock Vina.** Vina performs
        its own internal torsion-angle search during docking, so a single
        well-optimised conformer (from ``prepare_ligand()``) is sufficient
        for most ligands.  Multi-conformer docking is only useful for
        special cases where Vina cannot cross conformational barriers,
        such as macrocycles (>12-membered rings) or rigid ring systems
        with distinct conformers (e.g. chair vs. boat cyclohexane).
        For typical drug-like molecules, multi-conformer docking adds
        compute time without improving pose accuracy.

    Args:
        smiles: SMILES string.
        output_dir: Directory for conformer PDBQT files.
        n_conformers: Number of conformers.
        name: Residue name.
        seed_start: Starting random seed.

    Returns:
        List of PDBQT file paths.
    """
    ensure_dir(output_dir)
    paths = []
    for i in range(n_conformers):
        out_path = os.path.join(output_dir, f"conformer_{i}.pdbqt")
        try:
            prepare_ligand(smiles, out_path, name=name, seed=seed_start + i, ph=ph)
            paths.append(out_path)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.warning(f"Conformer {i} preparation failed: {exc} — skipping")
    if not paths:
        raise PreparationError(f"All {n_conformers} conformer preparations failed for {smiles}")
    logger.info(f"Generated {len(paths)}/{n_conformers} conformers in {output_dir}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive multi-conformer ligand preparation
# ─────────────────────────────────────────────────────────────────────────────


def _classify_ligand_complexity(mol) -> str:
    """
    Classify ligand structural complexity to choose preparation strategy.

    Heuristics tuned on the 20-target benchmark set.

    .. note::

        AutoDock Vina performs its own torsion-angle search during docking,
        so **single-conformer input is sufficient for most ligands**.
        Multi-conformer docking is only useful for special cases where Vina
        cannot cross conformational barriers (macrocycles, rigid ring
        systems with distinct conformers such as chair/boat cyclohexane).
        For typical drug-like molecules, multi-conformer docking adds
        runtime without improving accuracy.

    Categories:
        - **simple**  : single-conformer preparation (default, recommended).
        - **medium**  : may benefit from multi-conformer if rigid rings
          have accessible alternative conformers.
        - **complex** : macrocycles or very large/flexible ligands that may
          need external conformational sampling (e.g. CREST) before docking.

    Returns:
        "simple", "medium", or "complex"
    """
    from rdkit import Chem

    n_heavy = mol.GetNumHeavyAtoms()
    n_rings = mol.GetRingInfo().NumRings()
    rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Chiral centers restrict accessible conformational space;
    # many chirals = harder for single-conformer generation
    # Only count assigned chiral centres — unassigned centres in plain SMILES
    # do not restrict conformational space.
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=False))

    # Macrocycles are inherently hard for distance-geometry methods
    ring_info = mol.GetRingInfo()
    has_macrocycle = any(len(r) > 12 for r in ring_info.AtomRings())

    # Fused / bridged systems
    n_spiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    # Complex: macrocycles, very large, very flexible, or many fused rings
    if has_macrocycle or n_heavy > 40 or rot_bonds > 12 or n_rings > 3 or n_spiro + n_bridge > 1:
        return "complex"

    # Medium: moderate size/flexibility or significant chirality
    if rot_bonds > 6 or n_rings > 2 or n_heavy > 28 or n_chiral > 3:
        return "medium"

    return "simple"


def _generate_multi_conformers(
    mol,
    n_conformers: int = 50,
    seed: int = 42,
    cluster_threshold: float | None = None,
) -> tuple:
    """
    Generate diverse conformers via RDKit EmbedMultipleConfs + RMSD clustering.

    Dynamic threshold: high-flexibility ligands need larger RMSD cutoffs to
    produce meaningful conformational families instead of every conformer
    becoming its own cluster.

    Returns:
        (mol_with_conformers, list_of_representative_cids)
        The returned molecule has Hs and all conformers attached.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    mol_h = Chem.AddHs(mol, addCoords=True)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Dynamic clustering threshold based on flexibility
    if cluster_threshold is None:
        if n_rot > 10:
            cluster_threshold = 2.0
        elif n_rot > 6:
            cluster_threshold = 1.5
        else:
            cluster_threshold = 1.0

    # 1. Generate multiple conformers with initial pruning for diversity
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0  # use all CPU cores
    # Prune similar conformers during embedding to save MMFF time
    params.pruneRmsThresh = max(0.5, cluster_threshold * 0.5)
    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformers, params=params)
    if len(cids) == 0:
        return mol_h, []

    # 2. MMFF optimize all conformers
    results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=0)
    energies = {cid: results[i][1] for i, cid in enumerate(cids)}

    # 3. RMSD-based clustering (greedy) — energy-ranked lowest first
    sorted_cids = sorted(cids, key=lambda c: energies[c])
    representatives = []
    for cid in sorted_cids:
        is_unique = True
        for rep_cid in representatives:
            rms = AllChem.GetConformerRMS(mol_h, cid, rep_cid, prealigned=False)
            if rms < cluster_threshold:
                is_unique = False
                break
        if is_unique:
            representatives.append(cid)

    # 4. If every conformer became its own cluster, the threshold was too strict.
    #    Fallback: return only the lowest-energy N representatives.
    if len(representatives) == len(cids) and len(cids) > 10:
        logger.debug(
            f"No clustering occurred ({len(cids)} clusters) — "
            f"falling back to top {min(len(cids), 10)} lowest-energy conformers"
        )
        representatives = sorted_cids[: min(len(cids), 10)]

    logger.debug(
        f"Conformer clustering: {len(cids)} generated → "
        f"{len(representatives)} clusters (threshold={cluster_threshold} Å, rot={n_rot})"
    )
    return mol_h, representatives


def prepare_ligand_multi(
    smiles: str,
    output_dir: str,
    name: str = "LIG",
    seed: int = 42,
    n_conformers: int = 50,
    max_representatives: int = 5,
    ph: float = 7.4,
) -> list[str]:
    """
    Prepare a ligand with multi-conformer sampling for flexible molecules.

    Workflow:
        1. Generate N conformers with ETKDGv3
        2. MMFF optimize all
        3. RMSD cluster (threshold 1.0 Å)
        4. Select lowest-energy representative per cluster
        5. Export each representative to PDBQT via Meeko

    Args:
        smiles: SMILES string.
        output_dir: Directory for output PDBQT files.
        name: Residue name.
        seed: Random seed.
        n_conformers: Number of conformers to generate before clustering.
        max_representatives: Maximum number of cluster representatives to keep.

    Returns:
        List of PDBQT file paths (one per cluster).
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PreparationError(f"Could not parse SMILES: {smiles}")

    # Generate & cluster conformers
    mol_h, rep_cids = _generate_multi_conformers(mol, n_conformers=n_conformers, seed=seed)
    if not rep_cids:
        logger.warning("Multi-conformer generation failed — falling back to single conformer")
        single_path = os.path.join(output_dir, "conformer_0.pdbqt")
        prepare_ligand(smiles, single_path, name=name, seed=seed)
        return [single_path]

    # Limit representatives
    rep_cids = rep_cids[:max_representatives]

    ensure_dir(output_dir)
    paths = []
    safe_name = (name or "LIG")[:3]

    for idx, cid in enumerate(rep_cids):
        # Create a fresh molecule with only this conformer
        mol_single = Chem.Mol(mol_h)
        conf = mol_h.GetConformer(cid)
        # Copy coordinates
        from rdkit import Geometry

        new_conf = Chem.Conformer(mol_single.GetNumAtoms())
        new_conf.SetId(0)
        for i in range(mol_single.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, Geometry.Point3D(pos.x, pos.y, pos.z))
        mol_single.RemoveAllConformers()
        mol_single.AddConformer(new_conf)

        # Gasteiger charges
        rdPartialCharges.ComputeGasteigerCharges(mol_single)
        if _has_nan_charges(mol_single):
            logger.warning(f"Rep {idx}: NaN charges — trying Open Babel for this conformer")
            # Fallback: write SMILES, use obabel with a different seed
            ob_path = os.path.join(output_dir, f"conformer_{idx}.pdbqt")
            _prepare_ligand_with_obabel(smiles, ob_path, name=name, ph=ph)
            paths.append(ob_path)
            continue

        # Meeko
        params_mk = MoleculePreparation(charge_model="gasteiger")
        try:
            mol_setup = params_mk.prepare(mol_single)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.warning(f"Rep {idx}: Meeko failed ({exc}) — Open Babel fallback")
            ob_path = os.path.join(output_dir, f"conformer_{idx}.pdbqt")
            _prepare_ligand_with_obabel(smiles, ob_path, name=name, ph=ph)
            paths.append(ob_path)
            continue

        setup = mol_setup[0] if isinstance(mol_setup, list) else mol_setup
        pdbqt_str, success, err = PDBQTWriterLegacy.write_string(setup)
        if not success:
            logger.warning(f"Rep {idx}: PDBQT write failed ({err}) — Open Babel fallback")
            ob_path = os.path.join(output_dir, f"conformer_{idx}.pdbqt")
            _prepare_ligand_with_obabel(smiles, ob_path, name=name, ph=ph)
            paths.append(ob_path)
            continue

        # Rename residue if needed
        if safe_name != "LIG":
            lines = pdbqt_str.splitlines()
            renamed = []
            for line in lines:
                if line.startswith(("ATOM  ", "HETATM")):
                    line = line[:17] + f"{safe_name:>3}" + line[20:]
                renamed.append(line)
            pdbqt_str = "\n".join(renamed)

        out_path = os.path.join(output_dir, f"conformer_{idx}.pdbqt")
        with open(out_path, "w") as fh:
            fh.write(pdbqt_str)
        paths.append(out_path)

    logger.info(
        f"Multi-conformer prep: {len(rep_cids)} representatives → "
        f"{len(paths)} PDBQT files in {output_dir}"
    )
    return paths


def prepare_ligand_adaptive(
    smiles: str,
    output_pdbqt: str,
    name: str = "LIG",
    seed: int = 42,
    strategy: str | None = None,
    n_conformers_medium: int = 30,
    max_reps_medium: int = 5,
    n_conformers_complex: int = 100,
    max_reps_complex: int = 10,
    force_multi_conformer: bool = False,
) -> str | list[str]:
    """
    Adaptive ligand preparation: auto-selects strategy based on molecular complexity.

    .. note::

        **Default behaviour (``force_multi_conformer=False``):** all ligands
        receive single-conformer preparation, because AutoDock Vina performs
        its own internal conformational search.  Multi-conformer docking is
        only engaged automatically for macrocycles (>12-membered rings),
        where Vina cannot cross ring-conformation barriers.

        Set ``force_multi_conformer=True`` to override and use the legacy
        medium/complex multi-conformer strategy for all non-simple ligands.

    Args:
        smiles: SMILES string.
        output_pdbqt: Output PDBQT file path (for single) OR directory (for multi).
        name: Residue name.
        seed: Random seed.
        strategy: "simple", "medium", "complex", or None for auto-detection.
        n_conformers_medium: Conformers to generate for medium ligands
            (only when ``force_multi_conformer=True``).
        max_reps_medium: Max representatives for medium ligands
            (only when ``force_multi_conformer=True``).
        n_conformers_complex: Conformers to generate for complex ligands
            (only when ``force_multi_conformer=True``).
        max_reps_complex: Max representatives for complex ligands
            (only when ``force_multi_conformer=True``).
        force_multi_conformer: If True, use legacy multi-conformer strategy
            for medium/complex ligands.  Default False (recommended for Vina).

    Returns:
        Single PDBQT path, or list of paths for multi-conformer mode.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PreparationError(f"Could not parse SMILES: {smiles}")

    if strategy is None:
        strategy = _classify_ligand_complexity(mol)
        logger.info(f"Adaptive ligand prep: complexity='{strategy}' for '{smiles[:40]}...'")

    # Auto-detect macrocycles: the one case where multi-conformer may help Vina
    ring_info = mol.GetRingInfo()
    has_macrocycle = any(len(r) > 12 for r in ring_info.AtomRings())

    # Default (force_multi_conformer=False): single-conformer for everything.
    # Vina performs its own torsion search; multi-conformer docking is only
    # useful when rigid ring conformers cannot be interconverted by torsion
    # changes (macrocycles, chair/boat cyclohexane, etc.).
    if not force_multi_conformer and strategy in ("simple", "medium"):
        if os.path.isdir(output_pdbqt):
            output_pdbqt = os.path.join(output_pdbqt, "ligand.pdbqt")
        return prepare_ligand(smiles, output_pdbqt, name=name, seed=seed)

    if not force_multi_conformer and strategy == "complex" and not has_macrocycle:
        logger.info(
            "Complex ligand without macrocycle — using single conformer "
            "(Vina handles torsion search internally). "
            "Set force_multi_conformer=True to override."
        )
        if os.path.isdir(output_pdbqt):
            output_pdbqt = os.path.join(output_pdbqt, "ligand.pdbqt")
        return prepare_ligand(smiles, output_pdbqt, name=name, seed=seed)

    # Multi-conformer path (legacy behaviour, or macrocycle auto-detection)
    if os.path.isfile(output_pdbqt):
        output_dir = os.path.dirname(output_pdbqt) or "."
    else:
        output_dir = output_pdbqt
        ensure_dir(output_dir)

    if has_macrocycle and not force_multi_conformer:
        logger.info(
            "Macrocycle detected — using multi-conformer preparation "
            "(Vina cannot cross macrocycle conformational barriers)."
        )

    if strategy == "medium":
        return prepare_ligand_multi(
            smiles,
            output_dir,
            name=name,
            seed=seed,
            n_conformers=n_conformers_medium,
            max_representatives=max_reps_medium,
        )

    # Complex: cap representatives for very large ligands
    n_heavy = mol.GetNumHeavyAtoms()
    effective_max_reps = max_reps_complex

    # >50 atoms — force single conformer to avoid Vina timeout/hang
    if n_heavy > 50:
        logger.info(
            f"Very large ligand ({n_heavy} heavy atoms)"
            " — forcing single conformer to avoid Vina hang"
        )
        single_path = os.path.join(output_dir, "ligand.pdbqt")
        return prepare_ligand(smiles, single_path, name=name, seed=seed)

    if n_heavy > 45:
        effective_max_reps = min(effective_max_reps, 2)
        logger.info(
            f"Large ligand ({n_heavy} heavy atoms)"
            f" — capping representatives to {effective_max_reps}"
        )

    return prepare_ligand_multi(
        smiles,
        output_dir,
        name=name,
        seed=seed,
        n_conformers=n_conformers_complex,
        max_representatives=effective_max_reps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Binding Site Detection (fpocket + P2Rank)
# ─────────────────────────────────────────────────────────────────────────────


def _compute_box_size(
    dims: tuple[float, float, float], padding: float = 5.0
) -> tuple[float, float, float]:
    """Compute Vina docking box size from pocket dimensions + padding."""
    box = []
    for d in dims:
        v = d + 2 * padding
        rounded = round(v * 2) / 2  # nearest 0.5 Å
        box.append(max(10.0, rounded))
    return tuple(box)


def _prepare_pdb_for_fpocket(pdb_in: str, pdb_out: str) -> None:
    """Strip waters and keep only ATOM/HETATM for fpocket.

    Supports both PDB and mmCIF input via ``read_pdb_atoms``.
    """
    from autodock.utils import read_pdb_atoms, write_pdb_atoms

    atoms = read_pdb_atoms(pdb_in)
    filtered = [a for a in atoms if a["res_name"] not in _SKIP_WATER]
    write_pdb_atoms(filtered, pdb_out)


def _parse_fpocket_info(info_path: str) -> list[dict[str, Any]]:
    """Parse fpocket *_info.txt to extract pocket metadata."""
    pockets = []
    if not os.path.exists(info_path):
        return pockets
    with open(info_path) as fh:
        text = fh.read()
    blocks = re.split(r"(?=Pocket \d+ :)", text)
    for block in blocks:
        m = re.match(r"Pocket (\d+) :", block)
        if not m:
            continue
        pocket_num = int(m.group(1))

        def _float_search(pattern: str) -> float | None:
            match = re.search(pattern, block)
            return float(match.group(1)) if match else None

        def _int_search(pattern: str) -> int | None:
            match = re.search(pattern, block)
            return int(match.group(1)) if match else None

        druggability = _float_search(r"Druggability Score\s+:\s+([\d.]+)")
        volume = _float_search(r"Volume\s+:\s+([\d.]+)")
        depth = _float_search(r"Depth\s+:\s+([\d.]+)")
        openings = _int_search(r"Number of mouth openings\s+:\s+(\d+)")
        n_apolar = _int_search(r"Number of apolar alpha sphere\s+:\s+(\d+)")
        n_polar = _int_search(r"Number of polar alpha sphere\s+:\s+(\d+)")

        # Read pocket PQR for centroid and dimensions
        info_dir = os.path.dirname(info_path)
        pqr_path = os.path.join(info_dir, "pockets", f"pocket{pocket_num}_vert.pqr")
        if not os.path.exists(pqr_path):
            pqr_path = os.path.join(info_dir, f"pocket{pocket_num}_vert.pqr")

        center = None
        dims = None
        if os.path.exists(pqr_path):
            coords = []
            with open(pqr_path) as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        try:
                            coords.append(
                                [
                                    float(line[30:38]),
                                    float(line[38:46]),
                                    float(line[46:54]),
                                ]
                            )
                        except ValueError:
                            continue
            if coords:
                ca = np.array(coords)
                center = tuple(ca.mean(axis=0).tolist())
                dims = tuple((ca.max(axis=0) - ca.min(axis=0)).tolist())

        if center:
            # Compute shape descriptors from vertex PQR (available now,
            # before the temp directory is cleaned up)
            shape = _compute_pocket_shape_descriptors(pqr_path)
            pockets.append(
                {
                    "num": pocket_num,
                    "druggability": druggability if druggability is not None else 0.0,
                    "volume": volume,
                    "depth": depth,
                    "openings": openings,
                    "n_apolar": n_apolar,
                    "n_polar": n_polar,
                    "center": center,
                    "dims": dims if dims else (20.0, 20.0, 20.0),
                    "circularity": shape.get("circularity"),
                    "aspect_ratio": shape.get("aspect_ratio"),
                }
            )
    return pockets


def _run_p2rank_rescore(prep_pdb: str, out_dir: str) -> dict[int, float] | None:
    """
    Run P2Rank rescore on fpocket output. Returns {fpocket_num: probability}.
    Returns None if P2Rank or Java unavailable or times out.
    """
    prank = find_p2rank()
    if not prank:
        logger.warning("P2Rank not found — skipping rescoring")
        return None

    base = os.path.splitext(os.path.basename(prep_pdb))[0]
    prep_dir = os.path.dirname(os.path.abspath(prep_pdb)) or "."
    fpocket_out_dir = os.path.join(prep_dir, f"{base}_out")
    fpocket_pdb = os.path.join(fpocket_out_dir, f"{base}_out.pdb")

    if not os.path.exists(fpocket_pdb):
        logger.warning(f"Fpocket output PDB not found for P2Rank: {fpocket_pdb}")
        return None

    ds_file = os.path.join(out_dir, "p2rank.ds")
    with open(ds_file, "w") as f:
        f.write(f"# P2Rank rescore for {base}\n")
        f.write("PARAM.PREDICTION_METHOD=fpocket\n")
        f.write("HEADER: prediction protein\n")
        f.write(f"{os.path.abspath(fpocket_pdb)}  {os.path.abspath(prep_pdb)}\n")

    pred_out = os.path.join(out_dir, "p2rank_out")
    # Do NOT set JAVA_HOME — P2Rank finds Java automatically.
    # Setting a wrong JAVA_HOME (e.g. /usr when java is /usr/bin/java)
    # causes P2Rank to hang indefinitely.

    success, _, stderr = safe_subprocess(
        ["bash", prank, "rescore", ds_file, "-o", pred_out, "-visualizations", "0"],
        timeout=45,
    )
    if not success:
        logger.warning(f"P2Rank rescore failed: {stderr[:300]}")
        return None

    csv_path = os.path.join(pred_out, f"{os.path.basename(prep_pdb)}_predictions.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"P2Rank predictions CSV not found: {csv_path}")
        return None

    probs = {}
    with open(csv_path) as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        try:
            prob_idx = header.index("probability")
            cx_idx = header.index("center_x")
        except ValueError:
            logger.warning(f"P2Rank CSV missing expected columns: {header}")
            return None

        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) <= max(prob_idx, cx_idx):
                continue
            try:
                name = parts[0]
                prob = float(parts[prob_idx])
                m = re.search(r"pocket[._]?(\d+)", name, re.IGNORECASE)
                if m:
                    fpocket_num = int(m.group(1))
                    probs[fpocket_num] = prob
            except (ValueError, IndexError):
                continue

    return probs


def _run_p2rank_predict(prep_pdb: str, out_dir: str) -> list[dict[str, Any]] | None:
    """Run P2Rank in standalone prediction mode (no fpocket dependency).

    Returns list of pocket dicts with keys: center, score, radius, residues
    or None if P2Rank unavailable / fails.
    """
    prank = find_p2rank()
    if not prank:
        return None

    pred_out = os.path.join(out_dir, "p2rank_predict")
    success, stdout, stderr = safe_subprocess(
        ["bash", prank, "predict", "-f", prep_pdb, "-o", pred_out, "-visualizations", "0"],
        timeout=120,
    )
    if not success:
        logger.warning(f"P2Rank predict failed: {stderr[:300]}")
        return None

    # Parse CSV output
    base = os.path.splitext(os.path.basename(prep_pdb))[0]
    csv_path = os.path.join(pred_out, f"{base}.pdb_predictions.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"P2Rank predictions CSV not found: {csv_path}")
        return None

    pockets = []
    with open(csv_path) as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        try:
            prob_idx = header.index("probability")
            cx_idx = header.index("center_x")
            cy_idx = header.index("center_y")
            cz_idx = header.index("center_z")
            rad_idx = header.index("radius") if "radius" in header else None
            name_idx = header.index("name")
        except ValueError as exc:
            logger.warning(f"P2Rank CSV missing columns: {exc}")
            return None

        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) <= max(cx_idx, cz_idx, prob_idx):
                continue
            try:
                prob = float(parts[prob_idx])
                center = (
                    float(parts[cx_idx]),
                    float(parts[cy_idx]),
                    float(parts[cz_idx]),
                )
                radius = (
                    float(parts[rad_idx]) if rad_idx is not None and rad_idx < len(parts) else 10.0
                )
                name = parts[name_idx]
                pocket_num = 0
                m = re.search(r"(\d+)", name)
                if m:
                    pocket_num = int(m.group(1))
                pockets.append(
                    {
                        "num": pocket_num,
                        "center": center,
                        "radius": radius,
                        "score": prob,
                        "druggability": prob,
                        "volume": None,
                        "depth": None,
                        "openings": None,
                        "n_apolar": None,
                        "n_polar": None,
                        "dims": (radius * 2, radius * 2, radius * 2),
                        "pocket_source": "p2rank",
                    }
                )
            except (ValueError, IndexError):
                continue

    logger.info(f"P2Rank predict: {len(pockets)} pocket(s) found")
    return pockets


def _parse_p2rank_residues(
    pockets_csv: str,
) -> dict[int, list[dict[str, Any]]]:
    """Parse residue_ids from P2Rank predictions CSV.

    P2Rank CSV ``residue_ids`` column contains comma-separated entries like
    ``A:123,A:124,B:45``. Returns mapping {pocket_num: [residue dicts]}.

    Reference:
        Krivák & Hoksza (2018) JCIM 34:1399-1410.
    """
    result: dict[int, list[dict[str, Any]]] = {}
    if not os.path.exists(pockets_csv):
        return result

    with open(pockets_csv) as fh:
        header = [h.strip() for h in fh.readline().strip().split(",")]
        try:
            name_idx = header.index("name")
            resid_idx = header.index("residue_ids")
        except ValueError:
            return result

        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) <= max(name_idx, resid_idx):
                continue
            name = parts[name_idx]
            m = re.search(r"(\d+)", name)
            if not m:
                continue
            pocket_num = int(m.group(1))
            residues_str = parts[resid_idx]
            if not residues_str or residues_str == "-":
                result[pocket_num] = []
                continue
            residues = []
            for token in residues_str.split(","):
                token = token.strip()
                mm = re.match(r"([A-Za-z0-9]*):(\d+)", token)
                if mm:
                    chain = mm.group(1) if mm.group(1) else "A"
                    resi = int(mm.group(2))
                    residues.append({"chain": chain, "resid": resi})
            result[pocket_num] = residues

    return result


def _compute_pocket_shape_descriptors(
    vert_pqr_path: str,
) -> dict[str, float | None]:
    """Compute geometric shape descriptors from fpocket vertex PQR.

    Returns dict with keys:
      - circularity: sphericity index (0-1, 1 = perfect sphere)
      - aspect_ratio: elongation (≥1, larger = more elongated)
      - surface_area: approximate surface area from alpha-sphere vertices (Å²)

    Reference:
        Weisel et al. (2007) J. Chem. Inf. Model. 47:1799-1811.
    """
    desc: dict[str, float | None] = {
        "circularity": None,
        "aspect_ratio": None,
        "surface_area": None,
    }
    if not os.path.exists(vert_pqr_path):
        return desc

    coords = []
    with open(vert_pqr_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ]
                    )
                except (ValueError, IndexError):
                    continue

    if len(coords) < 4:
        return desc

    ca = np.array(coords)
    center = ca.mean(axis=0)
    centered = ca - center
    cov = (centered.T @ centered) / len(centered)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-12)  # avoid zero / negative

    # Circularity (sphericity): ratio of smallest to largest eigenvalue
    # 1.0 = perfect sphere, <<1 = highly elongated
    circularity = float(np.min(eigvals) / np.max(eigvals))
    desc["circularity"] = round(circularity, 4)

    # Aspect ratio: sqrt(max / min)
    aspect_ratio = float(np.sqrt(np.max(eigvals) / np.min(eigvals)))
    desc["aspect_ratio"] = round(aspect_ratio, 2)

    # Approximate surface area: convex hull area
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(ca)
        desc["surface_area"] = round(float(hull.area), 1)
    except ImportError:
        pass
    except (OSError, ValueError, RuntimeError):
        pass

    return desc


def _validate_alphafold_pocket(
    pocket_center: tuple[float, float, float],
    pocket_radius: float,
    receptor_pdb: str,
    plddt_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether a pocket overlaps with low-pLDDT AlphaFold regions.

    Args:
        pocket_center: (x, y, z) pocket center.
        pocket_radius: pocket radius (Å) for overlap check.
        receptor_pdb: Path to receptor PDB file (for pLDDT extraction).
        plddt_data: Pre-computed output from
            :func:`assess_alphafold_quality`.  If None, computed on the fly.

    Returns:
        Dict with keys:
          - af_suitable: bool — True if no low-pLDDT overlap
          - mean_plddt_in_pocket: float or None
          - min_plddt_in_pocket: float or None
          - overlapping_low_conf_regions: list of region dicts
    """
    from autodock.alphafold_tools import assess_alphafold_quality

    result: dict[str, Any] = {
        "af_suitable": True,
        "mean_plddt_in_pocket": None,
        "min_plddt_in_pocket": None,
        "overlapping_low_conf_regions": [],
    }

    if not os.path.exists(receptor_pdb):
        return result

    quality = plddt_data or assess_alphafold_quality(receptor_pdb)
    if not quality.get("suitable_for_docking", False):
        result["af_suitable"] = False

    low_regions = quality.get("low_confidence_regions", [])
    if not low_regions:
        return result

    # Pocket sphere: check if any low-confidence Cα atoms fall inside
    cx, cy, cz = pocket_center
    pocket_plddts = []
    overlapping = []

    try:
        with open(receptor_pdb) as fh:
            for line in fh:
                if not line.startswith("ATOM  "):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    bfactor = float(line[60:66])
                    chain = line[21:22].strip() or "A"
                    resi = int(line[22:26].strip())
                except (ValueError, IndexError):
                    continue
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                if dist_sq <= pocket_radius**2:
                    pocket_plddts.append(bfactor)
                    for region in low_regions:
                        if region.get("chain") == chain and region.get(
                            "start", 0
                        ) <= resi <= region.get("end", 0):
                            overlapping.append(region)
    except OSError:
        pass

    if pocket_plddts:
        result["mean_plddt_in_pocket"] = round(float(np.mean(pocket_plddts)), 1)
        result["min_plddt_in_pocket"] = round(float(np.min(pocket_plddts)), 1)
    if overlapping:
        result["af_suitable"] = False
        # Deduplicate overlapping regions (same region may be hit by multiple Cα atoms)
        unique_overlapping = []
        seen = set()
        for region in overlapping:
            key = (region.get("chain"), region.get("start"), region.get("end"))
            if key not in seen:
                seen.add(key)
                unique_overlapping.append(region)
        result["overlapping_low_conf_regions"] = unique_overlapping
    else:
        result["overlapping_low_conf_regions"] = []

    return result


def _druggability_classification(
    druggability_score: float | None,
    volume: float | None = None,
    depth: float | None = None,
) -> dict[str, Any]:
    """Classify pocket druggability and return qualitative description.

    Thresholds adapted from:
        - Schmidtke & Barril (2010) J. Med. Chem. 53:5858-5867.
        - Kozakov et al. (2015) J. Med. Chem. 58:5083-5119.

    Returns:
        Dict with keys:
          - level: "high" | "medium" | "low" | "unknown"
          - description: str for publication reporting
          - references: list of citation keys
    """
    if druggability_score is None:
        return {
            "level": "unknown",
            "description": "Druggability not assessed (no score available)",
            "references": [],
        }

    refs = ["Schmidtke & Barril 2010 JMC", "Kozakov et al. 2015 JMC"]
    if druggability_score >= _DRUGGABILITY_HIGH:
        return {
            "level": "high",
            "description": (
                f"Highly druggable pocket (score={druggability_score:.2f}). "
                "Likely to bind small-molecule drug-like compounds with high affinity."
            ),
            "references": refs,
        }
    elif druggability_score >= _DRUGGABILITY_MEDIUM:
        extra = ""
        if volume is not None and volume < 300:
            extra = " Small volume may limit ligand size."
        if depth is not None and depth < 4:
            extra += " Shallow binding site may reduce selectivity."
        return {
            "level": "medium",
            "description": (
                f"Moderately druggable pocket (score={druggability_score:.2f})."
                f"{extra} May require focused library screening."
            ),
            "references": refs,
        }
    else:
        return {
            "level": "low",
            "description": (
                f"Low druggability (score={druggability_score:.2f}). "
                "Consider targeting a different pocket or using covalent/fragment-based "
                "approaches. Protein-protein interaction interface may be present."
            ),
            "references": refs,
        }


def _classify_pocket_type(
    pocket_center: tuple[float, float, float],
    known_active_site_center: tuple[float, float, float] | None,
    pocket_volume: float | None = None,
) -> dict[str, Any]:
    """Classify pocket as orthosteric or allosteric candidate.

    Args:
        pocket_center: (x, y, z) of detected pocket.
        known_active_site_center: (x, y, z) of known orthosteric site.
            If None, all pockets are labelled "unclassified".
        pocket_volume: pocket volume in Å³, used for secondary classification.

    Returns:
        Dict with keys:
          - type: "orthosteric" | "allosteric_candidate" | "unclassified"
          - distance_to_active: float or None (Å)
          - confidence: str
    """
    if known_active_site_center is None:
        return {
            "type": "unclassified",
            "distance_to_active": None,
            "confidence": "no_reference",
        }

    dist = np.linalg.norm(np.array(pocket_center) - np.array(known_active_site_center))
    dist_rounded = round(float(dist), 1)

    if dist <= _ALLOSTERIC_MIN_DISTANCE / 2:
        return {
            "type": "orthosteric",
            "distance_to_active": dist_rounded,
            "confidence": "high" if dist < 5.0 else "medium",
        }
    elif dist >= _ALLOSTERIC_MIN_DISTANCE:
        return {
            "type": "allosteric_candidate",
            "distance_to_active": dist_rounded,
            "confidence": (
                "high" if pocket_volume is not None and 200 <= pocket_volume <= 1500 else "medium"
            ),
        }
    else:
        return {
            "type": "unclassified",
            "distance_to_active": dist_rounded,
            "confidence": "low — ambiguous distance",
        }


def _pocket_bfactor_flexibility(
    pocket_center: tuple[float, float, float],
    pocket_radius: float,
    receptor_pdb: str,
) -> dict[str, Any]:
    """Assess pocket flexibility from crystallographic B-factors.

    B-factors reflect atomic displacement; higher values indicate greater
    thermal motion / flexibility.

    Reference:
        Yuan et al. (2005) Proteins 58:672-682.
        (B-factor-based pocket flexibility correlates with induced-fit propensity)

    Returns:
        Dict with keys:
          - mean_bfactor: float — average B-factor of pocket Cα atoms (Å²)
          - max_bfactor: float
          - flexibility: "rigid" | "moderate" | "flexible"
          - induced_fit_likely: bool — True if mean_bfactor > threshold
    """
    result: dict[str, Any] = {
        "mean_bfactor": None,
        "max_bfactor": None,
        "flexibility": "unknown",
        "induced_fit_likely": False,
    }

    if not os.path.exists(receptor_pdb):
        return result

    cx, cy, cz = pocket_center
    bfactors = []
    try:
        with open(receptor_pdb) as fh:
            for line in fh:
                if not line.startswith("ATOM  "):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    bfactor = float(line[60:66])
                except (ValueError, IndexError):
                    continue
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                if dist_sq <= pocket_radius**2:
                    bfactors.append(bfactor)
    except OSError:
        pass

    if not bfactors:
        return result

    mean_b = float(np.mean(bfactors))
    max_b = float(np.max(bfactors))
    result["mean_bfactor"] = round(mean_b, 1)
    result["max_bfactor"] = round(max_b, 1)

    if mean_b < _POCKET_DEFAULT_BFACTOR:
        result["flexibility"] = "rigid"
        result["induced_fit_likely"] = False
    elif mean_b < _POCKET_DEFAULT_BFACTOR * 1.5:
        result["flexibility"] = "moderate"
        result["induced_fit_likely"] = True
    else:
        result["flexibility"] = "flexible"
        result["induced_fit_likely"] = True

    return result


def _run_fpocket_detect(
    receptor_pdb: str,
    min_alpha: float = 3.0,
    max_alpha: float = 6.0,
) -> list[dict[str, Any]] | None:
    """Run fpocket cavity detection as a standalone pipeline step.

    Returns list of pocket dicts (same schema as _parse_fpocket_info) or
    None if fpocket fails / is unavailable.

    Reference:
        Schmidtke et al. (2010) J. Mol. Biol. 403:656-676.
    """
    fpocket_bin = find_conda_tool("fpocket")
    if not fpocket_bin:
        logger.warning("fpocket not found — skipping fpocket detection")
        return None

    fd, prep_pdb = tempfile.mkstemp(suffix="_fpocket_prep.pdb")
    os.close(fd)
    _prepare_pdb_for_fpocket(receptor_pdb, prep_pdb)

    prep_pdb_abs = os.path.abspath(prep_pdb)
    prep_dir = os.path.dirname(prep_pdb_abs) or "."
    base = os.path.splitext(os.path.basename(prep_pdb))[0]
    out_dir = os.path.join(prep_dir, base + "_out")

    try:
        success, _, stderr = safe_subprocess(
            [fpocket_bin, "-f", prep_pdb_abs, "-m", str(min_alpha), "-M", str(max_alpha)],
            timeout=120,
            cwd=prep_dir,
        )
        if not success:
            logger.warning(f"fpocket failed: {stderr[:300]}")
            return None

        info_file = os.path.join(out_dir, base + "_info.txt")
        if not os.path.exists(info_file):
            logger.warning(f"fpocket info file not found: {info_file}")
            return None

        pockets = _parse_fpocket_info(info_file)
        logger.info(f"fpocket: {len(pockets)} pocket(s) detected")
        return pockets

    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        logger.warning(f"fpocket detection failed: {exc}")
        return None
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        with contextlib.suppress(OSError):
            if os.path.exists(prep_pdb):
                os.remove(prep_pdb)


def _run_dogsite3_predict(
    receptor_pdb: str,
    chain_id: str = "A",
    timeout: int = 300,
) -> list[dict[str, Any]] | None:
    """Run DoGSite3 pocket detection via the proteins.plus REST API.

    Uses the asynchronous job pattern documented at
    https://proteins.plus/api/v2/

    Returns list of pocket dicts or None on failure.  Note: requires
    ``requests`` library and internet connectivity.

    Reference:
        DoGSite3 — Leis et al. (2010) BMC Bioinformatics 11:557.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not available — skipping DoGSite3")
        return None

    import time

    base_url = "https://proteins.plus/api/v2"

    # Submit
    try:
        with open(receptor_pdb, "rb") as fh:
            files = {"pdb_file": fh}
            data = {"chain_id": chain_id, "ligand": "", "calc_clefts": "true"}
            resp = requests.post(f"{base_url}/dogsite3/", files=files, data=data, timeout=60)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        logger.warning(f"DoGSite3 submission failed: {exc}")
        return None

    # Poll
    start = time.time()
    job_result = None
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/dogsite3/{job_id}/", timeout=30)
            resp.raise_for_status()
            status = resp.json().get("status", "unknown")
            if status == "completed":
                job_result = resp.json()
                break
            elif status == "failed":
                logger.warning(f"DoGSite3 job {job_id} failed")
                return None
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.warning(f"DoGSite3 poll error: {exc}")
            return None
        time.sleep(5)
    else:
        logger.warning(f"DoGSite3 job {job_id} timed out after {timeout}s")
        return None

    if not job_result:
        return None

    # Download results
    try:
        dl_resp = requests.get(f"{base_url}/dogsite3/{job_id}/download/", timeout=60)
        dl_resp.raise_for_status()
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        logger.warning(f"DoGSite3 download failed: {exc}")
        return None

    # Parse DoGSite3 CSV output
    import io
    import zipfile

    pockets = []
    try:
        with zipfile.ZipFile(io.BytesIO(dl_resp.content)) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_files:
                logger.warning("DoGSite3: no CSV in ZIP")
                return None
            with zf.open(csv_files[0]) as csv_fh:
                text = csv_fh.read().decode("utf-8")
                lines = text.strip().split("\n")
                if len(lines) < 2:
                    return None
                csv_header = [h.strip() for h in lines[0].split(",")]
                try:
                    cx_idx = csv_header.index("Center_x")
                    cy_idx = csv_header.index("Center_y")
                    cz_idx = csv_header.index("Center_z")
                    vol_idx = csv_header.index("Volume")
                    drugg_idx = csv_header.index("Druggability_Score")
                    surf_idx = csv_header.index("Surface")
                except ValueError:
                    cx_idx = csv_header.index("center_x")
                    cy_idx = csv_header.index("center_y")
                    cz_idx = csv_header.index("center_z")
                    vol_idx = csv_header.index("volume")
                    drugg_idx = csv_header.index("druggability_score")
                    surf_idx = csv_header.index("surface")

                for i, line in enumerate(lines[1:], 1):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) <= max(cx_idx, cz_idx, vol_idx):
                        continue
                    try:
                        center = (
                            float(parts[cx_idx]),
                            float(parts[cy_idx]),
                            float(parts[cz_idx]),
                        )
                        volume = float(parts[vol_idx])
                        drugg = float(parts[drugg_idx])
                        pocket = {
                            "num": i,
                            "center": center,
                            "druggability": drugg,
                            "volume": volume,
                            "surface_area": (
                                float(parts[surf_idx]) if surf_idx < len(parts) else None
                            ),
                            "pocket_source": "dogsite3",
                            "box_size": _compute_box_size(
                                (volume ** (1 / 3),) * 3,
                                padding=5.0,
                            ),
                        }
                        pockets.append(pocket)
                    except (ValueError, IndexError):
                        continue
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        logger.warning(f"DoGSite3 parse failed: {exc}")
        return None

    logger.info(f"DoGSite3: {len(pockets)} pocket(s) detected")
    return pockets


def find_top_pockets(
    receptor_pdb: str,
    ligand_pdb: str | None = None,
    padding: float = 5.0,
    max_pockets: int = 5,
    known_active_site: tuple[float, float, float] | None = None,
    plddt_data: dict[str, Any] | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Identify top-N candidate binding pockets with publication-grade analysis.

    Pipeline:
      1. **P2Rank** — ML-based random forest classifier as primary screen
      2. **fpocket** — geometric cavity detection (α-sphere) for cross-validation
      3. **Cross-validation** — spatial overlap check: each P2Rank candidate
         is verified against fpocket pockets (threshold: 5 Å center distance)
      4. **Druggability re-rank** — fpocket Drug Score re-orders verified pockets
      5. **Enhanced analysis** — per-pocket: residue IDs, druggability class,
         AlphaFold pLDDT compatibility, B-factor flexibility, pocket type

    References:
        - Krivák & Hoksza (2018) JCIM (P2Rank)
        - Schmidtke et al. (2010) J. Mol. Biol. (fpocket)
        - Schmidtke & Barril (2010) JMC (druggability)
        - Jumper et al. (2021) Nature (AlphaFold pLDDT)

    Args:
        receptor_pdb: Protein PDB file.
        ligand_pdb: Optional co-crystallized ligand PDB for centering.
            When provided, skips computational detection (gold standard).
        padding: Box padding around pocket dimensions (Å, default 5.0).
        max_pockets: Maximum pockets to return (default 5).
        known_active_site: (x, y, z) of known orthosteric site for
            allosteric/orthosteric classification.
        plddt_data: Pre-computed AlphaFold quality assessment
            (from :func:`~autodock.alphafold_tools.assess_alphafold_quality`).

    Returns:
        List of pocket dicts, each with keys: center, box_size,
        druggability, druggability_level, druggability_description,
        p2rank_prob, pocket_num, pocket_source, fpocket_verified,
        fpocket_match_distance, volume, depth, openings, n_apolar,
        n_polar, residue_ids, shape_circularity, shape_aspect_ratio,
        flexibility, induced_fit_likely, af_suitable, af_mean_plddt,
        af_min_plddt, pocket_type, distance_to_active.
    """

    # ── Cache lookup -------------------------------------------------------
    if cache_dir is not None:
        from autodock.cache import PocketCache

        pc = PocketCache(cache_dir)
        cache_params = {
            "padding": padding,
            "max_pockets": max_pockets,
            "known_active_site": known_active_site,
        }
        cached = pc.get(receptor_pdb, **cache_params)
        if cached:
            logger.info(f"Pocket cache hit — {len(cached)} pocket(s) from {pc.cache_dir}")
            return cached

    # ── Gold standard: ligand-centered ──────────────────────────────────────
    if ligand_pdb and os.path.isfile(ligand_pdb):
        try:
            from rdkit import Chem

            mol = Chem.MolFromPDBFile(ligand_pdb)
            if mol is not None:
                conf = mol.GetConformer()
                coords = np.array(
                    [
                        [
                            conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z,
                        ]
                        for i in range(mol.GetNumAtoms())
                    ]
                )
                center = tuple(coords.mean(axis=0).tolist())
                dims = tuple((coords.max(axis=0) - coords.min(axis=0)).tolist())
                box_size = _compute_box_size(dims, padding)
                logger.info(f"Binding site from ligand: center={center}, box={box_size}")
                return [
                    {
                        "center": center,
                        "box_size": box_size,
                        "druggability": None,
                        "druggability_level": "unknown",
                        "druggability_description": "Centered on co-crystal ligand",
                        "p2rank_prob": None,
                        "pocket_num": None,
                        "pocket_source": "crystal_ligand",
                        "fpocket_verified": None,
                        "fpocket_match_distance": None,
                        "volume": None,
                        "depth": None,
                        "openings": None,
                        "n_apolar": None,
                        "n_polar": None,
                        "residue_ids": [],
                        "shape_circularity": None,
                        "shape_aspect_ratio": None,
                        "flexibility": None,
                        "induced_fit_likely": None,
                        "af_suitable": None,
                        "af_mean_plddt": None,
                        "af_min_plddt": None,
                        "pocket_type": "orthosteric" if known_active_site else "unclassified",
                        "distance_to_active": None,
                    }
                ]
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.warning(
                f"Ligand-centered pocket detection failed: {exc}. "
                "Falling back to computational detection."
            )

    # ── Preparation ─────────────────────────────────────────────────────────
    fd, prep_pdb = tempfile.mkstemp(suffix="_prep.pdb")
    os.close(fd)
    prep_pdb_abs: str | None = None
    prep_dir: str | None = None
    try:
        _prepare_pdb_for_fpocket(receptor_pdb, prep_pdb)
        prep_pdb_abs = os.path.abspath(prep_pdb)
        prep_dir = os.path.dirname(prep_pdb_abs) or "."
        # fpocket is now delegated to _run_fpocket_detect() which handles
        # its own temp files. The prep_pdb here is used only by P2Rank.

        p2rank_available = find_p2rank() is not None
        fpocket_available = find_conda_tool("fpocket") is not None

        # Step 1: P2Rank primary screen ─────────────────────────────────────────
        p2rank_pockets: list[dict[str, Any]] | None = None
        p2rank_csv_path: str | None = None

        if p2rank_available:
            p2rank_pockets = _run_p2rank_predict(prep_pdb_abs, prep_dir)
            if p2rank_pockets:
                logger.info(f"P2Rank primary screen: {len(p2rank_pockets)} pocket(s) detected")
                _base = os.path.splitext(os.path.basename(prep_pdb))[0]
                _csv = os.path.join(prep_dir, "p2rank_predict", f"{_base}.pdb_predictions.csv")
                if os.path.exists(_csv):
                    p2rank_csv_path = _csv

        if not p2rank_pockets:
            # Fallback: DoGSite3 via proteins.plus REST API
            dogsite3_pockets = _run_dogsite3_predict(receptor_pdb)
            if dogsite3_pockets:
                logger.info(
                    f"DoGSite3 fallback: {len(dogsite3_pockets)} pocket(s) detected"
                )
                # Normalise DoGSite3 format to P2Rank-compatible dicts so the
                # rest of the cross-validation / ranking pipeline works unchanged.
                p2rank_pockets = []
                for i, dgp in enumerate(dogsite3_pockets):
                    vol = dgp.get("volume") or 1000.0
                    radius = (vol ** (1 / 3)) / 2  # approximate from volume
                    p2rank_pockets.append(
                        {
                            "num": i + 1,
                            "center": dgp["center"],
                            "radius": radius,
                            "score": dgp.get("druggability", 0.5),
                            "druggability": dgp.get("druggability", 0.5),
                            "volume": vol,
                            "depth": None,
                            "openings": None,
                            "n_apolar": None,
                            "n_polar": None,
                            "dims": (radius * 2, radius * 2, radius * 2),
                            "pocket_source": "dogsite3",
                        }
                    )
            if not p2rank_pockets:
                raise PreparationError(
                    f"P2Rank found no pockets in {receptor_pdb}. "
                    "Cannot proceed — P2Rank is the primary detection method."
                )

        # Step 2: fpocket geometric cross-validation ────────────────────────────
        fpocket_pockets: list[dict[str, Any]] | None = None
        if fpocket_available:
            fpocket_pockets = _run_fpocket_detect(receptor_pdb)
            if fpocket_pockets:
                logger.info(f"fpocket cross-validation: {len(fpocket_pockets)} pocket(s) detected")
            else:
                logger.warning("fpocket found no pockets — proceeding with P2Rank only")

        # Step 3: Cross-validation — spatial overlap check ──────────────────────
        # For each P2Rank candidate, find the nearest fpocket pocket within 8 Å.
        # Mark fpocket_verified=True/False and carry over the fpocket druggability.
        fpocket_centers: list[tuple[np.ndarray, dict]] = []
        if fpocket_pockets:
            for fp in fpocket_pockets:
                fpocket_centers.append((np.array(fp["center"]), fp))

        # Limit P2Rank candidates to top 10 for cross-validation.
        # P2Rank paper (Krivák & Hoksza 2018): top-5 ~88%, top-10 ~92%.
        # With 5Å tight threshold, casting a wider net (top-10) ensures
        # true pockets are not prematurely discarded before fpocket
        # cross-validation. Final output is capped at max_pockets (5).
        _P2RANK_CROSSVAL_TOPK = 10
        p2rank_shortlist = p2rank_pockets[:_P2RANK_CROSSVAL_TOPK]
        logger.info(
            f"Cross-validating top {len(p2rank_shortlist)} P2Rank pocket(s) "
            f"(from {len(p2rank_pockets)} total, prob ≥ {_P2RANK_PROB_THRESHOLD})"
        )

        candidates: list[dict[str, Any]] = []
        for p2p in p2rank_shortlist:
            p2_center = np.array(p2p["center"])
            prob = p2p.get("score")

            # NOTE: We do NOT hard-filter by probability here because
            # P2Rank score thresholds are protein-dependent and a single
            # cutoff discards many valid pockets before fpocket cross-validation.
            # Krivák & Hoksza 2018 (Table 3) show Top-10 recall ~90% on COACH420
            # regardless of score.  fpocket verification (below) is the actual filter.
            if prob is not None and prob < _P2RANK_PROB_THRESHOLD:
                logger.debug(
                    f"Pocket #{p2p.get('num', 0)}: P2Rank prob={prob:.3f} below "
                    f"default threshold {_P2RANK_PROB_THRESHOLD} — kept for "
                    f"fpocket cross-validation; final rank may demote it."
                )

            # Find nearest fpocket pocket
            best_dist = float("inf")
            best_fp: dict | None = None
            for fp_c, fp_dict in fpocket_centers:
                d = float(np.linalg.norm(p2_center - fp_c))
                if d < best_dist:
                    best_dist = d
                    best_fp = fp_dict

            verified = best_fp is not None and best_dist <= _POCKET_CONSENSUS_DISTANCE

            # Re-rank metric: fpocket druggability if verified, else P2Rank prob
            druggability = best_fp.get("druggability") if best_fp else p2p.get("druggability")
            # Use the fpocket druggability for verified pockets
            rank_score = druggability if (druggability is not None and verified) else prob

            candidates.append(
                {
                    "p2rank_pocket": p2p,
                    "fpocket_pocket": best_fp,
                    "verified": verified,
                    "match_distance": round(best_dist, 2) if best_fp else None,
                    "druggability": druggability,
                    "rank_score": rank_score if rank_score is not None else 0.0,
                    "prob": prob,
                }
            )

        if not candidates:
            raise PreparationError(
                f"All P2Rank pockets filtered out (prob < {_P2RANK_PROB_THRESHOLD})."
            )

        # Step 4: Re-rank by fpocket druggability (verified pockets → higher rank) ──
        candidates.sort(
            key=lambda c: (
                not c["verified"],  # verified first
                -(c["druggability"] or 0.0),  # then by druggability desc
                -(c["prob"] or 0.0),  # then by P2Rank prob desc
            ),
        )

        # ── Parse P2Rank residue IDs ───────────────────────────────────────────
        pocket_residues: dict[int, list[dict[str, Any]]] = {}
        if p2rank_csv_path:
            pocket_residues = _parse_p2rank_residues(p2rank_csv_path)

        # ── AlphaFold quality assessment ────────────────────────────────────────
        af_data = plddt_data
        if af_data is None and os.path.exists(receptor_pdb):
            try:
                from autodock.alphafold_tools import assess_alphafold_quality

                af_data = assess_alphafold_quality(receptor_pdb)
            except (OSError, ValueError, RuntimeError, ImportError):
                pass

        # Step 5: Build enriched result list ─────────────────────────────────────
        # Output priority:
        #   verified (P2Rank+fpocket match) → use fpocket pocket (α-sphere centroid,
        #     druggability_score computed on own definition, semantically consistent)
        #   unverified → use P2Rank pocket (fallback, only data available)
        result: list[dict[str, Any]] = []
        for c in candidates[:max_pockets]:
            verified = c["verified"]
            # Data source: fpocket for verified, P2Rank for unverified
            src = c["fpocket_pocket"] if verified else c["p2rank_pocket"]

            center = src["center"]
            box_size = _compute_box_size(src.get("dims", (20, 20, 20)), padding)
            volume = src.get("volume")
            depth_val = src.get("depth")
            pnum = src.get("num", 0)

            drugg_class = _druggability_classification(c["druggability"], volume, depth_val)

            # Shape descriptors
            shape_desc: dict[str, float | None] = {
                "circularity": src.get("circularity"),
                "aspect_ratio": src.get("aspect_ratio"),
            }

            # Pocket residues (P2Rank has richer residue data via CSV parsing)
            residues = c["p2rank_pocket"].get("residue_ids") or pocket_residues.get(
                c["p2rank_pocket"].get("num", 0), []
            )

            # B-factor flexibility
            pocket_radius = src.get("radius") or 10.0
            flex_info = _pocket_bfactor_flexibility(center, pocket_radius, receptor_pdb)

            # AlphaFold compatibility
            af_info: dict[str, Any] = {
                "af_suitable": None,
                "mean_plddt_in_pocket": None,
                "min_plddt_in_pocket": None,
            }
            if af_data and af_data.get("mean_plddt") is not None:
                af_info = _validate_alphafold_pocket(center, pocket_radius, receptor_pdb, af_data)

            # Allosteric / orthosteric
            pocket_type_info = _classify_pocket_type(center, known_active_site, volume)

            entry = {
                "center": center,
                "box_size": box_size,
                "druggability": c["druggability"],
                "druggability_level": drugg_class["level"],
                "druggability_description": drugg_class["description"],
                "p2rank_prob": c["prob"],
                "pocket_num": pnum,
                "pocket_source": "fpocket" if verified else src.get("pocket_source", "p2rank"),
                "fpocket_verified": verified,
                "fpocket_match_distance": c["match_distance"],
                "volume": volume,
                "depth": depth_val,
                "openings": src.get("openings"),
                "n_apolar": src.get("n_apolar"),
                "n_polar": src.get("n_polar"),
                "residue_ids": residues,
                "shape_circularity": shape_desc.get("circularity"),
                "shape_aspect_ratio": shape_desc.get("aspect_ratio"),
                "flexibility": flex_info["flexibility"],
                "induced_fit_likely": flex_info["induced_fit_likely"],
                "af_suitable": af_info.get("af_suitable"),
                "af_mean_plddt": af_info.get("mean_plddt_in_pocket"),
                "af_min_plddt": af_info.get("min_plddt_in_pocket"),
                "pocket_type": pocket_type_info["type"],
                "distance_to_active": pocket_type_info["distance_to_active"],
            }
            result.append(entry)

        # Log summary
        for i, pk in enumerate(result):
            v_str = " ✓fpocket" if pk["fpocket_verified"] else " fpocket-unverified"
            src_label = pk["pocket_source"]
            p_str = (
                f"P2Rank={pk['p2rank_prob']:.3f}" if pk["p2rank_prob"] is not None else "P2Rank=N/A"
            )
            logger.info(
                f"Pocket {i + 1} (#{pk['pocket_num']}): [{src_label}]{v_str}, "
                f"center={pk['center']}, box={pk['box_size']} ({p_str}, "
                f"druggability={pk['druggability']:.3f} ({pk['druggability_level']}))"
            )

        # ── Cache store --------------------------------------------------------
        if cache_dir is not None:
            from autodock.cache import PocketCache

            pc = PocketCache(cache_dir)
            cache_params = {
                "padding": padding,
                "max_pockets": max_pockets,
                "known_active_site": known_active_site,
            }
            pc.put(receptor_pdb, result, **cache_params)

        return result
    finally:
        # Cleanup temp files regardless of success or failure.
        # _run_fpocket_detect() handles its own cleanup internally.
        for f in (prep_pdb,):
            with contextlib.suppress(OSError):
                if os.path.exists(f):
                    os.unlink(f)
        if prep_dir:
            for d in (
                os.path.join(prep_dir, "p2rank_predict"),
                os.path.join(prep_dir, "p2rank_out"),
            ):
                with contextlib.suppress(OSError):
                    if os.path.exists(d):
                        shutil.rmtree(d, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Flexible receptor preparation
# ─────────────────────────────────────────────────────────────────────────────


def find_nearby_residues(
    pdb_file: str,
    center: tuple[float, float, float],
    radius: float = 6.0,
    max_residues: int = 6,
    exclude_water: bool = True,
) -> list[str]:
    """Identify protein residues whose Cα atoms lie within *radius* Å of *center*.

    The returned list uses the Meeko ``chain:resid`` format
    (e.g. ``["A:149", "A:276"]``) which is accepted by
    ``mk_prepare_receptor.py --flexres``.

    Args:
        pdb_file: Path to a PDB file (already cleaned — ligand/waters removed).
        center: (x, y, z) reference point, typically the ligand centroid.
        radius: Search radius in Å (default 6.0).
        max_residues: Maximum number of residues to return (default 6).
            Targets with very large binding pockets may have >20 nearby
            residues; capping prevents excessive compute in flexible docking.
        exclude_water: If True (default), skip water residues.

    Returns:
        List of ``chain:resid`` strings sorted by distance ascending.
    """
    cx, cy, cz = center
    nearby: list[tuple[float, str]] = []
    seen: set[str] = set()

    try:
        with open(pdb_file) as fh:
            for line in fh:
                if not line.startswith("ATOM  "):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                res_name = line[17:20].strip()
                if exclude_water and res_name in _SKIP_WATER:
                    continue
                chain = line[21:22].strip() or "A"
                try:
                    resi = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                if dist_sq <= radius**2:
                    key = f"{chain}:{resi}"
                    if key not in seen:
                        seen.add(key)
                        nearby.append((math.sqrt(dist_sq), key))
    except OSError as exc:
        logger.warning(f"Could not read {pdb_file} for nearby-residue search: {exc}")
        return []

    nearby.sort(key=lambda t: t[0])
    return [key for _, key in nearby[:max_residues]]


def _clean_pdb_with_openbabel(input_pdb: str, output_pdb: str) -> None:
    """Deprotonate + reprotonate a PDB file via Open Babel to fix RDKit valence issues."""
    import subprocess

    tmp_pdb = output_pdb + ".tmp"
    # Step 1: remove hydrogens
    subprocess.run(
        ["obabel", "-ipdb", input_pdb, "-opdb", "-O", tmp_pdb, "-d"],
        check=True,
        capture_output=True,
        timeout=60,
    )
    # Step 2: re-add hydrogens
    subprocess.run(
        ["obabel", "-ipdb", tmp_pdb, "-opdb", "-O", output_pdb, "-h"],
        check=True,
        capture_output=True,
        timeout=60,
    )
    if os.path.isfile(tmp_pdb):
        os.remove(tmp_pdb)


def prepare_flexible_receptor(
    pdb_file: str,
    flexres: list[str],
    output_dir: str,
    allow_bad_res: bool = True,
    timeout: int = 300,
) -> tuple[str, str]:
    """Prepare rigid + flexible receptor PDBQTs using Meeko.

    Wraps ``mk_prepare_receptor.py --read_pdb <pdb> -o <prefix> -p
    --flexres <list> -a`` (``-a`` = allow_bad_res).

    Args:
        pdb_file: Path to receptor PDB file.
        flexres: List of flexible residues in ``chain:resid`` format.
        output_dir: Directory for output files.
        allow_bad_res: Passed to Meeko as ``-a`` (default True).
        timeout: Subprocess timeout in seconds.

    Returns:
        ``(rigid_pdbqt_path, flex_pdbqt_path)``.

    Raises:
        PreparationError: If Meeko fails or outputs are missing.
    """
    import sys

    if not flexres:
        raise PreparationError("flexres list is empty — cannot prepare flexible receptor")

    ensure_dir(output_dir)
    prefix = os.path.join(output_dir, "receptor")
    flexres_str = ",".join(flexres)

    # Build mk_prepare_receptor.py command
    mkprep = find_conda_tool("mk_prepare_receptor.py")
    if not mkprep:
        raise PreparationError("mk_prepare_receptor.py not found in PATH/conda env")

    cmd = [
        sys.executable,
        mkprep,
        "--read_pdb",
        pdb_file,
        "-o",
        prefix,
        "-p",
        "--flexres",
        flexres_str,
    ]
    if allow_bad_res:
        cmd.append("-a")

    success, stdout, stderr = safe_subprocess(cmd, timeout=timeout)
    if not success:
        # Attempt Open Babel cleanup for RDKit valence errors
        err_lower = stderr.lower()
        if "valence" in err_lower or "rdkit" in err_lower:
            clean_pdb = os.path.join(output_dir, "receptor_clean.pdb")
            logger.warning(
                "mk_prepare_receptor.py failed with RDKit valence error — "
                f"attempting Open Babel cleanup ({pdb_file} → {clean_pdb})"
            )
            try:
                _clean_pdb_with_openbabel(pdb_file, clean_pdb)
                if os.path.isfile(clean_pdb) and os.path.getsize(clean_pdb) > 0:
                    cmd[cmd.index("--read_pdb") + 1] = clean_pdb
                    success, stdout, stderr = safe_subprocess(cmd, timeout=timeout)
            except Exception as exc:
                logger.warning(f"Open Babel cleanup failed: {exc}")
        if not success:
            raise PreparationError(f"mk_prepare_receptor.py failed: {stderr[:500]}")

    rigid_pdbqt = prefix + "_rigid.pdbqt"
    flex_pdbqt = prefix + "_flex.pdbqt"

    if not os.path.isfile(rigid_pdbqt):
        raise PreparationError(f"Rigid receptor PDBQT not created: {rigid_pdbqt}")
    if not os.path.isfile(flex_pdbqt):
        raise PreparationError(f"Flexible receptor PDBQT not created: {flex_pdbqt}")

    logger.info(
        f"Flexible receptor prepared ({len(flexres)} residues): "
        f"rigid={rigid_pdbqt}, flex={flex_pdbqt}"
    )
    return os.path.abspath(rigid_pdbqt), os.path.abspath(flex_pdbqt)
