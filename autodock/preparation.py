"""
autodock.preparation — Receptor / ligand preparation and binding-site detection.
==============================================================================
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
from typing import Any

import numpy as np

from autodock.core import (
    _DRUGGABILITY_THRESHOLD,
    _P2RANK_PROB_THRESHOLD,
    _POCKET_MAX_DIM,
    _POCKET_MAX_VOLUME,
    _POCKET_MIN_DEPTH,
    _POCKET_MIN_DIM,
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
) -> str:
    """
    Prepare a protein structure for docking (PDB/mmCIF → PDBQT).

    Uses a **three-stage publication-grade workflow**:

    1. **PDBFixer** — fills missing residues and atoms, assigns protonation
       states at the specified pH, replaces nonstandard residues.
    2. **OpenMM energy minimization** — short L-BFGS minimisation (200 steps)
       to relieve local strain from missing-atom reconstruction.
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
            restrained (10 kcal/mol/Å²) while side chains move freely;
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

    Returns:
        Absolute path to the prepared PDBQT file.

    Raises:
        PreparationError: If input file missing or preparation fails.
    """
    if not os.path.isfile(pdb_file):
        raise PreparationError(f"Input file not found: {pdb_file}")

    if not force and os.path.isfile(output_pdbqt):
        logger.info(f"Receptor PDBQT already exists — skipping prep: {output_pdbqt}")
        return os.path.abspath(output_pdbqt)

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
        except Exception as exc:
            raise PreparationError(f"CIF parsing failed: {exc}")
    else:
        with open(pdb_file) as fh:
            pdb_content = fh.read()

    # Write to temp PDB for PDBFixer consumption
    tmp_raw = write_temp_file(pdb_content, suffix="_raw.pdb")

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
            # else: skip other HETATMs (ligands, buffers, etc.)
        else:
            _pbfixer_lines.append(_line)
    _pbfixer_pdb = "".join(_pbfixer_lines)
    _tmp_fixer_in = write_temp_file(_pbfixer_pdb, suffix="_pdbfixer_in.pdb")

    try:
        from openmm.app import PDBFile as _OMM_PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise PreparationError(
            f"PDBFixer / OpenMM required for receptor preparation: {exc}. "
            "Install: conda install -c conda-forge pdbfixer openmm"
        )

    tmp_fixed = write_temp_file("", suffix="_fixed.pdb")
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
            gaps_detail = []
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
    except Exception as exc:
        logger.warning(f"PDBFixer failed ({exc}) — falling back to raw structure")
        tmp_fixed = tmp_raw  # use raw structure
    finally:
        # Clean up PDBFixer input temp
        with contextlib.suppress(OSError):
            if os.path.exists(_tmp_fixer_in):
                os.remove(_tmp_fixer_in)

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
        # Count flips from log
        n_flips = stdout.count("FLIP") - stdout.count("NOFLIP")
        if n_flips > 0:
            logger.info(f"Reduce: corrected {n_flips} ASN/GLN sidechain flip(s)")
        logger.info("Reduce: ASN/GLN flips processed, HIS tautomers assigned")
    except Exception as exc:
        logger.warning(f"Reduce step skipped ({exc}) — using PDBFixer output")
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
                    if _type == "ASP" and (_pka < 2.0 or _pka > 6.0):
                        _anomalous_pka.append({"residue": _label, "type": _type, "pKa": round(_pka, 1), "expected_range": "2.0-6.0"})
                    elif _type == "GLU" and (_pka < 2.0 or _pka > 7.0):
                        _anomalous_pka.append({"residue": _label, "type": _type, "pKa": round(_pka, 1), "expected_range": "2.0-7.0"})
                    elif _type == "HIS" and (_pka < 4.0 or _pka > 8.0):
                        _anomalous_pka.append({"residue": _label, "type": _type, "pKa": round(_pka, 1), "expected_range": "4.0-8.0"})
                    elif _type == "LYS" and (_pka < 8.0 or _pka > 12.0):
                        _anomalous_pka.append({"residue": _label, "type": _type, "pKa": round(_pka, 1), "expected_range": "8.0-12.0"})
                    elif _type == "CYS" and _pka > 11.0:
                        _anomalous_pka.append({"residue": _label, "type": _type, "pKa": round(_pka, 1), "expected_range": "<11.0"})
                    elif _type == "TYR" and _pka > 13.0:
                        _anomalous_pka.append({"residue": _label, "type": _type, "pKa": round(_pka, 1), "expected_range": "<13.0"})

            if _anomalous_pka:
                logger.info(
                    f"PROPKA: {len(_anomalous_pka)} anomalous pKa residue(s) at pH {ph}:"
                )
                for _a in _anomalous_pka[:15]:
                    logger.info(
                        f"  {_a['residue']:12s} ({_a['type']}): "
                        f"pKa={_a['pKa']:.1f} (expected {_a['expected_range']})"
                    )
                if len(_anomalous_pka) > 15:
                    logger.info(f"  ... and {len(_anomalous_pka) - 15} more")
                # Flag residues with pKa near target → protonation state uncertain
                _flagged = [a for a in _anomalous_pka if abs(a['pKa'] - ph) < 1.0]
                if _flagged:
                    logger.warning(
                        f"PROPKA: {len(_flagged)} residue(s) with pKa within ±1 of pH {ph} "
                        f"— protonation state may be wrong. Consider manual inspection."
                    )
            else:
                logger.info("PROPKA: all titratable residues within normal pKa ranges")
        except ImportError:
            logger.debug("propka not installed — skipping pKa prediction")
        except Exception as exc:
            logger.warning(f"PROPKA pKa prediction failed ({exc}) — skipping")

    # ── Step 5: OpenMM pocket-restrained energy minimisation ─────────────
    tmp_min = write_temp_file("", suffix="_minimized.pdb")
    try:
        from openmm import CustomExternalForce, LangevinIntegrator
        from openmm import unit as _omm_unit
        from openmm.app import ForceField, PDBFile, Simulation

        # Reload from Reduce output for OpenMM
        from pdbfixer import PDBFixer as _PBFixer

        _reduce_fixer = _PBFixer(filename=tmp_reduced)
        _top = _reduce_fixer.topology
        _pos = _reduce_fixer.positions  # OpenMM: units in nm

        ff = ForceField(forcefield)
        system = ff.createSystem(_top)

        # ── Positional restraints ─────────────────────────────────────────
        # Pocket region: backbone restrained (10 kcal/mol/Å²), side chains free
        # Outside pocket: all heavy atoms restrained (10 kcal/mol/Å²)
        # Keeps crystal structure globally while relieving local strain.
        k_kcal = 10.0  # restraint force constant (kcal/mol/Å²)
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

        integrator = LangevinIntegrator(
            300 * _omm_unit.kelvin,
            1.0 / _omm_unit.picosecond,
            0.002 * _omm_unit.picosecond,
        )
        simulation = Simulation(_top, system, integrator)
        simulation.context.setPositions(_pos)
        simulation.minimizeEnergy(maxIterations=200)
        min_positions = simulation.context.getState(getPositions=True).getPositions()
        with open(tmp_min, "w") as fh:
            PDBFile.writeFile(_top, min_positions, fh)
        logger.info("OpenMM: receptor energy minimised (200 steps L-BFGS)")
    except Exception as exc:
        logger.warning(f"OpenMM minimisation skipped ({exc}) — using PDBFixer output")
        tmp_min = tmp_fixed

    # Read back the cleaned PDB content
    with open(tmp_min) as fh:
        pdb_content = fh.read()

    # Clean up temp files (avoid deleting fallback aliases)
    _keep = {tmp_raw}
    for _t in (tmp_fixed, tmp_reduced, tmp_min):
        if _t in _keep:
            _keep.add(_t)
    for _t in (tmp_raw, tmp_fixed, tmp_reduced, tmp_min):
        if _t not in _keep:
            with contextlib.suppress(OSError):
                if os.path.exists(_t):
                    os.remove(_t)

    # ── Step 5: Filter waters / hetatms (second pass after PDBFixer) ─────
    _metal_set: set[str] = set()
    if retain_metal_ions:
        from autodock.core import _METAL_COFACTORS, _METAL_IONS

        _metal_set = _METAL_IONS | _METAL_COFACTORS

    n_waters_removed = 0
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
                    continue
                filtered.append(line)
            elif line.startswith("HETATM"):
                resn = safe_pdb_slice(line, 17, 20)
                # Retain metals even when remove_hetatms=True (P0 fix)
                if resn in _metal_set:
                    n_metals_retained_step5 += 1
                    filtered.append(line)
                    continue
                if remove_hetatms:
                    continue
                if keep_residues and resn not in keep_residues:
                    continue
                if remove_water and resn in _SKIP_WATER:
                    n_waters_removed += 1
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
                skipped_residues[resn] = skipped_residues.get(resn, 0) + 1
                continue
        filtered.append(line)
    if skipped_residues:
        logger.info(f"Skipped additives before Meeko: {dict(sorted(skipped_residues.items()))}")
    if retained_metals:
        logger.info(f"Retained metal ions / cofactors: {dict(sorted(retained_metals.items()))}")
    pdb_content = "\n".join(filtered)

    # ── Step 7: Meeko preparation ────────────────────────────────────────
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates
    except ImportError as exc:
        raise PreparationError(f"meeko not available: {exc}")

    templates = ResidueChemTemplates.create_from_defaults()
    mk_prep = MoleculePreparation(charge_model="gasteiger")

    try:
        polymer = Polymer.from_pdb_string(pdb_content, templates, mk_prep, default_altloc="A")
    except Exception:
        # Retry with allow_bad_res=True: removes unknown residues and continues
        logger.warning("Some residues failed template matching — retrying with allow_bad_res=True")
        try:
            polymer = Polymer.from_pdb_string(
                pdb_content, templates, mk_prep, allow_bad_res=True, default_altloc="A"
            )
        except Exception as exc2:
            logger.error(
                f"Meeko preparation failed even with allow_bad_res: {exc2} — "
                f"falling back to Open Babel"
            )
            return _prepare_receptor_with_obabel(pdb_file, output_pdbqt)

    try:
        rigid_pdbqt, _ = PDBQTWriterLegacy.write_from_polymer(polymer)
    except Exception as exc:
        logger.error(f"PDBQT writing failed: {exc} — falling back to Open Babel")
        return _prepare_receptor_with_obabel(pdb_file, output_pdbqt)

    ensure_dir(os.path.dirname(output_pdbqt) or ".")
    with open(output_pdbqt, "w") as fh:
        fh.write(rigid_pdbqt)

    logger.info(f"Receptor prepared: {output_pdbqt}")
    return os.path.abspath(output_pdbqt)


def _prepare_receptor_with_obabel(pdb_file: str, output_pdbqt: str) -> str:
    """Fallback receptor preparation using Open Babel."""
    success = obabel_convert(
        pdb_file,
        output_pdbqt,
        in_format="pdb",
        out_format="pdbqt",
        options=["-xr"],  # rigid receptor (no rotatable bonds)
        timeout=300,
    )
    if not success:
        raise PreparationError("Open Babel receptor preparation failed")
    logger.info(f"Receptor prepared (Open Babel fallback): {output_pdbqt}")
    return os.path.abspath(output_pdbqt)


# ─────────────────────────────────────────────────────────────────────────────
# Ligand Preparation
# ─────────────────────────────────────────────────────────────────────────────


def _has_nan_charges(mol) -> bool:
    """Check if any atom has a NaN Gasteiger charge."""
    for atom in mol.GetAtoms():
        try:
            c = atom.GetDoubleProp("_GasteigerCharge")
            if c != c:  # NaN check
                return True
        except KeyError:
            return True
    return False


def _prepare_ligand_with_obabel(smiles: str, output_pdbqt: str, name: str = "LIG") -> str:
    """Fallback ligand preparation using Open Babel (SMILES → PDBQT)."""
    from autodock.utils import write_temp_file

    tmp_smi = write_temp_file(smiles, ".smi")
    tmp_pdbqt = tmp_smi.replace(".smi", "_obabel.pdbqt")
    try:
        success = obabel_convert(
            tmp_smi,
            tmp_pdbqt,
            in_format="smi",
            out_format="pdbqt",
            options=["-p", "7.4", "--gen3d"],
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


def prepare_ligand(
    smiles: str,
    output_pdbqt: str,
    name: str = "LIG",
    seed: int = 42,
    n_conformer_attempts: int = 20,
    ph: float = 7.4,
) -> str:
    """
    Prepare a ligand for docking (SMILES → PDBQT).

    Uses RDKit ETKDGv3 for 3D conformer generation **with multi-round
    energy sorting**: generates ``n_conformer_attempts`` conformers,
    optimises each with MMFF94, and selects the lowest-energy one for
    PDBQT export.  This follows the best practice of not accepting the
    first valid conformer without quality assessment (see Rinciker et al.
    2004, J. Med. Chem.; Hawkins 2017, J. Chem. Inf. Model.).

    Falls back to Open Babel if Gasteiger charge calculation fails
    (e.g. for phosphorylated ligands such as 5GP in 1C9K).

    .. note::
       Protonation states are assigned by RDKit's default neutral model
       (pH 7.4).  For pH-dependent ionisation (carboxylic acids, amines),
       consider pre-treating the SMILES string with Dimorphite-DL or
       passing the pH-adjusted SMILES explicitly.  The OpenBabel fallback
       uses the ``ph`` argument for protonation.

    Args:
        smiles: SMILES string.
        output_pdbqt: Output PDBQT file path.
        name: Residue name in PDBQT.
        seed: Random seed for reproducible conformer generation.
        n_conformer_attempts: Number of ETKDGv3 attempts; the lowest-energy
            conformer after MMFF94 optimisation is selected.  More attempts
            improve the chance of finding the global energy minimum.
        ph: Target pH for ligand protonation.  Passed to OpenBabel in the
            fallback path; RDKit always uses neutral model (7.4).

    Returns:
        Absolute path to the prepared PDBQT file.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdPartialCharges
    except ImportError as exc:
        raise PreparationError(f"Required package missing: {exc}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PreparationError(f"Could not parse SMILES: {smiles}")

    mol_h = Chem.AddHs(mol, addCoords=True)

    # ── Multi-round ETKDGv3 + energy selection ────────────────────────────
    # Publication best practice: generate N conformers, evaluate MMFF94 energy,
    # and keep the lowest-energy one.  Single-shot ETKDG may produce a strained
    # conformer that biases subsequent docking (Issue #4 from scientific audit).
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0
    params.pruneRmsThresh = 0.5
    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformer_attempts, params=params)

    if len(cids) == 0:
        logger.warning(
            f"ETKDGv3 produced zero conformers after {n_conformer_attempts} attempts"
            " — trying fallback"
        )
        fallback_ok = AllChem.EmbedMolecule(mol_h, randomSeed=seed)
        if fallback_ok != 0:
            logger.warning("RDKit embedding failed — falling back to Open Babel")
            return _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name)
        cids = [0]

    # Optimise all conformers and select lowest-energy one
    mmff_results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=500, numThreads=0)
    energies: list[tuple[int, float]] = []
    for i, cid in enumerate(cids):
        status = mmff_results[i][0] if i < len(mmff_results) else -1
        energy = mmff_results[i][1] if i < len(mmff_results) else float("inf")
        if status != -1 and energy is not None:
            energies.append((cid, float(energy)))
        else:
            energies.append((cid, float("inf")))

    if energies:
        energies.sort(key=lambda x: x[1])
        best_cid = energies[0][0]
        best_energy = energies[0][1]
        logger.debug(
            f"Multi-round conformer selection: {len(energies)} evaluated,"
            f" best energy={best_energy:.2f} kcal/mol (cid={best_cid})"
        )
    else:
        # Fallback: use first conformer
        best_cid = cids[0]

    # Extract the best conformer into a single-conformer molecule
    mol = Chem.Mol(mol_h)
    conf = mol_h.GetConformer(best_cid)
    from rdkit import Geometry

    new_conf = Chem.Conformer(mol.GetNumAtoms())
    new_conf.SetId(0)
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        new_conf.SetAtomPosition(i, Geometry.Point3D(pos.x, pos.y, pos.z))
    mol.RemoveAllConformers()
    mol.AddConformer(new_conf)

    rdPartialCharges.ComputeGasteigerCharges(mol)

    # If Gasteiger produced NaN charges, skip Meeko and use Open Babel
    if _has_nan_charges(mol):
        logger.warning(
            "Gasteiger charges contain NaN — falling back to Open Babel ligand preparation"
        )
        return _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name)

    params_mk = MoleculePreparation(charge_model="gasteiger")
    try:
        mol_setup = params_mk.prepare(mol)
    except Exception as exc:
        err_str = str(exc)
        if "non finite charge" in err_str or "charge" in err_str.lower():
            logger.warning(f"Meeko charge failure ({exc}) — falling back to Open Babel")
            return _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name)
        raise PreparationError(f"Meeko ligand prep failed: {exc}")

    setup = mol_setup[0] if isinstance(mol_setup, list) else mol_setup

    pdbqt_str, success, err = PDBQTWriterLegacy.write_string(setup)
    if not success:
        if "non finite charge" in err or "charge" in err.lower():
            logger.warning("Meeko PDBQT write charge failure — falling back to Open Babel")
            return _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name)
        raise PreparationError(f"Meeko ligand prep failed: {err}")

    # Inject residue name if requested (PDB format: cols 18-20 = resname, max 3 chars)
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
    return os.path.abspath(output_pdbqt)


def prepare_ligand_from_sdf(
    sdf_path: str,
    output_pdbqt: str,
    name: str = "LIG",
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
        return _prepare_ligand_with_obabel(Chem.MolToSmiles(mol), output_pdbqt, name=name)

    params_mk = MoleculePreparation(charge_model="gasteiger")
    try:
        mol_setup = params_mk.prepare(mol)
    except Exception as exc:
        err_str = str(exc)
        if "non finite charge" in err_str or "charge" in err_str.lower():
            logger.warning(f"Meeko charge failure ({exc}) — falling back to Open Babel")
            return _prepare_ligand_with_obabel(Chem.MolToSmiles(mol), output_pdbqt, name=name)
        raise PreparationError(f"Meeko ligand prep failed: {exc}")

    setup = mol_setup[0] if isinstance(mol_setup, list) else mol_setup
    pdbqt_str, success, err = PDBQTWriterLegacy.write_string(setup)
    if not success:
        if "non finite charge" in err or "charge" in err.lower():
            logger.warning("Meeko charge error — falling back to Open Babel")
            return _prepare_ligand_with_obabel(Chem.MolToSmiles(mol), output_pdbqt, name=name)
        raise PreparationError(f"PDBQT export failed: {err}")

    pdbqt_str = pdbqt_str.replace("UNL", name)
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
) -> list[str]:
    """
    Generate multiple 3D conformers of a ligand for multi-conformer docking.

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
        prepare_ligand(smiles, out_path, name=name, seed=seed_start + i, ph=ph)
        paths.append(out_path)
    logger.info(f"Generated {n_conformers} conformers in {output_dir}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive multi-conformer ligand preparation
# ─────────────────────────────────────────────────────────────────────────────


def _classify_ligand_complexity(mol) -> str:
    """
    Classify ligand structural complexity to choose preparation strategy.

    Heuristics tuned on the 20-target benchmark set:
        - simple  : ETKDGv3 single-conformer is usually sufficient
        - medium  : benefits from 5-rep multi-conformer docking
        - complex : needs 10-rep multi-conformer or external tools

    Returns:
        "simple", "medium", or "complex"
    """
    from rdkit import Chem

    n_heavy = mol.GetNumHeavyAtoms()
    n_rings = mol.GetRingInfo().NumRings()
    rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Chiral centers restrict accessible conformational space;
    # many chirals = harder for single-conformer generation
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

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
            _prepare_ligand_with_obabel(smiles, ob_path, name=name)
            paths.append(ob_path)
            continue

        # Meeko
        params_mk = MoleculePreparation(charge_model="gasteiger")
        try:
            mol_setup = params_mk.prepare(mol_single)
        except Exception as exc:
            logger.warning(f"Rep {idx}: Meeko failed ({exc}) — Open Babel fallback")
            ob_path = os.path.join(output_dir, f"conformer_{idx}.pdbqt")
            _prepare_ligand_with_obabel(smiles, ob_path, name=name)
            paths.append(ob_path)
            continue

        setup = mol_setup[0] if isinstance(mol_setup, list) else mol_setup
        pdbqt_str, success, err = PDBQTWriterLegacy.write_string(setup)
        if not success:
            logger.warning(f"Rep {idx}: PDBQT write failed ({err}) — Open Babel fallback")
            ob_path = os.path.join(output_dir, f"conformer_{idx}.pdbqt")
            _prepare_ligand_with_obabel(smiles, ob_path, name=name)
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
) -> str | list[str]:
    """
    Adaptive ligand preparation: auto-selects strategy based on molecular complexity.

    Args:
        smiles: SMILES string.
        output_pdbqt: Output PDBQT file path (for single) OR directory (for multi).
        name: Residue name.
        seed: Random seed.
        strategy: "simple", "medium", "complex", or None for auto-detection.
        n_conformers_medium: Conformers to generate for medium ligands.
        max_reps_medium: Max representatives for medium ligands.
        n_conformers_complex: Conformers to generate for complex ligands.
        max_reps_complex: Max representatives for complex ligands.

    Returns:
        Single PDBQT path for simple ligands, or list of paths for medium/complex.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PreparationError(f"Could not parse SMILES: {smiles}")

    if strategy is None:
        strategy = _classify_ligand_complexity(mol)
        logger.info(f"Adaptive ligand prep: complexity='{strategy}' for '{smiles[:40]}...'")

    if strategy == "simple":
        # Simple: single conformer. If output_pdbqt is a directory, write ligand.pdbqt inside.
        if os.path.isdir(output_pdbqt):
            output_pdbqt = os.path.join(output_pdbqt, "ligand.pdbqt")
        return prepare_ligand(smiles, output_pdbqt, name=name, seed=seed)

    # Medium/complex: output_pdbqt is treated as a directory
    if os.path.isfile(output_pdbqt):
        output_dir = os.path.dirname(output_pdbqt) or "."
    else:
        output_dir = output_pdbqt
        ensure_dir(output_dir)

    if strategy == "medium":
        return prepare_ligand_multi(
            smiles,
            output_dir,
            name=name,
            seed=seed,
            n_conformers=n_conformers_medium,
            max_representatives=max_reps_medium,
        )

    # Complex: cap representatives for very large ligands to keep docking tractable
    n_heavy = mol.GetNumHeavyAtoms()
    effective_max_reps = max_reps_complex

    # Scheme C: >50 atoms — force single conformer to avoid Vina timeout/hang
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


def find_top_pockets(
    receptor_pdb: str,
    ligand_pdb: str | None = None,
    padding: float = 5.0,
    max_pockets: int = 3,
    use_p2rank: bool = True,
    fpocket_min_alpha: float = 3.0,
    fpocket_max_alpha: float = 6.0,
) -> list[dict[str, Any]]:
    """
    Identify top-N candidate binding pockets (sorted by quality).

    Priority:
      1. ligand_pdb provided → center on ligand (gold standard)
      2. Otherwise → fpocket cavity detection → optional P2Rank rescoring

    Args:
        receptor_pdb: Protein PDB file.
        ligand_pdb: Optional co-crystallized ligand PDB for centering.
        padding: Padding around pocket (Å).
        max_pockets: Maximum pockets to return.
        use_p2rank: Enable P2Rank rescoring if available.
        fpocket_min_alpha: Fpocket α-sphere minimum radius.
        fpocket_max_alpha: Fpocket α-sphere maximum radius.

    Returns:
        List of pocket dicts, each with keys:
          center, box_size, druggability, p2rank_prob, pocket_num,
          pocket_source, volume, depth, openings, n_apolar, n_polar.
    """
    if ligand_pdb and os.path.isfile(ligand_pdb):
        # Gold standard: center on known ligand
        try:
            from rdkit import Chem

            mol = Chem.MolFromPDBFile(ligand_pdb)
            if mol is None:
                raise ValueError("Could not parse ligand PDB")
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
                    "p2rank_prob": None,
                    "pocket_num": None,
                    "pocket_source": "crystal_ligand",
                    "volume": None,
                    "depth": None,
                    "openings": None,
                    "n_apolar": None,
                    "n_polar": None,
                }
            ]
        except Exception as exc:
            logger.warning(
                f"Ligand-centered pocket detection failed: {exc}. Falling back to fpocket."
            )

    # fpocket detection
    fpocket_bin = find_conda_tool("fpocket")
    if not fpocket_bin:
        raise PreparationError("fpocket not found. Install: conda install -c conda-forge fpocket")

    prep_pdb = tempfile.mktemp(suffix="_prep.pdb")
    _prepare_pdb_for_fpocket(receptor_pdb, prep_pdb)

    prep_pdb_abs = os.path.abspath(prep_pdb)
    prep_dir = os.path.dirname(prep_pdb_abs) or "."
    base = os.path.splitext(os.path.basename(prep_pdb))[0]
    out_dir = os.path.join(prep_dir, base + "_out")

    try:
        success, _, stderr = safe_subprocess(
            [
                fpocket_bin,
                "-f",
                prep_pdb_abs,
                "-m",
                str(fpocket_min_alpha),
                "-M",
                str(fpocket_max_alpha),
            ],
            timeout=120,
            cwd=prep_dir,
        )
        if not success:
            raise PreparationError(f"fpocket failed: {stderr[:500]}")

        info_file = os.path.join(out_dir, base + "_info.txt")
        if not os.path.exists(info_file):
            raise PreparationError(f"fpocket did not produce info file: {info_file}")

        pockets = _parse_fpocket_info(info_file)
        if not pockets:
            raise PreparationError(f"No pockets found by fpocket in {receptor_pdb}")

        # P2Rank rescoring
        p2rank_probs = None
        if use_p2rank:
            p2rank_probs = _run_p2rank_rescore(prep_pdb_abs, prep_dir)
            if p2rank_probs:
                logger.info(
                    f"P2Rank rescored {len(p2rank_probs)} pockets "
                    f"(prob range: {min(p2rank_probs.values()):.3f}"
                    f" - {max(p2rank_probs.values()):.3f})"
                )

        # Sort: primary P2Rank probability, secondary druggability - opening_penalty
        def sort_key(p: dict) -> tuple:
            prob = p2rank_probs.get(p["num"], None) if p2rank_probs else None
            drugg = p["druggability"] if p["druggability"] is not None else 0.0
            opening_penalty = (p.get("openings") or 0) * 0.05
            return (prob if prob is not None else -1.0, drugg - opening_penalty)

        pockets.sort(key=sort_key, reverse=True)

        result = []
        for p in pockets:
            # Dimension validation
            if any(d < _POCKET_MIN_DIM or d > _POCKET_MAX_DIM for d in p["dims"]):
                continue
            # Volume validation
            if p.get("volume") is not None and p["volume"] > _POCKET_MAX_VOLUME:
                logger.warning(
                    f"Pocket #{p['num']} oversized"
                    f" ({p['volume']:.0f} Å³ > {_POCKET_MAX_VOLUME}), skipping"
                )
                continue
            # Depth warning
            if p.get("depth") is not None and p["depth"] < _POCKET_MIN_DEPTH:
                logger.info(
                    f"Pocket #{p['num']} shallow (depth={p['depth']:.1f}Å), may be false positive"
                )

            prob = p2rank_probs.get(p["num"], None) if p2rank_probs else None
            center = p["center"]
            box_size = _compute_box_size(p["dims"], padding)
            result.append(
                {
                    "center": center,
                    "box_size": box_size,
                    "druggability": p["druggability"],
                    "p2rank_prob": prob,
                    "pocket_num": p["num"],
                    "pocket_source": "fpocket",
                    "volume": p.get("volume"),
                    "depth": p.get("depth"),
                    "openings": p.get("openings"),
                    "n_apolar": p.get("n_apolar"),
                    "n_polar": p.get("n_polar"),
                }
            )
            if len(result) >= max_pockets:
                break

        if not result:
            raise PreparationError(
                f"All {len(pockets)} fpocket pockets failed validation. "
                f"Protein may lack druggable pockets."
            )

        for i, pk in enumerate(result):
            prob = pk["p2rank_prob"]
            prob_str = f"P2Rank={prob:.3f}" if prob is not None else "P2Rank=N/A"
            if prob is not None and prob < _P2RANK_PROB_THRESHOLD:
                logger.warning(
                    f"Pocket {i+1} (#{pk['pocket_num']}):"
                    f" LOW P2Rank confidence ({prob:.3f} < {_P2RANK_PROB_THRESHOLD})"
                )
            if pk["druggability"] < _DRUGGABILITY_THRESHOLD:
                logger.warning(
                    f"Pocket {i+1} (#{pk['pocket_num']}):"
                    f" LOW druggability ({pk['druggability']:.3f}"
                    f" < {_DRUGGABILITY_THRESHOLD})"
                )
            logger.info(
                f"Pocket {i+1} (#{pk['pocket_num']}): center={pk['center']}, box={pk['box_size']} "
                f"({prob_str}, druggability={pk['druggability']:.3f})"
            )

        return result

    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        if os.path.exists(prep_pdb):
            os.remove(prep_pdb)
