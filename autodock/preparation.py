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
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "2.0-6.0",
                            }
                        )
                    elif _type == "GLU" and (_pka < 2.0 or _pka > 7.0):
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "2.0-7.0",
                            }
                        )
                    elif _type == "HIS" and (_pka < 4.0 or _pka > 8.0):
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "4.0-8.0",
                            }
                        )
                    elif _type == "LYS" and (_pka < 8.0 or _pka > 12.0):
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "8.0-12.0",
                            }
                        )
                    elif _type == "CYS" and _pka > 11.0:
                        _anomalous_pka.append(
                            {
                                "residue": _label,
                                "type": _type,
                                "pKa": round(_pka, 1),
                                "expected_range": "<11.0",
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
            except Exception as exc:
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
        return _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name)

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
        except Exception:
            continue

    if best_mol is None:
        # All candidates had NaN charges — fall through to OBabel
        logger.warning("Gasteiger charges contain NaN — falling back to Open Babel")
        return _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name)

    if molscrub_states and len(candidate_mols) > 1:
        logger.info(f"Selected best state: {best_label} (MMFF={best_energy:.1f} kcal/mol)")

    # ── Meeko → PDBQT ─────────────────────────────────────────────────────
    params_mk = MoleculePreparation(charge_model="gasteiger")
    try:
        mol_setup = params_mk.prepare(best_mol)
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


def prepare_ligand_from_file(
    path: str,
    output_pdbqt: str,
    name: str = "LIG",
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
            raise PreparationError(f"Could not parse MOL2: {path}")

        smiles = Chem.MolToSmiles(mol)
        return prepare_ligand(
            smiles,
            output_pdbqt,
            name=name,
            molscrub_states=False,
            enumerate_stereo=False,
        )

    # SDF / MOL — use prepare_ligand_from_sdf
    return prepare_ligand_from_sdf(path, output_pdbqt, name=name)


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
    ph_range: float | None = 1.5,
    molscrub_states: bool = True,
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
                        "druggability": None,
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
    except Exception:
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
        result["overlapping_low_conf_regions"] = overlapping
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


def _compute_p2rank_druggability(
    predictors_csv: str,
) -> dict[int, float | None]:
    """Parse P2Rank druggability scores from residue-level CSV (*_residues.csv).

    P2Rank's residue-level output includes a ``druggability_score`` column.
    Returns {pocket_num: druggability_score} or empty dict if unavailable.

    Reference:
        Krivák & Hoksza (2018) JCIM — P2Rank druggability scoring.
    """
    result: dict[int, float | None] = {}
    if not os.path.exists(predictors_csv):
        return result

    # Try residues CSV first
    residues_csv = predictors_csv.replace("_predictions.csv", "_residues.csv")
    if not os.path.exists(residues_csv):
        return result

    with open(residues_csv) as fh:
        header = [h.strip() for h in fh.readline().strip().split(",")]
        try:
            pocket_idx = header.index("pocket")
            drugg_idx = header.index("druggability_score")
        except ValueError:
            return result

        pocket_scores: dict[int, list[float]] = {}
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) <= max(pocket_idx, drugg_idx):
                continue
            try:
                pnum = int(parts[pocket_idx])
                score = float(parts[drugg_idx])
            except (ValueError, IndexError):
                continue
            pocket_scores.setdefault(pnum, []).append(score)

    for pnum, scores in pocket_scores.items():
        result[pnum] = round(float(np.mean(scores)), 4)
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

    prep_pdb = tempfile.mktemp(suffix="_fpocket_prep.pdb")
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

    except Exception as exc:
        logger.warning(f"fpocket detection failed: {exc}")
        return None
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        with contextlib.suppress(OSError):
            if os.path.exists(prep_pdb):
                os.remove(prep_pdb)


def _ensemble_consensus(
    p2rank_pockets: list[dict[str, Any]] | None,
    fpocket_pockets: list[dict[str, Any]] | None,
    distance_threshold: float = _POCKET_CONSENSUS_DISTANCE,
) -> list[dict[str, Any]]:
    """Merge P2Rank (ML) and fpocket (geometric) pocket sets into consensus.

    For each P2Rank pocket, find the nearest fpocket pocket within
    ``distance_threshold`` Å.  Pockets with a match are marked as
    ``consensus=True``; unmatched P2Rank pockets retain their ML ranking;
    unmatched fpocket pockets are appended with lower priority.

    Reference:
        Ensemble docking best practice — Houston & Walkinshaw (2013)
        J. Chem. Inf. Model. 53:384-390.

    Returns:
        List of merged pocket dicts, sorted by consensus confidence then
        P2Rank probability.
    """
    if not p2rank_pockets and not fpocket_pockets:
        return []
    if not p2rank_pockets:
        for p in fpocket_pockets:
            p["consensus"] = False
            p["p2rank_prob"] = None
        return fpocket_pockets
    if not fpocket_pockets:
        for p in p2rank_pockets:
            p["consensus"] = False
        return p2rank_pockets

    merged = []
    used_fpocket_indices: set[int] = set()

    for p2p in p2rank_pockets:
        p2_center = np.array(p2p["center"])
        best_dist = float("inf")
        best_fp_idx = None

        for fi, fp in enumerate(fpocket_pockets):
            if fi in used_fpocket_indices:
                continue
            fp_center = np.array(fp["center"])
            dist = float(np.linalg.norm(p2_center - fp_center))
            if dist < best_dist:
                best_dist = dist
                best_fp_idx = fi

        is_consensus = best_fp_idx is not None and best_dist <= distance_threshold
        if is_consensus:
            used_fpocket_indices.add(best_fp_idx)

        p2p["consensus"] = is_consensus
        p2p["consensus_distance"] = round(best_dist, 2) if best_fp_idx is not None else None
        p2p["fpocket_match"] = best_fp_idx
        merged.append(p2p)

    # Append unmatched fpocket pockets
    for fi, fp in enumerate(fpocket_pockets):
        if fi not in used_fpocket_indices:
            fp["consensus"] = False
            fp["consensus_distance"] = None
            fp["p2rank_prob"] = None
            fp["fpocket_match"] = None
            fp["pocket_source"] = "fpocket_only"
            merged.append(fp)

    # Sort: consensus first (by P2Rank prob), then non-consensus (by prob)
    merged.sort(
        key=lambda p: (
            not p.get("consensus", False),
            -(p.get("p2rank_prob") or 0.0),
        )
    )

    return merged


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
    except Exception as exc:
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
        except Exception as exc:
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
    except Exception as exc:
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
    except Exception as exc:
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
) -> list[dict[str, Any]]:
    """
    Identify top-N candidate binding pockets with publication-grade analysis.

    Pipeline:
      1. **P2Rank** — ML-based random forest classifier as primary screen
      2. **fpocket** — geometric cavity detection (α-sphere) for cross-validation
      3. **Cross-validation** — spatial overlap check: each P2Rank candidate
         is verified against fpocket pockets (threshold: 8 Å center distance)
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
        except Exception as exc:
            logger.warning(
                f"Ligand-centered pocket detection failed: {exc}. "
                "Falling back to computational detection."
            )

    # ── Preparation ─────────────────────────────────────────────────────────
    prep_pdb = tempfile.mktemp(suffix="_prep.pdb")
    _prepare_pdb_for_fpocket(receptor_pdb, prep_pdb)
    prep_pdb_abs = os.path.abspath(prep_pdb)
    prep_dir = os.path.dirname(prep_pdb_abs) or "."
    base = os.path.splitext(os.path.basename(prep_pdb))[0]
    fpocket_out_dir = os.path.join(prep_dir, f"{base}_out")

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
        if prob is not None and prob < _P2RANK_PROB_THRESHOLD:
            logger.info(
                f"Skipping pocket #{p2p.get('num', 0)}: "
                f"P2Rank prob={prob:.3f} < {_P2RANK_PROB_THRESHOLD}"
            )
            continue

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
        except Exception:
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
            "pocket_source": "fpocket" if verified else "p2rank",
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
        p_str = f"P2Rank={pk['p2rank_prob']:.3f}" if pk["p2rank_prob"] is not None else "P2Rank=N/A"
        logger.info(
            f"Pocket {i + 1} (#{pk['pocket_num']}): [{src_label}]{v_str}, "
            f"center={pk['center']}, box={pk['box_size']} ({p_str}, "
            f"druggability={pk['druggability_level']})"
        )

    # Cleanup temp files
    for d in (fpocket_out_dir,):
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
    for f in (prep_pdb,):
        if os.path.exists(f):
            os.unlink(f)
    for d in (os.path.join(prep_dir, "p2rank_predict"), os.path.join(prep_dir, "p2rank_out")):
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

    return result
