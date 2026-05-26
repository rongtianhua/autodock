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
from autodock.utils import ensure_dir, obabel_convert

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
) -> str:
    """
    Prepare a protein structure for docking (PDB/mmCIF → PDBQT).

    Uses Meeko Polymer + PDBQTWriterLegacy for accurate atom typing and
    Gasteiger charge assignment.  Falls back to Open Babel if Meeko fails.

    Args:
        pdb_file: Input structure path (.pdb, .cif, .pdbx).
        output_pdbqt: Output PDBQT file path.
        remove_water: Remove HOH / WAT residues.
        remove_hetatms: Remove all HETATM records (keep only protein).
        input_format: 'auto' | 'pdb' | 'cif' | 'pdbx'.
        keep_residues: If provided, keep only these residue names.
        force: If False and output_pdbqt already exists, skip preparation.

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

    # Step 1: Convert mmCIF → PDB if needed
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

    # Step 2: Filter waters / hetatms
    if remove_water or remove_hetatms or keep_residues:
        tmp_pdb = tempfile.mktemp(suffix="_filtered.pdb")
        # Write filtered content to temp file
        lines = pdb_content.splitlines(keepends=True)
        filtered = []
        for line in lines:
            if line.startswith("ATOM  "):
                resn = line[17:20].strip()
                if keep_residues and resn not in keep_residues:
                    continue
                if remove_water and resn in _SKIP_WATER:
                    continue
                filtered.append(line)
            elif line.startswith("HETATM"):
                if remove_hetatms:
                    continue
                resn = line[17:20].strip()
                if keep_residues and resn not in keep_residues:
                    continue
                if remove_water and resn in _SKIP_WATER:
                    continue
                filtered.append(line)
            else:
                filtered.append(line)
        with open(tmp_pdb, "w") as fh:
            fh.writelines(filtered)
        pdb_content = "".join(filtered)
        with contextlib.suppress(Exception):
            os.remove(tmp_pdb)

    # Step 3: Remove known problematic additives that crash Meeko
    lines = pdb_content.splitlines()
    filtered = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            resn = line[17:20].strip()
            if resn in _SKIP_ADDITIVES:
                continue
        filtered.append(line)
    pdb_content = "\n".join(filtered)

    # Step 4: Meeko preparation
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
) -> str:
    """
    Prepare a ligand for docking (SMILES → PDBQT).

    Uses RDKit ETKDGv3 for 3D conformer + Meeko for PDBQT export.
    Falls back to Open Babel if Gasteiger charge calculation fails
    (e.g. for phosphorylated ligands such as 5GP in 1C9K).

    Args:
        smiles: SMILES string.
        output_pdbqt: Output PDBQT file path.
        name: Residue name in PDBQT.
        seed: Random seed for reproducible conformer generation.

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

    mol = Chem.AddHs(mol, addCoords=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    embed_ok = AllChem.EmbedMolecule(mol, params)
    if embed_ok != 0:
        logger.warning(f"ETKDGv3 embedding returned {embed_ok} — trying fallback")
        embed_ok2 = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if embed_ok2 != 0:
            logger.warning("RDKit embedding failed — falling back to Open Babel")
            return _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name)

    # Optimize geometry: try MMFF first, then UFF for exotic elements (P, S+4, etc.)
    mmff_ok = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    if mmff_ok == -1:
        logger.debug("MMFF unsupported for this molecule — trying UFF")
        uff_ok = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        if uff_ok == -1:
            logger.warning("RDKit force-field optimization failed — falling back to Open Babel")
            return _prepare_ligand_with_obabel(smiles, output_pdbqt, name=name)
    elif mmff_ok == 1:
        logger.debug("MMFF did not fully converge (status=1) — accepting partial optimization")

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


def prepare_ligand_conformers(
    smiles: str,
    output_dir: str,
    n_conformers: int = 10,
    name: str = "LIG",
    seed_start: int = 42,
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
        prepare_ligand(smiles, out_path, name=name, seed=seed_start + i)
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
        representatives = sorted_cids[:min(len(cids), 10)]

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
            f"Very large ligand ({n_heavy} heavy atoms) — forcing single conformer to avoid Vina hang"
        )
        single_path = os.path.join(output_dir, "ligand.pdbqt")
        return prepare_ligand(smiles, single_path, name=name, seed=seed)

    if n_heavy > 45:
        effective_max_reps = min(effective_max_reps, 2)
        logger.info(f"Large ligand ({n_heavy} heavy atoms) — capping representatives to {effective_max_reps}")

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
    """Strip waters and keep only ATOM/HETATM for fpocket."""
    with open(pdb_in) as fin, open(pdb_out, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM  ", "HETATM")):
                resn = line[17:20].strip()
                if resn not in _SKIP_WATER:
                    fout.write(line)


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
                    f"(prob range: {min(p2rank_probs.values()):.3f} - {max(p2rank_probs.values()):.3f})"
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
                    f"Pocket #{p['num']} oversized ({p['volume']:.0f} Å³ > {_POCKET_MAX_VOLUME}), skipping"
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
                    f"Pocket {i+1} (#{pk['pocket_num']}): LOW P2Rank confidence ({prob:.3f} < {_P2RANK_PROB_THRESHOLD})"
                )
            if pk["druggability"] < _DRUGGABILITY_THRESHOLD:
                logger.warning(
                    f"Pocket {i+1} (#{pk['pocket_num']}): LOW druggability ({pk['druggability']:.3f} < {_DRUGGABILITY_THRESHOLD})"
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
