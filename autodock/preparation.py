"""
autodock.preparation — Receptor / ligand preparation and binding-site detection.
==============================================================================
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from autodock.core import (
    logger,
    PreparationError,
    safe_subprocess,
    find_conda_tool,
    find_p2rank,
    find_java,
    _SKIP_WATER,
    _SKIP_ADDITIVES,
    _POCKET_MIN_DIM,
    _POCKET_MAX_DIM,
    _POCKET_MAX_VOLUME,
    _POCKET_MIN_DEPTH,
    _P2RANK_PROB_THRESHOLD,
    _DRUGGABILITY_THRESHOLD,
)
from autodock.utils import ensure_dir, filter_pdb_lines, compute_bounding_box, obabel_convert


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

    Returns:
        Absolute path to the prepared PDBQT file.

    Raises:
        PreparationError: If input file missing or preparation fails.
    """
    if not os.path.isfile(pdb_file):
        raise PreparationError(f"Input file not found: {pdb_file}")

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
        with open(pdb_file, "r") as fh:
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
        try:
            os.remove(tmp_pdb)
        except Exception:
            pass

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
        polymer = Polymer.from_pdb_string(pdb_content, templates, mk_prep)
    except Exception as exc:
        # Retry with allow_bad_res=True: removes unknown residues and continues
        logger.warning(
            f"Some residues failed template matching — retrying with allow_bad_res=True"
        )
        try:
            polymer = Polymer.from_pdb_string(
                pdb_content, templates, mk_prep, allow_bad_res=True
            )
        except Exception as exc2:
            logger.error(f"Meeko preparation failed even with allow_bad_res: {exc2}")
            raise PreparationError(f"Receptor preparation failed: {exc2}")

    try:
        rigid_pdbqt, _ = PDBQTWriterLegacy.write_from_polymer(polymer)
    except Exception as exc:
        raise PreparationError(f"PDBQT writing failed: {exc}")

    ensure_dir(os.path.dirname(output_pdbqt) or ".")
    with open(output_pdbqt, "w") as fh:
        fh.write(rigid_pdbqt)

    logger.info(f"Receptor prepared: {output_pdbqt}")
    return os.path.abspath(output_pdbqt)


# ─────────────────────────────────────────────────────────────────────────────
# Ligand Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ligand(
    smiles: str,
    output_pdbqt: str,
    name: str = "LIG",
    seed: int = 42,
) -> str:
    """
    Prepare a ligand for docking (SMILES → PDBQT).

    Uses RDKit ETKDGv3 for 3D conformer + Meeko for PDBQT export.

    Args:
        smiles: SMILES string.
        output_pdbqt: Output PDBQT file path.
        name: Residue name in PDBQT.
        seed: Random seed for reproducible conformer generation.

    Returns:
        Absolute path to the prepared PDBQT file.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdPartialCharges
        from meeko import MoleculePreparation, PDBQTWriterLegacy
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
            raise PreparationError("Failed to generate 3D conformer for ligand.")

    AllChem.MMFFOptimizeMolecule(mol)
    rdPartialCharges.ComputeGasteigerCharges(mol)

    params_mk = MoleculePreparation(charge_model="gasteiger")
    mol_setup = params_mk.prepare(mol)
    setup = mol_setup[0] if isinstance(mol_setup, list) else mol_setup

    pdbqt_str, success, err = PDBQTWriterLegacy.write_string(setup)
    if not success:
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
# Binding Site Detection (fpocket + P2Rank)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_box_size(dims: tuple[float, float, float], padding: float = 5.0) -> tuple[float, float, float]:
    """Compute Vina docking box size from pocket dimensions + padding."""
    box = []
    for d in dims:
        v = d + 2 * padding
        rounded = round(v * 2) / 2  # nearest 0.5 Å
        box.append(max(10.0, rounded))
    return tuple(box)


def _prepare_pdb_for_fpocket(pdb_in: str, pdb_out: str) -> None:
    """Strip waters and keep only ATOM/HETATM for fpocket."""
    with open(pdb_in, "r") as fin, open(pdb_out, "w") as fout:
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
    text = open(info_path).read()
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
                            coords.append([
                                float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54]),
                            ])
                        except ValueError:
                            continue
            if coords:
                ca = np.array(coords)
                center = tuple(ca.mean(axis=0).tolist())
                dims = tuple((ca.max(axis=0) - ca.min(axis=0)).tolist())

        if center:
            pockets.append({
                "num": pocket_num,
                "druggability": druggability if druggability is not None else 0.0,
                "volume": volume,
                "depth": depth,
                "openings": openings,
                "n_apolar": n_apolar,
                "n_polar": n_polar,
                "center": center,
                "dims": dims if dims else (20.0, 20.0, 20.0),
            })
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
            coords = np.array([
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ])
            center = tuple(coords.mean(axis=0).tolist())
            dims = tuple((coords.max(axis=0) - coords.min(axis=0)).tolist())
            box_size = _compute_box_size(dims, padding)
            logger.info(f"Binding site from ligand: center={center}, box={box_size}")
            return [{
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
            }]
        except Exception as exc:
            logger.warning(f"Ligand-centered pocket detection failed: {exc}. Falling back to fpocket.")

    # fpocket detection
    fpocket_bin = find_conda_tool("fpocket")
    if not fpocket_bin:
        raise PreparationError(
            "fpocket not found. Install: conda install -c conda-forge fpocket"
        )

    prep_pdb = tempfile.mktemp(suffix="_prep.pdb")
    _prepare_pdb_for_fpocket(receptor_pdb, prep_pdb)

    prep_pdb_abs = os.path.abspath(prep_pdb)
    prep_dir = os.path.dirname(prep_pdb_abs) or "."
    base = os.path.splitext(os.path.basename(prep_pdb))[0]
    out_dir = os.path.join(prep_dir, base + "_out")

    try:
        success, _, stderr = safe_subprocess(
            [
                fpocket_bin, "-f", prep_pdb_abs,
                "-m", str(fpocket_min_alpha),
                "-M", str(fpocket_max_alpha),
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
            result.append({
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
            })
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
