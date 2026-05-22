"""
autodock.interactions — Protein-ligand interaction detection.
===========================================================
PLIP (primary) and ProLIF (secondary) for comprehensive interaction profiling.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

from autodock.core import logger, VisualizationError, _HAVE_PLIP, _HAVE_MDANALYSIS, _HAVE_PROLIF


# ─────────────────────────────────────────────────────────────────────────────
# PLIP-based interaction detection (primary / authoritative)
# ─────────────────────────────────────────────────────────────────────────────

INTERACTION_CATEGORIES = {
    # (plip_attr, sub_attr, display_name, color)
    ("hbonds_ldon", None, "H-bond", "cyan"),
    ("hbonds_pdon", None, "H-bond", "cyan"),
    ("hydrophobic_contacts", None, "Hydrophobic", "orange"),
    ("pistacking", None, "π-π", "green"),
    ("pication_laro", None, "π-cation", "purple"),
    ("pication_paro", None, "π-cation", "purple"),
    ("saltbridge_lneg", None, "Salt bridge", "red"),
    ("saltbridge_pneg", None, "Salt bridge", "red"),
    ("halogen_bonds", None, "Halogen bond", "yellow"),
    ("water_bridges", None, "Water bridge", "blue"),
    ("metal_complexes", None, "Metal complex", "grey"),
}


def _build_complex_pdb(receptor_pdb: str, ligand_pdbqt: str, output_pdb: str) -> str:
    """
    Merge receptor PDB and ligand PDBQT into a single PDB for PLIP analysis.
    PLIP requires the ligand as HETATM records with a distinct residue name.
    """
    with open(receptor_pdb, "r") as fh:
        rec_lines = fh.readlines()

    # Strip END/ENDMDL from receptor
    rec_lines = [l for l in rec_lines if not l.strip().startswith(("END", "ENDMDL"))]

    # Parse ligand from PDBQT and rewrite as properly formatted HETATM
    lig_lines = []
    atom_num = 1
    with open(ligand_pdbqt, "r") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atom_name = line[12:16].strip() if len(line) > 16 else "C"
            elem = line[76:78].strip() if len(line) > 78 else atom_name[0]
            new_line = (
                f"HETATM{atom_num:5d} {atom_name:>4s} LIG A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
            )
            lig_lines.append(new_line)
            atom_num += 1

    with open(output_pdb, "w") as fh:
        fh.writelines(rec_lines)
        fh.write("TER\n")
        fh.writelines(lig_lines)
        fh.write("END\n")

    return output_pdb


def detect_interactions_plip(
    receptor_pdb: str,
    ligand_pdbqt: str,
    output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Detect protein-ligand interactions using PLIP 3.x.

    This is the PRIMARY / AUTHORITATIVE interaction detector.
    If PLIP is unavailable, raises VisualizationError (no silent fallback).

    Args:
        receptor_pdb: Receptor PDB file.
        ligand_pdbqt: Docked ligand PDBQT file.
        output_dir: Optional directory for intermediate files.

    Returns:
        List of interaction dicts, each with keys:
          type, color, resn, resi, chain, atom, distance, description
    """
    if not _HAVE_PLIP:
        raise VisualizationError(
            "PLIP not available. Install: conda install -c conda-forge plip"
        )

    from plip.structure.preparation import PDBComplex

    tmp_dir = output_dir or tempfile.mkdtemp(prefix="plip_")
    os.makedirs(tmp_dir, exist_ok=True)

    complex_pdb = os.path.join(tmp_dir, "complex.pdb")
    _build_complex_pdb(receptor_pdb, ligand_pdbqt, complex_pdb)

    try:
        my_mol = PDBComplex()
        my_mol.load_pdb(complex_pdb)
        my_mol.analyze()
    except Exception as exc:
        raise VisualizationError(f"PLIP analysis failed: {exc}")

    interactions: list[dict[str, Any]] = []

    for ligand in my_mol.ligands:
        if ligand.hetid != "LIG":
            continue
        key = f"{ligand.hetid}:{ligand.chain}:{ligand.position}"
        if key not in my_mol.interaction_sets:
            continue

        interaction_set = my_mol.interaction_sets[key]

        for plip_attr, sub_attr, display_type, color in INTERACTION_CATEGORIES:
            records = getattr(interaction_set, plip_attr, [])
            if sub_attr:
                records = getattr(records, sub_attr, [])
            for rec in records:
                try:
                    resn = getattr(rec, "restype", "UNK")
                    resi = getattr(rec, "resnr", 0)
                    chain = getattr(rec, "reschain", "A")
                    atom = getattr(rec, "atype", "")
                    distance = getattr(rec, "distance", None)
                    if distance is None:
                        distance = getattr(rec, "dist", None)

                    desc = f"{display_type}: {resn}{resi}.{chain}"
                    if atom:
                        desc += f" ({atom})"
                    if distance is not None:
                        desc += f" — {distance:.2f} Å"

                    interactions.append({
                        "type": display_type,
                        "color": color,
                        "resn": resn,
                        "resi": int(resi),
                        "chain": chain,
                        "atom": atom,
                        "distance": round(float(distance), 2) if distance is not None else None,
                        "description": desc,
                    })
                except Exception as exc:
                    logger.debug(f"Skipping malformed PLIP record: {exc}")
                    continue

    logger.info(f"PLIP detected {len(interactions)} interactions")
    return interactions


# ─────────────────────────────────────────────────────────────────────────────
# ProLIF-based interaction detection (secondary)
# ─────────────────────────────────────────────────────────────────────────────

def detect_interactions_prolif(
    receptor_pdb: str,
    ligand_pdbqt: str,
) -> list[dict[str, Any]]:
    """
    Detect interactions using ProLIF as a secondary / cross-validation source.

    Args:
        receptor_pdb: Receptor PDB file.
        ligand_pdbqt: Ligand PDBQT file.

    Returns:
        List of interaction dicts.
    """
    if not _HAVE_MDANALYSIS or not _HAVE_PROLIF:
        raise VisualizationError(
            "ProLIF requires MDAnalysis + prolif. Install: conda install mdanalysis; pip install prolif"
        )

    import MDAnalysis as mda
    import prolif as plf

    u = mda.Universe(receptor_pdb)
    protein = u.select_atoms("protein")

    lig_u = mda.Universe(ligand_pdbqt)
    ligand = lig_u.atoms

    fp = plf.Fingerprint()
    fp.run(u.trajectory, ligand, protein)
    df = fp.to_dataframe()

    interactions = []
    if df.empty:
        return interactions

    for col in df.columns:
        interaction_type = col[0]
        prot_res = col[2] if len(col) > 2 else str(col[1])
        count = df[col].sum()
        if count > 0:
            color = "cyan"
            interactions.append({
                "type": interaction_type.replace("_", " ").title(),
                "color": color,
                "resn": prot_res.split(":")[0] if ":" in str(prot_res) else str(prot_res),
                "resi": 0,
                "chain": "A",
                "atom": "",
                "distance": None,
                "description": f"{interaction_type}: {prot_res}",
            })

    logger.info(f"ProLIF detected {len(interactions)} interaction types")
    return interactions


# ─────────────────────────────────────────────────────────────────────────────
# Unified interaction detection with cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def detect_interactions(
    receptor_pdb: str,
    ligand_pdbqt: str,
    method: str = "plip",
    output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Detect protein-ligand interactions.

    Args:
        receptor_pdb: Receptor PDB file.
        ligand_pdbqt: Ligand PDBQT file.
        method: 'plip' | 'prolif' | 'both'.
        output_dir: Optional working directory.

    Returns:
        List of interaction dicts.
    """
    if method in ("plip", "both"):
        try:
            plip_intx = detect_interactions_plip(receptor_pdb, ligand_pdbqt, output_dir)
            if method == "plip":
                return plip_intx
        except Exception as exc:
            if method == "plip":
                raise
            logger.warning(f"PLIP failed, falling back to ProLIF: {exc}")
            plip_intx = []

    if method in ("prolif", "both"):
        try:
            prolif_intx = detect_interactions_prolif(receptor_pdb, ligand_pdbqt)
            if method == "prolif":
                return prolif_intx
        except Exception as exc:
            if method == "prolif":
                raise
            logger.warning(f"ProLIF failed: {exc}")
            prolif_intx = []

    # Both: merge and deduplicate by (type, resn, resi)
    seen = set()
    merged = []
    for i in plip_intx + prolif_intx:
        key = (i.get("type"), i.get("resn"), i.get("resi"), i.get("chain"))
        if key not in seen:
            seen.add(key)
            merged.append(i)

    logger.info(f"Merged interactions (PLIP + ProLIF): {len(merged)} unique")
    return merged
