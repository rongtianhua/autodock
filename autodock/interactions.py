"""
autodock.interactions — Protein-ligand interaction detection.
===========================================================
PLIP (primary) and ProLIF (secondary) for comprehensive interaction profiling.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from typing import Any

from autodock.core import (
    _HAVE_PLIP,
    _HAVE_PROLIF,
    _HAVE_RDKIT,
    VisualizationError,
    logger,
)
from autodock.utils import _AD4_ELEMENT_MAP, safe_pdb_slice

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


def _extract_ligand_atoms_plip(rec: Any, plip_attr: str) -> list[dict[str, Any]]:
    """Extract ligand atom coordinates from a PLIP interaction record."""
    atoms: list[dict[str, Any]] = []
    if plip_attr in ("hbonds_ldon",):
        # Ligand is H-bond donor
        atoms = [{"coords": rec.d.coords}]
    elif plip_attr in ("hbonds_pdon",):
        # Ligand is H-bond acceptor
        atoms = [{"coords": rec.a.coords}]
    elif plip_attr in ("hydrophobic_contacts",):
        atoms = [{"coords": rec.ligatom.coords}]
    elif plip_attr in ("pistacking",):
        if hasattr(rec, "ligandring") and rec.ligandring:
            atoms = [{"coords": a.coords} for a in rec.ligandring.atoms]
    elif plip_attr in ("pication_laro",):
        # Ligand is aromatic ring (protein has cation)
        if hasattr(rec, "ring") and rec.ring:
            atoms = [{"coords": a.coords} for a in rec.ring.atoms]
    elif plip_attr in ("pication_paro",):
        # Ligand is cation (protein has aromatic ring)
        if hasattr(rec, "charge") and rec.charge:
            atoms = [{"coords": a.coords} for a in rec.charge.atoms]
    elif plip_attr in ("saltbridge_lneg",):
        # Ligand is negatively charged
        if hasattr(rec, "negative") and rec.negative:
            atoms = [{"coords": a.coords} for a in rec.negative.atoms]
    elif plip_attr in ("saltbridge_pneg",):
        # Ligand is positively charged
        if hasattr(rec, "positive") and rec.positive:
            atoms = [{"coords": a.coords} for a in rec.positive.atoms]
    elif plip_attr in ("halogen_bonds",):
        if hasattr(rec, "don") and rec.don:
            atoms = [{"coords": rec.don.x.coords}]
    elif plip_attr in ("water_bridges",):
        atoms = [{"coords": rec.a.coords}] if rec.protisdon else [{"coords": rec.d.coords}]
    elif plip_attr in ("metal_complexes",) and hasattr(rec, "target") and rec.target:
        atoms = [{"coords": rec.target.atom.coords}]
    return atoms


def _build_complex_pdb(receptor_pdb: str, ligand_pdbqt: str, output_pdb: str) -> str:
    """
    Merge receptor PDB and ligand PDBQT into a single PDB for PLIP analysis.
    PLIP requires the ligand as HETATM records with a distinct residue name.
    """
    with open(receptor_pdb) as fh:
        rec_lines = fh.readlines()

    # Strip END/ENDMDL from receptor
    rec_lines = [line for line in rec_lines if not line.strip().startswith(("END", "ENDMDL"))]

    # Parse ligand from PDBQT and rewrite as properly formatted HETATM
    lig_lines = []
    atom_num = 1
    with open(ligand_pdbqt) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atom_name = safe_pdb_slice(line, 12, 16, default="C")
            # Use sanitized element: read last token (most robust across generators)
            stripped_tail = line[71:].strip() if len(line) > 71 else ""
            ad_type = stripped_tail.split()[-1] if stripped_tail else ""
            elem = _AD4_ELEMENT_MAP.get(ad_type, ad_type)
            if not elem:
                elem = atom_name[0] if atom_name else "C"
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
        raise VisualizationError("PLIP not available. Install: conda install -c conda-forge plip")

    from plip.structure.preparation import PDBComplex

    # CIF→PDB conversion must happen upstream in prepare_receptor(output_pdb=...).
    # This function accepts PDB format only.  Passing a .cif file will fail.
    # See preparation.py:prepare_receptor() for the canonical conversion path.
    _own_tmp = output_dir is None
    tmp_dir = output_dir or tempfile.mkdtemp(prefix="plip_")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        complex_pdb = os.path.join(tmp_dir, "complex.pdb")
        _build_complex_pdb(receptor_pdb, ligand_pdbqt, complex_pdb)

        try:
            my_mol = PDBComplex()
            my_mol.load_pdb(complex_pdb)
            my_mol.analyze()
        except (RuntimeError, OSError, ValueError, TypeError, ImportError) as exc:
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

                        ligand_atoms = _extract_ligand_atoms_plip(rec, plip_attr)
                        interactions.append(
                            {
                                "type": display_type,
                                "color": color,
                                "resn": resn,
                                "resi": int(resi),
                                "chain": chain,
                                "atom": atom,
                                "distance": (
                                    round(float(distance), 2) if distance is not None else None
                                ),
                                "description": desc,
                                "ligand_atoms": ligand_atoms,
                            }
                        )
                    except (ValueError, TypeError, IndexError, KeyError, AttributeError) as exc:
                        logger.debug(f"Skipping malformed PLIP record: {exc}")
                        continue

        logger.info(f"PLIP detected {len(interactions)} interactions")
        return interactions
    finally:
        if _own_tmp:
            with contextlib.suppress(OSError):
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# ProLIF-based interaction detection (secondary)
# ─────────────────────────────────────────────────────────────────────────────

_PROLIF_COLOR_MAP = {
    "Hydrophobic": "orange",
    "HBDonor": "cyan",
    "HBAcceptor": "cyan",
    "PiStacking": "green",
    "FaceToFace": "green",
    "EdgeToFace": "green",
    "PiCation": "purple",
    "CationPi": "purple",
    "Cationic": "red",
    "Anionic": "red",
    "VdWContact": "grey",
    "MetalDonor": "grey",
    "MetalAcceptor": "grey",
    "XBDonor": "yellow",
    "XBAcceptor": "yellow",
    "WaterBridge": "blue",
}

_PROLIF_DISPLAY_NAME = {
    "Hydrophobic": "Hydrophobic",
    "HBDonor": "H-bond",
    "HBAcceptor": "H-bond",
    "PiStacking": "π-π",
    "FaceToFace": "π-π",
    "EdgeToFace": "π-π",
    "PiCation": "π-cation",
    "CationPi": "π-cation",
    "Cationic": "Salt bridge",
    "Anionic": "Salt bridge",
    "VdWContact": "van der Waals",
    "MetalDonor": "Metal complex",
    "MetalAcceptor": "Metal complex",
    "XBDonor": "Halogen bond",
    "XBAcceptor": "Halogen bond",
    "WaterBridge": "Water bridge",
}


def _parse_smiles_from_pdbqt(ligand_pdbqt: str) -> str | None:
    """Extract REMARK SMILES from AutoDock Vina PDBQT output."""
    with open(ligand_pdbqt) as fh:
        for line in fh:
            if line.startswith("REMARK SMILES ") and not line.startswith("REMARK SMILES IDX"):
                return line.split(None, 2)[2].strip()
    return None


def _build_ligand_mol_for_prolif(ligand_pdbqt: str):
    """
    Build an RDKit molecule with explicit hydrogens and docking pose coordinates.

    Strategy:
      1. Try REMARK SMILES from PDBQT → full molecule with correct H count.
      2. MCS-align SMILES molecule to PDBQT coordinates.
      3. Fallback: read PDBQT directly (incomplete H, but functional).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFMCS

    smiles = _parse_smiles_from_pdbqt(ligand_pdbqt)

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not parse SMILES from PDBQT: {smiles!r}")
        else:
            mol = Chem.AddHs(mol)
            # Load PDBQT into RDKit for coordinate transfer
            mol_pdbqt = Chem.MolFromPDBFile(ligand_pdbqt, sanitize=False)
            if mol_pdbqt and mol_pdbqt.GetNumAtoms() > 0:
                try:
                    mcs = rdFMCS.FindMCS(
                        [mol, mol_pdbqt],
                        atomCompare=rdFMCS.AtomCompare.CompareAny,
                        bondCompare=rdFMCS.BondCompare.CompareAny,
                    )
                    if mcs.numAtoms > 0:
                        patt = Chem.MolFromSmarts(mcs.smartsString)
                        match_lig = mol.GetSubstructMatch(patt)
                        match_pdbqt = mol_pdbqt.GetSubstructMatch(patt)
                        if len(match_lig) == len(match_pdbqt) and len(match_lig) > 0:
                            coord_map = {}
                            conf_pdbqt = mol_pdbqt.GetConformer()
                            for i, j in zip(match_lig, match_pdbqt, strict=False):
                                coord_map[i] = conf_pdbqt.GetAtomPosition(j)
                            mol.RemoveAllConformers()
                            ret = AllChem.EmbedMolecule(mol, coordMap=coord_map)
                            if ret == 0:
                                return mol
                            # If coordMap fails (e.g. planar rings with missing H coords),
                            # fall back to full ETKDG and warn
                            mol.RemoveAllConformers()
                            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                            logger.warning(
                                "ProLIF ligand: coordMap ETKDG failed, using generated coordinates"
                            )
                            return mol
                except Exception as exc:
                    logger.debug(f"MCS coordinate transfer failed: {exc}")
            # If MCS fails, generate coordinates from scratch (geometry is lost)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            logger.warning("ProLIF ligand: using SMILES-generated coordinates (not docking pose)")
            return mol

    # Fallback: read PDBQT directly
    mol = Chem.MolFromPDBFile(ligand_pdbqt, sanitize=False)
    if mol is None:
        raise VisualizationError(f"Could not read ligand PDBQT: {ligand_pdbqt}")
    mol = Chem.AddHs(mol, addCoords=True)
    logger.warning("ProLIF ligand: using PDBQT direct read (hydrogens may be incomplete)")
    return mol


def detect_interactions_prolif(
    receptor_pdb: str,
    ligand_pdbqt: str,
) -> list[dict[str, Any]]:
    """
    Detect interactions using ProLIF v2.1.0 as a secondary / cross-validation source.

    Uses RDKit to add explicit hydrogens to both receptor and ligand, then
    runs ProLIF's Fingerprint.generate() for static structure analysis.

    Args:
        receptor_pdb: Receptor PDB file.
        ligand_pdbqt: Ligand PDBQT file (may contain REMARK SMILES).

    Returns:
        List of interaction dicts compatible with PLIP format.
    """
    if not _HAVE_RDKIT or not _HAVE_PROLIF:
        raise VisualizationError(
            "ProLIF requires rdkit + prolif." " Install: conda install rdkit; pip install prolif"
        )

    import prolif as plf
    from rdkit import Chem

    # ── Receptor ──────────────────────────────────────────────────────────────
    rec_mol = Chem.MolFromPDBFile(receptor_pdb, sanitize=False)
    if rec_mol is None:
        raise VisualizationError(f"Could not read receptor PDB: {receptor_pdb}")
    rec_h = Chem.AddHs(rec_mol, addCoords=True)
    prot_mol = plf.Molecule.from_rdkit(rec_h)

    # ── Ligand ────────────────────────────────────────────────────────────────
    lig_mol_rdkit = _build_ligand_mol_for_prolif(ligand_pdbqt)
    lig_mol = plf.Molecule.from_rdkit(lig_mol_rdkit)

    # ── Run ProLIF ────────────────────────────────────────────────────────────
    fp = plf.Fingerprint()
    ifp = fp.generate(lig_mol, prot_mol, metadata=True)

    interactions: list[dict[str, Any]] = []
    if not ifp:
        logger.info("ProLIF detected 0 interactions")
        return interactions

    ligand_conf = lig_mol_rdkit.GetConformer()

    for (_lig_resid, prot_resid), interaction_data in ifp.items():
        for interaction_name, metadatas in interaction_data.items():
            display_name = _PROLIF_DISPLAY_NAME.get(interaction_name, interaction_name)
            color = _PROLIF_COLOR_MAP.get(interaction_name, "grey")

            for metadata in metadatas:
                distance = metadata.get("distance")
                ligand_indices = metadata.get("parent_indices", {}).get("ligand", ())

                ligand_atoms = []
                for idx in ligand_indices:
                    if 0 <= idx < lig_mol_rdkit.GetNumAtoms():
                        pos = ligand_conf.GetAtomPosition(int(idx))
                        ligand_atoms.append({"coords": (pos.x, pos.y, pos.z)})

                resn = (
                    str(prot_resid.name)
                    if hasattr(prot_resid, "name")
                    else str(prot_resid).split(":")[0]
                )
                resi = int(prot_resid.number) if hasattr(prot_resid, "number") else 0
                chain = str(prot_resid.chain) if prot_resid.chain else "A"

                desc = f"{display_name}: {resn}{resi}.{chain}"
                if distance is not None:
                    desc += f" — {distance:.2f} Å"

                interactions.append(
                    {
                        "type": display_name,
                        "color": color,
                        "resn": resn,
                        "resi": resi,
                        "chain": chain,
                        "atom": "",
                        "distance": round(float(distance), 2) if distance is not None else None,
                        "description": desc,
                        "ligand_atoms": ligand_atoms,
                    }
                )

    logger.info(f"ProLIF detected {len(interactions)} interactions")
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
    plip_intx: list[dict[str, Any]] = []
    prolif_intx: list[dict[str, Any]] = []

    if method not in ("plip", "prolif", "both"):
        raise ValueError(
            f"Invalid interaction method: {method}. Choose 'plip', 'prolif', or 'both'."
        )

    if method in ("plip", "both"):
        try:
            plip_intx = detect_interactions_plip(receptor_pdb, ligand_pdbqt, output_dir)
            if method == "plip":
                return plip_intx
        except (
            RuntimeError,
            OSError,
            ValueError,
            TypeError,
            ImportError,
            VisualizationError,
        ) as exc:
            if method == "plip":
                raise
            logger.warning(f"PLIP failed, falling back to ProLIF: {exc}")
            plip_intx = []

    if method in ("prolif", "both"):
        try:
            prolif_intx = detect_interactions_prolif(receptor_pdb, ligand_pdbqt)
            if method == "prolif":
                return prolif_intx
        except (
            RuntimeError,
            OSError,
            ValueError,
            TypeError,
            ImportError,
            VisualizationError,
        ) as exc:
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
