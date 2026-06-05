"""
autodock.interactions — Protein-ligand interaction detection.
===========================================================
PLIP (primary) and ProLIF (secondary) for comprehensive interaction profiling.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import shutil
import tempfile
import time
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


def _generate_conect_from_pdbqt(pdbqt_path: str, atom_offset: int) -> list[str]:
    """Generate PDB CONECT records from a ligand PDBQT using RDKit topology.

    Args:
        pdbqt_path: Path to ligand PDBQT file.
        atom_offset: Number to add to each atom index (so ligand serials
            do not overlap with receptor atoms in the merged PDB).

    Returns:
        List of CONECT record strings. Empty list if RDKit fails or is unavailable.
    """
    if not _HAVE_RDKIT:
        return []

    from rdkit import Chem

    from autodock.utils import _sanitize_pdbqt_for_rdkit

    try:
        pdb_block = _sanitize_pdbqt_for_rdkit(pdbqt_path)
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
        if mol is None or mol.GetNumAtoms() == 0:
            return []
    except Exception:
        return []

    conect_lines: list[str] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx() + 1 + atom_offset  # PDB is 1-based
        j = bond.GetEndAtomIdx() + 1 + atom_offset
        conect_lines.append(f"CONECT{i:5d}{j:5d}\n")
    return conect_lines


def _build_complex_pdb(receptor_pdb: str, ligand_pdbqt: str, output_pdb: str) -> str:
    """
    Merge receptor PDB and ligand PDBQT into a single PDB for PLIP analysis.
    PLIP requires the ligand as HETATM records with a distinct residue name.

    To avoid atom-serial collisions and help Open Babel infer bonds for
    macrocycles / complex ring systems, ligand atoms are renumbered starting
    from ``max_receptor_serial + 1``, and optional CONECT records are injected.
    """
    with open(receptor_pdb) as fh:
        rec_lines = fh.readlines()

    # Strip END/ENDMDL from receptor
    rec_lines = [line for line in rec_lines if not line.strip().startswith(("END", "ENDMDL"))]

    # Find the highest atom serial in the receptor so ligand serials don't collide
    max_rec_serial = 0
    for line in rec_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                serial = int(line[6:11])
                if serial > max_rec_serial:
                    max_rec_serial = serial
            except ValueError:
                continue

    # Parse ligand from PDBQT and rewrite as properly formatted HETATM
    lig_lines = []
    atom_num = max_rec_serial + 1
    atom_count = 0
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
            atom_count += 1

    # Generate CONECT records to help PLIP/Open Babel with macrocycles
    conect_lines = _generate_conect_from_pdbqt(ligand_pdbqt, atom_offset=max_rec_serial)

    with open(output_pdb, "w") as fh:
        fh.writelines(rec_lines)
        fh.write("TER\n")
        fh.writelines(lig_lines)
        if conect_lines:
            fh.writelines(conect_lines)
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


def _get_smiles_from_pdbqt_via_openbabel(ligand_pdbqt: str) -> str | None:
    """Use Open Babel Python API to convert PDBQT → SMILES.

    This is a robust fallback when REMARK SMILES is missing from the PDBQT.
    Open Babel handles the PDBQT atom-type mapping internally.
    """
    try:
        from openbabel import openbabel as ob
    except ImportError:
        return None

    try:
        conv = ob.OBConversion()
        conv.SetInFormat("pdbqt")
        conv.SetOutFormat("smi")
        mol = ob.OBMol()
        if not conv.ReadFile(mol, ligand_pdbqt):
            return None
        # Remove title (after tab) to get pure SMILES
        smi = conv.WriteString(mol).strip()
        if "\t" in smi:
            smi = smi.split("\t")[0]
        return smi if smi else None
    except Exception as exc:
        logger.debug(f"Open Babel SMILES extraction failed: {exc}")
        return None


def _build_ligand_mol_for_prolif(ligand_pdbqt: str):
    """
    Build an RDKit molecule with explicit hydrogens and docking pose coordinates.

    Strategy:
      1. Try REMARK SMILES from PDBQT → full molecule with correct H count.
      2. If REMARK SMILES missing, try Open Babel → SMILES conversion.
      3. MCS-align SMILES molecule to PDBQT coordinates.
      4. Fallback: read PDBQT directly (incomplete H, but functional).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFMCS

    smiles = _parse_smiles_from_pdbqt(ligand_pdbqt)

    if not smiles:
        # Try Open Babel as robust fallback
        smiles = _get_smiles_from_pdbqt_via_openbabel(ligand_pdbqt)
        if smiles:
            logger.info(f"ProLIF ligand: SMILES recovered via Open Babel: {smiles}")

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


def _build_prolif_receptor_mol(receptor_pdb: str):
    """Build a ProLIF Molecule from a receptor PDB.

    Strategy (fastest-first):
      1. Try MDAnalysis → ProLIF ``from_mda`` (inferrer=None for no-H PDBs).
         This is ~2-3× faster than RDKit for large proteins.
      2. Fallback: RDKit ``MolFromPDBFile`` + ``AddHs`` + ``from_rdkit``.

    Returns:
        ``prolif.Molecule`` ready for fingerprint generation.
    """
    t0 = time.perf_counter()

    # ── Path A: MDAnalysis (fast, handles no-H PDBs via inferrer=None) ───────
    try:
        import MDAnalysis as mda
        import prolif as plf

        u = mda.Universe(receptor_pdb)
        prot = u.select_atoms("protein")
        if prot.n_atoms == 0:
            # Some PDBs don't have standard protein residues; try all atoms
            prot = u.atoms

        # inferrer=None disables bond-order inference (safe for no-H PDBs).
        # ProLIF interaction detection is distance-based, so exact bond orders
        # are not required for correct results.
        rec_mol = plf.Molecule.from_mda(prot, inferrer=None)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"ProLIF receptor loaded via MDAnalysis ({rec_mol.GetNumAtoms()} atoms, "
            f"{elapsed:.3f}s)"
        )
        return rec_mol
    except Exception as exc:
        logger.debug(f"MDAnalysis receptor loading failed: {exc}")

    # ── Path B: RDKit (legacy, slower for large proteins) ────────────────────
    import prolif as plf
    from rdkit import Chem

    rec_mol_rdkit = Chem.MolFromPDBFile(receptor_pdb, sanitize=False)
    if rec_mol_rdkit is None:
        raise VisualizationError(f"Could not read receptor PDB: {receptor_pdb}")
    rec_h = Chem.AddHs(rec_mol_rdkit, addCoords=True)
    rec_mol = plf.Molecule.from_rdkit(rec_h)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"ProLIF receptor loaded via RDKit ({rec_mol.GetNumAtoms()} atoms, " f"{elapsed:.3f}s)"
    )
    return rec_mol


def detect_interactions_prolif(
    receptor_pdb: str,
    ligand_pdbqt: str,
) -> list[dict[str, Any]]:
    """
    Detect interactions using ProLIF v2.1.0 as a secondary / cross-validation source.

    Uses MDAnalysis (primary) or RDKit (fallback) to load the receptor, and
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

    t0 = time.perf_counter()

    # ── Receptor ──────────────────────────────────────────────────────────────
    prot_mol = _build_prolif_receptor_mol(receptor_pdb)

    # ── Ligand ────────────────────────────────────────────────────────────────
    lig_mol_rdkit = _build_ligand_mol_for_prolif(ligand_pdbqt)
    lig_mol = plf.Molecule.from_rdkit(lig_mol_rdkit)

    # ── Run ProLIF ────────────────────────────────────────────────────────────
    fp = plf.Fingerprint()
    ifp = fp.generate(lig_mol, prot_mol, metadata=True)

    interactions: list[dict[str, Any]] = []
    if not ifp:
        logger.info("ProLIF detected 0 interactions")
        total_t = time.perf_counter() - t0
        logger.info(f"ProLIF total time: {total_t:.3f}s")
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

    total_t = time.perf_counter() - t0
    logger.info(f"ProLIF detected {len(interactions)} interactions ({total_t:.3f}s)")
    return interactions


# ─────────────────────────────────────────────────────────────────────────────
# Cross-engine discrepancy reporting
# ─────────────────────────────────────────────────────────────────────────────


# Cross-engine interaction-type normalisation.
# PLIP and ProLIF use different nomenclature and definitions for some
# interaction classes (e.g. PLIP "Hydrophobic" vs ProLIF "van der Waals").
# The mapping below is used for *loose* cross-engine comparison only.
_INTERACTION_TYPE_NORMALISATION = {
    # PLIP → canonical
    "H-bond": "H-bond",
    "Hydrophobic": "Hydrophobic",
    "π-π": "Aromatic",
    "π-cation": "Aromatic",
    "Salt bridge": "Ionic",
    "Halogen bond": "Halogen",
    "Water bridge": "H-bond",
    "Metal complex": "Metal",
    # ProLIF → canonical
    "van der Waals": "Hydrophobic",
    "π-π": "Aromatic",
    "π-cation": "Aromatic",
    "Cationic": "Ionic",
    "Anionic": "Ionic",
    "XBDonor": "Halogen",
    "XBAcceptor": "Halogen",
    "MetalDonor": "Metal",
    "MetalAcceptor": "Metal",
    "WaterBridge": "H-bond",
}


def _normalise_interaction_type(name: str) -> str:
    """Return a canonical interaction-type name for cross-engine comparison."""
    return _INTERACTION_TYPE_NORMALISATION.get(name, name)


def _make_interaction_key(
    item: dict[str, Any],
    normalise_type: bool = False,
) -> tuple:
    """Create a normalised key for cross-engine comparison.

    Parameters
    ----------
    item:
        Interaction dict (PLIP or ProLIF format).
    normalise_type:
        If *True*, map interaction-type names to a canonical vocabulary
        (e.g. PLIP "Hydrophobic" ↔ ProLIF "van der Waals").
        Default is *False* (strict type matching).

    Notes
    -----
    Ligand-atom coordinates are **intentionally excluded** from the key
    because PLIP reports global PDB coordinates whereas ProLIF reports
    ligand-local RDKit conformer coordinates — the two systems never
    overlap, so coordinate-based matching would always yield 0 %
    agreement.
    """
    itype = item.get("type", "")
    if normalise_type:
        itype = _normalise_interaction_type(itype)
    return (
        itype,
        item.get("resn", ""),
        item.get("resi", 0),
        item.get("chain", ""),
    )


def _residue_key(item: dict[str, Any]) -> tuple:
    """Return a residue-only key for residue-level agreement analysis."""
    return (
        item.get("resn", ""),
        item.get("resi", 0),
        item.get("chain", ""),
    )


def _generate_interaction_discrepancy_report(
    plip_results: list[dict[str, Any]],
    prolif_results: list[dict[str, Any]],
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Compare PLIP and ProLIF results and produce a structured discrepancy report.

    Returns a dict with:
      - ``summary``: high-level statistics (strict agreement, loose type-normalised
        agreement, residue-level agreement, counts).
      - ``only_plip``: interactions found by PLIP but not ProLIF (strict).
      - ``only_prolif``: interactions found by ProLIF but not PLIP (strict).
      - ``both``: interactions detected by both engines (strict).
      - ``json_path`` / ``csv_path``: paths to saved files (if *output_dir* given).
    """
    # ── Strict matching (type + residue) ──────────────────────────────────
    plip_keys = {_make_interaction_key(i): i for i in plip_results}
    prolif_keys = {_make_interaction_key(i): i for i in prolif_results}

    plip_set = set(plip_keys.keys())
    prolif_set = set(prolif_keys.keys())

    both_keys = plip_set & prolif_set
    only_plip_keys = plip_set - prolif_set
    only_prolif_keys = prolif_set - plip_set

    both = [plip_keys[k] for k in both_keys]
    only_plip = [plip_keys[k] for k in only_plip_keys]
    only_prolif = [prolif_keys[k] for k in only_prolif_keys]

    # ── Loose matching (normalised type + residue) ────────────────────────
    plip_loose = {_make_interaction_key(i, normalise_type=True): i for i in plip_results}
    prolif_loose = {_make_interaction_key(i, normalise_type=True): i for i in prolif_results}
    both_loose_keys = set(plip_loose.keys()) & set(prolif_loose.keys())
    only_plip_loose_keys = set(plip_loose.keys()) - set(prolif_loose.keys())
    only_prolif_loose_keys = set(prolif_loose.keys()) - set(plip_loose.keys())

    # ── Residue-level matching (any type, same residue) ───────────────────
    plip_residues = {_residue_key(i) for i in plip_results}
    prolif_residues = {_residue_key(i) for i in prolif_results}
    both_residues = plip_residues & prolif_residues
    only_plip_residues = plip_residues - prolif_residues
    only_prolif_residues = prolif_residues - plip_residues

    total_unique = len(plip_set | prolif_set)
    agreement_rate = len(both_keys) / total_unique if total_unique > 0 else 1.0
    loose_rate = len(both_loose_keys) / total_unique if total_unique > 0 else 1.0
    residue_rate = (
        len(both_residues) / len(plip_residues | prolif_residues)
        if (plip_residues | prolif_residues)
        else 1.0
    )

    report = {
        "summary": {
            "plip_count": len(plip_results),
            "prolif_count": len(prolif_results),
            "plip_unique": len(only_plip),
            "prolif_unique": len(only_prolif),
            "agreed": len(both),
            "agreement_rate": round(agreement_rate, 3),
            "loose_agreed": len(both_loose_keys),
            "loose_agreement_rate": round(loose_rate, 3),
            "residue_agreed": len(both_residues),
            "residue_agreement_rate": round(residue_rate, 3),
            "total_unique": total_unique,
        },
        "only_plip": only_plip,
        "only_prolif": only_prolif,
        "both": both,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # JSON report
        json_path = os.path.join(output_dir, "interaction_discrepancy.json")
        try:
            with open(json_path, "w") as fh:
                json.dump(report, fh, indent=2, default=str)
            report["json_path"] = json_path
        except (OSError, TypeError) as exc:
            logger.warning(f"Discrepancy JSON save failed: {exc}")

        # CSV flat-file for easy spreadsheet inspection
        csv_path = os.path.join(output_dir, "interaction_discrepancy.csv")
        try:
            rows: list[dict[str, Any]] = []
            for item in only_plip:
                row = dict(item)
                row["engine"] = "PLIP_only"
                rows.append(row)
            for item in only_prolif:
                row = dict(item)
                row["engine"] = "ProLIF_only"
                rows.append(row)
            for item in both:
                row = dict(item)
                row["engine"] = "Both"
                rows.append(row)

            if rows:
                # Flatten ligand_atoms for CSV
                for row in rows:
                    atoms = row.pop("ligand_atoms", [])
                    if atoms:
                        row["ligand_atom_coords"] = "; ".join(
                            f"({a['coords'][0]:.2f}, {a['coords'][1]:.2f}, {a['coords'][2]:.2f})"
                            for a in atoms
                        )
                    else:
                        row["ligand_atom_coords"] = ""

                with open(csv_path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
                    w.writeheader()
                    w.writerows(rows)
                report["csv_path"] = csv_path
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Discrepancy CSV save failed: {exc}")

    logger.info(
        f"Cross-engine agreement — strict: {len(both)}/{total_unique} "
        f"({agreement_rate*100:.1f}%), "
        f"loose: {len(both_loose_keys)}/{total_unique} ({loose_rate*100:.1f}%), "
        f"residue: {len(both_residues)}/{len(plip_residues | prolif_residues)} "
        f"({residue_rate*100:.1f}%) — "
        f"PLIP-only: {len(only_plip)}, ProLIF-only: {len(only_prolif)}"
    )
    return report


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
        output_dir: Optional working directory for discrepancy reports
            when *method* == ``"both"``.

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

    # Both: merge, deduplicate, and optionally generate discrepancy report.
    #
    # Cross-engine deduplication strategy:
    #   1. PLIP is the primary / authoritative engine — all PLIP interactions
    #      are always retained.
    #   2. ProLIF interactions are added only if they do NOT match an existing
    #      PLIP interaction at the *loose* level (normalised type + residue).
    #      This avoids double-counting the same physical contact while still
    #      preserving genuinely unique ProLIF findings.
    #
    # Coordinates are intentionally excluded from the dedup key because PLIP
    # uses global PDB coordinates and ProLIF uses ligand-local RDKit conformer
    # coordinates — they are never comparable.
    seen_plip = set()
    merged: list[dict[str, Any]] = []
    for i in plip_intx:
        key = _make_interaction_key(i)
        if key not in seen_plip:
            seen_plip.add(key)
            merged.append(i)

    seen_loose = {_make_interaction_key(i, normalise_type=True) for i in merged}
    for i in prolif_intx:
        loose_key = _make_interaction_key(i, normalise_type=True)
        if loose_key not in seen_loose:
            seen_loose.add(loose_key)
            merged.append(i)

    logger.info(
        f"Merged interactions (PLIP + ProLIF): {len(merged)} unique "
        f"({len(plip_intx)} PLIP, {len(prolif_intx)} ProLIF)"
    )

    # Discrepancy report for method="both"
    if method == "both" and (plip_intx or prolif_intx):
        _generate_interaction_discrepancy_report(plip_intx, prolif_intx, output_dir=output_dir)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Interaction Fingerprint (IFP) scoring for pose re-ranking
# ─────────────────────────────────────────────────────────────────────────────


def interaction_fingerprint(interactions: list[dict[str, Any]]) -> set[str]:
    """Convert interaction list to a fingerprint set of type:residue keys.

    Each key encodes the interaction type and the protein residue involved,
    e.g. ``"H-bond:GLU117.A"`` or ``"Hydrophobic:PHE121.A"``.  This coarse
    graining is intentionally robust to small geometric deviations between
    docked poses and the crystal structure.
    """
    fp: set[str] = set()
    for i in interactions:
        # Normalise type to a small canonical set for cross-engine compatibility
        itype = i.get("type", "Unknown")
        key = f"{itype}:{i.get('resn', 'UNK')}{i.get('resi', 0)}.{i.get('chain', 'A')}"
        fp.add(key)
    return fp


def ifp_tanimoto(ref_ifp: set[str], pose_ifp: set[str]) -> float:
    """Compute Tanimoto coefficient between two interaction fingerprints.

    Returns 1.0 when both fingerprints are empty (no interactions expected
    or detected), 0.0 when one is empty and the other is not, and the
    standard |intersection|/|union| otherwise.
    """
    if not ref_ifp and not pose_ifp:
        return 1.0
    if not ref_ifp or not pose_ifp:
        return 0.0
    intersection = len(ref_ifp & pose_ifp)
    union = len(ref_ifp | pose_ifp)
    return intersection / union if union > 0 else 0.0


def ifp_similarity_scores(
    receptor_pdb: str,
    all_poses_pdbqt: str,
    ref_ligand_pdb: str,
    method: str = "plip",
) -> list[tuple[int, float, float | None]]:
    """Score each pose in a multi-MODEL PDBQT by interaction-fingerprint similarity to a reference ligand.

    Args:
        receptor_pdb: Receptor PDB file path.
        all_poses_pdbqt: Multi-MODEL PDBQT with docked poses.
        ref_ligand_pdb: Reference ligand PDB (crystal pose) for computing the target IFP.
        method: Interaction detection engine (``"plip"`` | ``"prolif"`` | ``"both"``).

    Returns:
        List of ``(pose_index, tanimoto_score, vina_energy)`` tuples, sorted by
        descending Tanimoto score.  ``pose_index`` is 1-based to match Vina output.
        ``vina_energy`` is parsed from the PDBQT REMARK lines when available.
    """
    import re

    # 1. Reference IFP from crystal ligand
    ref_interactions = detect_interactions(receptor_pdb, ref_ligand_pdb, method=method)
    ref_ifp = interaction_fingerprint(ref_interactions)
    logger.info(f"Reference IFP: {len(ref_ifp)} interactions from crystal ligand")

    # 2. Split poses
    with open(all_poses_pdbqt) as fh:
        content = fh.read()
    models = re.split(r"MODEL\s+\d+\n", content)
    if len(models) <= 1:
        logger.warning("No poses found for IFP scoring")
        return []

    # 3. Score each pose
    scores: list[tuple[int, float, float | None]] = []
    for idx, model_block in enumerate(models[1:], start=1):
        # Write temporary pose PDBQT
        pose_path = tempfile.mktemp(suffix="_pose.pdbqt")
        try:
            with open(pose_path, "w") as fh:
                fh.write(model_block)

            # Parse Vina energy from REMARK
            energy: float | None = None
            for line in model_block.splitlines():
                if line.startswith("REMARK VINA RESULT:"):
                    try:
                        parts = line.split()
                        energy = float(parts[3])
                    except (IndexError, ValueError):
                        pass
                    break

            # Detect interactions
            try:
                pose_interactions = detect_interactions(receptor_pdb, pose_path, method=method)
                pose_ifp = interaction_fingerprint(pose_interactions)
                tanimoto = ifp_tanimoto(ref_ifp, pose_ifp)
                logger.debug(
                    f"Pose {idx}: IFP Tanimoto={tanimoto:.3f} "
                    f"({len(pose_ifp)} interactions, ref={len(ref_ifp)})"
                )
                scores.append((idx, tanimoto, energy))
            except Exception as exc:
                logger.debug(f"Pose {idx}: interaction detection failed ({exc}), Tanimoto=0.0")
                scores.append((idx, 0.0, energy))
        finally:
            with contextlib.suppress(OSError):
                os.unlink(pose_path)

    # Sort by descending Tanimoto
    scores.sort(key=lambda x: (-x[1], x[2] if x[2] is not None else 0.0))
    return scores
