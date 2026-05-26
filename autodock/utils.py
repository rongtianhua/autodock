"""
autodock.utils — General-purpose utilities.
===========================================
Coordinate math, file I/O helpers, PDB parsing, and format conversion.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from autodock.core import StructureFetchError, find_conda_tool, logger, safe_subprocess


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist; return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_temp_file(content: str, suffix: str = ".tmp") -> str:
    """Write content to a temporary file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
    except Exception:
        os.close(fd)
        raise
    return path


def read_pdb_atoms(pdb_path: str) -> list[dict[str, Any]]:
    """
    Parse ATOM / HETATM records from a PDB file.

    Returns a list of dicts with keys: record, atom_num, atom_name, res_name,
    chain, res_seq, x, y, z, element.
    """
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                atom = {
                    "record": line[:6].strip(),
                    "atom_num": int(line[6:11]),
                    "atom_name": line[12:16].strip(),
                    "res_name": line[17:20].strip(),
                    "chain": line[21].strip() or "A",
                    "res_seq": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": line[76:78].strip() or line[12:14].strip()[0],
                }
                atoms.append(atom)
            except (ValueError, IndexError):
                continue
    return atoms


def compute_bounding_box(
    atoms: list[dict[str, Any]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Compute (center, size) of a bounding box from atom coordinates.

    Returns:
        (center_xyz, size_xyz)
    """
    if not atoms:
        return (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)
    coords = np.array([(a["x"], a["y"], a["z"]) for a in atoms])
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    center = tuple((min_c + max_c) / 2)
    size = tuple(max_c - min_c)
    return center, size


def compute_bounding_box_from_pdb(
    pdb_path: str, residue_filter: set[str] | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Compute bounding box for atoms in a PDB file, optionally filtering by residue name.

    Args:
        pdb_path: Path to PDB file.
        residue_filter: If provided, only include atoms with res_name in this set.

    Returns:
        (center_xyz, size_xyz)
    """
    atoms = read_pdb_atoms(pdb_path)
    if residue_filter:
        atoms = [a for a in atoms if a["res_name"] in residue_filter]
    return compute_bounding_box(atoms)


def compute_bounding_box_from_pdbqt(
    pdbqt_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute bounding box from a PDBQT file (parses ATOM/HETATM lines)."""
    atoms = []
    with open(pdbqt_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                atoms.append(
                    {
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                    }
                )
            except ValueError:
                continue
    return compute_bounding_box(atoms)


def filter_pdb_lines(
    pdb_path: str,
    output_path: str,
    remove_water: bool = True,
    remove_hetatms: bool = False,
    keep_residues: set[str] | None = None,
) -> str:
    """
    Filter a PDB file, optionally removing waters and/or all HETATM records.

    Args:
        pdb_path: Input PDB file.
        output_path: Output PDB file.
        remove_water: Remove HOH/WAT/H2O residues.
        remove_hetatms: Remove all HETATM records.
        keep_residues: If provided, keep only these residue names (applies to both ATOM and HETATM).

    Returns:
        Path to output file.
    """
    water_names = {"HOH", "WAT", "H2O", "DOD", "TIP", "SOL"}
    out_lines = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM  "):
                res_name = line[17:20].strip()
                if keep_residues and res_name not in keep_residues:
                    continue
                if remove_water and res_name in water_names:
                    continue
                out_lines.append(line)
            elif line.startswith("HETATM"):
                res_name = line[17:20].strip()
                if remove_hetatms:
                    continue
                if keep_residues and res_name not in keep_residues:
                    continue
                if remove_water and res_name in water_names:
                    continue
                out_lines.append(line)
            else:
                out_lines.append(line)

    with open(output_path, "w") as fh:
        fh.writelines(out_lines)
    return output_path


# AutoDock atom type → element symbol mapping for PDBQT → RDKit parsing
_AD4_ELEMENT_MAP = {
    "A": "C",
    "OA": "O",
    "HD": "H",
    "NA": "N",
    "SA": "S",
    "N": "N",
    "O": "O",
    "C": "C",
    "H": "H",
    "S": "S",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "I": "I",
    "P": "P",
    "Mg": "Mg",
    "Ca": "Ca",
    "Mn": "Mn",
    "Fe": "Fe",
    "Zn": "Zn",
    "Na": "Na",
    "K": "K",
    "Cu": "Cu",
    "Co": "Co",
    "Ni": "Ni",
    "Se": "Se",
    # Open Babel sometimes emits G0 for unrecognized atoms (usually carbon)
    "G": "C",
    "G0": "C",
}


def _sanitize_pdbqt_for_rdkit(pdbqt_path: str) -> str:
    """
    Read a PDBQT file, keep only ATOM/HETATM lines, and replace AutoDock atom
    types with standard element symbols so RDKit can parse them.

    PDBQT appends the AutoDock atom type after the partial charge, which usually
    lands at columns 78-79 (0-based positions 77-78).  RDKit, however, reads the
    element symbol from the PDB element column at columns 77-78 (0-based 76-77).
    For two-letter elements (Cl, Br, …) an off-by-one placement causes RDKit to
    read only the first character and mis-assign the element (e.g. Cl → C).

    We therefore reconstruct each line, writing the element at the correct
    position (76-77) and stripping the trailing partial charge / atom type.

    Also fixes atom names that RDKit mis-interprets as element symbols
    (e.g. atom name 'G' causes RDKit to look up element 'G' and crash).
    """
    out_lines = []
    with open(pdbqt_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            # Read AutoDock atom type from the PDBQT extension position
            ad_type = line[77:79].strip() if len(line) > 78 else ""
            elem = _AD4_ELEMENT_MAP.get(ad_type, ad_type)

            # Strip trailing whitespace / newline so we can rebuild the line
            stripped = line.rstrip("\n\r")

            # Fix atom name (cols 13-16 = 0-based 12-15) if RDKit would choke on it
            atom_name = stripped[12:16].strip() if len(stripped) > 15 else ""
            if atom_name == "G":
                stripped = stripped[:12] + " C  " + stripped[16:]

            # Reconstruct with element at proper PDB position (cols 77-78 = 0-based 76-77).
            # Truncate anything from position 76 onward and append the element
            # right-justified in a 2-char field, then add newline.
            new_line = stripped[:76] + f"{elem:>2}\n"
            out_lines.append(new_line)
    return "".join(out_lines)


def obabel_convert(
    input_path: str,
    output_path: str,
    in_format: str | None = None,
    out_format: str | None = None,
    options: list[str] | None = None,
    timeout: int = 60,
) -> bool:
    """
    Convert molecular file formats using Open Babel CLI.

    Args:
        input_path: Input file path.
        output_path: Output file path.
        in_format: Input format (e.g., 'sdf', 'pdb'). Auto-detected from extension if None.
        out_format: Output format (e.g., 'pdbqt', 'sdf'). Auto-detected if None.
        options: Additional obabel options (e.g., ['-p', '7.4', '--gen3d']).
        timeout: Timeout in seconds.

    Returns:
        True on success, False on failure.
    """
    obabel = find_conda_tool("obabel")
    if not obabel:
        logger.error("Open Babel (obabel) not found in PATH.")
        return False

    cmd = [obabel, input_path, "-O", output_path]
    if in_format:
        cmd.extend(["-i", in_format])
    if out_format:
        cmd.extend(["-o", out_format])
    if options:
        cmd.extend(options)

    success, stdout, stderr = safe_subprocess(cmd, timeout=timeout)
    if not success:
        logger.error(f"obabel conversion failed: {stderr[:300]}")
        return False
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        logger.error(f"obabel produced empty output: {output_path}")
        return False
    return True


def extract_ligand_from_pdb(
    pdb_path: str,
    ligand_resname: str = "LIG",
    output_sdf: str | None = None,
) -> tuple[Any, str | None]:
    """
    Extract a ligand from a PDB complex file and optionally save as SDF.

    PDB asymmetric units often contain multiple copies of the same ligand
    (different chains / residue numbers).  We group by (chain, res_seq) and
    keep only the largest group so that the returned molecule is a single
    ligand instance.

    Returns:
        (rdkit_mol, sdf_path_or_none)
    """
    from rdkit import Chem

    with open(pdb_path) as fh:
        lines = fh.readlines()

    # Group HETATM lines by (chain, res_seq) to handle multi-copy ASUs
    from collections import defaultdict

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for line in lines:
        if line.startswith("HETATM") and ligand_resname in line[17:20]:
            chain = line[21].strip()
            res_seq = line[22:26].strip()
            groups[(chain, res_seq)].append(line)

    if not groups:
        logger.warning(f"No ligand '{ligand_resname}' found in {pdb_path}")
        return None, None

    # Pick the largest group (most atoms) — this is the primary ligand copy
    best_key = max(groups, key=lambda k: len(groups[k]))
    ligand_lines = groups[best_key]
    logger.debug(
        f"Extracted ligand '{ligand_resname}' from chain {best_key[0]} "
        f"residue {best_key[1]} ({len(ligand_lines)} atoms)"
    )

    ligand_pdb = "".join(ligand_lines)
    mol = Chem.MolFromPDBBlock(ligand_pdb)
    if mol is None:
        logger.warning(f"Could not parse ligand '{ligand_resname}' from PDB block.")
        return None, None

    mol = Chem.AddHs(mol, addCoords=True)

    # Sanity check: if the ligand still contains multiple fragments (e.g.
    # a covalent adduct split across residues), keep only the largest fragment.
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        logger.warning(
            f"Ligand '{ligand_resname}' has {len(frags)} fragments; " f"keeping the largest one."
        )
        mol = max(frags, key=lambda m: m.GetNumAtoms())
        # Re-add Hs because GetMolFrags may strip them
        mol = Chem.AddHs(mol, addCoords=True)

    if output_sdf:
        writer = Chem.SDWriter(output_sdf)
        writer.write(mol)
        writer.close()
        return mol, output_sdf
    return mol, None


def extract_chain_from_pdb(
    pdb_path: str,
    chain_id: str,
    output_pdb: str | None = None,
    include_connect: bool = True,
) -> str:
    """
    Extract a specific chain from a PDB file, preserving ATOM, HETATM,
    and optional CONECT records.

    Args:
        pdb_path: Input PDB file.
        chain_id: Chain identifier (e.g., 'C').
        output_pdb: Output PDB file path. If None, returns PDB block as string.
        include_connect: Whether to include CONECT records.

    Returns:
        Path to output PDB file, or PDB block string if output_pdb is None.
    """
    chain_id = chain_id.strip()
    with open(pdb_path) as fh:
        lines = fh.readlines()

    # Collect atom serial numbers in the target chain
    chain_atom_nums = set()
    out_lines = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            if line[21].strip() == chain_id:
                out_lines.append(line)
                with contextlib.suppress(ValueError):
                    chain_atom_nums.add(int(line[6:11]))
        elif line.startswith("TER   "):
            # Include TER if it matches the chain
            ter_chain = line[21].strip() if len(line) > 21 else ""
            if ter_chain == chain_id:
                out_lines.append(line)
        elif line.startswith("CONECT") and include_connect:
            # Include CONECT if any connected atom is in our chain
            try:
                nums = [
                    int(line[i : i + 5]) for i in range(6, len(line), 5) if line[i : i + 5].strip()
                ]
                if any(n in chain_atom_nums for n in nums):
                    out_lines.append(line)
            except ValueError:
                pass
        elif line.startswith(("HEADER", "COMPND", "SOURCE", "REMARK", "SEQRES")):
            # Keep metadata lines
            out_lines.append(line)
        elif line.startswith("END"):
            out_lines.append(line)

    pdb_block = "".join(out_lines)
    if output_pdb:
        with open(output_pdb, "w") as fh:
            fh.write(pdb_block)
        return output_pdb
    return pdb_block


def pdb_chain_to_smiles(pdb_path: str, chain_id: str) -> str | None:
    """
    Extract a chain from PDB and convert to SMILES using Open Babel.

    Returns:
        SMILES string or None on failure.
    """
    import subprocess
    import tempfile

    chain_pdb = tempfile.mktemp(suffix="_chain.pdb")
    extract_chain_from_pdb(pdb_path, chain_id, chain_pdb)

    obabel = find_conda_tool("obabel")
    if not obabel:
        logger.warning("Open Babel not found — cannot convert chain to SMILES")
        return None

    try:
        result = subprocess.run(
            [obabel, "-i", "pdb", chain_pdb, "-o", "smi"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Output format: "SMILES  filename"
            line = result.stdout.strip().split("\n")[0]
            smiles = line.split()[0] if line else None
            return smiles
    except Exception as exc:
        logger.warning(f"obabel SMILES conversion failed: {exc}")
    finally:
        if os.path.exists(chain_pdb):
            os.remove(chain_pdb)
    return None


def rmsd_matrix(poses: list[Any]) -> np.ndarray:
    """
    Compute pairwise RMSD matrix for a list of RDKit molecules (with 3D coords).

    Returns:
        NxN numpy array of RMSD values (Å).
    """
    from rdkit.Chem import AllChem

    n = len(poses)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            try:
                rms = AllChem.GetBestRMS(poses[i], poses[j])
                mat[i, j] = rms
                mat[j, i] = rms
            except Exception:
                mat[i, j] = 999.0
                mat[j, i] = 999.0
    return mat


def download_pdb(pdb_id: str, output_dir: str = ".") -> str:
    """
    Download a PDB file from RCSB.

    Returns:
        Path to downloaded PDB file.

    Raises:
        StructureFetchError: If download fails.
    """
    import urllib.request

    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise StructureFetchError(f"Invalid PDB ID: {pdb_id} (must be 4 characters)")

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    out_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    ensure_dir(output_dir)

    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as exc:
        raise StructureFetchError(f"Failed to download {pdb_id}: {exc}")

    if os.path.getsize(out_path) < 100:
        raise StructureFetchError(f"Downloaded file is empty or invalid: {out_path}")

    logger.info(f"Downloaded PDB: {out_path}")
    return out_path


def download_ligand_sdf_from_pdb(ligand_code: str, output_dir: str = ".") -> str:
    """
    Download ideal SDF for a ligand from PDB Ligand Expo.

    Returns:
        Path to SDF file.
    """
    import urllib.request

    ligand_code = ligand_code.strip().upper()
    url = f"https://files.rcsb.org/ligands/download/{ligand_code}_ideal.sdf"
    out_path = os.path.join(output_dir, f"{ligand_code}.sdf")
    ensure_dir(output_dir)

    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as exc:
        raise StructureFetchError(f"Failed to download ligand {ligand_code}: {exc}")

    if os.path.getsize(out_path) < 50:
        # Try model coordinates as fallback
        url = f"https://files.rcsb.org/ligands/download/{ligand_code}_model.sdf"
        try:
            urllib.request.urlretrieve(url, out_path)
        except Exception as exc:
            raise StructureFetchError(f"Failed to download ligand {ligand_code} (model): {exc}")

    logger.info(f"Downloaded ligand SDF: {out_path}")
    return out_path


class StructureCache:
    """
    Simple disk cache for downloaded structures.
    """

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.autodock/structure_cache")
        self.cache_dir = ensure_dir(cache_dir)

    def _cache_path(self, key: str, ext: str = ".pdb") -> Path:
        return self.cache_dir / f"{key}{ext}"

    def get(self, key: str, ext: str = ".pdb") -> str | None:
        path = self._cache_path(key, ext)
        if path.exists() and path.stat().st_size > 100:
            return str(path)
        return None

    def put(self, key: str, source_path: str, ext: str = ".pdb") -> str:
        dest = self._cache_path(key, ext)
        import shutil

        shutil.copy2(source_path, dest)
        return str(dest)

    def clear(self) -> int:
        """Remove all cached files. Returns number of files removed."""
        count = 0
        for f in self.cache_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        return count

    def info(self) -> dict[str, Any]:
        files = [(f.name, f.stat().st_size) for f in self.cache_dir.iterdir() if f.is_file()]
        total = sum(s for _, s in files)
        return {
            "cache_dir": str(self.cache_dir),
            "n_files": len(files),
            "total_bytes": total,
            "files": files,
        }
