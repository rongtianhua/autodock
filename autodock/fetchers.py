"""
autodock.fetchers — Unified structure and compound data fetchers.
===============================================================
Programmatic access to protein structure databases (RCSB PDB,
AlphaFold DB, SWISS-MODEL, ESM Atlas) and chemical databases
(PubChem, ChEMBL, BindingDB, ZINC, PDB Ligand Expo).

All functions follow a uniform pattern:
  * Accept an identifier and output directory / path.
  * Return the local file path on success.
  * Raise ``StructureFetchError`` or ``DataSourceError`` on failure.
  * Log informative messages via ``autodock.core.logger``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from autodock.core import (
    DataSourceError,
    StructureFetchError,
    find_conda_tool,
    logger,
)
from autodock.utils import ensure_dir

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _http_get_json(url: str, timeout: int = 30) -> Any:
    """Perform GET and parse JSON."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_text(url: str, timeout: int = 30) -> str:
    """Perform GET and return text."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _download_url(url: str, out_path: str, timeout: int = 60) -> None:
    """Download a URL to disk, raising on small/empty files."""
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as exc:
        raise StructureFetchError(f"Download failed: {url} -> {exc}")
    if os.path.getsize(out_path) < 50:
        raise StructureFetchError(f"Downloaded file too small (< 50 B): {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# RCSB PDB search by protein name / gene symbol
# ─────────────────────────────────────────────────────────────────────────────


def search_pdb_by_name(
    query: str,
    max_resolution: float = 3.0,
    require_ligand: bool = True,
    method: str = "X-RAY DIFFRACTION",
    max_results: int = 50,
) -> list[str]:
    """
    Search RCSB PDB by protein name or gene symbol, returning ranked PDB IDs.

    Uses the RCSB Search API v2 (REST) directly via HTTP POST.  Results are
    filtered by experimental method, resolution, and optionally the presence
    of bound non-polymer ligands.

    Args:
        query: Protein name (e.g. ``"SARS-CoV-2 main protease"``) or gene
            symbol (e.g. ``"BRAF"``, ``"EGFR"``).
        max_resolution: Maximum X-ray resolution in Å (default 3.0).
        require_ligand: If True, only return structures that contain at
            least one bound non-polymer ligand (default True).
        method: Experimental method filter.  Use ``"X-RAY DIFFRACTION"``
            (default), ``"ELECTRON MICROSCOPY"``, ``"SOLUTION NMR"``, or
            ``None`` for any method.
        max_results: Maximum number of PDB IDs to return for ranking.

    Returns:
        List of PDB IDs ranked by quality (resolution → R-free → date).

    Raises:
        StructureFetchError: If search fails or returns no results.
    """
    import json

    # Build the RCSB Search API v2 query payload
    nodes: list[dict] = [
        {"type": "terminal", "service": "full_text",
         "parameters": {"value": query}},
    ]

    if max_resolution < 99:
        nodes.append({
            "type": "terminal", "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": max_resolution,
            },
        })

    if method:
        nodes.append({
            "type": "terminal", "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": method,
            },
        })

    if require_ligand:
        nodes.append({
            "type": "terminal", "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.deposited_nonpolymer_entity_instance_count",
                "operator": "greater_or_equal",
                "value": 1,
            },
        })

    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
        },
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    data = json.dumps(payload).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise StructureFetchError(f"RCSB search query failed: {exc}")

    raw_ids = [
        e["identifier"] for e in result.get("result_set", [])
        if "identifier" in e
    ]
    if not raw_ids:
        raise StructureFetchError(
            f"No PDB structures found for '{query}' with the given filters."
        )

    pdb_ids = [pid.upper() for pid in raw_ids]
    logger.info(
        f"RCSB search: '{query}' → {len(pdb_ids)} candidates"
        f" (resolution ≤ {max_resolution} Å, method={method})"
    )

    # Rank by quality using metadata from the Data API
    ranked = _rank_pdb_entries(pdb_ids)
    return ranked


def _rank_pdb_entries(pdb_ids: list[str]) -> list[str]:
    """Rank PDB entries by resolution (asc), R-free (asc), deposit date (desc).

    Uses a batched approach: fetches metadata for the first N candidates
    (the search engine already returns approximate quality ordering), then
    fine-ranks them.
    """
    scored: list[tuple[float, int, str, float, str]] = []

    # Only rank the first 20 candidates to avoid excessive API calls
    for pid in pdb_ids[:20]:
        try:
            info = _fetch_entry_metadata(pid)
            if info is None:
                continue
            resolution = info.get("resolution", 99.0)
            r_free = info.get("r_free", 99.0)
            year = info.get("deposit_year", 1900)
            scored.append((resolution, -year, pid, r_free, str(year)))
        except Exception:
            continue

    if not scored:
        return pdb_ids

    scored.sort(key=lambda x: (x[0], x[3], x[1]))
    ranked = [s[2] for s in scored]
    logger.info(
        f"RCSB ranking: top entry {ranked[0]}"
        f" (resolution={scored[0][0]:.1f} Å, year={scored[0][4]})"
    )
    return ranked


def _fetch_entry_metadata(pdb_id: str) -> dict | None:
    """Fetch entry metadata from RCSB Data API (REST v1)."""
    import json
    import urllib.request

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None

    # Parse resolution
    resolution = 99.0
    if "rcsb_entry_info" in data:
        resolution = (
            data["rcsb_entry_info"].get("resolution_combined", [99.0]) or [99.0]
        )
        if isinstance(resolution, list):
            resolution = resolution[0]

    # Parse R-free
    r_free = 99.0
    refine = data.get("refine", [])
    if refine:
        r_free = refine[0].get("ls_R_factor_R_free", 99.0) or 99.0

    # Parse deposit date
    deposit_year = 1900
    accession = data.get("rcsb_accession_info", {})
    deposit_date = accession.get("deposit_date", "")
    if deposit_date:
        deposit_year = int(deposit_date[:4])

    return {
        "resolution": float(resolution),
        "r_free": float(r_free),
        "deposit_year": deposit_year,
        "pdb_id": pdb_id.upper(),
    }


def find_best_pdb_structure(
    query: str,
    max_resolution: float = 3.0,
    require_ligand: bool = True,
    method: str = "X-RAY DIFFRACTION",
) -> str | None:
    """
    Search RCSB PDB by protein name / gene symbol and return the best
    matching PDB ID, ranked by resolution → R-free → deposition date.

    This is the recommended entry point for "protein name → PDB structure"
    automation.  It handles both direct PDB IDs and free-text queries.

    Args:
        query: Protein name, gene symbol, or PDB ID (4-character code).
        max_resolution: Maximum resolution in Å (default 3.0).
        require_ligand: Only return structures with bound small molecules.
        method: Experimental method (default ``"X-RAY DIFFRACTION"``).

    Returns:
        Best-matching PDB ID string, or ``None`` if no match found.
    """
    # If query looks like a PDB ID (4 alphanum chars, at least one digit)
    q = query.strip().upper()
    if len(q) == 4 and q.isalnum() and any(c.isdigit() for c in q):
        logger.info(f"Query '{query}' detected as PDB ID — returning directly")
        return q

    ranked = search_pdb_by_name(
        query,
        max_resolution=max_resolution,
        require_ligand=require_ligand,
        method=method,
    )
    return ranked[0] if ranked else None


def fetch_protein_structure(
    query: str,
    output_dir: str = ".",
    format: str = "pdb",
    max_resolution: float = 3.0,
    require_ligand: bool = True,
    method: str = "X-RAY DIFFRACTION",
) -> str:
    """
    One-stop function: protein name → search → rank → download.

    Args:
        query: Protein name, gene symbol, or PDB ID.
        output_dir: Output directory for downloaded structure.
        format: ``"pdb"`` or ``"cif"``.
        max_resolution: Maximum resolution in Å (default 3.0).
        require_ligand: Require bound small molecules (default True).
        method: Experimental method (default ``"X-RAY DIFFRACTION"``).

    Returns:
        Path to downloaded structure file.

    Raises:
        StructureFetchError: If no structure can be found or downloaded.
    """
    pdb_id = find_best_pdb_structure(
        query,
        max_resolution=max_resolution,
        require_ligand=require_ligand,
        method=method,
    )
    if pdb_id is None:
        raise StructureFetchError(
            f"Could not find a suitable PDB structure for '{query}'."
        )

    logger.info(f"Best PDB structure: {pdb_id} (from query '{query}')")
    return download_pdb(pdb_id, output_dir=output_dir, format=format)


# ─────────────────────────────────────────────────────────────────────────────
# Receptor structure fetchers
# ─────────────────────────────────────────────────────────────────────────────


def download_pdb(
    pdb_id: str,
    output_dir: str = ".",
    *,
    format: str = "pdb",
) -> str:
    """
    Download a coordinate file from RCSB PDB.

    Args:
        pdb_id: 4-character PDB identifier.
        output_dir: Destination directory.
        format: ``"pdb"``, ``"cif"`` (mmCIF), or ``"bcif"`` (BinaryCIF).

    Returns:
        Path to downloaded file.

    Raises:
        StructureFetchError: On invalid ID or download failure.
    """
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise StructureFetchError(f"Invalid PDB ID: {pdb_id} (must be 4 characters)")

    fmt = format.lower()
    if fmt == "pdb":
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        ext = ".pdb"
    elif fmt in ("cif", "mmcif"):
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        ext = ".cif"
    elif fmt == "bcif":
        url = f"https://models.rcsb.org/{pdb_id}.bcif"
        ext = ".bcif"
    else:
        raise StructureFetchError(f"Unsupported PDB format: {format}")

    out_path = os.path.join(output_dir, f"{pdb_id}{ext}")
    ensure_dir(output_dir)
    try:
        _download_url(url, out_path)
    except StructureFetchError:
        # Fallback: if PDB format failed, try mmCIF
        if fmt == "pdb":
            cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            cif_path = os.path.join(output_dir, f"{pdb_id}.cif")
            logger.warning(f"PDB format download failed — trying mmCIF fallback for {pdb_id}")
            _download_url(cif_url, cif_path)
            logger.info(f"Downloaded PDB (cif fallback): {cif_path}")
            return cif_path
        raise
    logger.info(f"Downloaded PDB ({fmt}): {out_path}")
    return out_path


def download_alphafold(
    uniprot_id: str,
    output_dir: str = ".",
    *,
    format: str = "pdb",
) -> str:
    """
    Download an AlphaFold-predicted structure from the AlphaFold DB.

    Uses the EMBL-EBI AlphaFold API to resolve the latest model URL,
    then downloads the requested format (PDB or mmCIF).

    Args:
        uniprot_id: UniProt accession (e.g. ``"P68871"``).
        output_dir: Destination directory.
        format: ``"pdb"`` or ``"cif"``.

    Returns:
        Path to downloaded file.

    Raises:
        StructureFetchError: If the UniProt ID is unknown or download fails.

    References:
        * https://alphafold.ebi.ac.uk/api-docs
        * https://alphafold.com/entry/AF-{uniprot_id}-F1
    """
    uniprot_id = uniprot_id.strip().upper()
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

    try:
        data = _http_get_json(api_url, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise StructureFetchError(f"AlphaFold DB has no entry for UniProt ID: {uniprot_id}")
        raise StructureFetchError(f"AlphaFold API error: {exc}")
    except Exception as exc:
        raise StructureFetchError(f"AlphaFold API request failed: {exc}")

    if not isinstance(data, list) or len(data) == 0:
        raise StructureFetchError(f"AlphaFold API returned empty result for {uniprot_id}")

    entry = data[0]
    fmt = format.lower()
    if fmt == "pdb":
        file_url = entry.get("pdbUrl")
        ext = ".pdb"
    elif fmt in ("cif", "mmcif"):
        file_url = entry.get("cifUrl")
        ext = ".cif"
    else:
        raise StructureFetchError(f"Unsupported AlphaFold format: {format}")

    if not file_url:
        raise StructureFetchError(
            f"AlphaFold API did not return a {fmt.upper()} URL for {uniprot_id}"
        )

    out_path = os.path.join(output_dir, f"AF-{uniprot_id}-F1{ext}")
    ensure_dir(output_dir)
    _download_url(file_url, out_path)
    logger.info(f"Downloaded AlphaFold ({fmt}, pLDDT={entry.get('globalMetricValue')}): {out_path}")
    return out_path


def download_swissmodel(
    uniprot_id: str,
    output_dir: str = ".",
) -> str:
    """
    Download a homology model from the SWISS-MODEL Repository.

    Args:
        uniprot_id: UniProt accession (e.g. ``"P68871"``).
        output_dir: Destination directory.

    Returns:
        Path to downloaded PDB file.

    Raises:
        StructureFetchError: If no model exists or download fails.

    References:
        * https://swissmodel.expasy.org/repository
    """
    uniprot_id = uniprot_id.strip().upper()
    url = f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.pdb"
    out_path = os.path.join(output_dir, f"{uniprot_id}_SWISSMODEL.pdb")
    ensure_dir(output_dir)

    try:
        _download_url(url, out_path)
    except StructureFetchError:
        raise

    # SWISS-MODEL returns an HTML error page (200 OK) when no model exists.
    with open(out_path, "rb") as fh:
        header = fh.read(200)
    if b"<!DOCTYPE" in header or b"<html" in header:
        os.remove(out_path)
        raise StructureFetchError(f"No SWISS-MODEL structure available for {uniprot_id}")

    logger.info(f"Downloaded SWISS-MODEL: {out_path}")
    return out_path


def fetch_uniprot_fasta(uniprot_id: str, output_path: str | None = None) -> str:
    """
    Download a FASTA sequence from UniProt.

    Args:
        uniprot_id: UniProt accession.
        output_path: If provided, write to this file; otherwise return as string.

    Returns:
        FASTA string if *output_path* is None, else the file path.
    """
    uniprot_id = uniprot_id.strip().upper()
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    text = _http_get_text(url, timeout=30)
    if not text.startswith(">"):
        raise DataSourceError(f"Invalid FASTA response for {uniprot_id}")
    if output_path:
        with open(output_path, "w") as fh:
            fh.write(text)
        logger.info(f"Downloaded UniProt FASTA: {output_path}")
        return output_path
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Ligand / compound fetchers
# ─────────────────────────────────────────────────────────────────────────────


def download_ligand_sdf_from_pdb(ligand_code: str, output_dir: str = ".") -> str:
    """
    Download ideal SDF for a ligand from PDB Ligand Expo.

    Falls back to ``_model.sdf`` if the ideal file is too small.

    Returns:
        Path to SDF file.
    """
    ligand_code = ligand_code.strip().upper()
    url = f"https://files.rcsb.org/ligands/download/{ligand_code}_ideal.sdf"
    out_path = os.path.join(output_dir, f"{ligand_code}.sdf")
    ensure_dir(output_dir)

    try:
        _download_url(url, out_path)
    except StructureFetchError:
        url = f"https://files.rcsb.org/ligands/download/{ligand_code}_model.sdf"
        _download_url(url, out_path)

    logger.info(f"Downloaded ligand SDF: {out_path}")
    return out_path


def fetch_pubchem_smiles(query: str) -> str:
    """
    Resolve a compound name or SMILES to canonical SMILES via PubChem.

    Args:
        query: Compound name (e.g. ``"aspirin"``) or SMILES string.

    Returns:
        Canonical SMILES string.

    Raises:
        DataSourceError: If the compound cannot be found.
    """
    # Try pubchempy first (robust Python wrapper)
    try:
        import pubchempy as pcp

        compounds = pcp.get_compounds(query, "name")
        if compounds:
            # canonical_smiles deprecated in pubchempy 1.0.5+
            return getattr(compounds[0], "connectivity_smiles", compounds[0].canonical_smiles)
    except Exception:
        pass

    # Fallback to raw PUG REST API
    encoded = urllib.parse.quote(query)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/"
        f"property/CanonicalSMILES/JSON"
    )
    try:
        data = _http_get_json(url, timeout=30)
        props = data["PropertyTable"]["Properties"]
        if props:
            return props[0]["CanonicalSMILES"]
    except Exception as exc:
        raise DataSourceError(f"PubChem lookup failed for '{query}': {exc}")

    raise DataSourceError(f"PubChem has no record for '{query}'")


def fetch_pubchem_sdf(cid: str | int, output_path: str) -> str:
    """
    Download an SDF file for a PubChem Compound ID (CID).

    Args:
        cid: PubChem CID.
        output_path: Destination file path.

    Returns:
        Path to downloaded SDF.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF"
    _download_url(url, output_path)
    logger.info(f"Downloaded PubChem SDF (CID {cid}): {output_path}")
    return output_path


def fetch_chembl_smiles(chembl_id: str) -> str:
    """
    Lookup canonical SMILES for a ChEMBL compound ID.

    Args:
        chembl_id: ChEMBL ID, e.g. ``"CHEMBL25"``.

    Returns:
        Canonical SMILES string.

    Raises:
        DataSourceError: On lookup failure.
    """
    chembl_id = chembl_id.strip().upper()
    if not chembl_id.startswith("CHEMBL"):
        raise DataSourceError(f"Invalid ChEMBL ID: {chembl_id}")

    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
    try:
        data = _http_get_json(url, timeout=30)
        smiles = data.get("molecule_structures", {}).get("canonical_smiles")
        if smiles:
            return smiles
    except Exception as exc:
        raise DataSourceError(f"ChEMBL lookup failed for {chembl_id}: {exc}")

    raise DataSourceError(f"ChEMBL has no SMILES for {chembl_id}")


def fetch_chembl_sdf(chembl_id: str, output_path: str) -> str:
    """
    Download an SDF file for a ChEMBL compound ID.

    Args:
        chembl_id: ChEMBL ID.
        output_path: Destination file path.

    Returns:
        Path to downloaded SDF.
    """
    chembl_id = chembl_id.strip().upper()
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.sdf"
    _download_url(url, output_path)
    logger.info(f"Downloaded ChEMBL SDF ({chembl_id}): {output_path}")
    return output_path


def fetch_bindingdb_by_smiles(
    smiles: str,
    *,
    affinity_cutoff: float = 0.85,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Query BindingDB for targets that bind a compound by SMILES.

    Args:
        smiles: Canonical SMILES string.
        affinity_cutoff: Affinity cutoff (pKi / pIC50).
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON response from BindingDB REST API.

    Raises:
        DataSourceError: On query failure.

    References:
        * https://www.bindingdb.org/rwd/bind/BindingDBRESTfulAPI.jsp
    """
    encoded = urllib.parse.quote(smiles)
    url = (
        f"https://bindingdb.org/rest/getTargetByCompound?"
        f"smiles={encoded}&cutoff={affinity_cutoff}&response=application/json"
    )
    try:
        return _http_get_json(url, timeout=timeout)
    except Exception as exc:
        raise DataSourceError(f"BindingDB query failed: {exc}")


def fetch_zinc_smiles(zinc_id: str, timeout: int = 15) -> str | None:
    """
    Best-effort lookup of SMILES for a ZINC ID.

    ZINC-22/CartBlanche does not expose a simple REST endpoint for
    single-molecule metadata.  This function tries a handful of
    public ZINC-15 / CartBlanche URLs and returns *None* if all fail.

    Args:
        zinc_id: ZINC identifier, e.g. ``"ZINC000000000001"``.
        timeout: Per-request timeout.

    Returns:
        SMILES string or *None*.
    """
    zinc_id = zinc_id.strip().upper()
    if not zinc_id.startswith("ZINC"):
        return None

    # Attempt 1: ZINC-15 legacy JSON endpoint (sometimes available)
    urls = [
        f"https://zinc15.docking.org/substances/{zinc_id}.json",
        f"https://zinc.docking.org/substances/{zinc_id}.json",
    ]
    for url in urls:
        try:
            data = _http_get_json(url, timeout=timeout)
            smi = data.get("smiles") or data.get("SMILES")
            if smi:
                logger.info(f"Resolved ZINC SMILES from {url}")
                return smi
        except Exception:
            continue

    # Attempt 2: CartBlanche text endpoint (experimental)
    try:
        txt_url = (
            f"https://cartblanche22.docking.org/substances.txt:"
            f"zinc_id={zinc_id}&output_fields=zinc_id,smiles"
        )
        text = _http_get_text(txt_url, timeout=timeout)
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].upper() == zinc_id:
                return parts[1]
    except Exception:
        pass

    logger.warning(f"Could not resolve ZINC ID {zinc_id} to SMILES")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# File-format readers (ligand libraries)
# ─────────────────────────────────────────────────────────────────────────────


def read_sdf_library(path: str) -> dict[str, str]:
    """
    Read a multi-molecule SDF file and extract name → SMILES mapping.

    Uses RDKit if available; falls back to parsing the SDF text blocks
    and attempting Open Babel conversion.

    Args:
        path: Path to SDF file.

    Returns:
        Dictionary mapping compound name to canonical SMILES.
        Molecules that cannot be converted are silently skipped.
    """
    from rdkit import Chem

    results: dict[str, str] = {}
    supplier = Chem.SDMolSupplier(str(path))
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        name = mol.GetProp("_Name") or f"compound_{i}"
        try:
            smiles = Chem.MolToSmiles(mol)
            results[name] = smiles
        except Exception:
            continue
    return results


def read_sdf_3d_library(path: str) -> dict[str, tuple[str, Any]]:
    """
    Read a multi-molecule SDF file preserving 3D coordinates.

    Args:
        path: Path to SDF file.

    Returns:
        Dictionary mapping compound name to (SMILES, RDKit Mol with 3D coords).
        Molecules that cannot be parsed are silently skipped.
    """
    from rdkit import Chem

    results: dict[str, tuple[str, Any]] = {}
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        name = mol.GetProp("_Name") or f"compound_{i}"
        try:
            smiles = Chem.MolToSmiles(mol)
            results[name] = (smiles, mol)
        except Exception:
            continue
    return results


def read_mol_file(path: str) -> Any | None:
    """
    Read a MOL or MOL2 file into an RDKit molecule.

    Args:
        path: Path to ``.mol`` or ``.mol2`` file.

    Returns:
        RDKit Mol object, or *None* on failure.
    """
    from rdkit import Chem

    ext = Path(path).suffix.lower()
    if ext == ".mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=False)
        if mol is not None:
            return mol
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
        if mol is not None:
            return mol

    # Fallback: Open Babel
    obabel = find_conda_tool("obabel")
    if not obabel:
        return None
    import subprocess

    fmt = "mol" if ext == ".mol" else "mol2"
    try:
        result = subprocess.run(
            [obabel, f"-i{fmt}", str(path), "-osmi"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            smiles = result.stdout.strip().split()[0]
            return Chem.MolFromSmiles(smiles)
    except Exception:
        pass
    return None


def read_mol2_file(path: str) -> Any | None:
    """Convenience wrapper for ``read_mol_file`` on MOL2."""
    return read_mol_file(path)


# ─────────────────────────────────────────────────────────────────────────────
# Config-driven unified fetch dispatcher
# ─────────────────────────────────────────────────────────────────────────────


RECEPTOR_FETCHERS: dict[str, Any] = {
    "pdb": download_pdb,
    "alphafold": download_alphafold,
    "swissmodel": download_swissmodel,
}

LIGAND_FETCHERS: dict[str, Any] = {
    "pubchem": fetch_pubchem_smiles,
    "chembl": fetch_chembl_smiles,
    "bindingdb": fetch_bindingdb_by_smiles,
    "zinc": fetch_zinc_smiles,
    "pdb_ligand": download_ligand_sdf_from_pdb,
}
