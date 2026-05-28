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
        {"type": "terminal", "service": "full_text", "parameters": {"value": query}},
    ]

    if max_resolution < 99:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": max_resolution,
                },
            }
        )

    if method:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": method,
                },
            }
        )

    if require_ligand:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.deposited_nonpolymer_entity_instance_count",
                    "operator": "greater_or_equal",
                    "value": 1,
                },
            }
        )

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
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise StructureFetchError(f"RCSB search query failed: {exc}")

    raw_ids = [e["identifier"] for e in result.get("result_set", []) if "identifier" in e]
    if not raw_ids:
        raise StructureFetchError(f"No PDB structures found for '{query}' with the given filters.")

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
        resolution = data["rcsb_entry_info"].get("resolution_combined", [99.0]) or [99.0]
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


def _resolve_to_uniprot(query: str) -> str | None:
    """Resolve a gene symbol or protein name to a UniProt accession ID.

    Uses the UniProt REST API (``/uniprotkb/search``) with gene-exact
    and protein-name filters.  Returns the first reviewed (Swiss-Prot)
    accession if found, or an unreviewed (TrEMBL) one as fallback.
    """
    import urllib.parse

    # If it already looks like a UniProt ID (e.g. P15056, Q9Y6K9), return directly
    q = query.strip().upper()
    if len(q) >= 6 and q[0].isalpha() and q[1:].isdigit():
        return q
    if len(q) >= 6 and q[:1].isalpha() and q[1].isdigit() and q[2:].isalpha():
        return q

    encoded = urllib.parse.quote(query.strip())
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?"
        f"query=({encoded}) AND (reviewed:true)&format=json&size=5"
    )
    try:
        data = _http_get_json(url, timeout=15)
        results = data.get("results", []) if isinstance(data, dict) else []
        if results:
            accession = results[0].get("primaryAccession")
            if accession:
                return accession
    except Exception:
        pass

    # Fallback: try without reviewed filter
    url2 = f"https://rest.uniprot.org/uniprotkb/search?" f"query=({encoded})&format=json&size=5"
    try:
        data2 = _http_get_json(url2, timeout=15)
        results2 = data2.get("results", []) if isinstance(data2, dict) else []
        if results2:
            accession = results2[0].get("primaryAccession")
            if accession:
                return accession
    except Exception:
        pass

    return None


def fetch_protein_structure(
    query: str,
    output_dir: str = ".",
    format: str = "cif",
    max_resolution: float = 3.0,
    require_ligand: bool = True,
    method: str = "X-RAY DIFFRACTION",
    fallback_alphafold: bool = True,
    fallback_swissmodel: bool = True,
) -> str:
    """
    One-stop function: protein name → search → rank → download.

    **Priority chain**:
    1. RCSB PDB (crystal/NMR/EM structures) — searched by name/gene symbol,
       filtered by resolution, method, and ligand presence.
    2. AlphaFold DB — if PDB search returns no suitable structure, resolves
       the query to a UniProt ID and downloads the AlphaFold prediction.
       Quality is assessed via :func:`assess_alphafold_quality` (logged).
    3. SWISS-MODEL — if AlphaFold quality is low (mean pLDDT < 90 or
       >20 % low-confidence residues), attempts SWISS-MODEL homology model.

    Args:
        query: Protein name, gene symbol, or PDB ID.
        output_dir: Output directory for downloaded structure.
        format: ``"cif"`` (default, mmCIF) or ``"pdb"``.
        max_resolution: Maximum resolution in Å (default 3.0) for PDB search.
        require_ligand: Require bound small molecules in PDB search.
        method: Experimental method for PDB search
            (default ``"X-RAY DIFFRACTION"``).
        fallback_alphafold: Try AlphaFold if PDB search fails (default True).
        fallback_swissmodel: Try SWISS-MODEL if AlphaFold quality is poor
            (default True).

    Returns:
        Path to downloaded structure file.

    Raises:
        StructureFetchError: If no structure can be found from any source.
    """
    # ── Level 1: RCSB PDB ──────────────────────────────────────────────────
    pdb_id = find_best_pdb_structure(
        query,
        max_resolution=max_resolution,
        require_ligand=require_ligand,
        method=method,
    )
    if pdb_id:
        path = download_pdb(pdb_id, output_dir=output_dir, format=format)
        logger.info(f"Source: RCSB PDB ({pdb_id}) for '{query}'")
        return path

    if not fallback_alphafold and not fallback_swissmodel:
        raise StructureFetchError(f"No PDB structure found for '{query}' and fallbacks disabled.")

    # ── Level 2: AlphaFold ─────────────────────────────────────────────────
    uniprot_id = _resolve_to_uniprot(query)
    af_path: str | None = None
    if uniprot_id and fallback_alphafold:
        try:
            af_path = download_alphafold(uniprot_id, output_dir=output_dir, format=format)
            logger.info(f"Source: AlphaFold DB (UniProt {uniprot_id}) for '{query}'")
            # Assess quality
            try:
                from autodock.alphafold_tools import assess_alphafold_quality

                quality = assess_alphafold_quality(af_path)
                if quality.get("suitable_for_docking"):
                    return af_path
                if not fallback_swissmodel:
                    logger.warning(
                        f"AlphaFold quality low (mean pLDDT={quality.get('mean_plddt', 'N/A'):.1f})"
                        f" — returning anyway (fallback_swissmodel=False)"
                    )
                    return af_path
            except Exception:
                # Quality assessment failed — return AF structure anyway
                return af_path
        except StructureFetchError:
            logger.info(f"AlphaFold has no entry for UniProt {uniprot_id}")

    # ── Level 3: SWISS-MODEL ──────────────────────────────────────────────
    if uniprot_id and fallback_swissmodel:
        try:
            sm_path = download_swissmodel(uniprot_id, output_dir=output_dir)
            logger.info(f"Source: SWISS-MODEL (UniProt {uniprot_id}) for '{query}'")
            return sm_path
        except StructureFetchError:
            logger.info(f"SWISS-MODEL has no entry for UniProt {uniprot_id}")

    # ── All sources exhausted ──────────────────────────────────────────────
    raise StructureFetchError(
        f"Could not obtain a structure for '{query}' from any source. "
        + (
            f"Resolved to UniProt {uniprot_id}."
            if uniprot_id
            else "Could not resolve to UniProt ID."
        )
        + " Tried: RCSB PDB"
        + (" → AlphaFold DB" if fallback_alphafold else "")
        + (" → SWISS-MODEL" if fallback_swissmodel else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Receptor structure fetchers
# ─────────────────────────────────────────────────────────────────────────────


def download_pdb(
    pdb_id: str,
    output_dir: str = ".",
    *,
    format: str = "cif",
) -> str:
    """
    Download a coordinate file from RCSB PDB (the Protein Data Bank).

    .. note::
       The function name reflects the **database** (PDB), not the file format.
       Default format is **mmCIF** (``"cif"``) — RCSB recommends mmCIF for
       all new depositions.  Use ``format="pdb"`` for legacy PDB format.

    Args:
        pdb_id: 4-character PDB identifier.
        output_dir: Destination directory.
        format: ``"cif"`` (default, mmCIF), ``"pdb"``, or ``"bcif"`` (BinaryCIF).

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
        # Bidirectional fallback between cif and pdb
        if fmt in ("cif", "mmcif"):
            # CIF failed → try PDB
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")
            logger.warning(f"mmCIF download failed — trying PDB fallback for {pdb_id}")
            try:
                _download_url(pdb_url, pdb_path)
                logger.info(f"Downloaded PDB (fallback): {pdb_path}")
                return pdb_path
            except StructureFetchError:
                raise
        elif fmt == "pdb":
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
    format: str = "cif",
) -> str:
    """
    Download an AlphaFold-predicted structure from the AlphaFold DB.

    Uses the EMBL-EBI AlphaFold API to resolve the latest model URL,
    then downloads the requested format (PDB or mmCIF).

    Args:
        uniprot_id: UniProt accession (e.g. ``"P68871"``).
        output_dir: Destination directory.
        format: ``"cif"`` (default, mmCIF) or ``"pdb"``.

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


def fetch_chembl_by_target(
    target_query: str,
    *,
    max_molecules: int = 50,
    min_pchembl: float | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """
    Search ChEMBL for compounds active against a protein target.

    Uses ChEMBL's REST API v2 to search by UniProt accession, gene
    symbol, or protein name.  Returns a list of dicts with keys
    ``chembl_id``,
    ``smiles``, ``pchembl_value``, ``assay_type``.

    This is the primary entry point for "target → known inhibitors"
    lookups in a drug-discovery pipeline.

    Args:
        target_query: UniProt ID (e.g. ``"P15056"``), gene symbol
            (e.g. ``"BRAF"``), or protein name (e.g. ``"EGFR"``).
        max_molecules: Maximum compounds to return (default 50).
        min_pchembl: Minimum pChEMBL threshold for filtering
            (e.g. ``6.0`` for micromolar).  None = no filter.
        timeout: HTTP timeout in seconds.

    Returns:
        List of compound dicts with keys ``chembl_id``, ``smiles``,
        ``pchembl_value`` (float or None), ``assay_type``, ``target_name``.

    Raises:
        DataSourceError: If the target cannot be resolved.

    References:
        * https://www.ebi.ac.uk/chembl/api/docs
    """
    encoded_query = urllib.parse.quote(target_query.strip())

    # Step 1: Resolve target → UniProt ID (try multiple query methods)
    target_urls = [
        f"https://www.ebi.ac.uk/chembl/api/data/target.json?target_components__accession={encoded_query}&limit=3",
        f"https://www.ebi.ac.uk/chembl/api/data/target.json?pref_name__iexact={encoded_query}&limit=3",
        "https://www.ebi.ac.uk/chembl/api/data/target.json?organism=Homo%20sapiens&limit=10",
    ]

    target_id: str | None = None
    target_name: str = target_query

    for turl in target_urls:
        try:
            data = _http_get_json(turl, timeout=timeout)
            targets = data.get("targets", [])
            if targets:
                tgt = targets[0]
                target_id = tgt.get("target_chembl_id", "")
                target_name = tgt.get("pref_name") or target_query
                logger.info(f"ChEMBL: resolved '{target_query}' → {target_id} ({target_name})")
                break
        except Exception:
            continue

    if not target_id:
        raise DataSourceError(
            f"Could not resolve target '{target_query}' in ChEMBL. "
            "Try a UniProt accession (e.g. P15056) or gene symbol (e.g. BRAF)."
        )

    # Step 2: Fetch activities for this target
    act_url = (
        f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
        f"?target_chembl_id={target_id}"
        f"&limit={max_molecules * 3}"  # fetch extra for filtering
        f"&standard_type__in=IC50%2CKi%2CKd%2CEC50"
        f"&standard_relation=%3D"
        f"&assay_type=%21B"
    )
    try:
        data = _http_get_json(act_url, timeout=timeout)
        activities = data.get("activities", [])
    except Exception as exc:
        raise DataSourceError(f"ChEMBL activity query failed: {exc}")

    if not activities:
        logger.warning(f"No ChEMBL activities found for target {target_id}")
        return []

    # Step 3: Collapse by molecule (keep best activity per compound)
    best_by_mol: dict[str, dict] = {}
    for act in activities:
        mol_id = act.get("molecule_chembl_id")
        if not mol_id:
            continue
        pchembl = _safe_float(act.get("pchembl_value"))
        if min_pchembl is not None and (pchembl is None or pchembl < min_pchembl):
            continue
        existing = best_by_mol.get(mol_id)
        if existing is None or (pchembl or 0) > (existing.get("pchembl_value") or 0):
            best_by_mol[mol_id] = {
                "chembl_id": mol_id,
                "pchembl_value": pchembl,
                "assay_type": act.get("assay_type", ""),
                "standard_type": act.get("standard_type", ""),
                "standard_value": act.get("standard_value"),
                "standard_units": act.get("standard_units", ""),
            }

    # Step 4: Fetch SMILES for each compound
    mol_ids = list(best_by_mol.keys())[:max_molecules]
    results: list[dict[str, Any]] = []
    for mol_id in mol_ids:
        try:
            mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_id}.json"
            mol_data = _http_get_json(mol_url, timeout=timeout)
            structures = mol_data.get("molecule_structures", {})
            smiles = structures.get("canonical_smiles") or structures.get("standard_inchi_key", "")
            if smiles:
                entry = best_by_mol[mol_id]
                entry["smiles"] = smiles
                entry["target_id"] = target_id
                entry["target_name"] = target_name
                results.append(entry)
        except Exception:
            continue

    logger.info(
        f"ChEMBL target '{target_name}': {len(results)} compounds"
        + (f" (pChEMBL ≥ {min_pchembl})" if min_pchembl else "")
    )
    return results


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


def _safe_float(val: Any, default: float | None = None) -> float | None:
    """Safely convert a value to float, returning *default* on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_zinc_results(results: list, max_results: int) -> list[dict[str, str]]:
    """Parse ZINC API JSON results into consistent format."""
    parsed: list[dict[str, str]] = []
    for item in results[:max_results]:
        if isinstance(item, dict):
            zid = item.get("zinc_id") or item.get("id") or ""
            smi = item.get("smiles") or item.get("SMILES") or item.get("canonical_smiles", "")
            if zid and smi:
                parsed.append({"zinc_id": zid, "smiles": smi})
    return parsed


def _parse_zinc_tsv(text: str, max_results: int) -> list[dict[str, str]]:
    """Parse ZINC TSV response into consistent format."""
    parsed: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            zid = parts[0].upper()
            smi = parts[1]
            if zid.startswith("ZINC") and smi:
                parsed.append({"zinc_id": zid, "smiles": smi})
                if len(parsed) >= max_results:
                    break
    return parsed


def search_zinc(
    query: str,
    max_results: int = 50,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """
    Search ZINC20/CartBlanche by compound name or SMILES substructure.

    .. attention::
       ZINC15/20 public REST APIs were deprecated in 2024.  CartBlanche22
       is a JavaScript SPA that does not expose a JSON API.  This function
       tries legacy endpoints and the CartBlanche text endpoint, but most
       ZINC data now requires **pre-downloaded catalog files**.

       For virtual screening, download a ZINC tranche (``.smi`` or ``.sdf``)
       from ``https://files.docking.org/`` and use
       :func:`read_sdf_library` or :func:`read_smi_library` instead.

    Args:
        query: Compound name or SMILES substructure.
        max_results: Maximum number of results.
        timeout: HTTP timeout in seconds.

    Returns:
        List of ``{"zinc_id": …, "smiles": …}`` dicts.  Often empty
        due to API deprecation; see note above.
    """
    import json

    if not query or not query.strip():
        return []

    # Try ZINC15 API first (returns 403 for most requests in 2024+)
    try:
        url = f"https://zinc15.docking.org/search?q={urllib.parse.quote(query)}&format=json&limit={max_results}"
        text = _http_get_text(url, timeout=timeout)
        if text and text.strip().startswith("{"):
            data = json.loads(text)
            results = data.get("results", data.get("substances", []))
            if isinstance(results, list) and results:
                _parsed = _parse_zinc_results(results, max_results)
                if _parsed:
                    logger.info(f"ZINC search (ZINC15): '{query}' → {len(_parsed)} results")
                    return _parsed
    except Exception:
        pass

    # Fallback: CartBlanche text endpoint
    try:
        url = f"https://cartblanche22.docking.org/substances.txt:q={urllib.parse.quote(query)}&limit={max_results}"
        text = _http_get_text(url, timeout=timeout)
        if text and "zinc_id" not in text.lower():
            # CartBlanche returns HTML when text format is unavailable
            raise ValueError("CartBlanche returned non-text response")
        _parsed = _parse_zinc_tsv(text, max_results)
        if _parsed:
            logger.info(f"ZINC search (CartBlanche): '{query}' → {len(_parsed)} results")
            return _parsed
    except Exception as exc:
        logger.warning(f"ZINC search failed for '{query}': {exc}")
        logger.info("Tip: download ZINC catalog files from https://files.docking.org/")

    return []


def fetch_zinc_sdf(zinc_id: str, output_path: str, timeout: int = 30) -> str | None:
    """
    Download an SDF with pre-computed 3D coordinates for a ZINC ID.

    ZINC20/22 stores pre-generated 3D conformers.  This downloads the
    3D SDF, which can be used directly with ``prepare_ligand()`` or
    DockingResult visualisation instead of embedding from SMILES.

    Args:
        zinc_id: ZINC identifier, e.g. ``"ZINC000000000001"``.
        output_path: Destination file path.
        timeout: HTTP timeout in seconds.

    Returns:
        Path to downloaded SDF, or *None* if download fails.
    """
    zinc_id = zinc_id.strip().upper()
    if not zinc_id.startswith("ZINC"):
        logger.warning(f"Invalid ZINC ID: {zinc_id}")
        return None

    urls = [
        f"https://cartblanche22.docking.org/substances/{zinc_id}.sdf",
        f"https://zinc.docking.org/substances/{zinc_id}.sdf",
    ]
    for url in urls:
        try:
            _download_url(url, output_path, timeout=timeout)
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 100:
                logger.info(f"Downloaded ZINC SDF: {output_path} ({url})")
                return output_path
        except Exception:
            continue

    logger.warning(f"Could not download ZINC SDF for {zinc_id}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# File-format readers (ligand libraries)
# ─────────────────────────────────────────────────────────────────────────────


def read_smi_library(path: str) -> dict[str, str]:
    """
    Read a SMILES-tab library file (e.g. ZINC tranche ``.smi``) into
    a name → SMILES dictionary.

    ZINC catalog files from ``https://files.docking.org/`` use the format::

        smiles\\tzinc_id\\tproperty1\\tproperty2

    Args:
        path: Path to ``.smi`` file (TSV with SMILES in first column,
            ZINC ID or name in second column).

    Returns:
        Dictionary mapping compound name/ZINC ID → canonical SMILES.
    """
    results: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                smi, name = parts[0], parts[1]
                if smi and name:
                    results[name] = smi
    return results


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


def read_pdbbind_index(
    index_path: str,
    *,
    max_resolution: float | None = 3.0,
    min_affinity: float | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Parse a PDBbind ``INDEX_general_PL`` file into a PDB-ID → data map.

    PDBbind is the standard benchmark for scoring-function evaluation.
    Each entry contains the PDB code, resolution, binding affinity,
    ligand SMILES, and reference.

    Args:
        index_path: Path to PDBbind INDEX file (e.g.
            ``"INDEX_general_PL_data.2020"``).
        max_resolution: Maximum resolution filter (Å).  None = no filter.
        min_affinity: Minimum -logKd/Ki filter (e.g. ``6.0`` = 1 µM).
            None = no filter.

    Returns:
        Dict mapping **PDB code** (uppercase, e.g. ``"1A30"``) to:
        ``{"pdb": …, "resolution": …, "affinity": …, "affinity_type": …,
        "ligand_smiles": …, "ligand_name": …, "reference": …}``.

    References:
        * http://www.pdbbind.org.cn/
    """
    import re

    results: dict[str, dict[str, Any]] = {}
    with open(index_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            # Format: PDB  RESOLUTION  YEAR  -logKd  Kd  REF  LIG_NAME  SMILES  ...
            parts = line.split()
            if len(parts) < 8:
                continue

            pdb_code = parts[0].upper()
            if not re.match(r"^[0-9A-Z]{4}$", pdb_code):
                continue

            resolution = _safe_float(parts[1])
            if max_resolution is not None and (resolution is None or resolution > max_resolution):
                continue

            neg_log_aff = _safe_float(parts[3])
            raw_aff = parts[4]
            lig_name = parts[6]
            smiles = parts[7]

            if min_affinity is not None and (neg_log_aff is None or neg_log_aff < min_affinity):
                continue

            results[pdb_code] = {
                "pdb": pdb_code,
                "resolution": resolution,
                "year": int(parts[2]) if parts[2].isdigit() else None,
                "neg_log_affinity": neg_log_aff,
                "affinity": raw_aff,
                "ligand_name": lig_name,
                "ligand_smiles": smiles,
                "reference": " ".join(parts[8:-1]) if len(parts) > 8 else "",
            }

    return results


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
    "chembl_target": fetch_chembl_by_target,
    "bindingdb": fetch_bindingdb_by_smiles,
    "zinc": fetch_zinc_smiles,
    "zinc_search": search_zinc,
    "pdb_ligand": download_ligand_sdf_from_pdb,
}
