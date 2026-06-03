"""
autodock.cache — Disk cache for receptor / ligand / pocket preparation.
=========================================================
Eliminates redundant computation in batch docking by caching:

  * Receptor PDBQT  (PDBFixer + reduce + Meeko)
  * Ligand PDBQT    (RDKit ETKDG + Meeko / Open Babel)
  * Pocket definitions  (P2Rank + fpocket)

Cache keys are **parameter-sensitive** SHA-256 hashes so that a change in
pH, padding, or any other preparation parameter automatically invalidates
the cached result.

Usage::

    from autodock.cache import ReceptorCache, LigandCache, PocketCache

    rc = ReceptorCache()
    hit = rc.get("6lu7.pdb", ph=7.4, remove_water=True, ...)
    if hit:
        shutil.copy(hit["receptor.pdbqt"], "out/6lu7.pdbqt")
    else:
        prepare_receptor("6lu7.pdb", "out/6lu7.pdbqt", ...)
        rc.put("6lu7.pdb", {"receptor.pdbqt": "out/6lu7.pdbqt"}, ph=7.4, ...)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from autodock.utils import ensure_dir

DEFAULT_CACHE_ROOT = os.path.expanduser("~/.autodock/cache")


def _canonical_json(value: Any) -> str:
    """Return a canonical JSON string for hashing (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash_params(**params: Any) -> str:
    """Hash a set of keyword parameters to a short hex string."""
    return hashlib.sha256(_canonical_json(params).encode()).hexdigest()[:16]


def _hash_file(path: str, limit_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash the first *limit_bytes* of a file."""
    if not os.path.isfile(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
            if limit_bytes and fh.tell() >= limit_bytes:
                break
    return h.hexdigest()[:16]


class _BaseCache:
    """Shared logic for receptor, ligand, and pocket caches."""

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_ROOT
        self.cache_dir = Path(ensure_dir(cache_dir))

    def _entry_dir(self, category: str, key: str) -> Path:
        return self.cache_dir / category / key

    def _meta_path(self, category: str, key: str) -> Path:
        return self._entry_dir(category, key) / "meta.json"

    def _read_meta(self, category: str, key: str) -> dict[str, Any] | None:
        p = self._meta_path(category, key)
        if not p.exists():
            return None
        try:
            with open(p) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def _write_meta(self, category: str, key: str, meta: dict[str, Any]) -> None:
        d = self._entry_dir(category, key)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def clear(self, category: str | None = None) -> int:
        """Remove all cached entries.  Returns number of entries removed."""
        target = self.cache_dir / category if category else self.cache_dir
        if not target.exists():
            return 0
        count = 0
        for item in target.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                count += 1
            elif item.is_file():
                item.unlink()
                count += 1
        return count

    def info(self) -> dict[str, Any]:
        """Return cache statistics."""
        total_bytes = 0
        n_entries = 0
        for root, _dirs, files in os.walk(self.cache_dir):
            for f in files:
                total_bytes += os.path.getsize(os.path.join(root, f))
                if f == "meta.json":
                    n_entries += 1
        return {
            "cache_dir": str(self.cache_dir),
            "n_entries": n_entries,
            "total_bytes": total_bytes,
        }


class ReceptorCache(_BaseCache):
    """Cache for prepared receptor PDBQT files (and optional intermediate PDB)."""

    def _make_key(self, input_path: str, **params: Any) -> str:
        """Generate a cache key from input file hash + parameter hash."""
        file_hash = _hash_file(input_path)
        param_hash = _hash_params(**params)
        return f"{file_hash}_{param_hash}"

    def get(
        self,
        input_path: str,
        **params: Any,
    ) -> dict[str, str] | None:
        """Return cached file paths if all exist, else None.

        Returns a dict mapping logical names → absolute paths, e.g.
        ``{"receptor.pdbqt": "/path/to/cache/.../receptor.pdbqt"}``.
        """
        key = self._make_key(input_path, **params)
        meta = self._read_meta("receptors", key)
        if not meta:
            return None
        entry_dir = self._entry_dir("receptors", key)
        result: dict[str, str] = {}
        for name in meta.get("files", []):
            p = entry_dir / name
            if not p.exists() or p.stat().st_size == 0:
                return None
            result[name] = str(p)
        return result

    def put(
        self,
        input_path: str,
        files: dict[str, str],
        **params: Any,
    ) -> dict[str, str]:
        """Store prepared files in the cache.

        Args:
            input_path: Original input PDB file (used for key generation).
            files: Mapping of logical name → source file path.
            **params: Preparation parameters (stored in meta for debugging).

        Returns:
            Mapping of logical name → cached file path.
        """
        key = self._make_key(input_path, **params)
        entry_dir = self._entry_dir("receptors", key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, str] = {}
        for name, src in files.items():
            dest = entry_dir / name
            shutil.copy2(src, dest)
            result[name] = str(dest)
        self._write_meta(
            "receptors",
            key,
            {
                "created": datetime.now().isoformat(),
                "files": list(files.keys()),
                "input_hash": _hash_file(input_path),
                "params": params,
            },
        )
        return result


class LigandCache(_BaseCache):
    """Cache for prepared ligand PDBQT files."""

    def _make_key(self, smiles: str, **params: Any) -> str:
        """Generate a cache key from canonical SMILES hash + parameter hash."""
        smiles_hash = hashlib.sha256(smiles.encode()).hexdigest()[:16]
        param_hash = _hash_params(**params)
        return f"{smiles_hash}_{param_hash}"

    def get(self, smiles: str, **params: Any) -> dict[str, str] | None:
        """Return cached file paths if all exist, else None."""
        key = self._make_key(smiles, **params)
        meta = self._read_meta("ligands", key)
        if not meta:
            return None
        entry_dir = self._entry_dir("ligands", key)
        result: dict[str, str] = {}
        for name in meta.get("files", []):
            p = entry_dir / name
            if not p.exists() or p.stat().st_size == 0:
                return None
            result[name] = str(p)
        return result

    def put(
        self,
        smiles: str,
        files: dict[str, str],
        **params: Any,
    ) -> dict[str, str]:
        """Store prepared files in the cache."""
        key = self._make_key(smiles, **params)
        entry_dir = self._entry_dir("ligands", key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, str] = {}
        for name, src in files.items():
            dest = entry_dir / name
            shutil.copy2(src, dest)
            result[name] = str(dest)
        self._write_meta(
            "ligands",
            key,
            {
                "created": datetime.now().isoformat(),
                "files": list(files.keys()),
                "smiles_hash": hashlib.sha256(smiles.encode()).hexdigest()[:16],
                "params": params,
            },
        )
        return result


class PocketCache(_BaseCache):
    """Cache for pocket detection results (JSON-serialised list of dicts)."""

    def _make_key(self, receptor_pdb: str, **params: Any) -> str:
        file_hash = _hash_file(receptor_pdb)
        param_hash = _hash_params(**params)
        return f"{file_hash}_{param_hash}"

    def get(
        self,
        receptor_pdb: str,
        **params: Any,
    ) -> list[dict[str, Any]] | None:
        """Return cached pocket list if valid, else None."""
        key = self._make_key(receptor_pdb, **params)
        meta = self._read_meta("pockets", key)
        if not meta:
            return None
        entry_dir = self._entry_dir("pockets", key)
        pockets_path = entry_dir / "pockets.json"
        if not pockets_path.exists() or pockets_path.stat().st_size == 0:
            return None
        try:
            with open(pockets_path) as fh:
                data = json.load(fh)
            if isinstance(data, list) and len(data) > 0:
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return None

    def put(
        self,
        receptor_pdb: str,
        pockets: list[dict[str, Any]],
        **params: Any,
    ) -> str:
        """Store pocket list in the cache.  Returns the cached JSON path."""
        key = self._make_key(receptor_pdb, **params)
        entry_dir = self._entry_dir("pockets", key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        pockets_path = entry_dir / "pockets.json"
        with open(pockets_path, "w") as fh:
            json.dump(pockets, fh, indent=2, default=str)
        self._write_meta(
            "pockets",
            key,
            {
                "created": datetime.now().isoformat(),
                "files": ["pockets.json"],
                "input_hash": _hash_file(receptor_pdb),
                "params": params,
            },
        )
        return str(pockets_path)


def clear_all_caches(cache_dir: str | None = None) -> dict[str, int]:
    """Clear receptor, ligand, and pocket caches.

    Returns a dict with counts per category.
    """
    rc = ReceptorCache(cache_dir)
    lc = LigandCache(cache_dir)
    pc = PocketCache(cache_dir)
    return {
        "receptors": rc.clear("receptors"),
        "ligands": lc.clear("ligands"),
        "pockets": pc.clear("pockets"),
    }
