"""
Tests for autodock.fetchers — structure and compound database fetchers.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from autodock import fetchers
from autodock.core import DataSourceError, StructureFetchError


# ─────────────────────────────────────────────────────────────────────────────
# Receptor fetchers (mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestDownloadPDB:
    def test_invalid_pdb_id_length(self):
        with pytest.raises(StructureFetchError, match="Invalid PDB ID"):
            fetchers.download_pdb("6LU7A", "/tmp")

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_download_pdb_default_format(self, mock_ensure, mock_dl, tmp_path):
        out = fetchers.download_pdb("6LU7", str(tmp_path))
        assert out.endswith("6LU7.pdb")
        mock_dl.assert_called_once()

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_download_pdb_cif_format(self, mock_ensure, mock_dl, tmp_path):
        out = fetchers.download_pdb("6LU7", str(tmp_path), format="cif")
        assert out.endswith("6LU7.cif")

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_download_pdb_bcif_format(self, mock_ensure, mock_dl, tmp_path):
        out = fetchers.download_pdb("6LU7", str(tmp_path), format="bcif")
        assert out.endswith("6LU7.bcif")

    def test_download_pdb_unsupported_format(self, tmp_path):
        with pytest.raises(StructureFetchError, match="Unsupported PDB format"):
            fetchers.download_pdb("6LU7", str(tmp_path), format="xml")


class TestDownloadAlphaFold:
    @patch("autodock.fetchers._http_get_json")
    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_success(self, mock_ensure, mock_dl, mock_json, tmp_path):
        mock_json.return_value = [
            {
                "pdbUrl": "https://example.com/AF-P68871-F1-model_v6.pdb",
                "cifUrl": "https://example.com/AF-P68871-F1-model_v6.cif",
                "globalMetricValue": 97.0,
            }
        ]
        out = fetchers.download_alphafold("P68871", str(tmp_path))
        assert out.endswith("AF-P68871-F1.pdb")
        mock_dl.assert_called_once_with(
            "https://example.com/AF-P68871-F1-model_v6.pdb",
            out,
        )

    @patch("autodock.fetchers._http_get_json")
    def test_not_found(self, mock_json, tmp_path):
        import urllib.error

        mock_json.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )
        with pytest.raises(StructureFetchError, match="no entry for UniProt ID"):
            fetchers.download_alphafold("P68871", str(tmp_path))

    @patch("autodock.fetchers._http_get_json")
    def test_empty_response(self, mock_json, tmp_path):
        mock_json.return_value = []
        with pytest.raises(StructureFetchError, match="empty result"):
            fetchers.download_alphafold("P68871", str(tmp_path))


class TestDownloadSwissModel:
    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_success(self, mock_ensure, mock_dl, tmp_path):
        outdir = str(tmp_path)

        def side_effect(url, out_path):
            with open(out_path, "w") as fh:
                fh.write("ATOM      1  N   VAL A   1      0.000   0.000   0.000\n")

        mock_dl.side_effect = side_effect
        out = fetchers.download_swissmodel("P68871", outdir)
        assert out.endswith("P68871_SWISSMODEL.pdb")

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_html_fallback(self, mock_ensure, mock_dl, tmp_path):
        """If SWISS-MODEL returns HTML (no model), raise error."""
        outdir = str(tmp_path)

        def side_effect(url, out_path):
            with open(out_path, "w") as fh:
                fh.write("<!DOCTYPE html><html>...</html>")

        mock_dl.side_effect = side_effect
        with pytest.raises(StructureFetchError, match="No SWISS-MODEL structure"):
            fetchers.download_swissmodel("FAKE01", outdir)


class TestFetchUniProtFASTA:
    @patch("autodock.fetchers._http_get_text")
    def test_success(self, mock_text, tmp_path):
        mock_text.return_value = ">sp|P68871|HBB_HUMAN...\nMVHLTPEEKS..."
        out = fetchers.fetch_uniprot_fasta("P68871", str(tmp_path / "out.fasta"))
        assert out.endswith("out.fasta")

    @patch("autodock.fetchers._http_get_text")
    def test_invalid_response(self, mock_text):
        mock_text.return_value = "not a fasta"
        with pytest.raises(DataSourceError, match="Invalid FASTA"):
            fetchers.fetch_uniprot_fasta("P68871")


# ─────────────────────────────────────────────────────────────────────────────
# Ligand fetchers (mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestFetchPubChemSMILES:
    @patch("autodock.fetchers._http_get_json")
    def test_via_rest(self, mock_json):
        mock_json.return_value = {
            "PropertyTable": {"Properties": [{"CanonicalSMILES": "CC(=O)O"}]}
        }
        # Patch pubchempy so we exercise the REST fallback path
        with patch.dict("sys.modules", {"pubchempy": None}):
            smi = fetchers.fetch_pubchem_smiles("aspirin")
        assert smi == "CC(=O)O"

    @patch("autodock.fetchers._http_get_json")
    def test_not_found(self, mock_json):
        mock_json.side_effect = Exception("network error")
        with pytest.raises(DataSourceError):
            fetchers.fetch_pubchem_smiles("notarealcompound12345")


class TestFetchChemblSMILES:
    @patch("autodock.fetchers._http_get_json")
    def test_success(self, mock_json):
        mock_json.return_value = {
            "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"}
        }
        smi = fetchers.fetch_chembl_smiles("CHEMBL25")
        assert "CC(=O)O" in smi

    def test_invalid_id(self):
        with pytest.raises(DataSourceError, match="Invalid ChEMBL ID"):
            fetchers.fetch_chembl_smiles("INVALID")

    @patch("autodock.fetchers._http_get_json")
    def test_missing_smiles(self, mock_json):
        mock_json.return_value = {"molecule_structures": {}}
        with pytest.raises(DataSourceError, match="no SMILES"):
            fetchers.fetch_chembl_smiles("CHEMBL999999")


class TestFetchBindingDB:
    @patch("autodock.fetchers._http_get_json")
    def test_success(self, mock_json):
        mock_json.return_value = {"bdb.hit": "42"}
        result = fetchers.fetch_bindingdb_by_smiles("CC(=O)O")
        assert result["bdb.hit"] == "42"


class TestFetchZincSMILES:
    @patch("autodock.fetchers._http_get_json")
    def test_success(self, mock_json):
        mock_json.return_value = {"smiles": "CC(=O)O"}
        smi = fetchers.fetch_zinc_smiles("ZINC000000000001")
        assert smi == "CC(=O)O"

    def test_invalid_id(self):
        assert fetchers.fetch_zinc_smiles("NOTZINC") is None

    @patch("autodock.fetchers._http_get_json")
    @patch("autodock.fetchers._http_get_text")
    def test_all_fail(self, mock_text, mock_json):
        mock_json.side_effect = Exception("fail")
        mock_text.side_effect = Exception("fail")
        assert fetchers.fetch_zinc_smiles("ZINC000000000001") is None


# ─────────────────────────────────────────────────────────────────────────────
# File-format readers
# ─────────────────────────────────────────────────────────────────────────────


class TestReadSDFLibrary:
    def test_read_sdf(self, tmp_path):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("_Name", "ethanol")
        w = Chem.SDWriter(str(tmp_path / "lib.sdf"))
        w.write(mol)
        w.close()

        lib = fetchers.read_sdf_library(str(tmp_path / "lib.sdf"))
        assert "ethanol" in lib
        assert lib["ethanol"] == "CCO"


class TestReadMolFile:
    def test_read_mol(self, tmp_path):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        w = Chem.SDWriter(str(tmp_path / "test.mol"))
        w.write(mol)
        w.close()

        result = fetchers.read_mol_file(str(tmp_path / "test.mol"))
        assert result is not None

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("fake")
        assert fetchers.read_mol_file(str(path)) is None


# ─────────────────────────────────────────────────────────────────────────────
# Integration / live tests (optional, gated by environment)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("AUTODOCK_SKIP_LIVE_TESTS", "1") == "1",
    reason="Set AUTODOCK_SKIP_LIVE_TESTS=0 to run live API tests",
)
class TestLiveAPIs:
    """Hit real endpoints — useful for validating URL stability."""

    def test_live_alphafold_api(self, tmp_path):
        path = fetchers.download_alphafold("P68871", str(tmp_path))
        assert os.path.getsize(path) > 500

    def test_live_swissmodel(self, tmp_path):
        path = fetchers.download_swissmodel("P68871", str(tmp_path))
        assert os.path.getsize(path) > 500

    def test_live_uniprot_fasta(self, tmp_path):
        text = fetchers.fetch_uniprot_fasta("P68871")
        assert text.startswith(">")

    def test_live_pubchem_smiles(self):
        smi = fetchers.fetch_pubchem_smiles("aspirin")
        assert "C" in smi

    def test_live_chembl_smiles(self):
        smi = fetchers.fetch_chembl_smiles("CHEMBL25")
        assert "C" in smi
