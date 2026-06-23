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
        assert out.endswith("6LU7.cif")  # default changed from pdb to cif
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
        assert out.endswith("AF-P68871-F1.cif")
        mock_dl.assert_called_once_with(
            "https://example.com/AF-P68871-F1-model_v6.cif",
            out,
        )

    @patch("autodock.fetchers._http_get_json")
    def test_not_found(self, mock_json, tmp_path):
        import urllib.error

        mock_json.side_effect = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
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

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_download_url_raises_re_raises(self, mock_ensure, mock_dl, tmp_path):
        """Cover lines 677-678: _download_url raises StructureFetchError directly."""
        mock_dl.side_effect = StructureFetchError("fail")
        with pytest.raises(StructureFetchError, match="fail"):
            fetchers.download_swissmodel("P68871", str(tmp_path))


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

    @patch("autodock.fetchers._http_get_text")
    def test_returns_text(self, mock_text):
        """Cover line 712: fetch_uniprot_fasta returns string when output_path is None."""
        mock_text.return_value = ">sp|P68871|HBB_HUMAN\nMVHLTPEEKS"
        result = fetchers.fetch_uniprot_fasta("P68871")
        assert result == ">sp|P68871|HBB_HUMAN\nMVHLTPEEKS"


# ─────────────────────────────────────────────────────────────────────────────
# Ligand fetchers (mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestFetchPubChemSMILES:
    @patch("autodock.fetchers._http_get_json")
    def test_via_rest(self, mock_json):
        mock_json.return_value = {"PropertyTable": {"Properties": [{"CanonicalSMILES": "CC(=O)O"}]}}
        # Patch pubchempy so we exercise the REST fallback path
        with patch.dict("sys.modules", {"pubchempy": None}):
            smi = fetchers.fetch_pubchem_smiles("aspirin")
        assert smi == "CC(=O)O"

    @patch("autodock.fetchers._http_get_json")
    def test_not_found(self, mock_json):
        import urllib.error
        from unittest.mock import MagicMock

        mock_json.side_effect = urllib.error.URLError("network error")
        mock_pcp = MagicMock()
        mock_pcp.get_compounds.side_effect = urllib.error.URLError("network error")
        with patch.dict("sys.modules", {"pubchempy": mock_pcp}):
            with pytest.raises(DataSourceError):
                fetchers.fetch_pubchem_smiles("notarealcompound12345")

    @patch("autodock.fetchers._http_get_json")
    def test_empty_properties(self, mock_json):
        """Cover line 794: PubChem returns empty properties list."""
        mock_json.return_value = {"PropertyTable": {"Properties": []}}
        with patch.dict("sys.modules", {"pubchempy": None}):
            with pytest.raises(DataSourceError, match="no record"):
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

    @patch("autodock.fetchers._http_get_json")
    def test_lookup_error(self, mock_json):
        """Cover lines 885-891: _http_get_json raises in fetch_chembl_smiles."""
        import urllib.error

        mock_json.side_effect = urllib.error.URLError("fail")
        with pytest.raises(DataSourceError, match="lookup failed"):
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
        import urllib.error

        mock_json.side_effect = urllib.error.URLError("fail")
        mock_text.side_effect = urllib.error.URLError("fail")
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


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestHttpHelpers:
    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_http_get_json(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"key": "value"}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        result = fetchers._http_get_json("https://example.com/api")
        assert result == {"key": "value"}

    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_http_get_text(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"hello"
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        result = fetchers._http_get_text("https://example.com/text")
        assert result == "hello"

    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_http_get_text_retryable_then_fail(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 503, "Service Unavailable", {}, None
        )
        with pytest.raises(urllib.error.HTTPError):
            fetchers._http_get_text("https://example.com/text")

    @patch("autodock.fetchers.urllib.request.urlretrieve")
    def test_download_url_success(self, mock_urlretrieve, tmp_path):
        out = str(tmp_path / "out.txt")
        with open(out, "w") as fh:
            fh.write("x" * 100)
        fetchers._download_url("https://example.com/file", out)
        mock_urlretrieve.assert_called_once()

    @patch("autodock.fetchers.urllib.request.urlretrieve")
    def test_download_url_too_small(self, mock_urlretrieve, tmp_path):
        out = str(tmp_path / "out.txt")
        with open(out, "w") as fh:
            fh.write("x" * 10)
        with pytest.raises(StructureFetchError, match="too small"):
            fetchers._download_url("https://example.com/file", out)

    @patch("autodock.fetchers.urllib.request.urlretrieve")
    def test_download_url_html(self, mock_urlretrieve, tmp_path):
        out = str(tmp_path / "out.txt")
        with open(out, "w") as fh:
            fh.write("<!DOCTYPE html><html>...</html>" + "x" * 100)
        with pytest.raises(StructureFetchError, match="HTML instead"):
            fetchers._download_url("https://example.com/file", out)

    @patch("autodock.fetchers.urllib.request.urlretrieve")
    def test_download_url_error(self, mock_urlretrieve):
        import urllib.error

        mock_urlretrieve.side_effect = urllib.error.URLError("fail")
        with pytest.raises(StructureFetchError, match="Download failed"):
            fetchers._download_url("https://example.com/file", "/tmp/out.txt")

    @patch("autodock.fetchers.urllib.request.urlretrieve")
    def test_download_url_retryable_then_fail(self, mock_urlretrieve):
        import urllib.error

        mock_urlretrieve.side_effect = [
            urllib.error.HTTPError("url", 503, "Service Unavailable", {}, None),
            urllib.error.HTTPError("url", 502, "Bad Gateway", {}, None),
            urllib.error.URLError("fail"),
        ]
        with pytest.raises(StructureFetchError, match="Download failed"):
            fetchers._download_url("https://example.com/file", "/tmp/out.txt")
        assert mock_urlretrieve.call_count == 3


class TestIsRetryable:
    def test_http_error_429(self):
        import urllib.error

        exc = urllib.error.HTTPError("url", 429, "Too Many", {}, None)
        assert fetchers._is_retryable(exc) is True

    def test_http_error_404(self):
        import urllib.error

        exc = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        assert fetchers._is_retryable(exc) is False

    def test_urlerror(self):
        import urllib.error

        exc = urllib.error.URLError("fail")
        assert fetchers._is_retryable(exc) is True

    def test_oserror(self):
        exc = OSError("fail")
        assert fetchers._is_retryable(exc) is True

    def test_generic_exception(self):
        exc = ValueError("fail")
        assert fetchers._is_retryable(exc) is False


class TestSafeFloat:
    def test_none(self):
        assert fetchers._safe_float(None) is None

    def test_none_with_default(self):
        assert fetchers._safe_float(None, default=0.0) == 0.0

    def test_valid(self):
        assert fetchers._safe_float("3.14") == 3.14

    def test_value_error(self):
        assert fetchers._safe_float("abc", default=-1.0) == -1.0

    def test_type_error(self):
        assert fetchers._safe_float([1, 2], default=-1.0) == -1.0


class TestParseZincResults:
    def test_valid(self):
        results = [
            {"zinc_id": "ZINC1", "smiles": "CCO"},
            {"id": "ZINC2", "canonical_smiles": "CCC"},
        ]
        parsed = fetchers._parse_zinc_results(results, 10)
        assert len(parsed) == 2
        assert parsed[0] == {"zinc_id": "ZINC1", "smiles": "CCO"}

    def test_missing_smiles_skipped(self):
        results = [
            {"zinc_id": "ZINC1", "smiles": "CCO"},
            {"zinc_id": "ZINC2"},
        ]
        parsed = fetchers._parse_zinc_results(results, 10)
        assert len(parsed) == 1

    def test_not_dict_skipped(self):
        results = ["not_a_dict", {"zinc_id": "ZINC1", "smiles": "CCO"}]
        parsed = fetchers._parse_zinc_results(results, 10)
        assert len(parsed) == 1

    def test_max_results(self):
        results = [
            {"zinc_id": "ZINC1", "smiles": "CCO"},
            {"zinc_id": "ZINC2", "smiles": "CCC"},
        ]
        parsed = fetchers._parse_zinc_results(results, 1)
        assert len(parsed) == 1


class TestParseZincTsv:
    def test_valid(self):
        text = "ZINC0001\tCCO\tprop1\nZINC0002\tCCC\n"
        parsed = fetchers._parse_zinc_tsv(text, 10)
        assert len(parsed) == 2

    def test_comments_and_empty_skipped(self):
        text = "# comment\n\nZINC0001\tCCO\n"
        parsed = fetchers._parse_zinc_tsv(text, 10)
        assert len(parsed) == 1

    def test_non_zinc_prefix_skipped(self):
        text = "ABC\tCCO\nZINC0001\tCCC\n"
        parsed = fetchers._parse_zinc_tsv(text, 10)
        assert len(parsed) == 1

    def test_single_column_skipped(self):
        text = "ZINC0001\n"
        parsed = fetchers._parse_zinc_tsv(text, 10)
        assert len(parsed) == 0

    def test_max_results(self):
        text = "ZINC0001\tCCO\nZINC0002\tCCC\nZINC0003\tCCN\n"
        parsed = fetchers._parse_zinc_tsv(text, 2)
        assert len(parsed) == 2


class TestHttpHelpersRetry:
    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_http_get_json_retry_then_success(self, mock_urlopen):
        import urllib.error

        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("url", 429, "Too Many", {}, None),
            mock_resp,
        ]
        result = fetchers._http_get_json("https://example.com/api")
        assert result == {"ok": True}
        assert mock_urlopen.call_count == 2

    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_http_get_text_retry_then_success(self, mock_urlopen):
        import urllib.error

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"hello"
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("url", 503, "Service Unavailable", {}, None),
            mock_resp,
        ]
        result = fetchers._http_get_text("https://example.com/text")
        assert result == "hello"
        assert mock_urlopen.call_count == 2

    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_http_get_json_non_retryable_raises(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        with pytest.raises(urllib.error.HTTPError):
            fetchers._http_get_json("https://example.com/api")


class TestDownloadPDBFallback:
    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_cif_to_pdb_fallback(self, mock_ensure, mock_dl, tmp_path):
        """When CIF download fails, fallback to PDB format."""

        def _dl(url, path):
            if url.endswith(".cif"):
                raise StructureFetchError("cif fail")
            # write a valid file for pdb fallback
            with open(path, "w") as fh:
                fh.write("x" * 100)

        mock_dl.side_effect = _dl
        out = fetchers.download_pdb("6LU7", str(tmp_path), format="cif")
        assert out.endswith("6LU7.pdb")

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_cif_to_pdb_fallback_also_fails(self, mock_ensure, mock_dl, tmp_path):
        """Cover lines 571-572: CIF fails and PDB fallback also fails."""
        mock_dl.side_effect = StructureFetchError("fail")
        with pytest.raises(StructureFetchError, match="fail"):
            fetchers.download_pdb("6LU7", str(tmp_path), format="cif")

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_pdb_to_cif_fallback(self, mock_ensure, mock_dl, tmp_path):
        """When PDB download fails, fallback to CIF format."""

        def _dl(url, path):
            if url.endswith(".pdb"):
                raise StructureFetchError("pdb fail")
            with open(path, "w") as fh:
                fh.write("x" * 100)

        mock_dl.side_effect = _dl
        out = fetchers.download_pdb("6LU7", str(tmp_path), format="pdb")
        assert out.endswith("6LU7.cif")

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_pdb_to_cif_fallback_also_fails(self, mock_ensure, mock_dl, tmp_path):
        """Cover line 580: PDB fails and CIF fallback also fails."""
        mock_dl.side_effect = StructureFetchError("fail")
        with pytest.raises(StructureFetchError, match="fail"):
            fetchers.download_pdb("6LU7", str(tmp_path), format="pdb")

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_bcif_fallback_raises(self, mock_ensure, mock_dl, tmp_path):
        """Cover line 580: bcif format has no fallback."""
        mock_dl.side_effect = StructureFetchError("fail")
        with pytest.raises(StructureFetchError, match="fail"):
            fetchers.download_pdb("6LU7", str(tmp_path), format="bcif")


# ─────────────────────────────────────────────────────────────────────────────
# RCSB PDB search / ranking
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchPDBByName:
    @patch("autodock.fetchers.urllib.request.urlopen")
    @patch("autodock.fetchers._rank_pdb_entries")
    def test_success(self, mock_rank, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"result_set": [{"identifier": "1abc"}, {"identifier": "2def"}]}
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        mock_rank.return_value = ["2DEF", "1ABC"]
        result = fetchers.search_pdb_by_name("kinase")
        assert result == ["2DEF", "1ABC"]

    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_empty(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"result_set": []}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        with pytest.raises(StructureFetchError, match="No PDB structures found"):
            fetchers.search_pdb_by_name("fake")

    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_network_error(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("fail")
        with pytest.raises(StructureFetchError, match="RCSB search query failed"):
            fetchers.search_pdb_by_name("kinase")

    @patch("autodock.fetchers.urllib.request.urlopen")
    @patch("autodock.fetchers._rank_pdb_entries")
    def test_no_filters(self, mock_rank, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"result_set": [{"identifier": "1abc"}]}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        mock_rank.return_value = ["1ABC"]
        result = fetchers.search_pdb_by_name(
            "kinase", max_resolution=100, require_ligand=False, method=None
        )
        assert result == ["1ABC"]


class TestRankPDBEntries:
    @patch("autodock.fetchers._fetch_entry_metadata")
    def test_ranking(self, mock_meta):
        mock_meta.side_effect = [
            {"resolution": 2.0, "r_free": 0.2, "deposit_year": 2020},
            {"resolution": 1.5, "r_free": 0.25, "deposit_year": 2019},
            None,
        ]
        result = fetchers._rank_pdb_entries(["1ABC", "2DEF", "3GHI"])
        assert result == ["2DEF", "1ABC"]

    @patch("autodock.fetchers._fetch_entry_metadata")
    def test_all_meta_fail(self, mock_meta):
        import urllib.error

        mock_meta.side_effect = urllib.error.URLError("fail")
        result = fetchers._rank_pdb_entries(["1ABC"])
        assert result == ["1ABC"]

    def test_empty(self):
        assert fetchers._rank_pdb_entries([]) == []


class TestFetchEntryMetadata:
    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "rcsb_entry_info": {"resolution_combined": [1.8]},
                "refine": [{"ls_R_factor_R_free": 0.22}],
                "rcsb_accession_info": {"deposit_date": "2019-05-01"},
            }
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        result = fetchers._fetch_entry_metadata("1abc")
        assert result["resolution"] == 1.8
        assert result["r_free"] == 0.22
        assert result["deposit_year"] == 2019
        assert result["pdb_id"] == "1ABC"

    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_error(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("fail")
        assert fetchers._fetch_entry_metadata("1abc") is None

    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_defaults(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        result = fetchers._fetch_entry_metadata("1abc")
        assert result["resolution"] == 99.0
        assert result["r_free"] == 99.0
        assert result["deposit_year"] == 1900

    @patch("autodock.fetchers.urllib.request.urlopen")
    def test_resolution_float(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"rcsb_entry_info": {"resolution_combined": 2.5}}
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        result = fetchers._fetch_entry_metadata("1abc")
        assert result["resolution"] == 2.5


class TestFindBestPDBStructure:
    def test_direct_pdb_id(self):
        assert fetchers.find_best_pdb_structure("6LU7") == "6LU7"

    @patch("autodock.fetchers.search_pdb_by_name")
    def test_search(self, mock_search):
        mock_search.return_value = ["1ABC"]
        assert fetchers.find_best_pdb_structure("kinase") == "1ABC"

    @patch("autodock.fetchers.search_pdb_by_name")
    def test_empty(self, mock_search):
        mock_search.return_value = []
        assert fetchers.find_best_pdb_structure("kinase") is None

    @patch("autodock.fetchers.search_pdb_by_name")
    def test_four_chars_no_digit(self, mock_search):
        mock_search.return_value = ["1ABC"]
        assert fetchers.find_best_pdb_structure("ABCD") == "1ABC"


class TestResolveToUniprot:
    def test_direct(self):
        assert fetchers._resolve_to_uniprot("P68871") == "P68871"

    @patch("autodock.fetchers._http_get_json")
    def test_reviewed(self, mock_json):
        mock_json.return_value = {"results": [{"primaryAccession": "P68871"}]}
        assert fetchers._resolve_to_uniprot("hemoglobin") == "P68871"

    @patch("autodock.fetchers._http_get_json")
    def test_unreviewed_fallback(self, mock_json):
        import urllib.error

        def side_effect(url, timeout):
            if "reviewed:true" in url:
                raise urllib.error.URLError("fail")
            return {"results": [{"primaryAccession": "P68871"}]}

        mock_json.side_effect = side_effect
        assert fetchers._resolve_to_uniprot("hemoglobin") == "P68871"

    @patch("autodock.fetchers._http_get_json")
    def test_no_match(self, mock_json):
        mock_json.return_value = {"results": []}
        assert fetchers._resolve_to_uniprot("fake") is None

    def test_second_regex(self):
        # Covers line 366: second UniProt ID pattern (e.g. A1BCDEF)
        assert fetchers._resolve_to_uniprot("A1BCDEF") == "A1BCDEF"


class TestFetchProteinStructure:
    @patch("autodock.fetchers.find_best_pdb_structure")
    @patch("autodock.fetchers.download_pdb")
    def test_pdb(self, mock_dl, mock_find, tmp_path):
        mock_find.return_value = "6LU7"
        mock_dl.return_value = str(tmp_path / "6LU7.cif")
        result = fetchers.fetch_protein_structure("6LU7", output_dir=str(tmp_path))
        assert result == str(tmp_path / "6LU7.cif")

    @patch("autodock.fetchers.find_best_pdb_structure")
    @patch("autodock.fetchers._resolve_to_uniprot")
    @patch("autodock.fetchers.download_alphafold")
    def test_alphafold(self, mock_af, mock_resolve, mock_find, tmp_path):
        mock_find.return_value = None
        mock_resolve.return_value = "P68871"
        mock_af.return_value = str(tmp_path / "AF.cif")
        result = fetchers.fetch_protein_structure("hemoglobin", output_dir=str(tmp_path))
        assert result == str(tmp_path / "AF.cif")

    @patch("autodock.fetchers.find_best_pdb_structure")
    @patch("autodock.fetchers._resolve_to_uniprot")
    @patch("autodock.fetchers.download_alphafold")
    @patch("autodock.fetchers.download_swissmodel")
    def test_swissmodel_fallback(self, mock_sm, mock_af, mock_resolve, mock_find, tmp_path):
        mock_find.return_value = None
        mock_resolve.return_value = "P68871"
        mock_af.side_effect = StructureFetchError("fail")
        mock_sm.return_value = str(tmp_path / "SM.pdb")
        result = fetchers.fetch_protein_structure("hemoglobin", output_dir=str(tmp_path))
        assert result == str(tmp_path / "SM.pdb")

    @patch("autodock.fetchers.find_best_pdb_structure")
    @patch("autodock.fetchers._resolve_to_uniprot")
    @patch("autodock.fetchers.download_alphafold")
    @patch("autodock.fetchers.download_swissmodel")
    def test_alphafold_low_quality_swissmodel(
        self, mock_sm, mock_af, mock_resolve, mock_find, tmp_path
    ):
        mock_find.return_value = None
        mock_resolve.return_value = "P68871"
        mock_af.return_value = str(tmp_path / "AF.cif")
        mock_sm.return_value = str(tmp_path / "SM.pdb")
        with patch("autodock.alphafold_tools.assess_alphafold_quality") as mock_q:
            mock_q.return_value = {"suitable_for_docking": False}
            result = fetchers.fetch_protein_structure("hemoglobin", output_dir=str(tmp_path))
            assert result == str(tmp_path / "SM.pdb")

    @patch("autodock.fetchers.find_best_pdb_structure")
    @patch("autodock.fetchers._resolve_to_uniprot")
    @patch("autodock.fetchers.download_alphafold")
    def test_alphafold_quality_exception(self, mock_af, mock_resolve, mock_find, tmp_path):
        mock_find.return_value = None
        mock_resolve.return_value = "P68871"
        af_path = str(tmp_path / "AF.cif")
        mock_af.return_value = af_path
        with patch("autodock.alphafold_tools.assess_alphafold_quality") as mock_q:
            mock_q.side_effect = ValueError("fail")
            result = fetchers.fetch_protein_structure("hemoglobin", output_dir=str(tmp_path))
            assert result == af_path

    @patch("autodock.fetchers.find_best_pdb_structure")
    def test_no_fallback(self, mock_find, tmp_path):
        mock_find.return_value = None
        with pytest.raises(StructureFetchError, match="fallbacks disabled"):
            fetchers.fetch_protein_structure(
                "hemoglobin",
                output_dir=str(tmp_path),
                fallback_alphafold=False,
                fallback_swissmodel=False,
            )

    @patch("autodock.fetchers.find_best_pdb_structure")
    @patch("autodock.fetchers._resolve_to_uniprot")
    def test_unresolved(self, mock_resolve, mock_find, tmp_path):
        mock_find.return_value = None
        mock_resolve.return_value = None
        with pytest.raises(StructureFetchError, match="Could not obtain"):
            fetchers.fetch_protein_structure("fake", output_dir=str(tmp_path))


# ─────────────────────────────────────────────────────────────────────────────
# Ligand fetchers (supplemental)
# ─────────────────────────────────────────────────────────────────────────────


class TestDownloadLigandSDF:
    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_ideal(self, mock_ensure, mock_dl, tmp_path):
        result = fetchers.download_ligand_sdf_from_pdb("ATP", str(tmp_path))
        assert result.endswith("ATP.sdf")

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_model_fallback(self, mock_ensure, mock_dl, tmp_path):
        def side_effect(url, out):
            if "_ideal" in url:
                raise StructureFetchError("fail")

        mock_dl.side_effect = side_effect
        result = fetchers.download_ligand_sdf_from_pdb("ATP", str(tmp_path))
        assert result.endswith("ATP.sdf")

    @patch("autodock.fetchers._download_url")
    @patch("autodock.fetchers.ensure_dir")
    def test_both_fail(self, mock_ensure, mock_dl, tmp_path):
        mock_dl.side_effect = StructureFetchError("fail")
        with pytest.raises(StructureFetchError):
            fetchers.download_ligand_sdf_from_pdb("ATP", str(tmp_path))


class TestFetchPubchem:
    @patch("autodock.fetchers._download_url")
    def test_fetch_pubchem_sdf(self, mock_dl, tmp_path):
        out = str(tmp_path / "out.sdf")
        result = fetchers.fetch_pubchem_sdf("2244", out)
        assert result == out

    def test_fetch_pubchem_smiles_via_pubchempy(self):
        mock_pcp = MagicMock()
        mock_compound = MagicMock()
        mock_compound.connectivity_smiles = "CC(=O)O"
        mock_pcp.get_compounds.return_value = [mock_compound]
        with patch.dict("sys.modules", {"pubchempy": mock_pcp}):
            smi = fetchers.fetch_pubchem_smiles("aspirin")
        assert smi == "CC(=O)O"


class TestFetchCompoundSDF:
    @patch("autodock.fetchers.fetch_pubchem_sdf")
    def test_via_pubchempy(self, mock_sdf, tmp_path):
        mock_pcp = MagicMock()
        mock_compound = MagicMock()
        mock_compound.cid = "2244"
        mock_pcp.get_compounds.return_value = [mock_compound]
        out = str(tmp_path / "out.sdf")
        mock_sdf.return_value = out
        with patch.dict("sys.modules", {"pubchempy": mock_pcp}):
            result = fetchers.fetch_compound_sdf_by_name("aspirin", out)
        assert result == out

    @patch("autodock.fetchers.fetch_pubchem_smiles")
    def test_fallback_gen3d(self, mock_smiles, tmp_path):
        mock_pcp = MagicMock()
        mock_pcp.get_compounds.return_value = []  # empty → triggers fallback
        with patch.dict("sys.modules", {"pubchempy": mock_pcp}):
            mock_smiles.return_value = "CCO"
            out = str(tmp_path / "out.sdf")
            result = fetchers.fetch_compound_sdf_by_name("ethanol", out)
            assert result == out

    @patch("autodock.fetchers.fetch_pubchem_smiles")
    def test_invalid_smiles(self, mock_smiles, tmp_path):
        mock_pcp = MagicMock()
        mock_pcp.get_compounds.return_value = []  # empty → triggers fallback
        with patch.dict("sys.modules", {"pubchempy": mock_pcp}):
            mock_smiles.return_value = "invalid_smiles"
            out = str(tmp_path / "out.sdf")
            with pytest.raises(DataSourceError, match="Could not generate 3D"):
                fetchers.fetch_compound_sdf_by_name("fake", out)


class TestFetchChembl:
    @patch("autodock.fetchers._download_url")
    def test_fetch_chembl_sdf(self, mock_dl, tmp_path):
        out = str(tmp_path / "out.sdf")
        result = fetchers.fetch_chembl_sdf("CHEMBL25", out)
        assert result == out


class TestFetchChemblByTarget:
    @patch("autodock.fetchers._http_get_json")
    def test_success(self, mock_json):
        mock_json.side_effect = [
            {"targets": [{"target_chembl_id": "CHEMBL123", "pref_name": "Test"}]},
            {
                "activities": [
                    {
                        "molecule_chembl_id": "CHEMBL25",
                        "pchembl_value": 7.5,
                        "assay_type": "B",
                        "standard_type": "IC50",
                    }
                ]
            },
            {"molecule_structures": {"canonical_smiles": "CCO"}},
        ]
        result = fetchers.fetch_chembl_by_target("P15056", max_molecules=1)
        assert len(result) == 1
        assert result[0]["chembl_id"] == "CHEMBL25"

    @patch("autodock.fetchers._http_get_json")
    def test_no_target(self, mock_json):
        mock_json.return_value = {"targets": []}
        with pytest.raises(DataSourceError, match="Could not resolve target"):
            fetchers.fetch_chembl_by_target("FAKE")

    @patch("autodock.fetchers._http_get_json")
    def test_no_activities(self, mock_json):
        mock_json.side_effect = [
            {"targets": [{"target_chembl_id": "CHEMBL123", "pref_name": "Test"}]},
            {"activities": []},
        ]
        result = fetchers.fetch_chembl_by_target("P15056")
        assert result == []

    @patch("autodock.fetchers._http_get_json")
    def test_min_pchembl_filter(self, mock_json):
        mock_json.side_effect = [
            {"targets": [{"target_chembl_id": "CHEMBL123", "pref_name": "Test"}]},
            {
                "activities": [
                    {
                        "molecule_chembl_id": "CHEMBL25",
                        "pchembl_value": 5.0,
                        "assay_type": "B",
                        "standard_type": "IC50",
                    },
                    {
                        "molecule_chembl_id": "CHEMBL26",
                        "pchembl_value": 8.0,
                        "assay_type": "B",
                        "standard_type": "IC50",
                    },
                ]
            },
            {"molecule_structures": {"canonical_smiles": "CCO"}},
        ]
        result = fetchers.fetch_chembl_by_target("P15056", max_molecules=1, min_pchembl=6.0)
        assert len(result) == 1
        assert result[0]["chembl_id"] == "CHEMBL26"

    @patch("autodock.fetchers._http_get_json")
    def test_inchi_fallback(self, mock_json):
        mock_json.side_effect = [
            {"targets": [{"target_chembl_id": "CHEMBL123", "pref_name": "Test"}]},
            {
                "activities": [
                    {
                        "molecule_chembl_id": "CHEMBL25",
                        "pchembl_value": 7.5,
                        "assay_type": "B",
                        "standard_type": "IC50",
                    }
                ]
            },
            {"molecule_structures": {"standard_inchi_key": "ABC"}},
        ]
        result = fetchers.fetch_chembl_by_target("P15056", max_molecules=1)
        assert len(result) == 1
        assert result[0]["smiles"] == "ABC"


class TestFetchBindingDBErrors:
    @patch("autodock.fetchers._http_get_json")
    def test_error(self, mock_json):
        import urllib.error

        mock_json.side_effect = urllib.error.URLError("fail")
        with pytest.raises(DataSourceError, match="BindingDB query failed"):
            fetchers.fetch_bindingdb_by_smiles("CCO")


class TestFetchZinc:
    @patch("autodock.fetchers._http_get_json")
    def test_fetch_zinc_smiles_json(self, mock_json):
        mock_json.return_value = {"smiles": "CCO"}
        assert fetchers.fetch_zinc_smiles("ZINC1") == "CCO"

    @patch("autodock.fetchers._http_get_json")
    @patch("autodock.fetchers._http_get_text")
    def test_fetch_zinc_smiles_text(self, mock_text, mock_json):
        mock_json.return_value = {}  # JSON endpoint returns no smiles
        mock_text.return_value = "ZINC1\tCCO\n"
        assert fetchers.fetch_zinc_smiles("ZINC1") == "CCO"

    def test_search_zinc_empty(self):
        assert fetchers.search_zinc("") == []

    @patch("autodock.fetchers._http_get_text")
    def test_search_zinc_json_path(self, mock_text):
        mock_text.return_value = '{"results": [{"zinc_id": "ZINC1", "smiles": "CCO"}]}'
        result = fetchers.search_zinc("ethanol", max_results=1)
        assert len(result) == 1
        assert result[0]["zinc_id"] == "ZINC1"

    @patch("autodock.fetchers._http_get_text")
    def test_search_zinc_tsv_path(self, mock_text):
        mock_text.return_value = "zinc_id\tsmiles\nZINC1\tCCO\nZINC2\tCCC\n"
        result = fetchers.search_zinc("ethanol", max_results=2)
        assert len(result) == 2

    @patch("autodock.fetchers._download_url")
    def test_fetch_zinc_sdf(self, mock_dl, tmp_path):
        out = str(tmp_path / "zinc.sdf")
        with open(out, "w") as fh:
            fh.write("x" * 200)
        result = fetchers.fetch_zinc_sdf("ZINC1", out)
        assert result == out

    def test_fetch_zinc_sdf_invalid_id(self):
        """Cover lines 1277-1278: invalid ZINC ID returns None."""
        result = fetchers.fetch_zinc_sdf("NOTZINC", "/tmp/fake.sdf")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# File-format readers (supplemental)
# ─────────────────────────────────────────────────────────────────────────────


class TestReadLibraries:
    def test_read_smi_library(self, tmp_path):
        path = tmp_path / "lib.smi"
        path.write_text("CCO\tethanol\nCCC\tpropane\n# comment\n\tbadline\n")
        result = fetchers.read_smi_library(str(path))
        assert result == {"ethanol": "CCO", "propane": "CCC"}

    def test_read_smi_library_empty_and_whitespace(self, tmp_path):
        path = tmp_path / "lib.smi"
        path.write_text("\n  \nCCO\tethanol\n\n")
        result = fetchers.read_smi_library(str(path))
        assert result == {"ethanol": "CCO"}

    def test_read_smi_library_no_tab(self, tmp_path):
        path = tmp_path / "lib.smi"
        path.write_text("CCO ethanol\n")
        result = fetchers.read_smi_library(str(path))
        assert result == {}

    def test_read_smi_library_empty_name(self, tmp_path):
        path = tmp_path / "lib.smi"
        path.write_text("CCO\t\n")
        result = fetchers.read_smi_library(str(path))
        assert result == {}

    def test_read_sdf_3d_library(self, tmp_path):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("_Name", "ethanol")
        w = Chem.SDWriter(str(tmp_path / "lib.sdf"))
        w.write(mol)
        w.close()
        result = fetchers.read_sdf_3d_library(str(tmp_path / "lib.sdf"))
        assert "ethanol" in result
        assert result["ethanol"][0] == "CCO"

    @patch("rdkit.Chem.MolFromMol2File")
    def test_read_mol2_file(self, mock_mol2, tmp_path):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        mock_mol2.return_value = mol
        result = fetchers.read_mol2_file(str(tmp_path / "test.mol2"))
        assert result is not None


class TestReadPDBbindIndex:
    def test_basic(self, tmp_path):
        path = tmp_path / "index.txt"
        path.write_text(
            "# comment\n"
            "//\n"
            "1A30  1.80  2000  8.00  10nM  REF  LIG  CCO  extra ref text\n"
            "1B9S  4.00  2001  5.00  10uM  REF2  LIG2  CCC\n"
            "BAD   1.0   2000  1.0   1uM   REF  LIG  CCO\n"
        )
        result = fetchers.read_pdbbind_index(str(path), max_resolution=3.0)
        assert "1A30" in result
        assert "1B9S" not in result
        assert "BAD" not in result
        assert result["1A30"]["resolution"] == 1.8
        assert result["1A30"]["ligand_smiles"] == "CCO"
        assert result["1A30"]["reference"] == "extra ref"

    def test_no_filters(self, tmp_path):
        path = tmp_path / "index.txt"
        path.write_text("1A30  1.80  2000  8.00  10nM  REF  LIG  CCO\n")
        result = fetchers.read_pdbbind_index(str(path), max_resolution=None, min_affinity=None)
        assert "1A30" in result

    def test_min_affinity(self, tmp_path):
        path = tmp_path / "index.txt"
        path.write_text(
            "1A30  1.80  2000  8.00  10nM  REF  LIG  CCO\n"
            "1B9S  2.00  2001  5.00  10uM  REF  LIG  CCC\n"
        )
        result = fetchers.read_pdbbind_index(str(path), min_affinity=6.0)
        assert "1A30" in result
        assert "1B9S" not in result

    def test_short_line_skipped(self, tmp_path):
        """Cover line 1477: lines with fewer than 8 parts are skipped."""
        path = tmp_path / "index.txt"
        path.write_text("1A30  1.80\n")
        result = fetchers.read_pdbbind_index(str(path))
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# PDB biological assembly info (mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPdbAssemblyInfo:
    """Tests for get_pdb_assembly_info — RCSB biological assembly detection."""

    @patch("autodock.fetchers._http_get_json")
    def test_monomeric_multi_chain(self, mock_json):
        """4F9Z-like: 5 chains in asymmetric unit, monomeric assembly."""
        mock_json.side_effect = [
            # entry
            {
                "rcsb_entry_container_identifiers": {
                    "polymer_entity_ids": ["1"],
                    "assembly_ids": ["1"],
                }
            },
            # polymer entity 1
            {"rcsb_polymer_entity_container_identifiers": {"asym_ids": ["A", "B", "C", "D", "E"]}},
            # assembly 1
            {
                "pdbx_struct_assembly": {
                    "oligomeric_count": 1,
                    "oligomeric_details": "monomeric",
                },
                "pdbx_struct_assembly_gen": [{"asym_id_list": ["A", "B", "C", "D", "E"]}],
            },
        ]
        result = fetchers.get_pdb_assembly_info("4F9Z")
        assert result["is_monomeric"] is True
        assert result["oligomeric_count"] == 1
        assert result["asymmetric_chains"] == ["A", "B", "C", "D", "E"]
        assert result["chains_per_assembly"] == [["A", "B", "C", "D", "E"]]

    @patch("autodock.fetchers._http_get_json")
    def test_homodimeric(self, mock_json):
        """True homodimer: 2 chains per assembly, not monomeric."""
        mock_json.side_effect = [
            {
                "rcsb_entry_container_identifiers": {
                    "polymer_entity_ids": ["1"],
                    "assembly_ids": ["1"],
                }
            },
            {"rcsb_polymer_entity_container_identifiers": {"asym_ids": ["A", "B"]}},
            {
                "pdbx_struct_assembly": {
                    "oligomeric_count": 2,
                    "oligomeric_details": "homodimeric",
                },
                "pdbx_struct_assembly_gen": [{"asym_id_list": ["A", "B"]}],
            },
        ]
        result = fetchers.get_pdb_assembly_info("1ABC")
        assert result["is_monomeric"] is False
        assert result["oligomeric_count"] == 2

    @patch("autodock.fetchers._http_get_json")
    def test_api_failure_returns_empty(self, mock_json):
        mock_json.side_effect = Exception("timeout")
        result = fetchers.get_pdb_assembly_info("4F9Z")
        assert result == {}

    def test_invalid_pdb_id_returns_empty(self):
        result = fetchers.get_pdb_assembly_info("TOOLONG")
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# Single-chain extraction from mmCIF
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractSingleChainFromMmcif:
    """Tests for extract_single_chain_from_mmcif."""

    def test_extract_by_chain_id(self, tmp_path):
        from autodock.fetchers import extract_single_chain_from_mmcif

        # Use the real 4F9Z CIF we already downloaded for validation
        cif_path = "/tmp/4F9Z.cif"
        if not os.path.exists(cif_path):
            pytest.skip("4F9Z.cif not available for testing")
        out = tmp_path / "chain_A.pdb"
        result = extract_single_chain_from_mmcif(cif_path, chain_id="A", output_path=str(out))
        assert result == str(out)
        assert out.exists()
        content = out.read_text()
        assert "ATOM" in content
        lines = [line for line in content.splitlines() if line.startswith("ATOM")]
        assert len(lines) > 0
        assert lines[0][21:22] == "A"

    def test_auto_select_longest_chain(self, tmp_path):
        from autodock.fetchers import extract_single_chain_from_mmcif

        cif_path = "/tmp/4F9Z.cif"
        if not os.path.exists(cif_path):
            pytest.skip("4F9Z.cif not available for testing")
        out = tmp_path / "chain_auto.pdb"
        result = extract_single_chain_from_mmcif(cif_path, output_path=str(out))
        assert result == str(out)
        content = out.read_text()
        lines = [line for line in content.splitlines() if line.startswith("ATOM")]
        assert len(lines) > 0
        # All lines should have the same chain ID (auto-selected longest)
        chain_ids = {line[21:22] for line in lines}
        assert len(chain_ids) == 1

    def test_missing_chain_raises(self, tmp_path):
        from autodock.fetchers import extract_single_chain_from_mmcif

        cif_path = "/tmp/4F9Z.cif"
        if not os.path.exists(cif_path):
            pytest.skip("4F9Z.cif not available for testing")
        with pytest.raises(StructureFetchError, match="No chain found"):
            extract_single_chain_from_mmcif(cif_path, chain_id="Z")
