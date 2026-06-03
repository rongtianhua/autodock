"""Tests for autodock.cache."""

import os

from autodock.cache import LigandCache, PocketCache, ReceptorCache, clear_all_caches


class TestReceptorCache:
    """Tests for ReceptorCache."""

    def test_miss_returns_none(self, tmp_path):
        rc = ReceptorCache(str(tmp_path))
        assert rc.get("nonexistent.pdb", ph=7.4) is None

    def test_put_and_get_roundtrip(self, tmp_path):
        rc = ReceptorCache(str(tmp_path))
        # Create a fake input file
        inp = tmp_path / "input.pdb"
        inp.write_text("ATOM    1  N   ALA A   1")
        # Create a fake output
        out = tmp_path / "out.pdbqt"
        out.write_text("REMARK  receptor")

        rc.put(str(inp), {"receptor.pdbqt": str(out)}, ph=7.4)
        cached = rc.get(str(inp), ph=7.4)
        assert cached is not None
        assert "receptor.pdbqt" in cached
        assert os.path.exists(cached["receptor.pdbqt"])

    def test_param_sensitive_different_keys(self, tmp_path):
        rc = ReceptorCache(str(tmp_path))
        inp = tmp_path / "input.pdb"
        inp.write_text("ATOM")
        out = tmp_path / "out.pdbqt"
        out.write_text("REMARK")

        rc.put(str(inp), {"receptor.pdbqt": str(out)}, ph=7.4)
        # Different ph → different key → miss
        assert rc.get(str(inp), ph=5.5) is None

    def test_meta_written(self, tmp_path):
        rc = ReceptorCache(str(tmp_path))
        inp = tmp_path / "input.pdb"
        inp.write_text("ATOM")
        out = tmp_path / "out.pdbqt"
        out.write_text("REMARK")

        rc.put(str(inp), {"receptor.pdbqt": str(out)}, ph=7.4)
        meta = rc._read_meta("receptors", rc._make_key(str(inp), ph=7.4))
        assert meta is not None
        assert meta["files"] == ["receptor.pdbqt"]
        assert meta["params"]["ph"] == 7.4

    def test_clear_removes_entries(self, tmp_path):
        rc = ReceptorCache(str(tmp_path))
        inp = tmp_path / "input.pdb"
        inp.write_text("ATOM")
        out = tmp_path / "out.pdbqt"
        out.write_text("REMARK")

        rc.put(str(inp), {"receptor.pdbqt": str(out)}, ph=7.4)
        assert rc.clear("receptors") >= 1
        assert rc.get(str(inp), ph=7.4) is None


class TestLigandCache:
    """Tests for LigandCache."""

    def test_miss_returns_none(self, tmp_path):
        lc = LigandCache(str(tmp_path))
        assert lc.get("CCO", ph=7.4) is None

    def test_put_and_get_roundtrip(self, tmp_path):
        lc = LigandCache(str(tmp_path))
        out = tmp_path / "lig.pdbqt"
        out.write_text("REMARK  ligand")

        lc.put("CCO", {"ligand.pdbqt": str(out)}, ph=7.4)
        cached = lc.get("CCO", ph=7.4)
        assert cached is not None
        assert os.path.exists(cached["ligand.pdbqt"])

    def test_param_sensitive(self, tmp_path):
        lc = LigandCache(str(tmp_path))
        out = tmp_path / "lig.pdbqt"
        out.write_text("REMARK")

        lc.put("CCO", {"ligand.pdbqt": str(out)}, ph=7.4)
        assert lc.get("CCO", ph=5.5) is None


class TestPocketCache:
    """Tests for PocketCache."""

    def test_miss_returns_none(self, tmp_path):
        pc = PocketCache(str(tmp_path))
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM")
        assert pc.get(str(rec), padding=5.0) is None

    def test_put_and_get_roundtrip(self, tmp_path):
        pc = PocketCache(str(tmp_path))
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM")
        pockets = [{"center": (1.0, 2.0, 3.0), "box_size": (20.0, 20.0, 20.0), "druggability": 0.5}]

        pc.put(str(rec), pockets, padding=5.0)
        cached = pc.get(str(rec), padding=5.0)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["center"] == [1.0, 2.0, 3.0]

    def test_param_sensitive(self, tmp_path):
        pc = PocketCache(str(tmp_path))
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM")
        pockets = [{"center": (1.0, 2.0, 3.0)}]

        pc.put(str(rec), pockets, padding=5.0)
        assert pc.get(str(rec), padding=10.0) is None


class TestClearAllCaches:
    """Tests for clear_all_caches."""

    def test_clears_all_categories(self, tmp_path):
        rc = ReceptorCache(str(tmp_path))
        lc = LigandCache(str(tmp_path))
        pc = PocketCache(str(tmp_path))

        inp = tmp_path / "input.pdb"
        inp.write_text("ATOM")
        out = tmp_path / "out.pdbqt"
        out.write_text("REMARK")

        rc.put(str(inp), {"receptor.pdbqt": str(out)}, ph=7.4)
        lc.put("CCO", {"ligand.pdbqt": str(out)}, ph=7.4)
        pc.put(str(inp), [{"center": (1.0, 2.0, 3.0)}], padding=5.0)

        counts = clear_all_caches(str(tmp_path))
        assert counts["receptors"] >= 1
        assert counts["ligands"] >= 1
        assert counts["pockets"] >= 1
