"""Tests for autodock.analysis."""

from unittest.mock import MagicMock, patch

import pytest

from autodock import analysis


class TestExtractAffinity:
    """Tests for _extract_affinity."""

    def test_extracts_from_remark(self):
        text = "REMARK VINA RESULT:      -8.236      0.000      0.000"
        assert analysis._extract_affinity(text) == pytest.approx(-8.236)

    def test_returns_none_when_missing(self):
        assert analysis._extract_affinity("ATOM 1 C") is None

    def test_extracts_positive_affinity(self):
        text = "REMARK VINA RESULT:       2.500      0.000      0.000"
        assert analysis._extract_affinity(text) == pytest.approx(2.5)


class TestParseAllPoses:
    """Tests for _parse_all_poses."""

    def test_empty_file_returns_empty(self, tmp_path):
        pdbqt = tmp_path / "poses.pdbqt"
        pdbqt.write_text("")
        assert analysis._parse_all_poses(str(pdbqt)) == []

    def test_missing_file_returns_empty(self, tmp_path):
        assert analysis._parse_all_poses(str(tmp_path / "missing.pdbqt")) == []

    def test_single_pose_no_model(self, tmp_path):
        pdbqt = tmp_path / "poses.pdbqt"
        content = "REMARK VINA RESULT:      -7.500      0.000      0.000\nATOM 1 C\n"
        pdbqt.write_text(content)
        results = analysis._parse_all_poses(str(pdbqt))
        assert len(results) == 1
        assert results[0][0] == pytest.approx(-7.5)

    def test_multi_model_parsing(self, tmp_path):
        pdbqt = tmp_path / "poses.pdbqt"
        content = (
            "REMARK VINA RESULT:      -7.500      0.000      0.000\n"
            "MODEL 1\n"
            "REMARK VINA RESULT:      -8.000      0.000      0.000\n"
            "ATOM 1 C\n"
            "ENDMDL\n"
            "MODEL 2\n"
            "REMARK VINA RESULT:      -6.500      0.000      0.000\n"
            "ATOM 2 O\n"
            "ENDMDL\n"
        )
        pdbqt.write_text(content)
        results = analysis._parse_all_poses(str(pdbqt))
        assert len(results) == 2
        assert results[0][0] == pytest.approx(-8.0)
        assert results[1][0] == pytest.approx(-6.5)

    def test_skips_empty_models(self, tmp_path):
        pdbqt = tmp_path / "poses.pdbqt"
        content = (
            "MODEL 1\n"
            "REMARK VINA RESULT:      -8.000      0.000      0.000\n"
            "ATOM 1 C\n"
            "ENDMDL\n"
            "MODEL 2\n"
            "ENDMDL\n"
        )
        pdbqt.write_text(content)
        results = analysis._parse_all_poses(str(pdbqt))
        assert len(results) == 1


class TestAnalyzeScoringBias:
    """Tests for analyze_scoring_bias."""

    def test_skips_missing_files(self, tmp_path):
        out = tmp_path / "benchmark"
        out.mkdir()
        (out / "1ABC").mkdir()
        # No docking_all_poses.pdbqt or crystal_ligand.pdb
        with patch.object(analysis, "compute_rmsd_to_crystal"):
            results = analysis.analyze_scoring_bias(str(out), target_ids=["1ABC"])
        assert "1ABC" not in results

    def test_analyzes_valid_target(self, tmp_path):
        out = tmp_path / "benchmark"
        target_dir = out / "1ABC"
        target_dir.mkdir(parents=True)

        # Write all_poses file
        poses = target_dir / "docking_all_poses.pdbqt"
        content = (
            "MODEL 1\n"
            "REMARK VINA RESULT:      -8.000      0.000      0.000\n"
            "ATOM 1 C\n"
            "ENDMDL\n"
            "MODEL 2\n"
            "REMARK VINA RESULT:      -6.000      0.000      0.000\n"
            "ATOM 2 O\n"
            "ENDMDL\n"
        )
        poses.write_text(content)

        # Write crystal ligand
        crystal = target_dir / "crystal_ligand.pdb"
        crystal.write_text("ATOM 1 C\n")

        # Pre-import matplotlib.pyplot so patch() can find it
        import matplotlib.pyplot  # noqa: F401

        with patch.object(analysis, "compute_rmsd_to_crystal", return_value=1.5):
            with patch("matplotlib.pyplot") as mock_plt:
                fig = MagicMock()
                ax = MagicMock()
                mock_plt.subplots.return_value = (fig, ax)
                results = analysis.analyze_scoring_bias(str(out), target_ids=["1ABC"])

        assert "1ABC" in results
        assert len(results["1ABC"]["poses"]) == 2
        assert results["1ABC"]["top1_affinity"] == pytest.approx(-8.0)
        assert results["1ABC"]["best_rmsd"] == pytest.approx(1.5)
        assert results["1ABC"]["top1_vs_best"] == pytest.approx(0.0)

    def test_matplotlib_unavailable_skips_plot(self, tmp_path):
        out = tmp_path / "benchmark"
        target_dir = out / "1ABC"
        target_dir.mkdir(parents=True)

        poses = target_dir / "docking_all_poses.pdbqt"
        poses.write_text(
            "MODEL 1\n"
            "REMARK VINA RESULT:      -8.000      0.000      0.000\n"
            "ATOM 1 C\n"
            "ENDMDL\n"
        )
        crystal = target_dir / "crystal_ligand.pdb"
        crystal.write_text("ATOM 1 C\n")

        with patch.object(analysis, "compute_rmsd_to_crystal", return_value=1.0):
            with patch.dict("sys.modules", {"matplotlib": None}):
                results = analysis.analyze_scoring_bias(str(out), target_ids=["1ABC"])
        assert "1ABC" in results
        assert "figure_path" not in results["1ABC"]
class TestComputeLigandEfficiency:
    def test_none_affinity_returns_none(self):
        result = analysis.compute_ligand_efficiency(None, 10)
        assert result["le"] is None
        assert result["n_heavy"] == 10

    def test_zero_heavy_atoms_returns_none(self):
        result = analysis.compute_ligand_efficiency(-8.0, 0)
        assert result["le"] is None

    def test_basic_le(self):
        result = analysis.compute_ligand_efficiency(-8.0, 10)
        assert result["le"] == pytest.approx(0.8)
        assert result["lle"] is None
        assert result["lem"] is None
        assert result["le_rb"] is None

    def test_with_mw(self):
        result = analysis.compute_ligand_efficiency(-8.0, 10, molecular_weight=200.0)
        assert result["lem"] == pytest.approx(0.04)

    def test_with_rotatable_bonds(self):
        result = analysis.compute_ligand_efficiency(-8.0, 10, n_rotatable_bonds=4)
        assert result["le_rb"] == pytest.approx(2.0)

    def test_all_optional(self):
        result = analysis.compute_ligand_efficiency(
            -8.0, 10, n_rotatable_bonds=4, molecular_weight=200.0
        )
        assert result["le"] == pytest.approx(0.8)
        assert result["lem"] == pytest.approx(0.04)
        assert result["le_rb"] == pytest.approx(2.0)
        assert result["lle"] is None
        assert result["n_heavy"] == 10
        assert result["n_rotatable"] == 4
        assert result["mw"] == 200.0


class TestComputeSpearmanCorrelation:
    def test_too_few_returns_none(self):
        result = analysis.compute_spearman_correlation([1.0, 2.0], [1.0, 2.0])
        assert result["rho"] is None
        assert result["pvalue"] is None
        assert result["n"] == 2

    def test_success(self):
        result = analysis.compute_spearman_correlation(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        )
        assert result["rho"] is not None
        assert result["pvalue"] is not None
        assert result["n"] == 5
        assert result["rho"] == pytest.approx(-1.0, abs=1e-6)



class TestComputeEnrichmentFactor:
    def test_empty_input(self):
        result = analysis.compute_enrichment_factor([], {"A"})
        assert result["ef"] is None
        assert result["n_total"] == 0

    def test_no_actives(self):
        result = analysis.compute_enrichment_factor(
            [("A", -8.0), ("B", -7.0)], set()
        )
        assert result["ef"] is None
        assert result["n_total"] == 2

    def test_basic_ef(self):
        compounds = [(f"C{i}", -float(i)) for i in range(1, 101)]
        actives = {f"C{i}" for i in range(91, 101)}  # top 10 by score
        result = analysis.compute_enrichment_factor(compounds, actives, ef_percent=10.0)
        assert result["ef"] is not None
        assert result["ef"] > 0
        assert result["n_total"] == 100
        assert result["n_actives"] == 10
        assert result["n_top"] == 10
        assert result["n_top_actives"] == 10

    def test_auc_roc_computed(self):
        compounds = [(f"C{i}", -float(i)) for i in range(1, 101)]
        actives = {f"C{i}" for i in range(1, 11)}
        result = analysis.compute_enrichment_factor(compounds, actives)
        assert result["auc_roc"] is not None

    def test_sklearn_not_available(self):
        compounds = [("A", -8.0), ("B", -7.0)]
        actives = {"A"}
        with patch("sklearn.metrics.roc_auc_score", side_effect=ImportError):
            result = analysis.compute_enrichment_factor(compounds, actives)
        assert result["ef"] is not None
        assert result["auc_roc"] is None


class TestComputeInteractionFingerprint:
    def test_empty_interactions(self):
        result = analysis.compute_interaction_fingerprint([])
        assert result["fingerprint"].shape == (0, 8)
        assert result["n_interactions"] == 0
        assert result["density"] == 0.0

    def test_residue_key_variants(self):
        interactions = [
            {"residue": "A:42:GLU", "type": "H-bond"},
            {"restype_ligand": "A:43:ASP", "type": "Salt bridge"},
            {"resnr": "A:44", "type": "Hydrophobic"},
            {"reschain": "A", "type": "π-π"},
            {"protisnr": 45, "protchain": "A", "restype": "PHE", "type": "Halogen bond"},
            {"resnr": 46, "reschain": "B", "type": "Water bridge"},
            {"type": "Metal complex"},  # no residue key
        ]
        result = analysis.compute_interaction_fingerprint(interactions)
        assert result["n_interactions"] > 0
        assert result["density"] > 0
        assert len(result["residues"]) > 0

    def test_residue_order_filter(self):
        interactions = [
            {"residue": "A:42:GLU", "type": "H-bond"},
            {"residue": "A:43:ASP", "type": "H-bond"},
        ]
        result = analysis.compute_interaction_fingerprint(
            interactions, residue_order=["A:43:ASP", "A:42:GLU"]
        )
        assert result["residues"] == ["A:43:ASP", "A:42:GLU"]

    def test_custom_types(self):
        interactions = [{"residue": "A:1:ALA", "type": "Custom"}]
        result = analysis.compute_interaction_fingerprint(
            interactions, interaction_types=("Custom",)
        )
        assert result["fingerprint"].shape == (1, 1)
        assert result["fingerprint"][0, 0]


class TestAnalyzeScoringBiasBranches:
    def test_target_ids_none(self, tmp_path):
        out = tmp_path / "benchmark"
        for pdb_id in ["1ABC", "2DEF"]:
            target_dir = out / pdb_id
            target_dir.mkdir(parents=True)
            poses = target_dir / "docking_all_poses.pdbqt"
            poses.write_text(
                "MODEL 1\nREMARK VINA RESULT:      -8.000\nATOM 1 C\nENDMDL\n"
            )
            crystal = target_dir / "crystal_ligand.pdb"
            crystal.write_text("ATOM 1 C\n")

        with patch.object(analysis, "compute_rmsd_to_crystal", return_value=1.5):
            results = analysis.analyze_scoring_bias(str(out))
        assert "1ABC" in results
        assert "2DEF" in results

    def test_no_poses_found(self, tmp_path):
        out = tmp_path / "benchmark"
        target_dir = out / "1ABC"
        target_dir.mkdir(parents=True)
        poses = target_dir / "docking_all_poses.pdbqt"
        poses.write_text("MODEL 1\nENDMDL\n")  # empty model
        crystal = target_dir / "crystal_ligand.pdb"
        crystal.write_text("ATOM 1 C\n")

        results = analysis.analyze_scoring_bias(str(out), target_ids=["1ABC"])
        assert "1ABC" not in results

    def test_no_valid_rmsd(self, tmp_path):
        out = tmp_path / "benchmark"
        target_dir = out / "1ABC"
        target_dir.mkdir(parents=True)
        poses = target_dir / "docking_all_poses.pdbqt"
        poses.write_text(
            "MODEL 1\nREMARK VINA RESULT:      -8.000\nATOM 1 C\nENDMDL\n"
        )
        crystal = target_dir / "crystal_ligand.pdb"
        crystal.write_text("ATOM 1 C\n")

        with patch.object(analysis, "compute_rmsd_to_crystal", return_value=None):
            results = analysis.analyze_scoring_bias(str(out), target_ids=["1ABC"])
        assert "1ABC" not in results

    def test_all_poses_filename_variant(self, tmp_path):
        out = tmp_path / "benchmark"
        target_dir = out / "1ABC"
        target_dir.mkdir(parents=True)
        poses = target_dir / "all_poses.pdbqt"
        poses.write_text(
            "MODEL 1\nREMARK VINA RESULT:      -8.000\nATOM 1 C\nENDMDL\n"
        )
        crystal = target_dir / "crystal_ligand.pdb"
        crystal.write_text("ATOM 1 C\n")

        with patch.object(analysis, "compute_rmsd_to_crystal", return_value=1.5):
            results = analysis.analyze_scoring_bias(str(out), target_ids=["1ABC"])
        assert "1ABC" in results
