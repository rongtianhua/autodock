"""Tests for autodock.rendering — PyMOL and 2D interaction rendering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autodock import rendering as rend
from autodock.core import VisualizationError


def _make_min_png(path: Path) -> None:
    """Write a minimal valid 1x1 PNG using PIL."""
    from PIL import Image

    img = Image.new("RGB", (1, 1), color=(0, 0, 0))
    img.save(path)


class TestBuildPymolScript:
    def test_returns_string(self):
        script = rend._build_pymol_script(
            "rec.pdb",
            "lig.pdbqt",
            "out.png",
            scene="complex",
            interactions=[],
        )
        assert isinstance(script, str)
        assert "load" in script

    def test_interaction_scene(self):
        intx = [
            {
                "type": "H-bond",
                "resn": "SER",
                "resi": 1,
                "chain": "A",
                "atom": "OG",
                "color": "cyan",
                "description": "test",
            }
        ]
        script = rend._build_pymol_script(
            "rec.pdb",
            "lig.pdbqt",
            "out.png",
            scene="interaction",
            center=(1.0, 2.0, 3.0),
            interactions=intx,
        )
        assert "distance" in script.lower()

    def test_pocket_scene(self):
        script = rend._build_pymol_script(
            "rec.pdb",
            "lig.pdbqt",
            "out.png",
            scene="pocket",
            center=(1.0, 2.0, 3.0),
        )
        assert "pocket_surf" in script


class TestRenderScenePymol:
    @patch("autodock.rendering._PYMOL_EXE", "/fake/pymol")
    @patch("autodock.rendering.safe_subprocess")
    @patch("os.path.exists")
    def test_calls_pymol(self, mock_exists, mock_subprocess, tmp_path):
        mock_subprocess.return_value = (True, "", "")
        mock_exists.return_value = True
        out_png = tmp_path / "scene.png"
        rend.render_scene_pymol("rec.pdb", "lig.pdbqt", str(out_png), scene="complex")
        assert mock_subprocess.called
        cmd = mock_subprocess.call_args[0][0]
        assert "pymol" in cmd[0]

    def test_missing_pymol_raises(self, tmp_path):
        with patch("autodock.rendering._PYMOL_EXE", None):
            with pytest.raises(VisualizationError, match="PyMOL"):
                rend.render_scene_pymol("rec.pdb", "lig.pdbqt", str(tmp_path / "out.png"))


def _have_rdkit() -> bool:
    try:
        import rdkit
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_rdkit(), reason="rdkit not installed")
class TestRenderInteractions2d:
    @patch("rdkit.Chem.MolFromSmiles")
    @patch("rdkit.Chem.RemoveHs")
    @patch("rdkit.Chem.AllChem.Compute2DCoords")
    @patch("rdkit.Chem.Draw.MolDraw2DCairo")
    @patch("PIL.Image.open")
    @patch("PIL.ImageDraw.Draw")
    @patch("PIL.ImageFont.truetype")
    def test_basic(
        self,
        mock_font,
        mock_draw,
        mock_img_open,
        mock_drawer,
        mock_2d,
        mock_remhs,
        mock_mol,
        tmp_path,
    ):
        mock_mol_instance = MagicMock()
        mock_mol.return_value = mock_mol_instance
        mock_remhs.return_value = mock_mol_instance
        mock_drawer_instance = MagicMock()
        mock_drawer.return_value = mock_drawer_instance
        mock_drawer_instance.GetDrawingText.return_value = b"\x89PNG\r\n\x1a\n"
        mock_img = MagicMock()
        mock_img_open.return_value = mock_img

        ligand = tmp_path / "lig.pdbqt"
        ligand.write_text("REMARK SMILES CC\nATOM 1 C 0 0 0\n")
        out = tmp_path / "out.png"
        rend.render_interactions_2d(
            "rec.pdb",
            str(ligand),
            interactions=[{"type": "H-bond", "resn": "SER", "resi": 1}],
            output_png=str(out),
        )
        assert mock_img.save.called

    def test_parse_failure_raises(self, tmp_path):
        ligand = tmp_path / "lig.pdbqt"
        ligand.write_text("ATOM 1 XX 0 0 0\n")
        with pytest.raises(VisualizationError):
            rend.render_interactions_2d(
                "rec.pdb",
                str(ligand),
                interactions=[],
                output_png=str(tmp_path / "out.png"),
            )


class TestCompositeSummary:
    def test_basic(self, tmp_path):
        img1 = tmp_path / "a.png"
        _make_min_png(img1)
        img2 = tmp_path / "b.png"
        _make_min_png(img2)
        out = tmp_path / "composite.png"

        rend.composite_summary([str(img1), str(img2)], str(out))
        assert out.exists()

    def test_empty_panels_raises(self, tmp_path):
        with pytest.raises(VisualizationError, match="No panels"):
            rend.composite_summary([], str(tmp_path / "out.png"))

    @patch("PIL.Image.open")
    def test_no_valid_images_raises(self, mock_open, tmp_path):
        # Image.open raises an exception for every path, so images list stays empty.
        mock_open.side_effect = Exception("bad image")
        with pytest.raises(VisualizationError, match="No valid panel"):
            rend.composite_summary([str(tmp_path / "bad.png")], str(tmp_path / "out.png"))
