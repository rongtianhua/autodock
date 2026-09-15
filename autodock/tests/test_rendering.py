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
    def test_calls_pymol_with_resolution_flags(self, mock_exists, mock_subprocess, tmp_path):
        mock_subprocess.return_value = (True, "", "")
        mock_exists.return_value = True
        out_png = tmp_path / "scene.png"
        rend.render_scene_pymol(
            "rec.pdb",
            "lig.pdbqt",
            str(out_png),
            scene="complex",
            width=2400,
            height=1800,
        )
        assert mock_subprocess.called
        cmd = mock_subprocess.call_args[0][0]
        assert "pymol" in cmd[0]
        # Headless PyMOL must receive -W/-H to avoid 640x480 fallback.
        assert "-W" in cmd
        assert "-H" in cmd
        assert cmd[cmd.index("-W") + 1] == "2400"
        assert cmd[cmd.index("-H") + 1] == "1800"

    def test_missing_pymol_raises(self, tmp_path):
        with patch("autodock.rendering._PYMOL_EXE", None):
            with pytest.raises(VisualizationError, match="PyMOL"):
                rend.render_scene_pymol("rec.pdb", "lig.pdbqt", str(tmp_path / "out.png"))


def _have_rdkit() -> bool:
    try:
        import importlib.util as _iu

        return _iu.find_spec("rdkit") is not None
    except (ImportError, OSError):
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

    def test_adaptive_row_heights(self, tmp_path):
        from PIL import Image

        # Create two wide panels and one tall panel.
        wide = tmp_path / "wide.png"
        Image.new("RGB", (400, 100), color=(255, 0, 0)).save(wide)
        tall = tmp_path / "tall.png"
        Image.new("RGB", (100, 400), color=(0, 255, 0)).save(tall)
        out = tmp_path / "composite.png"

        rend.composite_summary([str(wide), str(tall)], str(out), ncols=1)
        composite = Image.open(out)
        # Two rows; total height ≈ title + pad + row1(100) + pad + row2(400) + pad.
        assert composite.height > 500
        # Width should match the widest panel (400) plus padding.
        assert composite.width >= 400

    def test_empty_panels_raises(self, tmp_path):
        with pytest.raises(VisualizationError, match="No panels"):
            rend.composite_summary([], str(tmp_path / "out.png"))

    @patch("PIL.Image.open")
    def test_no_valid_images_raises(self, mock_open, tmp_path):
        # Image.open raises an exception for every path, so images list stays empty.
        mock_open.side_effect = Exception("bad image")
        with pytest.raises(VisualizationError, match="No valid panel"):
            rend.composite_summary([str(tmp_path / "bad.png")], str(tmp_path / "out.png"))


class TestParsePdbqtHelpers:
    def test_parse_smiles_idx(self, tmp_path):
        pdbqt = tmp_path / "lig.pdbqt"
        pdbqt.write_text(
            "REMARK SMILES IDX 1 5 2 6 3 7\n"
            "ATOM      1  C   UNL     1       0.000   0.000   0.000\n"
        )
        mapping = rend._parse_smiles_idx_from_pdbqt(str(pdbqt))
        assert mapping == {5: 1, 6: 2, 7: 3}

    def test_parse_smiles_idx_empty(self, tmp_path):
        pdbqt = tmp_path / "lig.pdbqt"
        pdbqt.write_text("ATOM      1  C   UNL     1       0.000   0.000   0.000\n")
        mapping = rend._parse_smiles_idx_from_pdbqt(str(pdbqt))
        assert mapping == {}

    def test_parse_pdbqt_coords(self, tmp_path):
        pdbqt = tmp_path / "lig.pdbqt"
        pdbqt.write_text(
            "ATOM      1  C   UNL     1       1.234   2.345   3.456\n"
            "ATOM      2  N   UNL     1       4.567   5.678   6.789\n"
        )
        coords = rend._parse_pdbqt_coords(str(pdbqt))
        assert len(coords) == 2
        assert coords[(1.234, 2.345, 3.456)] == 1
        assert coords[(4.567, 5.678, 6.789)] == 2

    def test_parse_pdbqt_coords_skips_malformed(self, tmp_path):
        pdbqt = tmp_path / "lig.pdbqt"
        pdbqt.write_text(
            "ATOM      1  C   UNL     1       1.234   2.345   3.456\n"
            "ATOM    BAD  C   UNL     1       xxxxx   yyyyy   zzzzz\n"
        )
        coords = rend._parse_pdbqt_coords(str(pdbqt))
        assert len(coords) == 1
        assert coords[(1.234, 2.345, 3.456)] == 1


class TestInteractionScenePocketCartoon:
    """3D interaction scene: pocket-window cartoon, not whole-protein transparency."""

    def test_interaction_scene_hides_nonpocket_cartoon(self):
        script = rend._build_pymol_script(
            "rec.pdb",
            "lig.pdbqt",
            "out.png",
            scene="interaction",
            center=(1.0, 2.0, 3.0),
            interactions=[],
        )
        # 8 Å pocket window — 15 Å covered ~80 % of small receptors in the wild
        assert "within 8.0 of ligand" in script
        assert "within 15.0 of ligand" not in script
        # Cartoon hidden outside pocket and re-shown inside (solid), not
        # transparency-graded over the whole protein
        assert "cmd.hide('cartoon', 'receptor')" in script
        assert "cmd.show('cartoon', 'pocket_vis')" in script
        assert "cmd.set('cartoon_transparency', 1.0" not in script
        # Pocket side chains visible as sticks
        assert "cmd.show('sticks', 'pocket_vis" in script


class TestComputeLabelPositions:
    """2D label layout: one label per residue, natural direction, in-canvas."""

    def test_same_residue_multiple_types_share_position(self):
        groups = [
            {"type": "H-bond", "resn": "PHE", "resi": 109, "chain": "A", "rdkit_atoms": {0, 1}},
            {"type": "Hydrophobic", "resn": "PHE", "resi": 109, "chain": "A", "rdkit_atoms": {2}},
            {"type": "H-bond", "resn": "SER", "resi": 96, "chain": "A", "rdkit_atoms": {3}},
        ]
        atom_coords = {0: (100, 100), 1: (130, 100), 2: (115, 140), 3: (520, 420)}
        pos = rend._compute_label_positions(groups, atom_coords, 1200, 900, margin=80)
        # PHE109 must get ONE shared position for both interaction types
        assert pos[0] == pos[1]
        # Every placeable group got a position
        assert set(pos) == {0, 1, 2}

    def test_positions_stay_inside_canvas(self):
        import random

        rng = random.Random(0)
        atom_coords = {i: (rng.uniform(400, 800), rng.uniform(300, 600)) for i in range(8)}
        groups = [
            {"type": "H-bond", "resn": "ARG", "resi": 10 + i, "chain": "A", "rdkit_atoms": {i}}
            for i in range(8)
        ]
        canvas_w, canvas_h = 2400, 1800
        pos = rend._compute_label_positions(groups, atom_coords, canvas_w, canvas_h, margin=80)
        assert len(pos) == 8
        for x, y in pos.values():
            assert 0 <= x <= canvas_w
            assert 0 <= y <= canvas_h


@pytest.mark.skipif(not _have_rdkit(), reason="rdkit not installed")
class TestFillNoninteractingAromatic:
    def test_noninteracting_aromatic_atoms_filled_grey(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1")
        highlight_atoms: set[int] = set()
        highlight_atom_colors: dict = {}
        highlight_bonds: set[int] = set()
        highlight_bond_colors: dict = {}
        # Atom 0 is the "interacting" atom with an interaction colour
        highlight_atom_colors[0] = (0.0, 0.65, 0.0)

        rend._fill_noninteracting_aromatic(
            mol, highlight_atoms, highlight_atom_colors, highlight_bonds, highlight_bond_colors
        )

        # Interacting atom keeps its colour; the rest of the ring goes grey
        assert highlight_atom_colors[0] == (0.0, 0.65, 0.0)
        for i in range(1, 6):
            assert i in highlight_atoms
            assert highlight_atom_colors[i] == (0.88, 0.88, 0.88)
        # All aromatic-aromatic bonds filled (none pre-coloured)
        assert len(highlight_bonds) == mol.GetNumBonds()
        assert all(c == (0.88, 0.88, 0.88) for c in highlight_bond_colors.values())

    def test_pre_coloured_bond_not_overwritten(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1")
        highlight_atoms: set[int] = set()
        highlight_atom_colors: dict = {}
        highlight_bonds: set[int] = set()
        highlight_bond_colors: dict = {0: (1.0, 0.0, 0.0)}

        rend._fill_noninteracting_aromatic(
            mol, highlight_atoms, highlight_atom_colors, highlight_bonds, highlight_bond_colors
        )

        assert highlight_bond_colors[0] == (1.0, 0.0, 0.0)
        assert len(highlight_bonds) == mol.GetNumBonds() - 1
