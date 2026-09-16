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
        assert "_dashed_line" in script
        assert "targets" in script

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


class TestSceneScriptHygiene:
    """Scene-wide fixes: no leftover lines specks, opaque background."""

    def test_lines_representation_hidden(self):
        script = rend._build_pymol_script("rec.pdb", "lig.pdbqt", "out.png", scene="complex")
        assert "cmd.hide('lines', 'receptor')" in script
        assert "cmd.hide('nonbonded', 'receptor')" in script

    def test_ray_opaque_background(self):
        script = rend._build_pymol_script("rec.pdb", "lig.pdbqt", "out.png", scene="complex")
        assert "cmd.set('ray_opaque_background', 1)" in script

    def test_complex_scene_orients_camera(self):
        """Whole-complex view must orient first (no wide empty margins)."""
        script = rend._build_pymol_script("rec.pdb", "lig.pdbqt", "out.png", scene="complex")
        assert "cmd.orient('receptor or ligand')" in script
        assert "cmd.zoom('(receptor or ligand)', 1.0)" in script

    def test_complex_default_resolution_bump(self):
        """Complex scene without explicit size renders at 3200x2400; explicit size wins."""
        with (
            patch("autodock.rendering._PYMOL_EXE", "/fake/pymol"),
            patch("autodock.rendering.safe_subprocess") as mock_sub,
            patch("os.path.exists", return_value=True),
        ):
            mock_sub.return_value = (True, "", "")
            rend.render_scene_pymol("rec.pdb", "lig.pdbqt", "out_complex.png", scene="complex")
            cmd = mock_sub.call_args[0][0]
            assert cmd[cmd.index("-W") + 1] == "3200"
            assert cmd[cmd.index("-H") + 1] == "2400"

            rend.render_scene_pymol(
                "rec.pdb",
                "lig.pdbqt",
                "out_complex.png",
                scene="complex",
                width=2400,
                height=1800,
            )
            cmd = mock_sub.call_args[0][0]
            assert cmd[cmd.index("-W") + 1] == "2400"
            assert cmd[cmd.index("-H") + 1] == "1800"

    def test_interaction_legend_overlay(self, tmp_path):
        """Legend box is composited onto the interaction PNG (bottom-left)."""
        from PIL import Image

        png = tmp_path / "scene.png"
        Image.new("RGB", (2400, 1800), (255, 255, 255)).save(png)
        before = Image.open(png).convert("RGB").load()
        assert before[20, 1800 - 20] == (255, 255, 255)

        interactions = [
            {"type": "H-bond", "resn": "ARG", "resi": 70},
            {"type": "Hydrophobic", "resn": "LEU", "resi": 69},
        ]
        rend._overlay_interaction_legend(str(png), interactions)

        after = Image.open(png).convert("RGB").load()
        # Bottom-left corner now carries the semi-transparent legend box
        assert after[20, 1800 - 20] != (255, 255, 255)


class TestInteractionSceneLabelsAndLines:
    """Interaction scene: per-type per-atom dashed lines + pocket residue labels."""

    def _interactions(self):
        return [
            {
                "type": "H-bond",
                "resn": "ARG",
                "resi": 70,
                "chain": "A",
                "ligand_atoms": [{"coords": (-20.7, 5.2, 47.6)}],
            },
            {
                "type": "Hydrophobic",
                "resn": "LEU",
                "resi": 69,
                "chain": "A",
                "ligand_atoms": [{"coords": (-12.6, 2.2, 46.3)}],
            },
        ]

    def test_per_atom_dashed_lines_with_type_colors(self):
        script = rend._build_pymol_script(
            "rec.pdb",
            "lig.pdbqt",
            "out.png",
            scene="interaction",
            center=(0.0, 0.0, 0.0),
            interactions=self._interactions(),
        )
        # One dashed line per ligand atom, colored by interaction type
        assert "targets = [(-20.7, 5.2, 47.6)]" in script
        assert "(0.0, 1.0, 1.0)" in script  # H-bond cyan
        assert "(1.0, 0.502, 0.0)" in script  # Hydrophobic orange (128/255)
        # Dashes thinner, endpoint element preference per interaction type
        assert "radius=0.06" in script
        assert "('N', 'O')" in script  # H-bond prefers N/O endpoints
        assert "('C',)" in script  # Hydrophobic prefers carbon endpoints
        assert "_elem(a) in _pref" in script
        # Distance label only on the closest pair
        assert "cmd.pseudoatom('dist_0'" in script

    def test_generated_python_blocks_compile(self):
        """Every 'python ... python end' block must be syntactically valid."""
        script = rend._build_pymol_script(
            "rec.pdb",
            "lig.pdbqt",
            "out.png",
            scene="interaction",
            center=(0.0, 0.0, 0.0),
            interactions=self._interactions(),
        )
        blocks = []
        in_block = False
        for line in script.splitlines():
            if line.strip() == "python":
                in_block = True
                blocks.append([])
                continue
            if line.strip() == "python end":
                in_block = False
                continue
            if in_block:
                blocks[-1].append(line)
        assert blocks, "expected at least one python block"
        for block in blocks:
            compile("\n".join(block), "<pymol-python-block>", "exec")

    def test_surrounding_pocket_residues_labelled(self):
        script = rend._build_pymol_script(
            "rec.pdb",
            "lig.pdbqt",
            "out.png",
            scene="interaction",
            center=(0.0, 0.0, 0.0),
            interactions=self._interactions(),
            color_scheme="presentation_black",
        )
        assert "pocket_vis and name CA" in script
        assert "amb_lbl_" in script
        # Interacting residues get the prominent label, surroundings the dim one
        assert "grey70" in script

    def test_surrounding_pocket_residues_labelled_white_scheme(self):
        # White (publication) schemes use grey50 dim labels and grey50 stick
        # carbons so they stay visible on the white background.
        script = rend._build_pymol_script(
            "rec.pdb",
            "lig.pdbqt",
            "out.png",
            scene="interaction",
            center=(0.0, 0.0, 0.0),
            interactions=self._interactions(),
        )
        assert "grey50" in script
        assert "cmd.bg_color('white')" in script


class TestClipSegmentToBox:
    def test_enters_box(self):
        # Segment from (0,0) to (100,0); box starts at x=50
        x, y = rend._clip_segment_to_box(0, 0, 100, 0, (50, -10, 60, 10))
        assert x == pytest.approx(50)
        assert y == pytest.approx(0)

    def test_no_intersection_returns_end(self):
        x, y = rend._clip_segment_to_box(0, 0, 10, 0, (50, -10, 60, 10))
        assert (x, y) == (10, 0)

    def test_start_inside_box_returns_end(self):
        x, y = rend._clip_segment_to_box(55, 0, 100, 0, (50, -10, 60, 10))
        assert (x, y) == (100, 0)


class TestLegendLayout:
    def test_geometry_scales_with_canvas(self):
        header = ("Interactions", 120, 20)
        rows = [("H-bond: 1", 90, 18), ("Hydrophobic: 4", 150, 18)]
        small = rend._legend_layout(header, rows, 1800, 1400, scale=1.0)
        large = rend._legend_layout(header, rows, 5400, 4200, scale=3.6)
        # Box must contain all rows + header at any scale
        for layout in (small, large):
            assert layout["h"] >= layout["header_h"] + layout["row_h"] * len(rows)
            assert layout["w"] >= 150 + layout["pad_x"] * 2
        # Larger canvas → proportionally larger legend
        assert large["row_h"] > small["row_h"]
        assert large["pad_x"] > small["pad_x"]
        # Box stays inside the canvas
        assert large["x"] + large["w"] <= 5400
        assert large["y"] + large["h"] <= 4200


class TestComputeLabelPositionsReserved:
    def test_labels_avoid_reserved_rect(self):
        groups = [
            {"type": "H-bond", "resn": "ARG", "resi": i, "chain": "A", "rdkit_atoms": {i}}
            for i in range(4)
        ]
        atom_coords = {0: (100, 100), 1: (1100, 100), 2: (100, 800), 3: (1100, 800)}
        canvas_w, canvas_h = 1200, 900
        # Legend-like reserved region bottom-right
        reserved = [(canvas_w - 300, canvas_h - 200, canvas_w - 10, canvas_h - 10)]
        pos = rend._compute_label_positions(
            groups, atom_coords, canvas_w, canvas_h, margin=80, reserved_rects=reserved
        )
        assert pos
        est_tw, est_th = 120, 30
        for x, y in pos.values():
            box = (x - 6, y - 6, x + est_tw + 6, y + est_th + 6)
            rx1, ry1, rx2, ry2 = reserved[0]
            overlaps = not (box[2] < rx1 or box[0] > rx2 or box[3] < ry1 or box[1] > ry2)
            assert not overlaps, f"label at {(x, y)} overlaps reserved rect"


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
