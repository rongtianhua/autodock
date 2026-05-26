"""
autodock.rendering — Publication-quality visualization.
=======================================================
3D rendering via PyMOL CLI and 2D interaction diagrams via RDKit + Cairo.
"""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from typing import Any

from autodock.core import (
    DEFAULT_DPI,
    DEFAULT_RAY_HEIGHT,
    DEFAULT_RAY_WIDTH,
    VisualizationError,
    find_pymol,
    logger,
    safe_subprocess,
)
from autodock.utils import ensure_dir

# ─────────────────────────────────────────────────────────────────────────────
# PyMOL 3D Rendering (CLI-based)
# ─────────────────────────────────────────────────────────────────────────────

_PYMOL_EXE = find_pymol()

# Leipzig standard dash parameters
DASH_PRESETS = {
    "fine": {
        "dash_gap": 0.4,
        "dash_radius": 0.05,
        "dash_length": 0.3,
        "dash_as_cylinders": True,
        "dash_round_ends": True,
    },
}

INTERACTION_COLORS = {
    "H-bond": "cyan",
    "Hydrophobic": "orange",
    "π-π": "green",
    "π-cation": "purple",
    "Salt bridge": "red",
    "Halogen bond": "yellow",
    "Water bridge": "blue",
    "Metal complex": "grey",
}


def _build_pymol_script(
    receptor_pdb: str,
    ligand_pdbqt: str,
    output_png: str,
    scene: str = "pocket",
    center: tuple[float, float, float] | None = None,
    interactions: list[dict[str, Any]] | None = None,
    width: int = DEFAULT_RAY_WIDTH,
    height: int = DEFAULT_RAY_HEIGHT,
    pocket_distance: float = 5.0,
) -> str:
    """Build a PyMOL command script for publication-quality rendering."""

    lines = []
    lines.append("cmd.delete('all')")
    lines.append(f'cmd.load("{receptor_pdb}", "receptor")')
    lines.append(f'cmd.load("{ligand_pdbqt}", "ligand")')

    # Background
    if scene in ("pocket", "interaction", "ligand_closeup"):
        lines.append("cmd.bg_color('black')")
    else:
        lines.append("cmd.bg_color('white')")

    # Protein representation
    lines.append("cmd.show('cartoon', 'receptor')")
    if scene == "complex":
        lines.append("cmd.color('cold', 'receptor and elem C')")
    else:
        lines.append("cmd.color('grey80', 'receptor and elem C')")

    # Transparency for pocket scenes
    if scene in ("pocket", "interaction"):
        lines.append("cmd.set('cartoon_transparency', 0.2, 'receptor')")

    # Pocket surface
    if scene == "pocket" and center:
        cx, cy, cz = center
        lines.append(
            f"cmd.select('pocket_surf', 'br. receptor and center {cx},{cy},{cz} around {pocket_distance}')"
        )
        lines.append("cmd.show('surface', 'pocket_surf')")
        lines.append("cmd.set('transparency', 0.25, 'pocket_surf')")
        lines.append("cmd.color('bluewhite', 'pocket_surf and elem C')")
        lines.append("cmd.set('surface_quality', 1)")

    # Ligand
    lines.append("cmd.show('sticks', 'ligand')")
    lines.append("cmd.set('stick_radius', 0.2, 'ligand')")
    lines.append("cmd.color('gold', 'ligand and elem C')")
    lines.append("cmd.color('red', 'ligand and elem O')")
    lines.append("cmd.color('blue', 'ligand and elem N')")
    lines.append("cmd.color('yellow', 'ligand and elem S')")

    if scene == "pocket":
        lines.append("cmd.show('spheres', 'ligand and name C')")
        lines.append("cmd.set('sphere_scale', 0.3, 'ligand')")

    # Interaction dashed lines
    if scene == "interaction" and interactions and center:
        dash = DASH_PRESETS["fine"]
        lines.append(f"cmd.set('dash_gap', {dash['dash_gap']})")
        lines.append(f"cmd.set('dash_radius', {dash['dash_radius']})")
        lines.append(f"cmd.set('dash_length', {dash['dash_length']})")
        lines.append(f"cmd.set('dash_as_cylinders', {int(dash['dash_as_cylinders'])})")

        for idx, inter in enumerate(interactions):
            itype = inter.get("type", "")
            if itype not in INTERACTION_COLORS:
                continue
            resn = inter.get("resn", "")
            resi = inter.get("resi", "")
            chain = inter.get("chain", "A")
            atom = inter.get("atom", "CA")

            pair_name = f"int_{idx}"
            prot_sel = (
                f"(receptor and resn {resn} and resi {resi} and chain {chain} and name {atom})"
            )
            lig_sel = "ligand"

            lines.append(f"try: cmd.distance('{pair_name}', '{prot_sel}', '{lig_sel}')")
            lines.append("except: pass")
            lines.append(
                f"try: cmd.set('dash_color', '{INTERACTION_COLORS[itype]}', '{pair_name}')"
            )
            lines.append("except: pass")
            lines.append(f"try: cmd.set('dash_width', 2.5, '{pair_name}')")
            lines.append("except: pass")
            lines.append(f"try: cmd.hide('labels', '{pair_name}')")
            lines.append("except: pass")

    # Labels for interacting residues
    if scene == "interaction" and interactions:
        resi_list = []
        for inter in interactions:
            resi = inter.get("resi", "")
            chain = inter.get("chain", "A")
            if resi:
                resi_list.append(f"(receptor and resi {resi} and chain {chain} and name CA)")
        if resi_list:
            sel = " or ".join(resi_list)
            lines.append(f"cmd.select('lab_sel', '{sel}')")
            lines.append("cmd.label('lab_sel', '\"%s-%s\" % (resn, resi)')")
            lines.append("cmd.set('label_color', 'white')")
            lines.append("cmd.set('label_size', 20)")

    # Camera: center on ligand
    lines.append("cmd.center('ligand')")
    if scene == "pocket" and center:
        cx, cy, cz = center
        lines.append(f"cmd.origin(center=[{cx}, {cy}, {cz}])")

    # Ray tracing settings
    lines.append("cmd.set('ray_trace_mode', 1)")
    lines.append("cmd.set('ray_shadows', 0)")
    lines.append("cmd.set('antialias', 2)")
    lines.append("cmd.set('ambient', 0.5)")
    lines.append("cmd.set('specular', 0.6)")
    lines.append("cmd.set('shininess', 55)")

    # Render
    lines.append(f"cmd.ray({width}, {height})")
    lines.append(f'cmd.png("{output_png}", dpi={DEFAULT_DPI})')
    lines.append("cmd.quit()")

    return "\n".join(lines)


def render_scene_pymol(
    receptor_pdb: str,
    ligand_pdbqt: str,
    output_png: str,
    scene: str = "pocket",
    center: tuple[float, float, float] | None = None,
    interactions: list[dict[str, Any]] | None = None,
    width: int = DEFAULT_RAY_WIDTH,
    height: int = DEFAULT_RAY_HEIGHT,
) -> str:
    """
    Render a 3D scene using PyMOL CLI.

    Args:
        receptor_pdb: Receptor PDB file.
        ligand_pdbqt: Ligand PDBQT file.
        output_png: Output PNG path.
        scene: 'complex' | 'pocket' | 'interaction' | 'ligand_closeup'.
        center: Pocket center for camera positioning.
        interactions: List of interaction dicts (for 'interaction' scene).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Path to output PNG.
    """
    if not _PYMOL_EXE:
        raise VisualizationError(
            "PyMOL executable not found. Install: conda install -c conda-forge pymol-open-source"
        )

    ensure_dir(os.path.dirname(output_png) or ".")

    script = _build_pymol_script(
        receptor_pdb,
        ligand_pdbqt,
        output_png,
        scene=scene,
        center=center,
        interactions=interactions,
        width=width,
        height=height,
    )

    script_path = tempfile.mktemp(suffix=".pml")
    with open(script_path, "w") as fh:
        fh.write(script)

    try:
        success, stdout, stderr = safe_subprocess(
            [_PYMOL_EXE, "-cq", script_path],
            timeout=300,
        )
        if not success:
            raise VisualizationError(f"PyMOL rendering failed: {stderr[:500]}")
    finally:
        with contextlib.suppress(Exception):
            os.remove(script_path)

    if not os.path.exists(output_png):
        raise VisualizationError(f"PyMOL did not produce output: {output_png}")

    logger.info(f"3D scene rendered: {output_png}")
    return output_png


# ─────────────────────────────────────────────────────────────────────────────
# RDKit 2D Interaction Diagram
# ─────────────────────────────────────────────────────────────────────────────


def _parse_smiles_idx_from_pdbqt(ligand_pdbqt: str) -> dict[int, int]:
    """Parse REMARK SMILES IDX lines from PDBQT.

    Returns mapping: PDB serial number -> SMILES atom index (1-based).
    """
    smiles_idx_map: dict[int, int] = {}
    with open(ligand_pdbqt) as fh:
        for line in fh:
            if line.startswith("REMARK SMILES IDX"):
                parts = line.strip().split()
                nums = [int(x) for x in parts[3:]]
                for i in range(0, len(nums), 2):
                    smiles_idx = nums[i]
                    pdb_serial = nums[i + 1]
                    smiles_idx_map[pdb_serial] = smiles_idx
    return smiles_idx_map


def _parse_pdbqt_coords(ligand_pdbqt: str) -> dict[tuple[float, float, float], int]:
    """Parse ATOM/HETATM coordinates from PDBQT.

    Returns mapping: rounded (x, y, z) -> PDB serial number.
    """
    coords_map: dict[tuple[float, float, float], int] = {}
    with open(ligand_pdbqt) as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                serial = int(line[6:11].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords_map[(round(x, 3), round(y, 3), round(z, 3))] = serial
    return coords_map


def render_interactions_2d(
    receptor_pdb: str,
    ligand_pdbqt: str,
    interactions: list[dict[str, Any]],
    output_png: str,
    width: int = 1200,
    height: int = 900,
    dpi: int = DEFAULT_DPI,
) -> str:
    """
    Render a 2D interaction diagram using RDKit + PIL.

    Draws the ligand 2D structure with highlighted interacting atoms
    colored by interaction type, plus residue labels positioned near
    the corresponding atoms.

    Args:
        receptor_pdb: Receptor PDB (not directly used, kept for API consistency).
        ligand_pdbqt: Ligand PDBQT (parsed for structure).
        interactions: List of interaction dicts from detect_interactions().
        output_png: Output PNG path.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        dpi: Image DPI.

    Returns:
        Path to output PNG.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
    except ImportError as exc:
        raise VisualizationError(f"Required packages missing for 2D rendering: {exc}")

    # ── Parse ligand structure ────────────────────────────────────────────────
    smiles = None
    with open(ligand_pdbqt) as fh:
        for line in fh:
            if line.startswith("REMARK SMILES ") and not line.startswith("REMARK SMILES IDX"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    smiles = parts[2]
                break

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
    else:
        with open(ligand_pdbqt) as fh:
            lines = fh.readlines()
        clean = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
        mol = Chem.MolFromPDBBlock("".join(clean))

    if mol is None:
        raise VisualizationError("Could not parse ligand structure for 2D rendering")

    mol = Chem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol)

    # ── Build mappings from PDBQT ─────────────────────────────────────────────
    smiles_idx_map = _parse_smiles_idx_from_pdbqt(ligand_pdbqt)
    pdbqt_coords = _parse_pdbqt_coords(ligand_pdbqt)

    # ── Map interactions to RDKit atoms ───────────────────────────────────────
    # Group by (type, resn, resi, chain) to deduplicate labels
    interaction_groups: dict[tuple, dict[str, Any]] = {}
    for inter in interactions:
        key = (inter.get("type"), inter.get("resn"), inter.get("resi"), inter.get("chain"))
        if key not in interaction_groups:
            interaction_groups[key] = {
                "type": inter.get("type"),
                "resn": inter.get("resn"),
                "resi": inter.get("resi"),
                "chain": inter.get("chain"),
                "color": inter.get("color"),
                "rdkit_atoms": set(),
            }
        for atom_info in inter.get("ligand_atoms", []):
            coords = tuple(round(c, 3) for c in atom_info["coords"])
            pdb_serial = pdbqt_coords.get(coords)
            if pdb_serial and pdb_serial in smiles_idx_map:
                rdkit_idx = smiles_idx_map[pdb_serial] - 1
                if 0 <= rdkit_idx < mol.GetNumAtoms():
                    interaction_groups[key]["rdkit_atoms"].add(rdkit_idx)

    # ── Build RDKit highlight dictionaries ────────────────────────────────────
    highlight_atoms: set[int] = set()
    highlight_atom_colors: dict[int, tuple[float, float, float]] = {}

    color_rgb_float = {
        "cyan": (0.0, 0.75, 0.75),
        "orange": (1.0, 0.55, 0.0),
        "green": (0.0, 0.65, 0.0),
        "purple": (0.55, 0.15, 0.85),
        "red": (0.85, 0.1, 0.15),
        "yellow": (0.9, 0.75, 0.0),
        "blue": (0.15, 0.45, 1.0),
        "grey": (0.5, 0.5, 0.5),
    }

    for group in interaction_groups.values():
        color_name = group["color"]
        rgb = color_rgb_float.get(color_name, (0.5, 0.5, 0.5))
        for rdkit_idx in group["rdkit_atoms"]:
            highlight_atoms.add(rdkit_idx)
            # If atom already has a color, keep the first one encountered
            if rdkit_idx not in highlight_atom_colors:
                highlight_atom_colors[rdkit_idx] = rgb

    # ── Draw molecule with highlights ─────────────────────────────────────────
    # Scale canvas by DPI for publication-quality output (RDKit Cairo works in px)
    canvas_w = int(width * dpi / 100)
    canvas_h = int(height * dpi / 100)
    drawer = Draw.MolDraw2DCairo(canvas_w, canvas_h)
    drawer.drawOptions().highlightRadius = 0.25
    drawer.drawOptions().clearBackground = True

    if highlight_atoms:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_atoms),
            highlightAtomColors=highlight_atom_colors,
        )
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()

    img = Image.open(__import__("io").BytesIO(png_data))
    draw = ImageDraw.Draw(img)

    # ── Fonts ─────────────────────────────────────────────────────────────────
    def _load_font(size: int):
        """Cross-platform font loader with sensible fallbacks."""
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",      # macOS
            "/System/Library/Fonts/HelveticaNeue.ttc",  # macOS alt
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/usr/share/fonts/TTF/DejaVuSans.ttf",      # Linux alt
            "C:/Windows/Fonts/arial.ttf",               # Windows
            "C:/Windows/Fonts/segoeui.ttf",             # Windows alt
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    font = _load_font(16)

    # ── Draw residue labels near atoms ────────────────────────────────────────
    # Sort groups so that labels with fewer atoms (more specific) are placed first
    sorted_groups = sorted(
        interaction_groups.values(),
        key=lambda g: len(g["rdkit_atoms"]),
    )

    placed_boxes: list[tuple[int, int, int, int]] = []

    # Count how many labels anchor to each atom (for fan-out placement)
    atom_label_counts: dict[int, int] = {}
    for group in sorted_groups:
        if not group["rdkit_atoms"]:
            continue
        best_atom = min(group["rdkit_atoms"], key=lambda idx: drawer.GetDrawCoords(idx).y)
        atom_label_counts[best_atom] = atom_label_counts.get(best_atom, 0) + 1

    atom_label_indices: dict[int, int] = {}

    for group in sorted_groups:
        if not group["rdkit_atoms"]:
            continue

        label = f"{group['resn']}{group['resi']}"
        color_name = group["color"]
        rgb = tuple(int(c * 255) for c in color_rgb_float.get(color_name, (0.5, 0.5, 0.5)))

        # Pick the atom closest to the top of the canvas for this group
        best_atom = min(group["rdkit_atoms"], key=lambda idx: drawer.GetDrawCoords(idx).y)
        pos = drawer.GetDrawCoords(best_atom)
        ax, ay = int(pos.x), int(pos.y)

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 3

        # Fan-out labels that share the same anchor atom
        total_for_atom = atom_label_counts.get(best_atom, 1)
        idx_for_atom = atom_label_indices.get(best_atom, 0)
        atom_label_indices[best_atom] = idx_for_atom + 1

        base_angle = -math.pi / 2  # straight up
        if total_for_atom > 1:
            spread = min(math.pi * 0.6, 0.35 * total_for_atom)
            base_angle = -math.pi / 2 + (idx_for_atom - (total_for_atom - 1) / 2) * spread / (
                total_for_atom - 1
            )

        dist = 55
        lx = ax + int(dist * math.cos(base_angle)) - text_w // 2
        ly = ay + int(dist * math.sin(base_angle)) - text_h // 2

        # Check overlap with already-placed boxes and nudge if needed
        for _ in range(15):
            box = (lx - padding, ly - padding, lx + text_w + padding, ly + text_h + padding)
            overlap = False
            for bx1, by1, bx2, by2 in placed_boxes:
                if not (box[2] < bx1 or box[0] > bx2 or box[3] < by1 or box[1] > by2):
                    overlap = True
                    lx += 10
                    ly -= 6
                    break
            if not overlap:
                break

        # Keep within canvas margins
        lx = max(5, min(lx, width - text_w - 5))
        ly = max(5, min(ly, height - text_h - 5))

        # Draw leader line from atom to label centre
        draw.line([(ax, ay), (lx + text_w // 2, ly + text_h // 2)], fill=rgb, width=1)

        # Draw label background box
        draw.rectangle(
            [(lx - padding, ly - padding), (lx + text_w + padding, ly + text_h + padding)],
            fill=(255, 255, 255, 220),
            outline=rgb,
        )
        draw.text((lx, ly), label, fill=(0, 0, 0), font=font)

        placed_boxes.append(
            (
                lx - padding,
                ly - padding,
                lx + text_w + padding,
                ly + text_h + padding,
            )
        )

    # ── Draw legend box ───────────────────────────────────────────────────────
    color_rgb_int = {
        "cyan": (0, 190, 190),
        "orange": (255, 140, 0),
        "green": (0, 170, 0),
        "purple": (140, 40, 230),
        "red": (210, 30, 50),
        "yellow": (230, 190, 0),
        "blue": (40, 115, 255),
        "grey": (128, 128, 128),
    }

    type_counts: dict[str, int] = {}
    for group in interaction_groups.values():
        t = group["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    legend_x = width - 280
    legend_y = 20
    legend_h = 30 + len(type_counts) * 24
    draw.rectangle(
        [(legend_x - 10, legend_y - 10), (legend_x + 260, legend_y + legend_h)],
        fill=(255, 255, 255, 230),
        outline=(100, 100, 100),
    )
    draw.text((legend_x, legend_y), "Interactions", fill=(0, 0, 0), font=font)

    y_off = legend_y + 28
    for itype, count in sorted(type_counts.items()):
        color_name = INTERACTION_COLORS.get(itype, "grey")
        rgb = color_rgb_int.get(color_name, (128, 128, 128))
        draw.ellipse([(legend_x, y_off), (legend_x + 14, y_off + 14)], fill=rgb)
        draw.text((legend_x + 20, y_off), f"{itype}: {count}", fill=(0, 0, 0), font=font)
        y_off += 24

    ensure_dir(os.path.dirname(output_png) or ".")
    img.save(output_png, dpi=(dpi, dpi))
    logger.info(f"2D interaction diagram rendered: {output_png}")
    return output_png


# ─────────────────────────────────────────────────────────────────────────────
# Composite figure assembly
# ─────────────────────────────────────────────────────────────────────────────


def composite_summary(
    panel_paths: list[str],
    output_png: str,
    ncols: int = 2,
    panel_titles: list[str] | None = None,
    figure_title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> str:
    """
    Assemble multiple panel images into a single composite figure.

    Args:
        panel_paths: List of PNG file paths.
        output_png: Output composite PNG path.
        ncols: Number of columns.
        panel_titles: Optional titles for each panel.
        figure_title: Optional overall figure title.
        dpi: Output DPI.

    Returns:
        Path to output PNG.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not panel_paths:
        raise VisualizationError("No panels provided for composite figure")

    images = [Image.open(p) for p in panel_paths if os.path.exists(p)]
    if not images:
        raise VisualizationError("No valid panel images found")

    nrows = (len(images) + ncols - 1) // ncols
    panel_w = max(img.width for img in images)
    panel_h = max(img.height for img in images)

    title_h = 60 if figure_title else 0
    label_h = 40 if panel_titles else 0
    total_w = panel_w * ncols + 20 * (ncols + 1)
    total_h = panel_h * nrows + label_h * nrows + title_h + 20 * (nrows + 1)

    composite = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(composite)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    if figure_title:
        draw.text((20, 20), figure_title, fill=(0, 0, 0), font=title_font)

    y_offset = title_h + 20
    for row in range(nrows):
        x_offset = 20
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= len(images):
                break
            img = images[idx]
            # Center the image in its cell
            x_pos = x_offset + (panel_w - img.width) // 2
            y_pos = y_offset + label_h + (panel_h - img.height) // 2
            composite.paste(img, (x_pos, y_pos))

            if panel_titles and idx < len(panel_titles):
                draw.text(
                    (x_offset + 10, y_offset),
                    panel_titles[idx],
                    fill=(0, 0, 0),
                    font=label_font,
                )

            x_offset += panel_w + 20
        y_offset += panel_h + label_h + 20

    ensure_dir(os.path.dirname(output_png) or ".")
    composite.save(output_png, dpi=(dpi, dpi))
    logger.info(f"Composite figure saved: {output_png}")
    return output_png
