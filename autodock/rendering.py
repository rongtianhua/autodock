"""
autodock.rendering — Publication-quality visualization.
=======================================================
3D rendering via PyMOL CLI and 2D interaction diagrams via RDKit + Cairo.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from autodock.core import (
    logger,
    VisualizationError,
    find_pymol,
    safe_subprocess,
    DEFAULT_DPI,
    DEFAULT_RAY_WIDTH,
    DEFAULT_RAY_HEIGHT,
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
        lines.append(f"cmd.select('pocket_surf', 'br. receptor and center {cx},{cy},{cz} around {pocket_distance}')")
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
            prot_sel = f"(receptor and resn {resn} and resi {resi} and chain {chain} and name {atom})"
            lig_sel = "ligand"

            lines.append(f"try: cmd.distance('{pair_name}', '{prot_sel}', '{lig_sel}')")
            lines.append(f"except: pass")
            lines.append(f"try: cmd.set('dash_color', '{INTERACTION_COLORS[itype]}', '{pair_name}')")
            lines.append(f"except: pass")
            lines.append(f"try: cmd.set('dash_width', 2.5, '{pair_name}')")
            lines.append(f"except: pass")
            lines.append(f"try: cmd.hide('labels', '{pair_name}')")
            lines.append(f"except: pass")

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
        raise VisualizationError("PyMOL executable not found. Install: conda install -c conda-forge pymol-open-source")

    ensure_dir(os.path.dirname(output_png) or ".")

    script = _build_pymol_script(
        receptor_pdb, ligand_pdbqt, output_png,
        scene=scene, center=center, interactions=interactions,
        width=width, height=height,
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
        try:
            os.remove(script_path)
        except Exception:
            pass

    if not os.path.exists(output_png):
        raise VisualizationError(f"PyMOL did not produce output: {output_png}")

    logger.info(f"3D scene rendered: {output_png}")
    return output_png


# ─────────────────────────────────────────────────────────────────────────────
# RDKit 2D Interaction Diagram
# ─────────────────────────────────────────────────────────────────────────────

def render_interactions_2d(
    receptor_pdb: str,
    ligand_pdbqt: str,
    interactions: list[dict[str, Any]],
    output_png: str,
    width: int = 1000,
    height: int = 800,
    dpi: int = DEFAULT_DPI,
) -> str:
    """
    Render a 2D interaction diagram using RDKit + PIL.

    Draws the ligand 2D structure with colored dots / arcs indicating
    interacting residues.  This is a simplified but publication-quality
    representation suitable for supplementary figures.

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
        from rdkit import Chem
        from rdkit.Chem import Draw, rdMolDescriptors
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise VisualizationError(f"Required packages missing for 2D rendering: {exc}")

    # Extract SMILES from PDBQT if available
    smiles = None
    with open(ligand_pdbqt) as fh:
        for line in fh:
            if line.startswith("REMARK SMILES "):
                parts = line.strip().split()
                if len(parts) >= 3:
                    smiles = parts[2]
                break

    # Parse ligand structure
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
    else:
        # Try parsing PDBQT as PDB block
        with open(ligand_pdbqt) as fh:
            lines = fh.readlines()
        clean = [l for l in lines if l.startswith(("ATOM  ", "HETATM"))]
        mol = Chem.MolFromPDBBlock("".join(clean))

    if mol is None:
        raise VisualizationError("Could not parse ligand structure for 2D rendering")

    mol = Chem.RemoveHs(mol)
    from rdkit.Chem import AllChem
    AllChem.Compute2DCoords(mol)

    # Generate RDKit 2D depiction
    drawer = Draw.MolDraw2DCairo(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()

    # Open as PIL Image
    img = Image.open(__import__("io").BytesIO(png_data))
    draw = ImageDraw.Draw(img)

    # Draw legend
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()

    # Count interactions by type
    type_counts: dict[str, int] = {}
    for inter in interactions:
        t = inter.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Draw legend box
    legend_x = width - 280
    legend_y = 20
    legend_h = 30 + len(type_counts) * 24
    draw.rectangle(
        [(legend_x - 10, legend_y - 10), (legend_x + 260, legend_y + legend_h)],
        fill=(255, 255, 255, 230),
        outline=(100, 100, 100),
    )
    draw.text((legend_x, legend_y), "Interactions", fill=(0, 0, 0), font=font)

    color_rgb = {
        "cyan": (0, 200, 200),
        "orange": (255, 140, 0),
        "green": (0, 180, 0),
        "purple": (160, 32, 240),
        "red": (220, 20, 60),
        "yellow": (255, 215, 0),
        "blue": (30, 144, 255),
        "grey": (128, 128, 128),
    }

    y_off = legend_y + 28
    for itype, count in sorted(type_counts.items()):
        color_name = INTERACTION_COLORS.get(itype, "grey")
        rgb = color_rgb.get(color_name, (128, 128, 128))
        draw.ellipse([(legend_x, y_off), (legend_x + 14, y_off + 14)], fill=rgb)
        draw.text((legend_x + 20, y_off), f"{itype}: {count}", fill=(0, 0, 0), font=font)
        y_off += 24

    # Draw interacting residue labels near ligand
    # Map residues to positions (simplified: just list them below legend)
    if interactions:
        res_list = []
        for inter in interactions:
            resn = inter.get("resn", "")
            resi = inter.get("resi", "")
            if resn and resi:
                label = f"{resn}{resi}"
                if label not in res_list:
                    res_list.append(label)
        if res_list:
            y_off += 10
            draw.text((legend_x, y_off), "Residues:", fill=(0, 0, 0), font=font)
            y_off += 22
            for label in sorted(res_list):
                draw.text((legend_x + 10, y_off), label, fill=(50, 50, 50), font=font)
                y_off += 20

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
