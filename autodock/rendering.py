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

# ─────────────────────────────────────────────────────────────────────────────
# Publication-quality rendering presets
# ─────────────────────────────────────────────────────────────────────────────

DASH_PRESETS = {
    "fine": {
        "dash_gap": 0.35,
        "dash_radius": 0.04,
        "dash_length": 0.25,
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

# Publication-grade colour schemes aligned with Nature/Science conventions
COLOR_SCHEMES: dict[str, dict[str, Any]] = {
    "publication_white": {
        "bg": "white",
        "receptor_c": "lightblue",
        "ligand_c": "salmon",
        "label_c": "black",
        "pocket_surface": "lightblue",
        "receptor_style": "cartoon",
        "receptor_transparency": 0.0,
    },
    "publication_grey": {
        "bg": "white",
        "receptor_c": "grey80",
        "ligand_c": "salmon",
        "label_c": "black",
        "pocket_surface": "bluewhite",
        "receptor_style": "cartoon",
        "receptor_transparency": 0.15,
    },
    "presentation_black": {
        "bg": "black",
        "receptor_c": "grey80",
        "ligand_c": "gold",
        "label_c": "white",
        "pocket_surface": "bluewhite",
        "receptor_style": "cartoon",
        "receptor_transparency": 0.2,
    },
}

# High-quality ray-tracing defaults (PyMOL wiki + bionerdnotes best practice)
_RAY_QUALITY_PRESET = {
    "antialias": 3,
    "hash_max": 300,
    "ray_trace_fog": 0,
    "depth_cue": 0,
    "orthoscopic": "on",
    "ray_shadows": 0,
    "ambient": 0.35,
    "specular": 0.45,
    "shininess": 60,
    "direct": 0.55,
    "reflect": 0.15,
    "cartoon_fancy_helices": 1,
    "cartoon_fancy_sheets": 1,
    "cartoon_oval_length": 0.8,
    "cartoon_oval_width": 0.2,
    "cartoon_rect_length": 1.25,
    "cartoon_rect_width": 0.25,
    "cartoon_loop_radius": 0.15,
    "cartoon_dumbbell_length": 1.25,
    "cartoon_dumbbell_width": 0.25,
    "cartoon_dumbbell_radius": 0.18,
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
    save_pse: str | None = None,
    color_scheme: str = "presentation_black",
    receptor_source: str = "auto",
) -> str:
    """Build a PyMOL command script for publication-quality rendering.

    Parameters
    ----------
    color_scheme:
        One of ``publication_white`` (default), ``publication_grey``,
        ``presentation_black``.
    receptor_source:
        ``"AlphaFold"``, ``"PDB"``, ``"PDB_single_chain"``, or ``"file"``.
        Determines protein coloring: AlphaFold → pLDDT (B-factor) rainbow;
        PDB → chainbow (N→C blue→red).
    """
    scheme = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["presentation_black"])
    is_af = receptor_source in ("AlphaFold", "SWISS-MODEL")
    is_pdb = receptor_source in ("PDB", "PDB_single_chain", "file")

    lines: list[str] = []
    lines.append("cmd.delete('all')")
    lines.append(f'cmd.load("{receptor_pdb}", "receptor")')
    lines.append("cmd.show('cartoon', 'receptor')")
    lines.append("cmd.set('cartoon_side_chain_helper', 1)")
    lines.append("cmd.set('cartoon_discrete_colors', 0)")

    # Protein coloring based on source (BEFORE loading ligand to avoid PyMOL 3.x bug
    # where loading a second object wipes selection-level spectrum coloring)
    if is_af:
        # AlphaFold pLDDT coloring via B-factor (pLDDT range 0-100)
        # rainbow_rev palette: low B-factor (low confidence, pLDDT 0) → red/yellow,
        #                      high B-factor (high confidence, pLDDT 100) → blue
        lines.append(
            "cmd.spectrum('b', 'rainbow_rev', 'receptor', minimum=0, maximum=100)"
        )
    else:
        # PDB / crystal: chainbow (N-terminus blue → C-terminus red)
        lines.append("cmd.spectrum('count', 'rainbow', 'receptor')")

    # ── Load ligand AFTER spectrum ──
    lines.append(f'cmd.load("{ligand_pdbqt}", "ligand")')

    # ── Viewport (must be set BEFORE ray) ──
    lines.append(f"cmd.viewport({width}, {height})")

    # ── Background ──
    lines.append(f"cmd.bg_color('{scheme['bg']}')")

    # ── Global quality settings ──
    for key, val in _RAY_QUALITY_PRESET.items():
        if isinstance(val, str):
            lines.append(f"cmd.set('{key}', '{val}')")
        else:
            lines.append(f"cmd.set('{key}', {val})")

    # For interaction scene, use cartoon_transparency to dim non-pocket cartoon
    # instead of cmd.hide which wipes distance objects in PyMOL 3.1.8
    if scene == "interaction" and center:
        lines.append(
            "cmd.select('pocket_vis', 'byres (receptor within 15.0 of ligand)')"
        )
        lines.append("cmd.set('cartoon_transparency', 1.0, 'receptor and not pocket_vis')")
        lines.append("cmd.set('cartoon_transparency', 0.2, 'pocket_vis')")
    else:
        # Cartoon transparency for pocket / interaction scenes
        if scene in ("pocket", "interaction"):
            t = scheme.get("receptor_transparency", 0.15)
            lines.append(f"cmd.set('cartoon_transparency', {t}, 'receptor')")

    # ── Pocket surface ──
    if scene == "pocket" and center:
        cx, cy, cz = center
        lines.append(
            "cmd.select('pocket_surf',"
            f"'br. receptor and center {cx},{cy},{cz}"
            f" around {pocket_distance}')"
        )
        lines.append("cmd.show('surface', 'pocket_surf')")
        lines.append("cmd.set('transparency', 0.30, 'pocket_surf')")
        lines.append(f"cmd.color('{scheme['pocket_surface']}', 'pocket_surf and elem C')")
        lines.append("cmd.set('surface_quality', 2)")

    # ── Ligand: ball-and-stick with publication CPK colors ──
    # Carbon: grey (0x999999 ~ [0.6,0.6,0.6]), Oxygen: red, Nitrogen: blue,
    # Sulfur: yellow, Chlorine: green, Bromine: brown, Phosphorus: purple, Fluorine: magenta
    lines.append("cmd.show('sticks', 'ligand')")
    lines.append("cmd.show('spheres', 'ligand')")
    lines.append("cmd.set('stick_radius', 0.15, 'ligand')")
    lines.append("cmd.set('sphere_scale', 0.25, 'ligand')")
    # Publication-standard CPK colors with grey carbon
    lines.append("cmd.set_color('greyc', [0.6, 0.6, 0.6])")
    lines.append("cmd.color('greyc', 'ligand and elem C')")
    lines.append("cmd.color('red', 'ligand and elem O')")
    lines.append("cmd.color('blue', 'ligand and elem N')")
    lines.append("cmd.color('yellow', 'ligand and elem S')")
    lines.append("cmd.color('green', 'ligand and elem Cl')")
    lines.append("cmd.color('brown', 'ligand and elem Br')")
    lines.append("cmd.color('purple', 'ligand and elem P')")
    lines.append("cmd.color('magenta', 'ligand and elem F')")
    lines.append("cmd.set('sphere_quality', 3, 'ligand')")
    lines.append("cmd.set('stick_quality', 3, 'ligand')")

    # ── Interaction dashed lines (CGO-based, PyMOL 3.x Open Source compatible) ──
    if scene == "interaction" and interactions and center:
        lines.append("python")
        lines.append("from pymol import cmd")
        lines.append("from pymol.cgo import CYLINDER")
        lines.append("import math")
        lines.append("")
        lines.append("def _dashed_line(p1, p2, radius=0.08, dash_len=0.4, gap_len=0.2, color=(1.0, 0.5, 0.0)):")
        lines.append("    dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]")
        lines.append("    total = math.sqrt(dx*dx + dy*dy + dz*dz)")
        lines.append("    if total == 0: return []")
        lines.append("    ux, uy, uz = dx/total, dy/total, dz/total")
        lines.append("    cgo = []")
        lines.append("    seg = dash_len + gap_len")
        lines.append("    n = int(total / seg) + 1")
        lines.append("    for i in range(n):")
        lines.append("        s = i * seg")
        lines.append("        e = min(s + dash_len, total)")
        lines.append("        if s >= total: break")
        lines.append("        sx = p1[0] + ux * s; sy = p1[1] + uy * s; sz = p1[2] + uz * s")
        lines.append("        ex = p1[0] + ux * e; ey = p1[1] + uy * e; ez = p1[2] + uz * e")
        lines.append("        cgo.extend([CYLINDER, sx, sy, sz, ex, ey, ez, radius,")
        lines.append("                   color[0], color[1], color[2], color[0], color[1], color[2]])")
        lines.append("    return cgo")
        lines.append("")
        # Color mapping from name to RGB tuple
        color_map = {
            "cyan": "(0.0, 1.0, 1.0)",
            "orange": "(1.0, 0.5, 0.0)",
            "green": "(0.0, 1.0, 0.0)",
            "purple": "(1.0, 0.0, 1.0)",
            "red": "(1.0, 0.0, 0.0)",
            "yellow": "(1.0, 1.0, 0.0)",
            "blue": "(0.0, 0.0, 1.0)",
            "grey": "(0.5, 0.5, 0.5)",
        }
        for idx, inter in enumerate(interactions):
            itype = inter.get("type", "")
            if itype not in INTERACTION_COLORS:
                continue
            resn = inter.get("resn", "")
            resi = inter.get("resi", "")
            chain = inter.get("chain", "A")
            color_name = INTERACTION_COLORS[itype]
            color_rgb = color_map.get(color_name, "(1.0, 0.5, 0.0)")
            prot_sel = f"receptor and resn {resn} and resi {resi} and chain {chain}"
            lines.append(f"try:")
            lines.append(f"    m1 = cmd.get_model('{prot_sel}')")
            lines.append(f"    m2 = cmd.get_model('ligand')")
            lines.append(f"    if m1.atom and m2.atom:")
            lines.append(f"        dmin = 9999; pair = None")
            lines.append(f"        for a in m1.atom:")
            lines.append(f"            for b in m2.atom:")
            lines.append(f"                d = cmd.get_distance(f'receptor and index {{a.index}}', f'ligand and index {{b.index}}')")
            lines.append(f"                if d < dmin: dmin = d; pair = (a.coord, b.coord)")
            lines.append(f"        if pair and dmin <= 4.5:")
            lines.append(f"            obj = _dashed_line(pair[0], pair[1], radius=0.10, dash_len=0.4, gap_len=0.2, color={color_rgb})")
            lines.append(f"            if obj:")
            lines.append(f"                cmd.load_cgo(obj, 'int_{idx}')")
            lines.append(f"                cmd.show('cgo', 'int_{idx}')")
            lines.append("except: pass")
        lines.append("python end")

    # ── Labels for interacting residues (directional offset to avoid overlap) ──
    if scene == "interaction" and interactions:
        label_color = scheme.get("label_c", "white")
        # De-duplicate residues so each residue gets only one label
        seen_residues: set[tuple[str, str, str]] = set()
        for idx, inter in enumerate(interactions):
            resn = inter.get("resn", "")
            resi = str(inter.get("resi", ""))
            chain = inter.get("chain", "A")
            if not resi:
                continue
            res_key = (resn, resi, chain)
            if res_key in seen_residues:
                continue
            seen_residues.add(res_key)

            lab_name = f"lab_{idx}"
            sel = f"(receptor and resn {resn} and resi {resi} and chain {chain} and name CA)"
            pseudo = f"labpos_{idx}"
            lines.append("try:")
            lines.append(f"    cmd.select('{lab_name}', '{sel}')")
            # Create pseudoatom at CA
            lines.append(f"    cmd.pseudoatom('{pseudo}', '{lab_name}')")
            # Use directional offset based on residue position relative to ligand center
            # to spread labels outward and reduce overlap
            lines.append(f"    ca = cmd.get_model('{sel}').atom")
            lines.append(f"    if ca:")
            lines.append(f"        c = ca[0].coord")
            lines.append(f"        com = cmd.get_extent('ligand')")
            lines.append(f"        lig_c = [(com[0][i]+com[1][i])/2 for i in range(3)]")
            lines.append(f"        dx = c[0] - lig_c[0]; dy = c[1] - lig_c[1]; dz = c[2] - lig_c[2]")
            lines.append(f"        dist = math.sqrt(dx*dx + dy*dy + dz*dz)")
            lines.append(f"        if dist > 0:")
            lines.append(f"            # Normalize and scale offset to 4.0 Å outward from ligand")
            lines.append(f"            scale = 4.0 / dist")
            lines.append(f"            cmd.translate([dx*scale, dy*scale, dz*scale], '{pseudo}')")
            lines.append(f"        else:")
            lines.append(f"            cmd.translate([0, 0, 4.0], '{pseudo}')")
            # Label with residue name, number and chain: e.g. LYS211(A)
            lines.append(f"    cmd.label('{pseudo}', '\"{resn}{resi}({chain})\"')")
            # Label style: white text, bold sans-serif
            lines.append(f"    cmd.set('label_color', '{label_color}', '{pseudo}')")
            lines.append(f"    cmd.set('label_size', 40, '{pseudo}')")
            lines.append(f"    cmd.set('label_font_id', 10, '{pseudo}')")  # Sans-serif bold
            lines.append("except: pass")

    # ── Camera / viewport ──
    # Use zoom with buffer to control field of view per scene
    if scene == "complex":
        lines.append("cmd.zoom('(receptor or ligand)', 5)")
    elif scene == "pocket":
        lines.append("cmd.zoom('(receptor or ligand)', 3)")
        if center:
            cx, cy, cz = center
            lines.append(f"cmd.origin([{cx}, {cy}, {cz}])")
    elif scene == "interaction":
        # Zoom to ligand to crop away non-pocket cartoon and reduce clutter
        lines.append("cmd.zoom('ligand', 2)")
        if center:
            cx, cy, cz = center
            lines.append(f"cmd.origin([{cx}, {cy}, {cz}])")
    else:
        lines.append("cmd.center('ligand')")

    # ── Ray trace mode: 0 = normal high-quality (no outline) ──
    # Mode 1 adds black outlines which can look cartoonish;
    # mode 0 with antialias=3 gives the cleanest publication look.
    lines.append("cmd.set('ray_trace_mode', 0)")
    # Additional quality settings for crisp edges
    lines.append("cmd.set('ray_trace_color', 'black')")
    lines.append("cmd.set('ray_trace_disco_factor', 1.0)")

    # ── Render ──
    # viewport must be set before ray(); cmd.png with ray=1 triggers ray-tracing
    lines.append(f"cmd.ray({width}, {height})")
    lines.append(f'cmd.png("{output_png}", dpi={DEFAULT_DPI}, ray=1)')
    if save_pse:
        lines.append(f'cmd.save("{save_pse}")')
    lines.append("cmd.quit()")

    return "\n".join(lines)


def render_scene_pymol(
    receptor_pdb: str,
    ligand_pdbqt: str,
    output_png: str,
    output_pdf: str | None = None,
    scene: str = "pocket",
    center: tuple[float, float, float] | None = None,
    interactions: list[dict[str, Any]] | None = None,
    width: int = DEFAULT_RAY_WIDTH,
    height: int = DEFAULT_RAY_HEIGHT,
    save_pse: str | None = None,
    color_scheme: str = "presentation_black",
    receptor_source: str = "auto",
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
        width: Image width in pixels (default 2400).
        height: Image height in pixels (default 1800).
        save_pse: Optional path to save a PyMOL session (.pse) file.
        color_scheme: Colour preset — ``presentation_black`` (default),
            ``publication_white``, or ``publication_grey``.
        receptor_source: ``"AlphaFold"``, ``"PDB"``, ``"PDB_single_chain"``,
            or ``"file"``. Determines protein coloring.

    Returns:
        Path to output PNG.
    """
    if not _PYMOL_EXE:
        raise VisualizationError(
            "PyMOL executable not found. Install: conda install -c conda-forge pymol-open-source"
        )

    ensure_dir(os.path.dirname(output_png) or ".")
    if save_pse:
        ensure_dir(os.path.dirname(save_pse) or ".")

    script = _build_pymol_script(
        receptor_pdb,
        ligand_pdbqt,
        output_png,
        scene=scene,
        center=center,
        interactions=interactions,
        width=width,
        height=height,
        save_pse=save_pse,
        color_scheme=color_scheme,
        receptor_source=receptor_source,
    )

    fd, script_path = tempfile.mkstemp(suffix=".pml")
    os.close(fd)
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

    # Optional PDF output — PIL converts PNG raster to PDF
    if output_pdf:
        ensure_dir(os.path.dirname(output_pdf) or ".")
        try:
            from PIL import Image as _PILImage

            _img = _PILImage.open(output_png)
            _rgb = _img.convert("RGB")
            _rgb.save(output_pdf, dpi=(DEFAULT_DPI, DEFAULT_DPI), format="PDF")
            logger.info(f"3D scene (PDF): {output_pdf}")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"3D PDF output skipped: {exc}")

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
                try:
                    serial = int(line[6:11].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                coords_map[(round(x, 3), round(y, 3), round(z, 3))] = serial
    return coords_map


# ─────────────────────────────────────────────────────────────────────────────
# LigPlot+ style drawing primitives
# ─────────────────────────────────────────────────────────────────────────────


def _draw_dashed_line(
    draw: Any,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill: tuple[int, int, int],
    width: int = 2,
    dash_len: int = 6,
    gap_len: int = 4,
) -> None:
    """Draw a dashed line on a PIL ImageDraw."""
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    nx = dx / dist
    ny = dy / dist
    step = dash_len + gap_len
    n_dashes = int(dist / step)
    for i in range(n_dashes + 1):
        s0 = i * step
        s1 = min(s0 + dash_len, dist)
        draw.line(
            [
                (x1 + nx * s0, y1 + ny * s0),
                (x1 + nx * s1, y1 + ny * s1),
            ],
            fill=fill,
            width=width,
        )


def _draw_spoked_arc(
    draw: Any,
    cx: int,
    cy: int,
    radius: int,
    start_angle: float,
    end_angle: float,
    fill: tuple[int, int, int],
    width: int = 2,
    n_spokes: int = 7,
) -> None:
    """Draw a red spoked arc (semicircle with radial spikes) as used by
    LigPlot+ for hydrophobic contacts.
    """
    # Draw the arc using short line segments
    n_segments = max(24, int(radius * abs(end_angle - start_angle) / 3))
    arc_points: list[tuple[int, int]] = []
    for i in range(n_segments + 1):
        t = start_angle + (end_angle - start_angle) * i / n_segments
        arc_points.append((int(cx + radius * math.cos(t)), int(cy + radius * math.sin(t))))
    for i in range(len(arc_points) - 1):
        draw.line([arc_points[i], arc_points[i + 1]], fill=fill, width=width)

    # Draw radial spokes (spikes pointing outward)
    for i in range(n_spokes):
        t = start_angle + (end_angle - start_angle) * i / (n_spokes - 1)
        sx = cx + radius * math.cos(t)
        sy = cy + radius * math.sin(t)
        ex = cx + (radius + 12) * math.cos(t)
        ey = cy + (radius + 12) * math.sin(t)
        draw.line([(int(sx), int(sy)), (int(ex), int(ey))], fill=fill, width=width)


def _draw_rounded_label(
    draw: Any,
    x: int,
    y: int,
    text: str,
    font: Any,
    border_color: tuple[int, int, int],
    bg_color: tuple[int, int, int, int] = (255, 255, 255, 235),
    radius: int = 8,
    padding: int = 4,
) -> tuple[int, int, int, int]:
    """Draw a text label inside a rounded rectangle.

    Returns the bounding box (x1, y1, x2, y2).
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x1 = x - padding
    y1 = y - padding
    x2 = x + tw + padding
    y2 = y + th + padding

    # Subtle drop-shadow for depth
    shadow_offset = 2
    draw.rounded_rectangle(
        [(x1 + shadow_offset, y1 + shadow_offset), (x2 + shadow_offset, y2 + shadow_offset)],
        radius=radius,
        fill=(200, 200, 200, 80),
    )
    # Main rounded rectangle
    draw.rounded_rectangle(
        [(x1, y1), (x2, y2)],
        radius=radius,
        fill=bg_color,
        outline=border_color,
        width=2,
    )
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return (x1, y1, x2, y2)


def _compute_label_positions(
    groups: list[dict[str, Any]],
    atom_coords: dict[int, tuple[float, float]],
    canvas_w: int,
    canvas_h: int,
    margin: int = 100,
) -> dict[int, tuple[int, int]]:
    """Compute radial label positions around ligand centre.

    Uses angular sector assignment with collision nudging.
    Returns mapping: group index -> (x, y) top-left of label.
    """
    if not groups:
        return {}

    # Ligand centre (mean of ALL atoms, not just interacting ones)
    all_atom_x = [c[0] for c in atom_coords.values()]
    all_atom_y = [c[1] for c in atom_coords.values()]
    if all_atom_x:
        cx = sum(all_atom_x) / len(all_atom_x)
        cy = sum(all_atom_y) / len(all_atom_y)
    else:
        cx, cy = canvas_w / 2, canvas_h / 2

    # Compute centroid angle for each group (from ligand centre)
    group_angles: list[tuple[int, float]] = []
    for i, g in enumerate(groups):
        atoms = [a for a in g.get("rdkit_atoms", set()) if a in atom_coords]
        if not atoms:
            continue
        gx = sum(atom_coords[a][0] for a in atoms) / len(atoms)
        gy = sum(atom_coords[a][1] for a in atoms) / len(atoms)
        angle = math.atan2(gy - cy, gx - cx)
        group_angles.append((i, angle))

    # Sort by angle for even distribution
    group_angles.sort(key=lambda x: x[1])

    # Assign sectors evenly around the circle
    n = len(group_angles)
    positions: dict[int, tuple[int, int]] = {}
    placed: list[tuple[int, int, int, int]] = []

    for rank, (gi, _orig_angle) in enumerate(group_angles):
        # Even angular spacing starting from top
        sector_angle = 2 * math.pi * rank / n - math.pi / 2
        g = groups[gi]
        atoms = [a for a in g.get("rdkit_atoms", set()) if a in atom_coords]
        if not atoms:
            continue

        # Anchor at centroid of group's atoms
        gx = sum(atom_coords[a][0] for a in atoms) / len(atoms)
        gy = sum(atom_coords[a][1] for a in atoms) / len(atoms)

        # Estimate text size
        label = f"{g['resn']}{g['resi']}"
        est_tw = len(label) * 11
        est_th = 18

        # Distance from ligand centre (not from atom)
        base_dist = max(canvas_w, canvas_h) * 0.40
        # H-bonds need room for dashed line + distance text
        if g.get("type") == "H-bond":
            base_dist *= 1.02
        elif g.get("type") == "Hydrophobic":
            base_dist *= 1.08

        # Try angles with nudging for collision avoidance
        best_pos = None
        for nudge in range(0, 51):
            angle = sector_angle + (nudge * 0.05 if nudge % 2 == 1 else -nudge * 0.05)
            # Position relative to ligand centre
            lx = int(cx + base_dist * math.cos(angle)) - est_tw // 2
            ly = int(cy + base_dist * math.sin(angle)) - est_th // 2

            # Margin clamp
            lx = max(margin, min(lx, canvas_w - margin - est_tw))
            ly = max(margin, min(ly, canvas_h - margin - est_th))

            box = (lx - 6, ly - 6, lx + est_tw + 6, ly + est_th + 6)
            overlap = False
            for bx1, by1, bx2, by2 in placed:
                if not (box[2] < bx1 or box[0] > bx2 or box[3] < by1 or box[1] > by2):
                    overlap = True
                    break
            if not overlap:
                best_pos = (lx, ly)
                placed.append(box)
                break

        if best_pos is None:
            # Fallback: stack vertically at right margin
            best_pos = (canvas_w - margin - est_tw, margin + gi * 30)
            placed.append(
                (
                    best_pos[0] - 6,
                    best_pos[1] - 6,
                    best_pos[0] + est_tw + 6,
                    best_pos[1] + est_th + 6,
                )
            )

        positions[gi] = best_pos

    return positions


def render_interactions_2d(
    receptor_pdb: str,
    ligand_pdbqt: str,
    interactions: list[dict[str, Any]],
    output_png: str,
    output_pdf: str | None = None,
    width: int = 1800,
    height: int = 1400,
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
        output_pdf: Optional output PDF path (high-DPI vector via PIL).
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
                # SMILES may contain spaces — slice after the prefix instead of split()
                smiles = line[14:].strip()
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
                "distance": inter.get("distance"),
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
    highlight_bonds: set[int] = set()
    highlight_bond_colors: dict[int, tuple[float, float, float]] = {}

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

    # Build a quick atom-pair -> bond-idx lookup
    bond_lookup: dict[tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_lookup[(min(a1, a2), max(a1, a2))] = bond.GetIdx()

    for group in interaction_groups.values():
        color_name = group["color"]
        rgb = color_rgb_float.get(color_name, (0.5, 0.5, 0.5))
        group_atoms = group["rdkit_atoms"]
        for rdkit_idx in group_atoms:
            highlight_atoms.add(rdkit_idx)
            # If atom already has a color, keep the first one encountered
            if rdkit_idx not in highlight_atom_colors:
                highlight_atom_colors[rdkit_idx] = rgb
        # Highlight bonds where both atoms are in this interaction group
        group_atom_list = list(group_atoms)
        for i in range(len(group_atom_list)):
            for j in range(i + 1, len(group_atom_list)):
                a1, a2 = group_atom_list[i], group_atom_list[j]
                key = (min(a1, a2), max(a1, a2))
                if key in bond_lookup:
                    bidx = bond_lookup[key]
                    highlight_bonds.add(bidx)
                    if bidx not in highlight_bond_colors:
                        highlight_bond_colors[bidx] = rgb

    # ── Draw molecule with highlights ─────────────────────────────────────────
    # Scale canvas by DPI for publication-quality output (RDKit Cairo works in px)
    canvas_w = int(width * dpi / 100)
    canvas_h = int(height * dpi / 100)
    drawer = Draw.MolDraw2DCairo(canvas_w, canvas_h)
    drawer.drawOptions().highlightRadius = 0.30
    drawer.drawOptions().clearBackground = True
    drawer.drawOptions().bondLineWidth = 3  # Thicker bonds for publication quality

    if highlight_atoms:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_atoms),
            highlightAtomColors=highlight_atom_colors,
            highlightBonds=list(highlight_bonds) if highlight_bonds else None,
            highlightBondColors=highlight_bond_colors if highlight_bonds else None,
        )
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()

    img = Image.open(__import__("io").BytesIO(png_data))
    draw = ImageDraw.Draw(img)

    # ── Fonts (publication hierarchy) ─────────────────────────────────────────
    def _load_font(size: int):
        """Cross-platform font loader with sensible fallbacks."""
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/System/Library/Fonts/HelveticaNeue.ttc",  # macOS alt
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Linux alt
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "C:/Windows/Fonts/segoeui.ttf",  # Windows alt
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    # Scaled font sizes based on canvas (reference: 1200x900 -> 16px label)
    scale = max(canvas_w, canvas_h) / 1500
    font_label = _load_font(max(14, int(18 * scale)))
    font_distance = _load_font(max(11, int(13 * scale)))
    font_legend = _load_font(max(12, int(15 * scale)))
    font_symbol = _load_font(max(16, int(22 * scale)))
    font = font_label

    # ── Collect atom 2D coordinates from RDKit drawer ─────────────────────────
    atom_coords: dict[int, tuple[float, float]] = {}
    for i in range(mol.GetNumAtoms()):
        pos = drawer.GetDrawCoords(i)
        atom_coords[i] = (pos.x, pos.y)

    # ── LigPlot+ style residue labels ─────────────────────────────────────────
    group_list = list(interaction_groups.values())
    label_positions = _compute_label_positions(
        group_list, atom_coords, canvas_w, canvas_h, margin=80
    )

    # LigPlot+ canonical colors (int RGB)
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

    # Pre-compute label sizes
    label_sizes: dict[int, tuple[int, int]] = {}
    for gi, g in enumerate(group_list):
        label = f"{g['resn']}{g['resi']}"
        bbox = draw.textbbox((0, 0), label, font=font)
        label_sizes[gi] = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    # Draw each interaction group with LigPlot+ style graphics
    for gi, g in enumerate(group_list):
        atoms = [a for a in g.get("rdkit_atoms", set()) if a in atom_coords]
        if not atoms:
            continue

        itype = g.get("type", "")
        color_name = g.get("color", "grey")
        rgb_int = color_rgb_int.get(color_name, (128, 128, 128))
        label = f"{g['resn']}{g['resi']}"
        pos = label_positions.get(gi)
        if pos is None:
            continue
        lx, ly = pos
        tw, th = label_sizes[gi]

        # Atom centroid for this group
        ax = int(sum(atom_coords[a][0] for a in atoms) / len(atoms))
        ay = int(sum(atom_coords[a][1] for a in atoms) / len(atoms))

        # Label centre
        lcx = lx + tw // 2
        lcy = ly + th // 2

        if itype == "H-bond":
            # LigPlot+ style: green dashed line with distance label
            _draw_dashed_line(draw, ax, ay, lcx, lcy, (0, 170, 0), width=2)
            # Distance annotation at midpoint
            dist = g.get("distance")
            if dist is not None:
                mid_x = (ax + lcx) // 2
                mid_y = (ay + lcy) // 2
                dist_text = f"{dist:.1f}"
                db = draw.textbbox((0, 0), dist_text, font=font_distance)
                dw = db[2] - db[0]
                dh = db[3] - db[1]
                # White background for distance text
                draw.rectangle(
                    [
                        (mid_x - dw // 2 - 2, mid_y - dh // 2 - 2),
                        (mid_x + dw // 2 + 2, mid_y + dh // 2 + 2),
                    ],
                    fill=(255, 255, 255),
                )
                draw.text(
                    (mid_x - dw // 2, mid_y - dh // 2),
                    dist_text,
                    fill=(0, 100, 0),
                    font=font_distance,
                )

        elif itype == "Hydrophobic":
            # LigPlot+ style: red spoked arc emanating from ligand atom
            angle_to_label = math.atan2(lcy - ay, lcx - ax)
            arc_span = math.pi / 2.5
            # Dynamic radius: ~35% of distance to label, with minimum
            dist_to_label = math.hypot(lcx - ax, lcy - ay)
            arc_radius = max(70, int(dist_to_label * 0.38))
            _draw_spoked_arc(
                draw,
                ax,
                ay,
                radius=arc_radius,
                start_angle=angle_to_label - arc_span / 2,
                end_angle=angle_to_label + arc_span / 2,
                fill=(210, 30, 50),
                width=2,
                n_spokes=7,
            )
            # Leader line from arc end toward label
            mid_arc_x = int(ax + arc_radius * math.cos(angle_to_label))
            mid_arc_y = int(ay + arc_radius * math.sin(angle_to_label))
            draw.line(
                [(mid_arc_x, mid_arc_y), (lcx, lcy)],
                fill=(210, 30, 50),
                width=1,
            )

        elif itype == "Salt bridge":
            # Salt bridge: dashed line with +/- symbols
            _draw_dashed_line(draw, ax, ay, lcx, lcy, (210, 30, 50), width=2)
            # Place charge symbols near atom and label
            charge_font = font  # reuse same font
            # Atom side: ligand charge (assume negative for saltbridge_lneg)
            draw.text((ax - 8, ay - 12), "−", fill=(210, 30, 50), font=charge_font)
            # Label side: protein charge (positive)
            draw.text((lcx + 4, lcy - 12), "+", fill=(210, 30, 50), font=charge_font)

        elif itype == "π-π":
            # π-π stacking: purple dashed arc between aromatic systems
            _draw_dashed_line(draw, ax, ay, lcx, lcy, (140, 40, 230), width=2)
            mid_x = (ax + lcx) // 2
            mid_y = (ay + lcy) // 2
            pi_text = "π-π"
            pb = draw.textbbox((0, 0), pi_text, font=font_symbol)
            pw = pb[2] - pb[0]
            ph = pb[3] - pb[1]
            draw.rectangle(
                [
                    (mid_x - pw // 2 - 2, mid_y - ph // 2 - 2),
                    (mid_x + pw // 2 + 2, mid_y + ph // 2 + 2),
                ],
                fill=(255, 255, 255),
            )
            draw.text(
                (mid_x - pw // 2, mid_y - ph // 2),
                pi_text,
                fill=(100, 20, 160),
                font=font_symbol,
            )

        elif itype == "π-cation":
            # π-cation interaction
            _draw_dashed_line(draw, ax, ay, lcx, lcy, (140, 40, 230), width=2)
            mid_x = (ax + lcx) // 2
            mid_y = (ay + lcy) // 2
            pc_text = "π-cat"
            pb = draw.textbbox((0, 0), pc_text, font=font_symbol)
            pw = pb[2] - pb[0]
            ph = pb[3] - pb[1]
            draw.rectangle(
                [
                    (mid_x - pw // 2 - 2, mid_y - ph // 2 - 2),
                    (mid_x + pw // 2 + 2, mid_y + ph // 2 + 2),
                ],
                fill=(255, 255, 255),
            )
            draw.text(
                (mid_x - pw // 2, mid_y - ph // 2),
                pc_text,
                fill=(100, 20, 160),
                font=font_symbol,
            )

        elif itype == "Water bridge":
            # Water bridge: blue dashed line via water molecule
            # Draw water as small circle at midpoint
            mid_x = (ax + lcx) // 2
            mid_y = (ay + lcy) // 2
            _draw_dashed_line(draw, ax, ay, mid_x, mid_y, (40, 115, 255), width=2)
            _draw_dashed_line(draw, mid_x, mid_y, lcx, lcy, (40, 115, 255), width=2)
            # Water molecule symbol (larger for visibility)
            w_radius = max(12, int(16 * scale))
            draw.ellipse(
                [(mid_x - w_radius, mid_y - w_radius), (mid_x + w_radius, mid_y + w_radius)],
                fill=(200, 220, 255),
                outline=(40, 115, 255),
                width=2,
            )
            wb = draw.textbbox((0, 0), "W", font=font_symbol)
            ww = wb[2] - wb[0]
            wh = wb[3] - wb[1]
            draw.text(
                (mid_x - ww // 2, mid_y - wh // 2),
                "W",
                fill=(0, 60, 150),
                font=font_symbol,
            )

        elif itype == "Metal complex":
            # Metal complex: grey dashed line
            _draw_dashed_line(draw, ax, ay, lcx, lcy, (100, 100, 100), width=2)

        elif itype == "Halogen bond":
            # Halogen bond: cyan dashed line
            _draw_dashed_line(draw, ax, ay, lcx, lcy, (0, 190, 190), width=2)

        else:
            # Default: thin colored leader line
            draw.line([(ax, ay), (lcx, lcy)], fill=rgb_int, width=1)

        # Draw rounded label box (LigPlot+ style: prominent border)
        _draw_rounded_label(draw, lx, ly, label, font, border_color=rgb_int, radius=10, padding=5)

    # ── Title (top-center) ────────────────────────────────────────────────────
    title_text = "Ligand Interaction Diagram"
    title_bbox = draw.textbbox((0, 0), title_text, font=font_legend)
    title_w = title_bbox[2] - title_bbox[0]
    title_x = (canvas_w - title_w) // 2
    title_y = 10
    # Subtle shadow for readability over any background
    draw.text((title_x + 1, title_y + 1), title_text, fill=(180, 180, 180), font=font_legend)
    draw.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font_legend)

    # ── Legend box (bottom-right, with safe margin) ───────────────────────────
    type_counts: dict[str, int] = {}
    for g in group_list:
        t = g.get("type")
        if t:
            type_counts[t] = type_counts.get(t, 0) + 1

    if type_counts:
        # Compute max text width to size the legend box dynamically
        legend_display_rgb = {
            "H-bond": (0, 170, 0),
            "Hydrophobic": (210, 30, 50),
            "π-π": (140, 40, 230),
            "π-cation": (140, 40, 230),
            "Salt bridge": (210, 30, 50),
            "Halogen bond": (0, 190, 190),
            "Water bridge": (40, 115, 255),
            "Metal complex": (128, 128, 128),
        }
        _text_widths: list[int] = []
        for itype, count in sorted(type_counts.items()):
            text = f"{itype}: {count}"
            try:
                tb = draw.textbbox((0, 0), text, font=font_legend)
                _text_widths.append(int(tb[2] - tb[0]))
            except Exception:
                pass
        max_text_w = max(_text_widths + [80])

        legend_margin = 40  # safe margin from canvas edges
        legend_pad_x = 14
        legend_pad_y = 10
        legend_row_h = 22
        legend_header_h = 24
        legend_w = max(120, max_text_w + legend_pad_x * 2 + 20)  # +20 for swatch
        legend_h = legend_header_h + len(type_counts) * legend_row_h + legend_pad_y * 2
        legend_x = canvas_w - legend_w - legend_margin
        legend_y = canvas_h - legend_h - legend_margin

        draw.rounded_rectangle(
            [(legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h)],
            radius=6,
            fill=(255, 255, 255, 240),
            outline=(100, 100, 100),
            width=1,
        )
        draw.text(
            (legend_x + legend_pad_x, legend_y + legend_pad_y),
            "Interactions",
            fill=(0, 0, 0),
            font=font_legend,
        )

        y_off = legend_y + legend_pad_y + legend_header_h
        for itype, count in sorted(type_counts.items()):
            rgb = legend_display_rgb.get(
                itype,
                color_rgb_int.get(INTERACTION_COLORS.get(itype, "grey"), (128, 128, 128)),
            )
            # Swatch
            draw.rounded_rectangle(
                [(legend_x + legend_pad_x, y_off), (legend_x + legend_pad_x + 14, y_off + 14)],
                radius=3,
                fill=rgb,
            )
            draw.text(
                (legend_x + legend_pad_x + 20, y_off),
                f"{itype}: {count}",
                fill=(0, 0, 0),
                font=font_legend,
            )
            y_off += legend_row_h

    ensure_dir(os.path.dirname(output_png) or ".")
    img.save(output_png, dpi=(dpi, dpi))
    logger.info(f"2D interaction diagram rendered: {output_png}")

    # Optional PDF output — PIL converts the same high-DPI bitmap to PDF
    if output_pdf:
        ensure_dir(os.path.dirname(output_pdf) or ".")
        try:
            rgb_img = img.convert("RGB")
            rgb_img.save(output_pdf, dpi=(dpi, dpi), format="PDF")
            logger.info(f"2D interaction diagram (PDF): {output_pdf}")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"2D PDF output skipped: {exc}")

    return output_png


# ─────────────────────────────────────────────────────────────────────────────
# LigPlot+ 2D interaction diagram (external binary)
# ─────────────────────────────────────────────────────────────────────────────


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
    max_panel_width: int = 1600,
) -> str:
    """
    Assemble multiple panel images into a single composite figure.

    Each panel is scaled uniformly so that its width does not exceed
    *max_panel_width*, preserving aspect ratio.  This prevents the
    common problem where one huge panel forces all other panels into
    oversized cells with excessive whitespace.

    Args:
        panel_paths: List of PNG file paths.
        output_png: Output composite PNG path.
        ncols: Number of columns.
        panel_titles: Optional titles for each panel.
        figure_title: Optional overall figure title.
        dpi: Output DPI.
        max_panel_width: Maximum width (px) for each panel after scaling.

    Returns:
        Path to output PNG.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not panel_paths:
        raise VisualizationError("No panels provided for composite figure")

    raw_images = [Image.open(p) for p in panel_paths if os.path.exists(p)]
    if not raw_images:
        raise VisualizationError("No valid panel images found")

    # ── Scale all panels to a uniform maximum width ──
    images: list[Image.Image] = []
    for img in raw_images:
        if img.width > max_panel_width:
            ratio = max_panel_width / img.width
            new_h = int(img.height * ratio)
            images.append(img.resize((max_panel_width, new_h), Image.Resampling.LANCZOS))
        else:
            images.append(img)

    nrows = (len(images) + ncols - 1) // ncols
    panel_w = max(img.width for img in images)
    panel_h = max(img.height for img in images)

    pad = 24
    title_h = 60 if figure_title else 0
    label_h = 36 if panel_titles else 0
    total_w = panel_w * ncols + pad * (ncols + 1)
    total_h = panel_h * nrows + label_h * nrows + title_h + pad * (nrows + 1)

    composite = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(composite)

    def _load_font(size: int):
        """Cross-platform font loader with sensible fallbacks."""
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    title_font = _load_font(28)
    label_font = _load_font(18)

    if figure_title:
        draw.text((pad, pad), figure_title, fill=(0, 0, 0), font=title_font)

    y_offset = title_h + pad
    for row in range(nrows):
        x_offset = pad
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= len(images):
                break
            img = images[idx]
            # Center the image in its uniform cell
            x_pos = x_offset + (panel_w - img.width) // 2
            y_pos = y_offset + label_h + (panel_h - img.height) // 2
            composite.paste(img, (x_pos, y_pos))

            if panel_titles and idx < len(panel_titles):
                draw.text(
                    (x_offset, y_offset),
                    panel_titles[idx],
                    fill=(60, 60, 60),
                    font=label_font,
                )

            x_offset += panel_w + pad
        y_offset += panel_h + label_h + pad

    ensure_dir(os.path.dirname(output_png) or ".")
    composite.save(output_png, dpi=(dpi, dpi))
    logger.info(f"Composite figure saved: {output_png}")
    return output_png
