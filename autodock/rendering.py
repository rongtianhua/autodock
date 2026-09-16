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
from pathlib import Path
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

# RGB values for the INTERACTION_COLORS names (must match the color_map used in
# _build_pymol_script) — reused by the PIL legend overlay on interaction scenes.
INTERACTION_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "cyan": (0, 255, 255),
    "orange": (255, 128, 0),
    "green": (0, 255, 0),
    "purple": (255, 0, 255),
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "blue": (0, 0, 255),
    "grey": (128, 128, 128),
}

# Extra-resolution default for whole-complex scenes (the overview figure users
# crop and reuse). Applied only when the caller did not request an explicit size.
_COMPLEX_RAY_WIDTH = 3200
_COMPLEX_RAY_HEIGHT = 2400

# Element preferences for interaction-line endpoints: an H-bond line should
# point at the residue N/O actually involved, a hydrophobic line at carbons,
# π interactions at the aromatic C/N framework. Only a *preference* — when no
# residue atom matches, the search falls back to all heavy atoms of the residue
# (see _build_pymol_script).
_DASH_TARGET_ELEMENTS: dict[str, tuple[str, ...]] = {
    "H-bond": ("N", "O"),
    "Water bridge": ("N", "O"),
    "Salt bridge": ("N", "O"),
    "Halogen bond": ("O", "N", "S"),
    "Hydrophobic": ("C",),
    "π-π": ("C", "N"),
    "π-cation": ("C", "N"),
    "Metal complex": ("N", "O", "S"),
}

# Publication-grade colour schemes aligned with Nature/Science conventions
JOURNAL_PRESETS: dict[str, str] = {
    "nature": "publication_white",
    "cell": "presentation_black",
    "acs": "publication_grey",
    "science": "publication_white",
}

COLOR_SCHEMES: dict[str, dict[str, Any]] = {
    "publication_white": {
        "bg": "white",
        "receptor_c": "lightblue",
        "ligand_c": "salmon",
        "label_c": "black",
        # lightblue washes out to near-invisible on a white background —
        # skyblue keeps the pocket surface readable in print.
        "pocket_surface": "skyblue",
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
    color_scheme: str = "publication_white",
    receptor_source: str = "auto",
    show_distance_labels: bool = True,
) -> str:
    """Build a PyMOL command script for publication-quality rendering.

    Parameters
    ----------
    color_scheme:
        One of ``publication_white`` (default), ``publication_grey``,
        ``presentation_black``, or a journal preset (``nature``, ``cell``,
        ``acs``, ``science``).
    receptor_source:
        ``"AlphaFold"``, ``"PDB"``, ``"PDB_single_chain"``, or ``"file"``.
        Determines protein coloring: AlphaFold → pLDDT (B-factor) rainbow;
        PDB → chainbow (N→C blue→red).
    show_distance_labels:
        If True and ``scene == "interaction"``, annotate each dashed interaction
        line with its distance in Å.
    """
    color_scheme = JOURNAL_PRESETS.get(color_scheme, color_scheme)
    scheme = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["publication_white"])
    is_af = receptor_source in ("AlphaFold", "SWISS-MODEL")

    lines: list[str] = []
    lines.append("cmd.delete('all')")
    lines.append(f'cmd.load("{receptor_pdb}", "receptor")')
    lines.append("cmd.show('cartoon', 'receptor')")
    # PyMOL shows the 'lines' representation by default on load; cmd.show()
    # adds cartoon on top without removing it. On dark backgrounds the thin
    # spectrum-coloured lines survive as scattered "speckle" pixels around the
    # cartoon in ray-traced output — hide them explicitly.
    lines.append("cmd.hide('lines', 'receptor')")
    lines.append("cmd.hide('nonbonded', 'receptor')")
    lines.append("cmd.set('cartoon_side_chain_helper', 1)")
    lines.append("cmd.set('cartoon_discrete_colors', 0)")

    # Protein coloring based on source (BEFORE loading ligand to avoid PyMOL 3.x bug
    # where loading a second object wipes selection-level spectrum coloring)
    if is_af:
        # AlphaFold pLDDT coloring via B-factor (pLDDT range 0-100)
        # rainbow_rev palette: low B-factor (low confidence, pLDDT 0) → red/yellow,
        #                      high B-factor (high confidence, pLDDT 100) → blue
        lines.append("cmd.spectrum('b', 'rainbow_rev', 'receptor', minimum=0, maximum=100)")
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
    # Opaque background: PyMOL's default (ray_opaque_background=0) writes a
    # transparent alpha channel, so the PNG renders as white-on-some-viewers
    # and white residue/distance labels become invisible. Force a solid bg.
    lines.append("cmd.set('ray_opaque_background', 1)")

    # For the interaction scene, show only the pocket-region cartoon. The old
    # approach (cartoon_transparency=1.0 outside a 15 Å "pocket_vis" selection)
    # made the ENTIRE cartoon semi-transparent on small proteins (< 5000 atoms,
    # where 15 Å covers most of the structure): transparent cartoon z-fights
    # with the ball-and-stick ligand, β-strands render as distorted double
    # ribbons, and CGO dashed interaction lines get visually lost.
    # cmd.hide is safe here because interaction lines are CGO objects (not
    # "distance" objects), which hide/show of cartoon does not affect.
    if scene == "interaction" and center:
        # 8 Å window: covers the pocket plus a visual buffer around the ligand
        # without swallowing small proteins whole (15 Å covered ~80% of a
        # 1737-atom receptor in the wild).
        lines.append("cmd.select('pocket_vis', 'byres (receptor within 8.0 of ligand)')")
        lines.append("cmd.hide('cartoon', 'receptor')")
        lines.append("cmd.show('cartoon', 'pocket_vis')")
        lines.append("cmd.set('cartoon_transparency', 0.0, 'pocket_vis')")
        # Pocket side chains as sticks — without these the interaction scene
        # showed cartoon + ligand only, leaving the interacting residues
        # invisible.
        lines.append("cmd.show('sticks', 'pocket_vis and not (name C+N+O+CA)')")
        lines.append("cmd.set('stick_radius', 0.12, 'pocket_vis')")
        # White side-chain carbons disappear on a white background — use a
        # mid grey there and keep white only for dark schemes.
        stick_carbon = "white" if scheme.get("bg") == "black" else "grey50"
        lines.append(f"cmd.color('{stick_carbon}', 'pocket_vis and elem C')")
        lines.append("cmd.color('red', 'pocket_vis and elem O')")
        lines.append("cmd.color('blue', 'pocket_vis and elem N')")
        lines.append("cmd.color('yellow', 'pocket_vis and elem S')")
    else:
        # Cartoon transparency for pocket / interaction scenes
        if scene in ("pocket", "interaction"):
            t = scheme.get("receptor_transparency", 0.15)
            lines.append(f"cmd.set('cartoon_transparency', {t}, 'receptor')")

    # ── Pocket surface ──
    if scene == "pocket" and center:
        cx, cy, cz = center
        # The old selection 'br. receptor and center x,y,z around 5' is not
        # valid PyMOL selection syntax — it silently selected nothing, so the
        # pocket surface never rendered. Anchor a pseudoatom at the pocket
        # center and select byres around it instead.
        lines.append(f"cmd.pseudoatom('pocket_ctr', pos=[{cx}, {cy}, {cz}], label='')")
        lines.append(
            f"cmd.select('pocket_surf', 'byres (receptor within {pocket_distance} of pocket_ctr)')"
        )
        lines.append("cmd.show('surface', 'pocket_surf')")
        lines.append("cmd.set('transparency', 0.30, 'pocket_surf')")
        lines.append(f"cmd.color('{scheme['pocket_surface']}', 'pocket_surf and elem C')")
        lines.append("cmd.set('surface_quality', 2)")
        lines.append("cmd.delete('pocket_ctr')")

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
        # Coarse element classifier for model atoms: prefer the chempy symbol,
        # fall back to a first-letter atom-name heuristic (CA→C, NZ→N, OE1→O,
        # SD→S). Only used for interaction-line endpoint preference, so the
        # rare mis-classification of e.g. "CL" is harmless.
        lines.append("def _elem(a):")
        lines.append(
            "    s = (getattr(a, 'symbol', '') or getattr(a, 'elem', '') or '').strip().upper()"
        )
        lines.append("    if s:")
        lines.append("        return s")
        lines.append("    nm = ''.join(ch for ch in a.name.strip().upper() if ch.isalpha())")
        lines.append("    return nm[:1] if nm[:1] in ('N', 'O', 'S', 'C') else 'C'")
        lines.append("")
        lines.append(
            "def _dashed_line(p1, p2, radius=0.08, dash_len=0.4, gap_len=0.2, color=(1.0, 0.5, 0.0)):"
        )
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
        lines.append(
            "                   color[0], color[1], color[2], color[0], color[1], color[2]])"
        )
        lines.append("    return cgo")
        lines.append("")
        # Color mapping from name to RGB tuple — keep in sync with the
        # INTERACTION_COLOR_RGB legend overlay in render_scene_pymol.
        color_map = {
            name: f"({round(r / 255, 4)}, {round(g / 255, 4)}, {round(b / 255, 4)})"
            for name, (r, g, b) in INTERACTION_COLOR_RGB.items()
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
            # One dashed line per ligand atom involved in this interaction, each
            # to its closest residue atom. A single closest-pair line made all
            # interaction types of one residue overlap into one indistinguishable
            # dash; per-atom lines keep the per-type colours readable.
            lig_coords = [a["coords"] for a in inter.get("ligand_atoms", [])][:4]
            lines.append("try:")
            lines.append(f"    m1 = cmd.get_model('{prot_sel}')")
            lines.append("    m2 = cmd.get_model('ligand')")
            lines.append("    if m1.atom and m2.atom:")
            if lig_coords:
                lines.append(f"        targets = {lig_coords!r}")
            else:
                # Fallback: interaction record has no atom coordinates — use the
                # ligand heavy-atom coordinates from the loaded model.
                lines.append("        targets = [tuple(b.coord) for b in m2.atom]")
            lines.append("        pairs = []")
            lines.append("        for t in targets:")
            lines.append("            best = None; bd = 9999")
            # Element preference: the dashed line should end at the atom class
            # that actually mediates the interaction (H-bond → N/O, hydrophobic
            # → C, π → C/N). Fall back to all residue atoms when the preferred
            # class is absent (e.g. glycine has no side-chain N/O).
            pref = _DASH_TARGET_ELEMENTS.get(itype, ())
            lines.append(f"            _pref = {pref!r}")
            lines.append(
                "            pool = [a for a in m1.atom if _elem(a) in _pref] or list(m1.atom)"
            )
            lines.append("            for a in pool:")
            lines.append(
                "                d = math.sqrt(sum((a.coord[i]-t[i])**2 for i in range(3)))"
            )
            lines.append("                if d < bd: bd = d; best = a.coord")
            lines.append("            if best is not None and bd <= 4.5:")
            lines.append("                pairs.append((best, t, bd))")
            lines.append("        for pi, (c1, c2, dd) in enumerate(pairs):")
            lines.append(
                f"            obj = _dashed_line(c1, c2, radius=0.06, dash_len=0.4,"
                f" gap_len=0.2, color={color_rgb})"
            )
            lines.append("            if obj:")
            lines.append(f"                cmd.load_cgo(obj, 'int_{idx}_%d' % pi)")
            lines.append("                cmd.show('cgo', 'int_{idx}_*')")
            # Distance label on the closest pair only (one per interaction).
            lines.append("        if pairs:")
            lines.append("            pairs.sort(key=lambda p: p[2])")
            lines.append("            c1, c2, dd = pairs[0]")
            lines.append("            mid = [(c1[i]+c2[i])/2 for i in range(3)]")
            lines.append(f"            cmd.pseudoatom('dist_{idx}', pos=mid, label='%.2f'%dd)")
            lines.append(
                f"            cmd.set('label_color', '{scheme.get('label_c', 'white')}', 'dist_{idx}')"
            )
            lines.append(f"            cmd.set('label_size', 30, 'dist_{idx}')")
            lines.append(f"            cmd.hide('nonbonded', 'dist_{idx}')")
            lines.append("except: pass")
        lines.append("python end")

    # ── Labels for interacting residues + surrounding pocket residues ──
    if scene == "interaction" and interactions:
        label_color = scheme.get("label_c", "white")
        dim_color = "grey70" if scheme.get("bg") == "black" else "grey50"
        lines.append("python")
        lines.append("from pymol import cmd")
        lines.append("import math")
        lines.append("seen = set()")
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
            res_sel = f"(receptor and resn {resn} and resi {resi} and chain {chain})"
            pseudo = f"labpos_{idx}"
            lines.append("try:")
            lines.append(f"    cmd.select('{lab_name}', '{sel}')")
            # Anchor the label at the residue's CONTACTING atom (the heavy
            # atom nearest to the ligand), not the CA. GLU-type residues reach
            # back to the ligand through a long side chain while their CA
            # backbone points outward — a CA-anchored label plus an outward
            # offset floated in empty space, far from both the residue and its
            # dashed interaction line. Fall back to CA when the ligand is
            # unavailable.
            lines.append(f"    m_res = cmd.get_model('{res_sel}')")
            lines.append("    m_lig = cmd.get_model('ligand')")
            lines.append("    anchor = None")
            lines.append("    if m_res.atom and m_lig.atom:")
            lines.append("        lig_cs = [a.coord for a in m_lig.atom]")
            lines.append("        best_d = 1e9")
            lines.append("        for a in m_res.atom:")
            lines.append(
                "            d = min(sum((a.coord[i]-c[i])**2 for i in range(3)) for c in lig_cs)"
            )
            lines.append("            if d < best_d: best_d = d; anchor = a.coord")
            lines.append("    if anchor is None:")
            lines.append(f"        ca = cmd.get_model('{sel}').atom")
            lines.append("        if ca: anchor = ca[0].coord")
            lines.append("    if anchor is not None:")
            lines.append(f"        cmd.pseudoatom('{pseudo}', pos=list(anchor))")
            # Small outward offset from the ligand center to reduce overlap
            lines.append("        com = cmd.get_extent('ligand')")
            lines.append("        lig_c = [(com[0][i]+com[1][i])/2 for i in range(3)]")
            lines.append(
                "        dx = anchor[0] - lig_c[0]; dy = anchor[1] - lig_c[1]; dz = anchor[2] - lig_c[2]"
            )
            lines.append("        dist = math.sqrt(dx*dx + dy*dy + dz*dz)")
            lines.append("        if dist > 0:")
            lines.append("            # Normalize and scale offset to 2.5 Å outward from ligand")
            lines.append("            scale = 2.5 / dist")
            lines.append(f"            cmd.translate([dx*scale, dy*scale, dz*scale], '{pseudo}')")
            lines.append("        else:")
            lines.append(f"            cmd.translate([0, 0, 2.5], '{pseudo}')")
            # Label with residue name, number and chain: e.g. LYS211(A)
            lines.append(f"        cmd.label('{pseudo}', '\"{resn}{resi}({chain})\"')")
            # Label style: bold sans-serif
            lines.append(f"        cmd.set('label_color', '{label_color}', '{pseudo}')")
            lines.append(f"        cmd.set('label_size', 40, '{pseudo}')")
            lines.append(f"        cmd.set('label_font_id', 10, '{pseudo}')")  # Sans-serif bold
            lines.append(f"    seen.add(('{resn}', '{resi}', '{chain}'))")
            lines.append("except: pass")
        # Surrounding pocket residues (within the 8 Å pocket window) get smaller,
        # dimmer labels so the scene annotates the whole pocket, not just the
        # contacting residues. Failures here must not kill the scene.
        lines.append("try:")
        lines.append("    m = cmd.get_model('pocket_vis and name CA')")
        lines.append("    n_amb = 0")
        lines.append("    for a in m.atom:")
        lines.append("        key = (a.resn, str(a.resi), a.chain)")
        lines.append("        if key in seen: continue")
        lines.append("        seen.add(key)")
        lines.append("        n_amb += 1")
        lines.append("        nm = 'amb_lbl_%d' % n_amb")
        lines.append("        cmd.pseudoatom(nm, pos=list(a.coord))")
        lines.append("        cmd.label(nm, '\"%s%s\"' % (a.resn, a.resi))")
        lines.append(f"        cmd.set('label_color', '{dim_color}', nm)")
        lines.append("        cmd.set('label_size', 28, nm)")
        lines.append("except: pass")
        lines.append("python end")

    # ── Camera / viewport ──
    # Use zoom with buffer to control field of view per scene
    if scene == "complex":
        # Whole-complex view: orient fills the viewport efficiently (kills the
        # wide empty margins of a plain zoom); a minimal buffer + the PNG
        # autocrop below trims the rest of the dead space.
        lines.append("cmd.orient('receptor or ligand')")
        lines.append("cmd.zoom('(receptor or ligand)', 1.0)")
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


def _overlay_interaction_legend(
    png_path: str,
    interactions: list[dict[str, Any]],
    dark_bg: bool = True,
) -> None:
    """Composite an interaction-type colour legend onto a rendered PNG.

    PyMOL cannot place a reliable 2D legend inside a 3D scene, so after the
    ray-traced PNG is written a semi-transparent legend box (swatch + label
    per present interaction type) is drawn onto the bottom-left corner with
    PIL. Colours match INTERACTION_COLORS / INTERACTION_COLOR_RGB, i.e. the
    dashed-line colours in the scene itself.

    Args:
        png_path: Rendered PNG (overwritten in place with the legend composited).
        interactions: Interaction dicts (``type`` key read; deduplicated).
        dark_bg: True for dark-scene renders (translucent black box, white
            text); False for white-scene renders (translucent white box with a
            grey border, black text).
    """
    from PIL import Image, ImageDraw, ImageFont

    present: list[str] = []
    seen: set[str] = set()
    for inter in interactions:
        itype = inter.get("type", "")
        if itype in INTERACTION_COLORS and itype not in seen:
            seen.add(itype)
            present.append(itype)
    if not present:
        return

    with Image.open(png_path) as img:
        rgba = img.convert("RGBA")
    w, h = rgba.size

    # Scale the legend to the rendered image so it stays legible at any size.
    pad = max(14, w // 160)
    row_h = max(30, h // 40)
    box_w = max(260, w // 6)
    box_h = pad * 2 + row_h * len(present)
    x0, y0 = pad, h - box_h - pad

    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    if dark_bg:
        box_fill, border_fill, text_fill = (0, 0, 0, 150), None, (255, 255, 255, 255)
    else:
        box_fill, border_fill, text_fill = (255, 255, 255, 210), (0, 0, 0, 120), (0, 0, 0, 255)
    odraw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=box_fill)
    if border_fill is not None:
        odraw.rectangle(
            [x0, y0, x0 + box_w, y0 + box_h], outline=border_fill, width=max(1, w // 800)
        )
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, int(row_h * 0.55)))
    except OSError:
        font = ImageFont.load_default()

    sw = int(row_h * 0.6)
    y_off = y0 + pad
    for itype in present:
        rgb = INTERACTION_COLOR_RGB[INTERACTION_COLORS[itype]]
        odraw.rounded_rectangle(
            [x0 + pad, y_off, x0 + pad + sw, y_off + sw],
            radius=max(2, sw // 4),
            fill=(*rgb, 255),
        )
        odraw.text(
            (x0 + pad + sw + max(8, w // 300), y_off + (sw - int(row_h * 0.55)) // 2),
            itype,
            fill=text_fill,
            font=font,
        )
        y_off += row_h

    composited = Image.alpha_composite(rgba, overlay).convert("RGB")
    composited.save(png_path)


def _autocrop_png(png_path: str, margin_frac: float = 0.03) -> None:
    """Crop uniform-background borders off a rendered PNG, keeping a margin.

    PyMOL's ``cmd.zoom`` fits the object's bounding box, so a whole-complex
    view of a small or elongated protein always carries wide empty bands
    (e.g. the interior of a U-shaped fold stays black no matter the zoom).
    For the overview figure this trims the dead space instead of clipping the
    structure. Only applied to solid-background renders (enforced by
    ``ray_opaque_background=1``).

    Args:
        png_path: Rendered PNG (overwritten in place).
        margin_frac: Fraction of the cropped width/height kept as margin.
    """
    from PIL import Image, ImageChops

    with Image.open(png_path) as img:
        rgb = img.convert("RGB")
    # Border colour = mode of the four corners (robust for white/grey bg).
    corners = [
        rgb.getpixel(p)
        for p in [(0, 0), (rgb.width - 1, 0), (0, rgb.height - 1), (rgb.width - 1, rgb.height - 1)]
    ]
    bg = max(set(corners), key=corners.count)
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, bg))
    bbox = diff.getbbox()
    if bbox is None:
        return
    mx = int((bbox[2] - bbox[0]) * margin_frac)
    my = int((bbox[3] - bbox[1]) * margin_frac)
    cropped = rgb.crop(
        (
            max(0, bbox[0] - mx),
            max(0, bbox[1] - my),
            min(rgb.width, bbox[2] + mx),
            min(rgb.height, bbox[3] + my),
        )
    )
    if cropped.size != rgb.size:
        cropped.save(png_path)
        logger.info(
            f"Autocropped overview figure: {png_path} → {cropped.size[0]}x{cropped.size[1]}"
        )


# Minimum non-background content for a rendered scene to be considered valid.
# (min_coverage, min_span) — coverage is the fraction of pixels that differ from
# the background colour; span is the fraction of the canvas width/height that
# the content bounding box must occupy. PyMOL exits 0 even when a selection
# error silently emptied the scene, so a structural scene that ray-traced to a
# blank canvas must be rejected here, before legend overlay / autocrop / PDF.
_CONTENT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "complex": (0.02, 0.30),
    "pocket": (0.005, 0.15),
    "interaction": (0.003, 0.10),
    "ligand_closeup": (0.003, 0.10),
}

# PyMOL reports malformed selections on stderr but still exits 0 and writes a
# PNG — the silent-empty-render class of bug. Surface these as warnings.
_SELECTOR_ERROR_PATTERNS = ("selector-error", "malformed selection", "invalid selection")


def _validate_render_content(png_path: str, scene: str) -> None:
    """Reject a rendered PNG whose scene is empty or implausibly small.

    PyMOL returns exit code 0 and writes a valid PNG even when every object in
    the scene failed to load or a selection expression was malformed (e.g. a
    pocket surface that silently never appeared). File existence and pixel
    dimensions therefore say nothing about scene content; this check compares
    the non-background pixel coverage and bounding-box span against per-scene
    floors.

    Args:
        png_path: Rendered PNG to inspect.
        scene: Scene kind — must be a key of ``_CONTENT_THRESHOLDS``.

    Raises:
        VisualizationError: If the image is blank or below the scene floor.
    """
    from PIL import Image, ImageChops

    min_coverage, min_span = _CONTENT_THRESHOLDS.get(scene, (0.003, 0.10))
    try:
        with Image.open(png_path) as img:
            rgb = img.convert("RGB")
    except Exception as exc:
        raise VisualizationError(
            f"PyMOL output for '{scene}' could not be opened as an image " f"({png_path}): {exc}"
        ) from exc
    corners = [
        rgb.getpixel(p)
        for p in [(0, 0), (rgb.width - 1, 0), (0, rgb.height - 1), (rgb.width - 1, rgb.height - 1)]
    ]
    bg = max(set(corners), key=corners.count)
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, bg))
    bbox = diff.getbbox()
    if bbox is None:
        raise VisualizationError(
            f"PyMOL rendered an empty scene for '{scene}' ({png_path}): every pixel "
            "matches the background colour. This usually means a selection expression "
            "silently matched nothing — check PyMOL stderr above for Selector-Error."
        )
    span_w = (bbox[2] - bbox[0]) / rgb.width
    span_h = (bbox[3] - bbox[1]) / rgb.height
    # Downsample the diff mask for coverage: exactness is unnecessary at 300 DPI.
    small = diff.resize((256, 256))
    raw = small.tobytes()
    coverage = sum(1 for i in range(0, len(raw), 3) if raw[i] or raw[i + 1] or raw[i + 2]) / (
        256 * 256
    )
    if coverage < min_coverage or min(span_w, span_h) < min_span:
        raise VisualizationError(
            f"PyMOL scene '{scene}' ({png_path}) rendered with implausibly little "
            f"content: coverage {coverage:.2%} (floor {min_coverage:.2%}), span "
            f"{span_w:.2f}x{span_h:.2f} (floor {min_span:.2f}). Likely a silent "
            "selection/load failure — treat the figure as invalid."
        )


def render_scene_pymol(
    receptor_pdb: str,
    ligand_pdbqt: str,
    output_png: str,
    output_pdf: str | None = None,
    scene: str = "pocket",
    center: tuple[float, float, float] | None = None,
    interactions: list[dict[str, Any]] | None = None,
    width: int | None = None,
    height: int | None = None,
    save_pse: str | None = None,
    color_scheme: str = "publication_white",
    receptor_source: str = "auto",
    show_distance_labels: bool = True,
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
        width: Image width in pixels. None → 2400 (or 3200 for scene='complex'
            — the whole-complex overview gets extra pixels unless the caller
            requests an explicit size).
        height: Image height in pixels (default 1800).
        save_pse: Optional path to save a PyMOL session (.pse) file.
        color_scheme: Colour preset — ``publication_white`` (default),
            ``publication_grey``, ``presentation_black``, or a journal preset
            (``nature``, ``cell``, ``acs``, ``science``).
        receptor_source: ``"AlphaFold"``, ``"PDB"``, ``"PDB_single_chain"``,
            or ``"file"``. Determines protein coloring.
        show_distance_labels: If True and ``scene == "interaction"``, label
            each interaction with its distance in Å.

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

    # Whole-complex scenes are the single-panel overview figure users crop and
    # reuse, so they deserve extra pixels — but only when the caller did not
    # request an explicit size. With None sentinels we can tell "relied on the
    # default" apart from "explicitly asked for the default".
    if scene == "complex" and width is None and height is None:
        width, height = _COMPLEX_RAY_WIDTH, _COMPLEX_RAY_HEIGHT
    else:
        width = DEFAULT_RAY_WIDTH if width is None else width
        height = DEFAULT_RAY_HEIGHT if height is None else height

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
        show_distance_labels=show_distance_labels,
    )

    fd, script_path = tempfile.mkstemp(suffix=".pml")
    os.close(fd)
    with open(script_path, "w") as fh:
        fh.write(script)

    try:
        # Headless PyMOL needs explicit window/buffer size; -cq alone defaults to
        # 640x480 and ignores cmd.viewport(). Pass -W/-H to set the off-screen
        # framebuffer to the requested ray-tracing resolution.
        success, stdout, stderr = safe_subprocess(
            [_PYMOL_EXE, "-cq", "-W", str(width), "-H", str(height), script_path],
            timeout=300,
        )
        if not success:
            raise VisualizationError(f"PyMOL rendering failed: {stderr[:500]}")
        # Selector errors are non-fatal for PyMOL (exit 0, PNG still written)
        # but fatal for figure correctness — surface them loudly.
        _stderr_lower = (stderr or "").lower()
        if any(pat in _stderr_lower for pat in _SELECTOR_ERROR_PATTERNS):
            logger.warning(
                f"PyMOL reported a selection error while rendering {output_png}: "
                f"{stderr[:500].strip()}"
            )
    finally:
        with contextlib.suppress(Exception):
            os.remove(script_path)

    if not os.path.exists(output_png):
        raise VisualizationError(f"PyMOL did not produce output: {output_png}")

    # Content validation: PyMOL exits 0 even when the scene silently rendered
    # blank (e.g. malformed selection → no surface/object). Run before the
    # legend overlay and autocrop so a bad render fails instead of being
    # decorated and shipped.
    _validate_render_content(output_png, scene)

    # Validate that the output has the requested resolution. PyMOL silently
    # falls back to smaller buffers on some builds; catch this before it reaches
    # the publication PDF.
    try:
        from PIL import Image as _PILImage

        with _PILImage.open(output_png) as _img:
            actual_size = _img.size
    except Exception as exc:
        logger.warning(f"Could not validate rendered image size: {exc}")
        actual_size = (0, 0)
    if actual_size != (width, height):
        logger.warning(
            f"PyMOL rendered {output_png} at {actual_size}, expected ({width}, {height}). "
            "This usually means the PyMOL build ignored -W/-H flags."
        )

    logger.info(f"3D scene rendered: {output_png} ({actual_size[0]}x{actual_size[1]})")

    # Interaction-scene legend: the dashed lines are colour-coded by type but
    # the 3D scene itself has no legend. Composite one onto the PNG before the
    # PDF conversion below (which re-opens the PNG). Legend furniture adapts to
    # the scheme background (dark box on dark scenes, white box on white).
    if scene == "interaction" and interactions:
        try:
            _scheme = COLOR_SCHEMES.get(
                JOURNAL_PRESETS.get(color_scheme, color_scheme),
                COLOR_SCHEMES["publication_white"],
            )
            _overlay_interaction_legend(
                output_png, interactions, dark_bg=_scheme.get("bg") == "black"
            )
        except Exception as exc:
            logger.warning(f"Interaction legend overlay skipped: {exc}")

    # Whole-complex overview: trim the uniform-background borders that the
    # bounding-box zoom inevitably leaves around small/elongated receptors.
    if scene == "complex":
        try:
            _autocrop_png(output_png)
        except Exception as exc:
            logger.warning(f"Overview autocrop skipped: {exc}")

    # Optional PDF output — PIL converts PNG raster to PDF at the requested DPI.
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


def _fill_noninteracting_aromatic(
    mol: Any,
    highlight_atoms: set[int],
    highlight_atom_colors: dict[int, tuple[float, float, float]],
    highlight_bonds: set[int],
    highlight_bond_colors: dict[int, tuple[float, float, float]],
) -> None:
    """Fill non-interacting aromatic atoms/bonds with a light grey background.

    When only a subset of aromatic atoms interact (common for fused or
    multi-ring ligands such as flavonoids: one ring contacts the pocket, the
    other does not), RDKit highlights only the interacting ring. The
    un-highlighted ring then reads as bare line strokes and the molecule
    looks visually "split in half". Filling all non-interacting aromatic
    atoms/bonds with light grey keeps the full structure continuous;
    interacting atoms keep their interaction colours (they are added first,
    so their colours win).
    """
    grey_bg = (0.88, 0.88, 0.88)
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        if not atom.GetIsAromatic():
            continue
        if atom_idx not in highlight_atom_colors:
            highlight_atoms.add(atom_idx)
            highlight_atom_colors[atom_idx] = grey_bg
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetIsAromatic() and bond.GetIdx() not in highlight_bond_colors:
                highlight_bonds.add(bond.GetIdx())
                highlight_bond_colors[bond.GetIdx()] = grey_bg


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


def _clip_segment_to_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return the point where the segment (x1,y1)->(x2,y2) enters ``box``.

    Liang–Barsky clipping. Used to stop interaction connector lines at the
    label border instead of drawing them underneath the label box. Falls
    back to the original end point when the segment never reaches the box
    or starts inside it.
    """
    bx1, by1, bx2, by2 = box
    if bx1 <= x1 <= bx2 and by1 <= y1 <= by2:
        return (x2, y2)
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - bx1), (dx, bx2 - x1), (-dy, y1 - by1), (dy, by2 - y1)):
        if p == 0:
            if q < 0:
                return (x2, y2)
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return (x2, y2)
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    return (x1 + dx * t0, y1 + dy * t0)


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
    spoke_len: int = 12,
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
        ex = cx + (radius + spoke_len) * math.cos(t)
        ey = cy + (radius + spoke_len) * math.sin(t)
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
    border_width: int = 2,
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
    shadow_offset = max(2, border_width)
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
        width=border_width,
    )
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return (x1, y1, x2, y2)


def _legend_layout(
    header: tuple[str, int, int],
    rows: list[tuple[str, int, int]],
    canvas_w: int,
    canvas_h: int,
    scale: float,
) -> dict[str, Any]:
    """Compute legend box geometry scaled to the canvas and font size.

    All paddings, row heights and the swatch scale with ``scale`` so the
    legend stays proportional on publication-size canvases (the previous
    fixed-pixel geometry collapsed into overlapping text once fonts were
    scaled up). ``header``/``rows`` are ``(text, width, height)`` tuples
    measured with the real font.
    """
    pad_x = int(14 * scale)
    pad_y = int(10 * scale)
    header_h = header[2] + int(8 * scale)
    swatch = int(14 * scale)
    row_gap = int(8 * scale)
    margin = int(40 * scale)
    max_text_w = max([header[1], *(w for _, w, _ in rows), int(80 * scale)])
    text_h = max([*(h for _, _, h in rows), int(15 * scale)])
    row_h = text_h + row_gap
    box_w = int(max(120 * scale, max_text_w + pad_x * 2 + swatch + int(8 * scale)))
    box_h = int(header_h + pad_y * 2 + row_h * len(rows))
    x = canvas_w - box_w - margin
    y = canvas_h - box_h - margin
    return {
        "x": x,
        "y": y,
        "w": box_w,
        "h": box_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "header_h": header_h,
        "swatch": swatch,
        "row_h": row_h,
    }


def _residue_label(item: dict[str, Any]) -> str:
    """2D diagram residue label: ``RESN RESI`` with chain ID when available.

    Mirrors the 3D scene label format (``GLU72(A)``) so multi-chain receptors
    stay unambiguous in the 2D figure.
    """
    chain = item.get("chain")
    base = f"{item.get('resn', '')}{item.get('resi', '')}"
    return f"{base}({chain})" if chain else base


def _compute_label_positions(
    groups: list[dict[str, Any]],
    atom_coords: dict[int, tuple[float, float]],
    canvas_w: int,
    canvas_h: int,
    margin: int = 100,
    reserved_rects: list[tuple[int, int, int, int]] | None = None,
) -> dict[int, tuple[int, int]]:
    """Compute radial label positions around ligand centre.

    Labels are placed along the natural direction of each residue's atom
    centroid (with angular nudging on collision), at a distance adapted to
    the ligand's actual extent — not a fixed fraction of the canvas, which
    pushed labels off-ligand on large publication canvases. Residues with
    several interaction types share ONE label position. If no non-overlapping
    position can be found, the label is skipped rather than drawn on top of
    another one.

    ``reserved_rects`` are canvas regions already claimed by other furniture
    (e.g. the legend box); labels are never placed inside them.

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

    # ── Merge groups by residue: one label per (resn, resi, chain) ────────────
    # A residue with e.g. both an H-bond and a hydrophobic contact used to get
    # one label per interaction group (grouped by type), each anchored at a
    # different atom centroid — producing duplicated, scattered residue labels.
    seen_res: dict[tuple[Any, Any, Any], int] = {}
    merged: list[dict[str, Any]] = []
    members: list[list[int]] = []
    for i, g in enumerate(groups):
        key = (g.get("resn"), g.get("resi"), g.get("chain"))
        if key in seen_res:
            mi = seen_res[key]
            merged[mi]["rdkit_atoms"] = merged[mi]["rdkit_atoms"] | g.get("rdkit_atoms", set())
            members[mi].append(i)
        else:
            seen_res[key] = len(merged)
            merged.append(
                {
                    "resn": g.get("resn"),
                    "resi": g.get("resi"),
                    "chain": g.get("chain"),
                    "rdkit_atoms": set(g.get("rdkit_atoms", set())),
                }
            )
            members.append([i])

    # Merged centroid and natural angle per residue
    merged_info: list[tuple[int, float, float, float]] = []  # (mi, gx, gy, angle)
    for mi, g in enumerate(merged):
        atoms = [a for a in g["rdkit_atoms"] if a in atom_coords]
        if not atoms:
            continue
        gx = sum(atom_coords[a][0] for a in atoms) / len(atoms)
        gy = sum(atom_coords[a][1] for a in atoms) / len(atoms)
        merged_info.append((mi, gx, gy, math.atan2(gy - cy, gx - cx)))

    # Sort by natural angle for clockwise distribution
    merged_info.sort(key=lambda t: t[3])

    # ── Adaptive base distance from the ligand's actual 2D extent ─────────────
    # Old behaviour: base_dist = max(canvas) * 0.40, which on a 5400×4200
    # publication canvas puts labels ~2200 px from the ligand centre — far
    # outside the drawn molecule and prone to clipping at the canvas edge.
    scale = max(canvas_w, canvas_h) / 1500.0
    numeric_x = [c[0] for c in atom_coords.values() if isinstance(c[0], (int, float))]
    numeric_y = [c[1] for c in atom_coords.values() if isinstance(c[1], (int, float))]
    if numeric_x:
        ligand_extent = max(
            max(numeric_x) - min(numeric_x),
            max(numeric_y) - min(numeric_y),
            1.0,
        )
    else:
        # No numeric coordinates (degenerate/mocked inputs) — fall back to a
        # conservative canvas fraction.
        ligand_extent = max(canvas_w, canvas_h) * 0.25
    max_radius = min(canvas_w, canvas_h) / 2 - margin
    base_dist = min(max(ligand_extent * 1.6, 200 * scale), max(200.0, max_radius * 0.85))

    char_w = max(8, int(9 * scale))
    line_h = max(16, int(20 * scale))

    positions: dict[int, tuple[int, int]] = {}
    placed: list[tuple[int, int, int, int]] = list(reserved_rects or [])

    for mi, _gx, _gy, natural_angle in merged_info:
        g = merged[mi]
        label = _residue_label(g)
        est_tw = len(label) * char_w + 8
        est_th = line_h

        best_pos: tuple[int, int] | None = None
        # Multi-pass: natural angle → angular nudges → radius expansion
        for radius_mult in (1.0, 1.15, 1.3, 1.5):
            for nudge_deg in (0, -12, 12, -24, 24, -38, 38, -55, 55, -75, 75):
                angle = natural_angle + math.radians(nudge_deg)
                radius = base_dist * radius_mult
                lx = int(cx + radius * math.cos(angle)) - est_tw // 2
                ly = int(cy + radius * math.sin(angle)) - est_th // 2

                # Margin clamp
                lx = max(margin, min(lx, canvas_w - margin - est_tw))
                ly = max(margin, min(ly, canvas_h - margin - est_th))

                box = (lx - 6, ly - 6, lx + est_tw + 6, ly + est_th + 6)
                overlap = any(
                    not (box[2] < bx1 or box[0] > bx2 or box[3] < by1 or box[1] > by2)
                    for bx1, by1, bx2, by2 in placed
                )
                if not overlap:
                    best_pos = (lx, ly)
                    placed.append(box)
                    break
            if best_pos is not None:
                break

        if best_pos is None:
            # Fallback: right margin, first free vertical slot
            for free_y in range(margin, canvas_h - margin - est_th, est_th + 8):
                lx = canvas_w - margin - est_tw
                ly = free_y
                box = (lx - 6, ly - 6, lx + est_tw + 6, ly + est_th + 6)
                if not any(
                    not (box[2] < bx1 or box[0] > bx2 or box[3] < by1 or box[1] > by2)
                    for bx1, by1, bx2, by2 in placed
                ):
                    best_pos = (lx, ly)
                    placed.append(box)
                    break

        if best_pos is None:
            # Last resort: skip this label instead of drawing it overlapping
            continue

        # Share the position across all original groups of this residue
        for oi in members[mi]:
            positions[oi] = best_pos

    return positions


def _draw_molecule_svg(
    mol: Any,
    highlight_atoms: set[int],
    highlight_atom_colors: dict[int, tuple[float, float, float]],
    highlight_bonds: set[int],
    highlight_bond_colors: dict[int, tuple[float, float, float]],
    width: int,
    height: int,
) -> str:
    """Render an RDKit molecule to SVG with highlighted atoms/bonds."""
    from rdkit.Chem import Draw

    drawer = Draw.MolDraw2DSVG(width, height)
    drawer.drawOptions().highlightRadius = 0.30
    drawer.drawOptions().clearBackground = True
    drawer.drawOptions().bondLineWidth = 3
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
    return drawer.GetDrawingText()


def render_interactions_2d(
    receptor_pdb: str,
    ligand_pdbqt: str,
    interactions: list[dict[str, Any]],
    output_png: str,
    output_pdf: str | None = None,
    output_svg: str | None = None,
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
        output_pdf: Optional output PDF path (high-DPI raster via PIL).
        output_svg: Optional output SVG path (true vector molecule diagram).
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
            itype = inter.get("type")
            interaction_groups[key] = {
                "type": itype,
                "resn": inter.get("resn"),
                "resi": inter.get("resi"),
                "chain": inter.get("chain"),
                "color": inter.get("color") or INTERACTION_COLORS.get(itype, "grey"),
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

    # Fill non-interacting aromatic rings with light grey so the full
    # structure stays visually continuous (skipped when no interaction was
    # mapped, to avoid implying non-existent contacts).
    if interaction_groups:
        _fill_noninteracting_aromatic(
            mol, highlight_atoms, highlight_atom_colors, highlight_bonds, highlight_bond_colors
        )

    # ── Draw molecule with highlights ─────────────────────────────────────────
    # Scale canvas by DPI for publication-quality output (RDKit Cairo works in px)
    canvas_w = int(width * dpi / 100)
    canvas_h = int(height * dpi / 100)

    # Standard-layout pipeline (v3): CoordGen 2D coordinates (ChemDraw-lineage
    # template library — the de-facto "standard" chemical drawing layout, with
    # axis-aligned rings and even bond lengths) drawn in ACS 1996 style as an
    # SVG vector graphic, then rasterised at the exact target size with
    # cairosvg. Earlier versions rasterised RDKit's tiny natural-size flexi
    # canvas (~150 px) and upscaled ~10-30x with LANCZOS, which blurred bonds
    # and labels; the vector route keeps every edge crisp at any DPI.
    try:
        from rdkit.Chem import rdCoordGen

        mol = Chem.Mol(mol)
        rdCoordGen.AddCoords(mol)
    except Exception as exc:
        logger.warning(f"CoordGen layout failed ({exc}) — falling back to RDKit depictor")
        from rdkit.Chem import rdDepictor

        rdDepictor.Compute2DCoords(mol)

    def _mean_bond_length(mol: Chem.Mol) -> float:
        conf = mol.GetConformer()
        lengths = []
        for bond in mol.GetBonds():
            p1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
            p2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
            lengths.append(math.hypot(p1.x - p2.x, p1.y - p2.y))
        return sum(lengths) / len(lengths) if lengths else 1.0

    _mol_img: Image.Image | None = None
    _mol_w = _mol_h = 0
    _raw_coords: dict[int, tuple[float, float]] = {}
    try:
        import io
        import re

        import cairosvg

        _svg_drawer = Draw.MolDraw2DSVG(-1, -1)  # flexi canvas: content-sized viewBox
        Draw.SetACS1996Mode(_svg_drawer.drawOptions(), _mean_bond_length(mol))
        _svg_drawer.drawOptions().clearBackground = False
        if highlight_atoms:
            _svg_drawer.DrawMolecule(
                mol,
                highlightAtoms=list(highlight_atoms),
                highlightBonds=list(highlight_bonds) if highlight_bonds else None,
                highlightAtomColors=highlight_atom_colors or None,
                highlightBondColors=highlight_bond_colors or None,
            )
        else:
            _svg_drawer.DrawMolecule(mol)
        _svg_drawer.FinishDrawing()
        _svg_text = _svg_drawer.GetDrawingText()

        _vb = re.search(r"viewBox\s*=\s*['\"]\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", _svg_text)
        _mol_w = float(_vb.group(1)) if _vb else 145.0
        _mol_h = float(_vb.group(2)) if _vb else 79.0
        for i in range(mol.GetNumAtoms()):
            pos = _svg_drawer.GetDrawCoords(i)
            _raw_coords[i] = (pos.x, pos.y)
        _vector_base = (_svg_text, _mol_w, _mol_h)
    except ImportError:
        logger.warning(
            "cairosvg not installed — falling back to low-resolution flexi-Cairo 2D base "
            "(pip install cairosvg for crisp vector-rasterised structures)"
        )
        _vector_base = None
        flexi = Draw.MolDraw2DCairo(-1, -1)
        flexi.drawOptions().clearBackground = True
        if highlight_atoms:
            Draw.DrawMoleculeACS1996(
                flexi,
                mol,
                "",
                list(highlight_atoms),
                list(highlight_bonds) if highlight_bonds else None,
                highlight_atom_colors or None,
                highlight_bond_colors or None,
            )
        else:
            Draw.DrawMoleculeACS1996(flexi, mol)
        flexi.FinishDrawing()
        _mol_img = Image.open(io.BytesIO(flexi.GetDrawingText())).convert("RGB")
        _mol_w, _mol_h = _mol_img.size
        for i in range(mol.GetNumAtoms()):
            pos = flexi.GetDrawCoords(i)
            _raw_coords[i] = (pos.x, pos.y)

    # Fit the molecule into the central area, leaving margins for residue
    # labels (sides/top) and the interaction legend (bottom-right).
    margin = int(0.10 * min(canvas_w, canvas_h))
    _target_w = canvas_w - 2 * margin
    _target_h = canvas_h - 2 * margin
    _s = min(_target_w / _mol_w, _target_h / _mol_h)
    _new_w = max(1, int(_mol_w * _s))
    _new_h = max(1, int(_mol_h * _s))
    _ox = (canvas_w - _new_w) // 2
    _oy = (canvas_h - _new_h) // 2

    if _vector_base is not None:
        # Rasterise the SVG at exactly the pasted size — vector-crisp, no resize.
        import io

        import cairosvg

        _svg_text, _, _ = _vector_base
        _png_bytes = cairosvg.svg2png(
            bytestring=_svg_text.encode(),
            output_width=_new_w,
            output_height=_new_h,
            background_color="white",
        )
        mol_img = Image.open(io.BytesIO(_png_bytes)).convert("RGB")
    else:
        mol_img = _mol_img.resize((_new_w, _new_h), Image.LANCZOS)  # type: ignore[union-attr]

    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    img.paste(mol_img, (_ox, _oy))
    draw = ImageDraw.Draw(img)

    atom_coords: dict[int, tuple[float, float]] = {
        i: (_ox + x * _s, _oy + y * _s) for i, (x, y) in _raw_coords.items()
    }

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

    # Ligand centroid — the fallback reference point for outward-pointing
    # graphics (hydrophobic arcs must emanate away from the ring centre,
    # not toward it).
    _all_pts = list(atom_coords.values())
    lig_cx = sum(p[0] for p in _all_pts) / len(_all_pts)
    lig_cy = sum(p[1] for p in _all_pts) / len(_all_pts)

    # Per-atom ring centroids. A hydrophobic arc anchored on a ring atom must
    # bulge away from THAT ring's centre (outward through the atom), not away
    # from the whole-ligand centroid — for fused ring systems the two can
    # disagree, and the arc would otherwise sweep across a neighbouring ring.
    atom_ring_centroid: dict[int, tuple[float, float]] = {}
    try:
        for ring in mol.GetRingInfo().AtomRings():
            pts = [atom_coords[a] for a in ring if a in atom_coords]
            if not pts:
                continue
            rc = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
            for a in ring:
                atom_ring_centroid.setdefault(a, rc)
    except Exception:
        atom_ring_centroid = {}

    # ── LigPlot+ style residue labels ─────────────────────────────────────────
    group_list = list(interaction_groups.values())

    # Legend furniture is laid out FIRST so its rectangle can be passed to the
    # label placer as a reserved region (labels must never overlap the legend).
    type_counts: dict[str, int] = {}
    for g in group_list:
        t = g.get("type")
        if t:
            type_counts[t] = type_counts.get(t, 0) + 1

    def _measure(text: str, font: Any) -> tuple[int, int]:
        try:
            tb = draw.textbbox((0, 0), text, font=font)
            return (max(1, int(tb[2] - tb[0])), max(1, int(tb[3] - tb[1])))
        except (TypeError, ValueError):  # mocked/incomplete font backends
            return (max(1, len(text) * 12), 16)

    legend_rows = [
        (f"{itype}: {count}", *_measure(f"{itype}: {count}", font_legend))
        for itype, count in sorted(type_counts.items())
    ]
    legend_header = ("Interactions", *_measure("Interactions", font_legend))
    legend = _legend_layout(legend_header, legend_rows, canvas_w, canvas_h, scale)
    legend_rect = (legend["x"], legend["y"], legend["x"] + legend["w"], legend["y"] + legend["h"])

    label_positions = _compute_label_positions(
        group_list,
        atom_coords,
        canvas_w,
        canvas_h,
        margin=int(80 * scale),
        reserved_rects=[legend_rect],
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
        label = _residue_label(g)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_sizes[gi] = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _group_geom(gi: int, g: dict[str, Any]) -> tuple | None:
        """Shared per-group geometry; None when the group is not drawable.

        Returns ``(itype, rgb_int, label, lx, ly, ax, ay, tx, ty, lw, dist_bg)``.
        """
        atoms = [a for a in g.get("rdkit_atoms", set()) if a in atom_coords]
        if not atoms:
            return None
        pos = label_positions.get(gi)
        if pos is None:
            return None
        itype = g.get("type", "")
        color_name = g.get("color", "grey")
        rgb_int = color_rgb_int.get(color_name, (128, 128, 128))
        label = _residue_label(g)
        lx, ly = pos
        tw, th = label_sizes[gi]

        # Atom centroid for this group
        ax = int(sum(atom_coords[a][0] for a in atoms) / len(atoms))
        ay = int(sum(atom_coords[a][1] for a in atoms) / len(atoms))

        # Label centre
        lcx = lx + tw // 2
        lcy = ly + th // 2

        # Connectors stop at the label border instead of crossing under the
        # label box; midpoint annotations sit on the visible segment.
        lbl_pad = int(6 * scale)
        label_box = (lx - lbl_pad, ly - lbl_pad, lx + tw + lbl_pad, ly + th + lbl_pad)
        tx, ty = _clip_segment_to_box(ax, ay, lcx, lcy, label_box)
        tx, ty = int(tx), int(ty)
        lw = max(2, int(2 * scale))
        dist_bg = int(3 * scale)
        return itype, rgb_int, label, lx, ly, ax, ay, tx, ty, lw, dist_bg

    # Pass 1 — connectors, arcs and symbols for every group. Label boxes are
    # drawn in pass 2 so that no arc or leader line overpaints another group's
    # label (the old single-pass loop let hydrophobic arcs sweep across labels
    # that had already been drawn).
    for gi, g in enumerate(group_list):
        geom = _group_geom(gi, g)
        if geom is None:
            continue
        itype, rgb_int, label, lx, ly, ax, ay, tx, ty, lw, dist_bg = geom

        if itype == "H-bond":
            # LigPlot+ style: green dashed line with distance label
            _draw_dashed_line(draw, ax, ay, tx, ty, (0, 170, 0), width=lw)
            # Distance annotation at midpoint
            dist = g.get("distance")
            if dist is not None:
                mid_x = (ax + tx) // 2
                mid_y = (ay + ty) // 2
                dist_text = f"{dist:.1f}"
                db = draw.textbbox((0, 0), dist_text, font=font_distance)
                dw = db[2] - db[0]
                dh = db[3] - db[1]
                # White background for distance text
                draw.rectangle(
                    [
                        (mid_x - dw // 2 - dist_bg, mid_y - dh // 2 - dist_bg),
                        (mid_x + dw // 2 + dist_bg, mid_y + dh // 2 + dist_bg),
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
            # LigPlot+ style: red spoked arc emanating from the ligand atom.
            # The arc's convex side (middle spoke) must point AWAY from the
            # aromatic ring it contacts — i.e. outward through the atom, along
            # the reverse of the atom→ring-centre line — so it never sweeps
            # across the ring or the residue label. Fall back to the ligand
            # centroid direction for non-ring (aliphatic) contact atoms.
            rc = next(
                (
                    atom_ring_centroid[a]
                    for a in g.get("rdkit_atoms", set())
                    if a in atom_ring_centroid
                ),
                None,
            )
            if rc is not None:
                out_angle = math.atan2(ay - rc[1], ax - rc[0])
            else:
                out_angle = math.atan2(ay - lig_cy, ax - lig_cx)
            arc_span = math.pi / 2.5
            # Dynamic radius: ~38% of distance to label, with scaled minimum
            dist_to_label = math.hypot(tx - ax, ty - ay)
            arc_radius = max(int(45 * scale), int(dist_to_label * 0.38))
            _draw_spoked_arc(
                draw,
                ax,
                ay,
                radius=arc_radius,
                start_angle=out_angle - arc_span / 2,
                end_angle=out_angle + arc_span / 2,
                fill=(210, 30, 50),
                width=lw,
                n_spokes=7,
                spoke_len=int(12 * scale),
            )
            # Leader line from arc tip toward label
            mid_arc_x = int(ax + arc_radius * math.cos(out_angle))
            mid_arc_y = int(ay + arc_radius * math.sin(out_angle))
            draw.line(
                [(mid_arc_x, mid_arc_y), (tx, ty)],
                fill=(210, 30, 50),
                width=max(1, lw - 1),
            )

        elif itype == "Salt bridge":
            # Salt bridge: dashed line with +/- symbols
            _draw_dashed_line(draw, ax, ay, tx, ty, (210, 30, 50), width=lw)
            # Place charge symbols near atom and label
            charge_font = font  # reuse same font
            # Atom side: ligand charge (assume negative for saltbridge_lneg)
            draw.text(
                (ax - int(8 * scale), ay - int(12 * scale)),
                "−",
                fill=(210, 30, 50),
                font=charge_font,
            )
            # Label side: protein charge (positive)
            draw.text(
                (tx + int(4 * scale), ty - int(12 * scale)),
                "+",
                fill=(210, 30, 50),
                font=charge_font,
            )

        elif itype == "π-π":
            # π-π stacking: purple dashed arc between aromatic systems
            _draw_dashed_line(draw, ax, ay, tx, ty, (140, 40, 230), width=lw)
            mid_x = (ax + tx) // 2
            mid_y = (ay + ty) // 2
            pi_text = "π-π"
            pb = draw.textbbox((0, 0), pi_text, font=font_symbol)
            pw = pb[2] - pb[0]
            ph = pb[3] - pb[1]
            draw.rectangle(
                [
                    (mid_x - pw // 2 - dist_bg, mid_y - ph // 2 - dist_bg),
                    (mid_x + pw // 2 + dist_bg, mid_y + ph // 2 + dist_bg),
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
            _draw_dashed_line(draw, ax, ay, tx, ty, (140, 40, 230), width=lw)
            mid_x = (ax + tx) // 2
            mid_y = (ay + ty) // 2
            pc_text = "π-cat"
            pb = draw.textbbox((0, 0), pc_text, font=font_symbol)
            pw = pb[2] - pb[0]
            ph = pb[3] - pb[1]
            draw.rectangle(
                [
                    (mid_x - pw // 2 - dist_bg, mid_y - ph // 2 - dist_bg),
                    (mid_x + pw // 2 + dist_bg, mid_y + ph // 2 + dist_bg),
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
            mid_x = (ax + tx) // 2
            mid_y = (ay + ty) // 2
            _draw_dashed_line(draw, ax, ay, mid_x, mid_y, (40, 115, 255), width=lw)
            _draw_dashed_line(draw, mid_x, mid_y, tx, ty, (40, 115, 255), width=lw)
            # Water molecule symbol (larger for visibility)
            w_radius = max(12, int(16 * scale))
            draw.ellipse(
                [(mid_x - w_radius, mid_y - w_radius), (mid_x + w_radius, mid_y + w_radius)],
                fill=(200, 220, 255),
                outline=(40, 115, 255),
                width=lw,
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
            _draw_dashed_line(draw, ax, ay, tx, ty, (100, 100, 100), width=lw)

        elif itype == "Halogen bond":
            # Halogen bond: cyan dashed line
            _draw_dashed_line(draw, ax, ay, tx, ty, (0, 190, 190), width=lw)

        else:
            # Default: thin colored leader line
            draw.line([(ax, ay), (tx, ty)], fill=rgb_int, width=max(1, lw - 1))

    # Pass 2 — rounded residue label boxes on top of all connectors.
    for gi, g in enumerate(group_list):
        geom = _group_geom(gi, g)
        if geom is None:
            continue
        itype, rgb_int, label, lx, ly = geom[0], geom[1], geom[2], geom[3], geom[4]
        # Draw rounded label box (LigPlot+ style: prominent border)
        _draw_rounded_label(
            draw,
            lx,
            ly,
            label,
            font,
            border_color=rgb_int,
            radius=max(6, int(8 * scale)),
            padding=max(4, int(5 * scale)),
            border_width=max(2, int(2 * scale)),
        )

    # ── Title (top-center) ────────────────────────────────────────────────────
    title_text = "Ligand Interaction Diagram"
    title_bbox = draw.textbbox((0, 0), title_text, font=font_legend)
    title_w = title_bbox[2] - title_bbox[0]
    title_x = (canvas_w - title_w) // 2
    title_y = 10
    # Subtle shadow for readability over any background
    draw.text((title_x + 1, title_y + 1), title_text, fill=(180, 180, 180), font=font_legend)
    draw.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font_legend)

    # ── Legend box (bottom-right, geometry pre-computed before label placement) ──
    if type_counts:
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
        lx0, ly0 = legend["x"], legend["y"]
        draw.rounded_rectangle(
            [(lx0, ly0), (lx0 + legend["w"], ly0 + legend["h"])],
            radius=max(4, int(6 * scale)),
            fill=(255, 255, 255, 240),
            outline=(100, 100, 100),
            width=max(1, int(scale)),
        )
        draw.text(
            (lx0 + legend["pad_x"], ly0 + legend["pad_y"]),
            "Interactions",
            fill=(0, 0, 0),
            font=font_legend,
        )

        y_off = ly0 + legend["pad_y"] + legend["header_h"]
        sw = legend["swatch"]
        for itype, count in sorted(type_counts.items()):
            rgb = legend_display_rgb.get(
                itype,
                color_rgb_int.get(INTERACTION_COLORS.get(itype, "grey"), (128, 128, 128)),
            )
            # Swatch
            draw.rounded_rectangle(
                [(lx0 + legend["pad_x"], y_off), (lx0 + legend["pad_x"] + sw, y_off + sw)],
                radius=max(2, int(3 * scale)),
                fill=rgb,
            )
            draw.text(
                (lx0 + legend["pad_x"] + sw + int(6 * scale), y_off),
                f"{itype}: {count}",
                fill=(0, 0, 0),
                font=font_legend,
            )
            y_off += legend["row_h"]

    ensure_dir(os.path.dirname(output_png) or ".")
    img.save(output_png, dpi=(dpi, dpi))
    logger.info(f"2D interaction diagram rendered: {output_png}")

    # Optional PDF output — PIL converts the same high-DPI bitmap to PDF.
    if output_pdf:
        ensure_dir(os.path.dirname(output_pdf) or ".")
        try:
            rgb_img = img.convert("RGB")
            rgb_img.save(output_pdf, dpi=(dpi, dpi), format="PDF")
            logger.info(f"2D interaction diagram (PDF): {output_pdf}")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"2D PDF output skipped: {exc}")

    # Optional SVG output — true vector rendering of the highlighted molecule.
    # Residue labels are not yet overlaid; this provides a publication-quality
    # starting layer that can be edited in Illustrator/Inkscape.
    if output_svg:
        ensure_dir(os.path.dirname(output_svg) or ".")
        try:
            svg_text = _draw_molecule_svg(
                mol,
                highlight_atoms,
                highlight_atom_colors,
                highlight_bonds,
                highlight_bond_colors,
                canvas_w,
                canvas_h,
            )
            Path(output_svg).write_text(svg_text, encoding="utf-8")
            logger.info(f"2D interaction diagram (SVG): {output_svg}")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"2D SVG output skipped: {exc}")

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
    # Convert("RGB") flattens any residual alpha channel (e.g. panels rendered
    # before ray_opaque_background was enforced) so transparent regions paste
    # as their intended background instead of raw garbage/black.
    images: list[Image.Image] = []
    for img in raw_images:
        img = img.convert("RGB")
        if img.width > max_panel_width:
            ratio = max_panel_width / img.width
            new_h = int(img.height * ratio)
            images.append(img.resize((max_panel_width, new_h), Image.Resampling.LANCZOS))
        else:
            images.append(img)

    nrows = (len(images) + ncols - 1) // ncols
    # Use a consistent column width (the widest scaled panel) so panels align,
    # but allow each row to have its own height to avoid excessive whitespace
    # when 3D (4:3) and 2D (tall) images are mixed.
    panel_w = max(img.width for img in images)
    row_heights: list[int] = []
    for row in range(nrows):
        row_imgs = images[row * ncols : (row + 1) * ncols]
        row_heights.append(max(img.height for img in row_imgs))

    pad = 24
    title_h = 60 if figure_title else 0
    label_h = 36 if panel_titles else 0
    total_w = panel_w * ncols + pad * (ncols + 1)
    total_h = title_h + sum(row_heights) + label_h * nrows + pad * (nrows + 1)

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
    for row, row_h in enumerate(row_heights):
        x_offset = pad
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= len(images):
                break
            img = images[idx]
            # Centre each panel in its column; top-align within the row.
            x_pos = x_offset + (panel_w - img.width) // 2
            y_pos = y_offset + label_h
            composite.paste(img, (x_pos, y_pos))

            if panel_titles and idx < len(panel_titles):
                draw.text(
                    (x_offset, y_offset),
                    panel_titles[idx],
                    fill=(60, 60, 60),
                    font=label_font,
                )

            x_offset += panel_w + pad
        y_offset += row_h + label_h + pad

    ensure_dir(os.path.dirname(output_png) or ".")
    composite.save(output_png, dpi=(dpi, dpi))
    logger.info(f"Composite figure saved: {output_png}")
    return output_png
