"""
autodock.reporting — Publication-ready report generation.
=========================================================
PDF reports (reportlab) and Excel tables (openpyxl).
"""

from __future__ import annotations

import os
from datetime import datetime

from autodock.core import DockingResult, logger
from autodock.utils import ensure_dir


def generate_pdf_report(
    result: DockingResult,
    output_pdf: str,
    figure_paths: list[str] | None = None,
) -> str:
    """
    Generate a publication-quality PDF report for a docking result.

    Args:
        result: DockingResult object.
        output_pdf: Output PDF path.
        figure_paths: Optional list of figure image paths to embed.

    Returns:
        Path to output PDF.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(f"reportlab required: {exc}")

    ensure_dir(os.path.dirname(output_pdf) or ".")

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=20,
    )
    story.append(Paragraph(f"Molecular Docking Report: {result.compound_name}", title_style))
    story.append(
        Paragraph(
            f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    # Docking parameters
    story.append(Paragraph("<b>Docking Parameters</b>", styles["Heading2"]))
    param_data = [
        ["Parameter", "Value"],
        ["Method", result.method_label],
        ["Exhaustiveness", str(result.exhaustiveness)],
        ["N poses", str(result.n_poses)],
        ["Seed", str(result.seed)],
        ["Box center", f"({result.center[0]:.2f}, {result.center[1]:.2f}, {result.center[2]:.2f})"],
        [
            "Box size",
            f"({result.box_size[0]:.1f}, {result.box_size[1]:.1f}, {result.box_size[2]:.1f})",
        ],
    ]
    if result.receptor_source:
        param_data.append(["Receptor source", result.receptor_source])

    param_table = Table(param_data, colWidths=[6 * cm, 10 * cm])
    param_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(param_table)
    story.append(Spacer(1, 0.5 * cm))

    # Scores
    story.append(Paragraph("<b>Docking Scores</b>", styles["Heading2"]))
    score_data = [["Metric", "Value"]]
    if result.best_affinity is not None:
        score_data.append(["Best affinity (kcal/mol)", f"{result.best_affinity:.3f}"])
    if result.consensus_affinity is not None:
        score_data.append(["Consensus affinity (kcal/mol)", f"{result.consensus_affinity:.3f}"])
    for sf, val in result.all_scores.items():
        score_data.append([f"{sf} score (kcal/mol)", f"{val:.3f}"])
    if result.pre_dock_score is not None:
        score_data.append(["Pre-dock score", f"{result.pre_dock_score:.3f}"])
    if result.score_improvement is not None:
        score_data.append(["Score improvement", f"{result.score_improvement:.3f}"])

    if len(score_data) > 1:
        score_table = Table(score_data, colWidths=[6 * cm, 10 * cm])
        score_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(score_table)
    story.append(Spacer(1, 0.5 * cm))

    # Validation
    story.append(Paragraph("<b>Validation</b>", styles["Heading2"]))
    val_data = [["Check", "Result"]]
    if result.posebusters_pass is not None:
        val_data.append(["PoseBusters", "PASS" if result.posebusters_pass else "FAIL"])
    if result.clash_score is not None:
        val_data.append(["Clash score (Å)", f"{result.clash_score:.3f}"])
    if result.clash_acceptable is not None:
        val_data.append(["Clash acceptable", "YES" if result.clash_acceptable else "NO"])
    if result.rmsd_from_crystal is not None:
        val_data.append(["RMSD from crystal (Å)", f"{result.rmsd_from_crystal:.3f}"])
    if result.protocol_valid is not None:
        val_data.append(["Protocol valid", "YES" if result.protocol_valid else "NO"])

    if len(val_data) > 1:
        val_table = Table(val_data, colWidths=[6 * cm, 10 * cm])
        val_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e94560")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(val_table)
    story.append(Spacer(1, 0.5 * cm))

    # Interactions
    story.append(Paragraph("<b>Interactions</b>", styles["Heading2"]))
    summary = result.interaction_summary
    intx_data = [["Type", "Count"]]
    for itype, count in summary.items():
        intx_data.append([itype, str(count)])

    if len(intx_data) > 1:
        intx_table = Table(intx_data, colWidths=[6 * cm, 10 * cm])
        intx_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#533483")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(intx_table)

    # Figures
    if figure_paths:
        story.append(PageBreak())
        story.append(Paragraph("<b>Figures</b>", styles["Heading2"]))
        for fig_path in figure_paths:
            if os.path.exists(fig_path):
                img = Image(fig_path, width=16 * cm, height=12 * cm)
                story.append(img)
                story.append(Spacer(1, 0.3 * cm))

    doc.build(story)
    logger.info(f"PDF report generated: {output_pdf}")
    return output_pdf


def generate_excel_report(
    results: list[DockingResult],
    output_xlsx: str,
) -> str:
    """
    Generate an Excel report from a list of DockingResult objects.

    Args:
        results: List of DockingResult.
        output_xlsx: Output Excel path.

    Returns:
        Path to output Excel file.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(f"pandas required: {exc}")

    ensure_dir(os.path.dirname(output_xlsx) or ".")

    df = pd.DataFrame() if not results else pd.DataFrame([r.to_dataframe_row() for r in results])

    df.to_excel(output_xlsx, index=False, float_format="%.4f")
    logger.info(f"Excel report generated: {output_xlsx}")
    return output_xlsx


def generate_csv_report(
    results: list[DockingResult],
    output_csv: str,
) -> str:
    """Generate a CSV report from DockingResult objects."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(f"pandas required: {exc}")

    ensure_dir(os.path.dirname(output_csv) or ".")

    df = pd.DataFrame() if not results else pd.DataFrame([r.to_dataframe_row() for r in results])

    df.to_csv(output_csv, index=False, float_format="%.4f")
    logger.info(f"CSV report generated: {output_csv}")
    return output_csv
