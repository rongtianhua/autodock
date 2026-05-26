"""Tests for autodock.reporting — PDF/Excel/CSV report generation."""

from __future__ import annotations

from pathlib import Path

from autodock import reporting as rep
from autodock.core import DockingResult


def _make_min_png(path: Path) -> None:
    """Write a minimal valid 1x1 PNG using PIL."""
    from PIL import Image

    img = Image.new("RGB", (1, 1), color=(0, 0, 0))
    img.save(path)


class TestGenerateCsvReport:
    def test_basic(self, tmp_path):
        result = DockingResult(
            compound_name="aspirin",
            receptor="rec.pdbqt",
            best_affinity=-7.5,
            center=(1.0, 2.0, 3.0),
            box_size=(20.0, 20.0, 20.0),
        )
        path = tmp_path / "out.csv"
        rep.generate_csv_report([result], str(path))
        assert path.exists()
        text = path.read_text()
        assert "aspirin" in text
        assert "-7.5" in text

    def test_empty_results(self, tmp_path):
        path = tmp_path / "empty.csv"
        rep.generate_csv_report([], str(path))
        assert path.exists()


class TestGenerateExcelReport:
    def test_basic(self, tmp_path):
        result = DockingResult(
            compound_name="aspirin",
            receptor="rec.pdbqt",
            best_affinity=-7.5,
        )
        path = tmp_path / "out.xlsx"
        rep.generate_excel_report([result], str(path))
        assert path.exists()

    def test_empty_results(self, tmp_path):
        path = tmp_path / "empty.xlsx"
        rep.generate_excel_report([], str(path))
        assert path.exists()


class TestGeneratePdfReport:
    def test_basic(self, tmp_path):
        result = DockingResult(
            compound_name="aspirin",
            receptor="rec.pdbqt",
            best_affinity=-7.5,
            interactions=[
                {
                    "type": "H-bond",
                    "resn": "SER",
                    "resi": 1,
                    "chain": "A",
                    "atom": "OG",
                    "distance": 2.8,
                    "description": "H-bond: SER1.A (OG) — 2.80 Å",
                }
            ],
        )
        path = tmp_path / "out.pdf"
        rep.generate_pdf_report(result, str(path), figure_paths=[])
        assert path.exists()

    def test_with_figures(self, tmp_path):
        fig = tmp_path / "fig.png"
        _make_min_png(fig)
        result = DockingResult(
            compound_name="aspirin",
            receptor="rec.pdbqt",
            best_affinity=-7.5,
            interactions=[],
        )
        path = tmp_path / "out.pdf"
        rep.generate_pdf_report(result, str(path), figure_paths=[str(fig)])
        assert path.exists()
