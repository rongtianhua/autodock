"""Tests for autodock.heatmap."""

from unittest.mock import MagicMock, patch

import pytest

from autodock import heatmap
from autodock.core import DockingResult


class TestPlotEnergyHeatmap:
    """Tests for plot_energy_heatmap."""

    def _make_result(self, compound_name="aspirin", affinity=-7.5):
        return DockingResult(
            compound_name=compound_name,
            receptor="6LU7",
            best_affinity=affinity,
        )

    def test_empty_batch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            heatmap.plot_energy_heatmap({}, output_dir=str(tmp_path))

    def test_no_ligands_raises(self, tmp_path):
        batch = {"rec1": []}
        with pytest.raises(ValueError, match="No ligands"):
            heatmap.plot_energy_heatmap(batch, output_dir=str(tmp_path))

    def test_no_valid_values_raises(self, tmp_path):
        r = self._make_result(affinity=None)
        batch = {"rec1": [r]}
        with pytest.raises(ValueError, match="No valid affinity"):
            heatmap.plot_energy_heatmap(batch, output_dir=str(tmp_path))

    def test_generates_png_and_pdf(self, tmp_path):
        batch = {
            "rec1": [self._make_result("aspirin", -8.0)],
            "rec2": [self._make_result("aspirin", -7.0)],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    result = heatmap.plot_energy_heatmap(
                        batch, output_dir=str(tmp_path), output_prefix="test"
                    )
        assert "png" in result
        assert "pdf" in result
        assert result["png"].endswith("test.png")
        assert result["pdf"].endswith("test.pdf")
        fig.savefig.assert_called()
        mock_plt.close.assert_called_once_with(fig)

    def test_matplotlib_import_error(self, tmp_path):
        with patch.dict("sys.modules", {"matplotlib": None}):
            with pytest.raises(RuntimeError, match="matplotlib"):
                heatmap.plot_energy_heatmap(
                    {"rec1": [self._make_result()]},
                    output_dir=str(tmp_path),
                )

    def test_auto_figsize_computed(self, tmp_path):
        batch = {
            "rec1": [self._make_result(f"lig{i}", float(i)) for i in range(10)],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    heatmap.plot_energy_heatmap(batch, output_dir=str(tmp_path), figsize=None)
                    # figsize should be auto-computed
                    call_args = mock_plt.subplots.call_args
                    assert call_args[1]["figsize"] is not None

    def test_annotations_disabled(self, tmp_path):
        batch = {
            "rec1": [self._make_result("aspirin", -8.0)],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    heatmap.plot_energy_heatmap(
                        batch,
                        output_dir=str(tmp_path),
                        annotate=False,
                    )
                    # ax.text should not be called for annotations
                    ax.text.assert_not_called()

    def test_custom_vrange(self, tmp_path):
        batch = {
            "rec1": [self._make_result("aspirin", -8.0)],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    with patch("matplotlib.colors.Normalize") as mock_norm:
                        heatmap.plot_energy_heatmap(
                            batch,
                            output_dir=str(tmp_path),
                            vrange=(-10.0, -5.0),
                        )
                        # abs_max = max(10, 5) = 10 → symmetric range
                        mock_norm.assert_called_once_with(vmin=-10.0, vmax=10.0)

    def test_elife_palette(self, tmp_path):
        batch = {
            "rec1": [self._make_result("aspirin", -8.0)],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    heatmap.plot_energy_heatmap(
                        batch,
                        output_dir=str(tmp_path),
                        palette="elife",
                    )
                    imshow_call = ax.imshow.call_args
                    assert imshow_call[1]["cmap"] == "PiYG"

    def test_viridis_palette(self, tmp_path):
        batch = {
            "rec1": [self._make_result("aspirin", -8.0)],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    heatmap.plot_energy_heatmap(
                        batch,
                        output_dir=str(tmp_path),
                        palette="viridis",
                    )
                    imshow_call = ax.imshow.call_args
                    assert imshow_call[1]["cmap"] == "viridis"

    def test_multiple_receptors_and_ligands(self, tmp_path):
        batch = {
            "6LU7": [
                self._make_result("aspirin", -8.0),
                self._make_result("ibuprofen", -7.5),
            ],
            "1B9S": [
                self._make_result("aspirin", -7.0),
                self._make_result("ibuprofen", -6.5),
            ],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    result = heatmap.plot_energy_heatmap(
                        batch,
                        output_dir=str(tmp_path),
                        output_prefix="multi",
                    )
                    # Matrix should be 2 receptors × 2 ligands
                    imshow_call = ax.imshow.call_args
                    matrix = imshow_call[0][0]
                    assert matrix.shape == (2, 2)
                    assert result["png"].endswith("multi.png")

    def test_xlabel_rotation_many_ligands(self, tmp_path):
        batch = {
            "rec1": [self._make_result(f"lig{i}", float(-i)) for i in range(8)],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    heatmap.plot_energy_heatmap(batch, output_dir=str(tmp_path))
                    xticks_call = ax.set_xticklabels.call_args
                    assert xticks_call[1]["rotation"] == 45

    def test_xlabel_no_rotation_few_ligands(self, tmp_path):
        batch = {
            "rec1": [self._make_result(f"lig{i}", float(-i)) for i in range(3)],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    heatmap.plot_energy_heatmap(batch, output_dir=str(tmp_path))
                    xticks_call = ax.set_xticklabels.call_args
                    assert xticks_call[1]["rotation"] == 0

    def test_text_color_white_for_high_lightness(self, tmp_path):
        batch = {
            "rec1": [self._make_result("aspirin", -8.0)],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    heatmap.plot_energy_heatmap(batch, output_dir=str(tmp_path))
                    text_calls = ax.text.call_args_list
                    assert len(text_calls) == 1
                    # val=-8.0, abs_max at least 8, lightness = 8/8 = 1.0 > 0.6 → white
                    assert text_calls[0][1]["color"] == "white"

    def test_text_color_dark_for_low_lightness(self, tmp_path):
        # Two ligands: one strong (-8.0), one weak (-3.0)
        # abs_max = 8, so weak cell has lightness = 3/8 = 0.375 < 0.6 → dark text
        batch = {
            "rec1": [
                self._make_result("strong", -8.0),
                self._make_result("weak", -3.0),
            ],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    heatmap.plot_energy_heatmap(batch, output_dir=str(tmp_path))
                    text_calls = ax.text.call_args_list
                    assert len(text_calls) == 2
                    colors = [c[1]["color"] for c in text_calls]
                    assert "white" in colors
                    assert "#333333" in colors

    def test_missing_data_cells_skipped(self, tmp_path):
        batch = {
            "rec1": [
                self._make_result("aspirin", -8.0),
                self._make_result("ibuprofen", None),
            ],
        }
        with patch.object(heatmap, "matplotlib", create=True):
            with patch("matplotlib.use"):
                with patch("matplotlib.pyplot") as mock_plt:
                    fig = MagicMock()
                    ax = MagicMock()
                    mock_plt.subplots.return_value = (fig, ax)
                    cbar = MagicMock()
                    fig.colorbar.return_value = cbar
                    cbar.ax = MagicMock()
                    heatmap.plot_energy_heatmap(batch, output_dir=str(tmp_path))
                    # Only one valid cell → one text call
                    assert ax.text.call_count == 1


class TestGgtheme:
    """Direct tests for _ggtheme."""

    def test_hides_top_right_spines(self):
        ax = MagicMock()
        ax.spines = {
            "top": MagicMock(),
            "right": MagicMock(),
            "bottom": MagicMock(),
            "left": MagicMock(),
        }
        heatmap._ggtheme(ax)
        ax.spines["top"].set_visible.assert_called_once_with(False)
        ax.spines["right"].set_visible.assert_called_once_with(False)

    def test_sets_bottom_left_spine_style(self):
        ax = MagicMock()
        ax.spines = {
            "top": MagicMock(),
            "right": MagicMock(),
            "bottom": MagicMock(),
            "left": MagicMock(),
        }
        heatmap._ggtheme(ax)
        for spine in [ax.spines["bottom"], ax.spines["left"]]:
            assert spine.set_linewidth.call_count == 1
            assert spine.set_color.call_count == 1

    def test_sets_tick_params(self):
        ax = MagicMock()
        ax.spines = {
            "top": MagicMock(),
            "right": MagicMock(),
            "bottom": MagicMock(),
            "left": MagicMock(),
        }
        heatmap._ggtheme(ax)
        ax.tick_params.assert_any_call(
            axis="both", which="major", labelsize=9, colors="#4d4d4d", length=3, width=0.5
        )
        ax.tick_params.assert_any_call(axis="both", which="minor", length=2, width=0.3)

    def test_enables_grid(self):
        ax = MagicMock()
        ax.spines = {
            "top": MagicMock(),
            "right": MagicMock(),
            "bottom": MagicMock(),
            "left": MagicMock(),
        }
        heatmap._ggtheme(ax)
        ax.grid.assert_called_once_with(
            True,
            which="major",
            axis="both",
            linestyle="--",
            linewidth=0.3,
            alpha=0.4,
            color="#cccccc",
        )
        ax.set_axisbelow.assert_called_once_with(True)
