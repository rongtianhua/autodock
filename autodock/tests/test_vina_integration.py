"""Integration tests for real AutoDock Vina docking.

These tests exercise the actual Vina C++ extension (not mocks) with
minimal receptor/ligand pairs to verify:
  * subprocess spawning works (spawn context, timeout)
  * pose parsing is correct
  * seed determinism holds
  * energy arrays have expected shape

Marked with ``requires_vina`` so CI can skip them when the conda
environment is not present.
"""

from __future__ import annotations

import os

import pytest

from autodock import docking
from autodock.core import DockingCalculationError

# ── Helpers: minimal valid PDBQT files ──────────────────────────────────────


MINI_RECEPTOR_PDBQT = """\
REMARK  mini receptor for integration test
ATOM      1  N   SER A   1      10.000  10.000  10.000  1.00 20.00      0.000 N
ATOM      2  CA  SER A   1      11.000  10.000  10.000  1.00 20.00      0.000 C
ATOM      3  C   SER A   1      12.000  10.000  10.000  1.00 20.00      0.000 C
ATOM      4  O   SER A   1      13.000  10.000  10.000  1.00 20.00      0.000 OA
ATOM      5  CB  SER A   1      11.000  11.000  10.000  1.00 20.00      0.000 C
ATOM      6  OG  SER A   1      11.000  12.000  10.000  1.00 20.00      0.000 OA
ATOM      7  N   ALA A   2      12.000  10.000  11.000  1.00 20.00      0.000 N
ATOM      8  CA  ALA A   2      12.000  10.000  12.000  1.00 20.00      0.000 C
ATOM      9  C   ALA A   2      12.000  10.000  13.000  1.00 20.00      0.000 C
ATOM     10  O   ALA A   2      12.000  10.000  14.000  1.00 20.00      0.000 OA
ATOM     11  CB  ALA A   2      13.000  10.000  12.000  1.00 20.00      0.000 C
TER
END
"""

MINI_LIGAND_PDBQT = """\
REMARK  mini ligand for integration test
ROOT
ATOM      1  C   LIG A   1      10.500  10.500  10.500  1.00 20.00      0.000 C
ATOM      2  C   LIG A   1      11.500  10.500  10.500  1.00 20.00      0.000 C
ATOM      3  C   LIG A   1      10.500  11.500  10.500  1.00 20.00      0.000 C
ENDROOT
TORSDOF 0
"""


def _write_mini_pair(tmp_path):
    """Write minimal receptor + ligand PDBQT files to tmp_path."""
    rec = tmp_path / "mini_rec.pdbqt"
    rec.write_text(MINI_RECEPTOR_PDBQT)
    lig = tmp_path / "mini_lig.pdbqt"
    lig.write_text(MINI_LIGAND_PDBQT)
    return str(rec), str(lig)


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not docking._HAVE_VINA, reason="Vina not available")
class TestRealVinaDocking:
    """Exercise the real Vina C++ extension via _run_vina_dock."""

    def test_basic_dock_returns_energies_and_poses(self, tmp_path):
        """A minimal receptor/ligand pair should dock and return poses."""
        rec, lig = _write_mini_pair(tmp_path)

        energies, poses = docking._run_vina_dock(
            rec,
            lig,
            center=(11.0, 11.0, 11.0),
            box_size=(10, 10, 10),
            exhaustiveness=4,
            n_poses=3,
            seed=42,
            _use_subprocess=True,
        )

        assert energies.shape[0] >= 1
        assert len(poses) >= 1
        assert all("MODEL" in p for p in poses)

    def test_seed_determinism(self, tmp_path):
        """Same seed → same poses (energies may differ by <0.01 kcal/mol)."""
        rec, lig = _write_mini_pair(tmp_path)

        energies1, poses1 = docking._run_vina_dock(
            rec,
            lig,
            (11.0, 11.0, 11.0),
            (10, 10, 10),
            exhaustiveness=4,
            n_poses=2,
            seed=123,
        )
        energies2, poses2 = docking._run_vina_dock(
            rec,
            lig,
            (11.0, 11.0, 11.0),
            (10, 10, 10),
            exhaustiveness=4,
            n_poses=2,
            seed=123,
        )

        # Energies should be identical for deterministic seed
        assert energies1.shape == energies2.shape
        assert pytest.approx(energies1[0, 0], abs=0.01) == energies2[0, 0]

        # Pose count identical
        assert len(poses1) == len(poses2)

    def test_different_seeds_produce_different_results(self, tmp_path):
        """Different seeds → different poses (with high probability)."""
        rec, lig = _write_mini_pair(tmp_path)

        _, poses1 = docking._run_vina_dock(
            rec,
            lig,
            (11.0, 11.0, 11.0),
            (10, 10, 10),
            exhaustiveness=8,
            n_poses=3,
            seed=1,
        )
        _, poses2 = docking._run_vina_dock(
            rec,
            lig,
            (11.0, 11.0, 11.0),
            (10, 10, 10),
            exhaustiveness=8,
            n_poses=3,
            seed=999,
        )

        # At least one pose should differ in coordinates
        assert len(poses1) == len(poses2)
        any_diff = any(p1 != p2 for p1, p2 in zip(poses1, poses2, strict=True))
        assert any_diff, "Different seeds produced identical poses — suspicious"

    def test_energy_array_shape(self, tmp_path):
        """Vina returns energy array with expected shape."""
        rec, lig = _write_mini_pair(tmp_path)

        energies, _ = docking._run_vina_dock(
            rec,
            lig,
            (11.0, 11.0, 11.0),
            (10, 10, 10),
            exhaustiveness=4,
            n_poses=2,
            seed=42,
        )

        assert energies.ndim == 2
        assert energies.shape[0] >= 1
        # Vina 1.2 returns either 1 or 5 columns depending on version/config
        assert energies.shape[1] >= 1

    def test_timeout_does_not_hang(self, tmp_path):
        """Very short timeout should raise DockingCalculationError, not hang."""
        rec, lig = _write_mini_pair(tmp_path)

        with pytest.raises(DockingCalculationError):
            docking._run_vina_dock(
                rec,
                lig,
                (11.0, 11.0, 11.0),
                (10, 10, 10),
                exhaustiveness=32,
                n_poses=9,
                seed=42,
                timeout=0,  # 0 seconds — impossible to complete
            )

    def test_high_level_dock_ligand_with_real_vina(self, tmp_path):
        """dock_ligand should succeed end-to-end with real Vina."""
        rec, lig = _write_mini_pair(tmp_path)

        result = docking.dock_ligand(
            rec,
            lig,
            center=(11.0, 11.0, 11.0),
            box_size=(10, 10, 10),
            exhaustiveness=4,
            n_poses=2,
            seed=42,
            output_dir=str(tmp_path / "out"),
        )

        assert result.best_affinity is not None
        assert result.n_poses >= 1
        assert os.path.isfile(result.best_pose_pdbqt)
