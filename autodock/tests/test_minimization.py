"""Tests for autodock.minimization.

Because OpenMM / OpenFF / RDKit are heavy dependencies that may not be
available in all CI environments, these tests mock the scientific libraries
and focus on API contract, parameter validation, and graceful fallback paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from autodock import minimization


class TestMinimizeDockedPose:
    """Tests for minimize_docked_pose() public API."""

    def test_returns_error_when_openff_unavailable(self, tmp_path):
        """If OpenFF toolkit is missing, return a structured error dict."""
        with patch.object(minimization, "_HAVE_OPENFF", False):
            result = minimization.minimize_docked_pose(
                receptor_pdb=str(tmp_path / "rec.pdb"),
                ligand_pdbqt=str(tmp_path / "lig.pdbqt"),
                ligand_smiles="CCO",
            )
        assert result["success"] is False
        assert "OpenFF" in result["error"]

    def test_returns_error_when_openmm_unavailable(self, tmp_path):
        """If OpenMM is missing, return a structured error dict."""
        with (
            patch.object(minimization, "_HAVE_OPENFF", True),
            patch.object(minimization, "_HAVE_OPENMM", False),
        ):
            result = minimization.minimize_docked_pose(
                receptor_pdb=str(tmp_path / "rec.pdb"),
                ligand_pdbqt=str(tmp_path / "lig.pdbqt"),
                ligand_smiles="CCO",
            )
        assert result["success"] is False
        assert "OpenMM" in result["error"]

    def test_returns_error_when_rdkit_unavailable(self, tmp_path):
        """If RDKit is missing, return a structured error dict."""
        with (
            patch.object(minimization, "_HAVE_OPENFF", True),
            patch.object(minimization, "_HAVE_OPENMM", True),
            patch.object(minimization, "_HAVE_RDKIT", False),
        ):
            result = minimization.minimize_docked_pose(
                receptor_pdb=str(tmp_path / "rec.pdb"),
                ligand_pdbqt=str(tmp_path / "lig.pdbqt"),
                ligand_smiles="CCO",
            )
        assert result["success"] is False
        assert "RDKit" in result["error"]

    def test_creates_temp_output_when_none_provided(self, tmp_path):
        """When output_pdb is None, a temporary file should be created."""
        ligand_pdbqt = tmp_path / "lig.pdbqt"
        ligand_pdbqt.write_text("ATOM   ")

        with (
            patch.object(minimization, "_HAVE_OPENFF", True),
            patch.object(minimization, "_HAVE_OPENMM", True),
            patch.object(minimization, "_HAVE_RDKIT", True),
        ):
            with patch.object(minimization, "_build_ligand", return_value=(None, None)):
                result = minimization.minimize_docked_pose(
                    receptor_pdb=str(tmp_path / "rec.pdb"),
                    ligand_pdbqt=str(ligand_pdbqt),
                    ligand_smiles="CCO",
                    output_pdb=None,
                )
        # _build_ligand returning None triggers early return
        assert result["success"] is False
        assert "output_pdb" in result
        assert result["output_pdb"] is not None

    def test_ph_parameter_passed_to_minimize_complex(self, tmp_path):
        """The ph parameter should be forwarded to _minimize_complex."""
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM")

        mock_offmol = MagicMock()
        mock_positions = MagicMock()

        with (
            patch.object(minimization, "_HAVE_OPENFF", True),
            patch.object(minimization, "_HAVE_OPENMM", True),
            patch.object(minimization, "_HAVE_RDKIT", True),
        ):
            with patch.object(
                minimization, "_build_ligand", return_value=(mock_offmol, mock_positions)
            ):
                with patch.object(
                    minimization, "_minimize_complex", return_value={"success": True}
                ) as mock_complex:
                    minimization.minimize_docked_pose(
                        receptor_pdb=str(rec),
                        ligand_pdbqt=str(lig),
                        ligand_smiles="CCO",
                        include_receptor=True,
                        ph=6.5,
                    )
        # Verify _minimize_complex was called with ph=6.5
        assert mock_complex.called
        call_kwargs = mock_complex.call_args
        assert call_kwargs[0][-1] == 6.5  # ph is the last positional arg

    def test_ligand_only_path_called_when_include_receptor_false(self, tmp_path):
        """When include_receptor=False, _minimize_ligand_only should be called."""
        rec = tmp_path / "rec.pdb"
        rec.write_text("ATOM")
        lig = tmp_path / "lig.pdbqt"
        lig.write_text("ATOM")

        mock_offmol = MagicMock()
        mock_positions = MagicMock()

        with (
            patch.object(minimization, "_HAVE_OPENFF", True),
            patch.object(minimization, "_HAVE_OPENMM", True),
            patch.object(minimization, "_HAVE_RDKIT", True),
        ):
            with patch.object(
                minimization, "_build_ligand", return_value=(mock_offmol, mock_positions)
            ):
                with patch.object(
                    minimization, "_minimize_ligand_only", return_value={"success": True}
                ) as mock_ligand_only:
                    result = minimization.minimize_docked_pose(
                        receptor_pdb=str(rec),
                        ligand_pdbqt=str(lig),
                        ligand_smiles="CCO",
                        include_receptor=False,
                    )
        assert mock_ligand_only.called
        assert result["success"] is True

    def test_build_ligand_failure_returns_early(self, tmp_path):
        """If _build_ligand fails, return early with error info."""
        with (
            patch.object(minimization, "_HAVE_OPENFF", True),
            patch.object(minimization, "_HAVE_OPENMM", True),
            patch.object(minimization, "_HAVE_RDKIT", True),
        ):
            with patch.object(minimization, "_build_ligand", return_value=(None, None)):
                result = minimization.minimize_docked_pose(
                    receptor_pdb=str(tmp_path / "rec.pdb"),
                    ligand_pdbqt=str(tmp_path / "lig.pdbqt"),
                    ligand_smiles="CCO",
                )
        assert result["success"] is False
        assert "Failed to build ligand" in result["error"]


class TestMinimizeComplex:
    """Tests for _minimize_complex internal function."""

    def test_returns_error_when_pdbfixer_unavailable(self, tmp_path):
        """If PDBFixer is not installed, return a graceful error."""
        with patch.dict("sys.modules", {"pdbfixer": None}):
            result = minimization._minimize_complex(
                offmol=MagicMock(),
                ligand_positions=MagicMock(),
                receptor_pdb=str(tmp_path / "rec.pdb"),
                output_pdb=str(tmp_path / "out.pdb"),
                max_iterations=500,
                restraint_k=10000.0,
                force_field="amber14-all.xml",
                small_molecule_forcefield="openff-2.2.0",
                ph=7.4,
            )
        assert result["success"] is False
        assert "PDBFixer not available" in result["error"]


class TestBuildLigand:
    """Tests for _build_ligand helper."""

    def test_returns_none_none_when_all_inputs_missing(self, tmp_path):
        """If no ligand source is provided, return (None, None)."""
        pdbqt = tmp_path / "lig.pdbqt"
        pdbqt.write_text("ATOM")
        with patch.object(minimization, "_HAVE_RDKIT", True):
            result = minimization._build_ligand(
                ligand_pdbqt=str(pdbqt),
                ligand_smiles=None,
                ligand_sdf=None,
            )
        assert result[0] is None

    def test_uses_smiles_when_provided(self, tmp_path):
        """When ligand_smiles is provided, it should be used."""
        mock_mol = MagicMock()
        mock_mol.to_topology.return_value.to_openmm.return_value = "topology"
        mock_mol.n_atoms = 3
        mock_mol.atoms = []
        mock_mol.bonds = []
        pdbqt = tmp_path / "lig.pdbqt"
        pdbqt.write_text("ATOM    1  C   LIG A   1       0.000   0.000   0.000")

        with patch.object(minimization, "_HAVE_RDKIT", True):
            with patch("autodock.minimization.Chem") as mock_chem:
                mock_chem.MolFromSmiles.return_value = MagicMock()
                mock_chem.AddHs.return_value = mock_chem.MolFromSmiles.return_value
                mock_chem.RemoveHs.return_value = mock_chem.MolFromSmiles.return_value
                mock_chem.MolFromPDBBlock.return_value = mock_chem.MolFromSmiles.return_value
                mock_chem.MolFromSmiles.return_value.GetConformer.return_value = MagicMock()
                mock_chem.MolFromSmiles.return_value.GetNumAtoms.return_value = 1
                mock_chem.MolFromSmiles.return_value.GetAtoms.return_value = []
                with patch.object(minimization, "OpenFFMolecule", create=True) as mock_off:
                    mock_off.from_smiles.return_value = mock_mol
                    result = minimization._build_ligand(
                        ligand_pdbqt=str(pdbqt),
                        ligand_smiles="CCO",
                        ligand_sdf=None,
                    )
        assert result[0] is mock_mol


class TestBuildLigandKabsch:
    """Tests for the Kabsch alignment path in _build_ligand (SDF case)."""

    def test_sdf_path_kabsch_aligns_coordinates(self, tmp_path):
        """SDF path should successfully Kabsch-align and return an OpenFF mol."""
        import numpy as np

        pdbqt = tmp_path / "lig.pdbqt"
        pdbqt.write_text("ATOM    1  C   LIG A   1       0.000   0.000   0.000")
        sdf = tmp_path / "lig.sdf"
        sdf.write_text("dummy")

        # Docked coordinates: 3-atom triangle in XY plane
        docked_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        # Template coordinates: same triangle rotated 90° around Z
        template_coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.5, 0.0]])

        def _make_mock_mol(coords, n_atoms=None):
            mol = MagicMock()
            conf = MagicMock()
            conf.GetAtomPosition.side_effect = lambda i: MagicMock(
                x=coords[i][0], y=coords[i][1], z=coords[i][2]
            )
            mol.GetConformer.return_value = conf
            mol.GetNumAtoms.return_value = len(coords) if n_atoms is None else n_atoms
            mol.GetAtoms.return_value = []
            return mol, conf

        docked_mol, _ = _make_mock_mol(docked_coords)
        docked_no_h, docked_no_h_conf = _make_mock_mol(docked_coords)
        template_mol, template_conf = _make_mock_mol(template_coords)
        template_no_h, template_no_h_conf = _make_mock_mol(template_coords)

        # Substructure match: map template atom i -> docked atom i (identity)
        template_no_h.GetSubstructMatch.return_value = (0, 1, 2)

        mock_offmol = MagicMock()
        mock_offmol.n_atoms = 3
        mock_offmol.atoms = []
        mock_offmol.bonds = []
        mock_offmol.conformers = [MagicMock()]
        mock_offmol.conformers[0].m = template_coords.tolist()

        aligned_rw_mol = MagicMock()
        aligned_rw_mol.GetNumAtoms.return_value = 3

        with patch.object(minimization, "_HAVE_RDKIT", True):
            with patch("autodock.minimization.Chem") as mock_chem:
                mock_chem.MolFromPDBBlock.return_value = docked_mol
                mock_chem.RemoveHs.side_effect = [docked_no_h, template_no_h]
                mock_chem.SDMolSupplier.return_value = iter([template_mol])
                mock_chem.RWMol.return_value = aligned_rw_mol
                with patch.object(minimization, "OpenFFMolecule", create=True) as mock_off:
                    mock_off.from_file.return_value = mock_offmol
                    offmol, positions = minimization._build_ligand(
                        ligand_pdbqt=str(pdbqt),
                        ligand_smiles=None,
                        ligand_sdf=str(sdf),
                    )
        assert offmol is mock_offmol
        assert len(positions) == 3

    def test_kabsch_formula_produces_proper_rotation(self):
        """Pure-numpy verification of the Kabsch formula used in _build_ligand.

        ``minimization.py`` uses the convention ``H = docked_c.T @ matched_c``
        (template coordinates are the structure to be rotated).  For this
        convention the optimal rotation is ``R = Vt.T @ U.T`` — *not*
        ``U @ Vt``, which is correct only when ``H = Q.T @ P``.
        """
        import numpy as np

        # Non-coplanar 4-atom set so the SVD gives a well-determined 3-D rotation
        docked = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # 90° rotation around Z
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        template = (docked - docked.mean(axis=0)) @ rot.T + docked.mean(axis=0)

        d_center = docked.mean(axis=0)
        t_center = template.mean(axis=0)
        docked_c = docked - d_center
        template_c = template - t_center

        H = docked_c.T @ template_c
        U, _S, Vt = np.linalg.svd(H)
        # This is the formula used in minimization.py (NOT clustering.py)
        R = Vt.T @ U.T

        # Must be a *proper* rotation (det = +1), not an improper one (det = -1)
        assert np.isclose(np.linalg.det(R), 1.0)

        # After alignment, template should coincide with docked
        aligned = (template - t_center) @ R + d_center
        np.testing.assert_allclose(aligned, docked, atol=1e-10)


class TestModuleAvailabilityFlags:
    """Tests for optional dependency flag consistency."""

    def test_flags_are_boolean(self):
        """_HAVE_* flags should be booleans."""
        assert isinstance(minimization._HAVE_OPENFF, bool)
        assert isinstance(minimization._HAVE_OPENMM, bool)
        assert isinstance(minimization._HAVE_RDKIT, bool)


def _have_rdkit() -> bool:
    try:
        import importlib.util as _iu

        return _iu.find_spec("rdkit") is not None
    except (ImportError, OSError):
        return False


def _have_scipy() -> bool:
    try:
        import importlib.util as _iu

        return _iu.find_spec("scipy") is not None
    except (ImportError, OSError):
        return False


@pytest.mark.skipif(not _have_rdkit(), reason="rdkit not installed")
class TestMatchHeavyAtoms:
    """Atom mapping between template and docked topology."""

    def test_substructure_fast_path(self):
        from rdkit import Chem

        template = Chem.RemoveHs(Chem.MolFromSmiles("CCO"))
        docked = Chem.RemoveHs(Chem.MolFromSmiles("CCO"))
        match = minimization._match_heavy_atoms(template, docked)
        assert match is not None
        assert len(match) == 3

    def test_mismatched_atom_counts_return_none(self):
        from rdkit import Chem

        # Docked query larger than template: substructure match impossible,
        # atom counts differ → no mapping
        template = Chem.RemoveHs(Chem.MolFromSmiles("CC"))
        docked = Chem.RemoveHs(Chem.MolFromSmiles("CCC"))
        assert minimization._match_heavy_atoms(template, docked) is None

    @pytest.mark.skipif(not _have_scipy(), reason="scipy not installed")
    def test_coordinate_assignment_recovers_mapping_after_rigid_transform(self):
        """Rotation + translation of the same molecule must still map exactly."""
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc(O)cc1"))  # phenol
        AllChem.EmbedMolecule(mol, randomSeed=7)
        template = Chem.RemoveHs(Chem.Mol(mol))
        docked = Chem.RemoveHs(Chem.Mol(mol))

        # Rigid-body transform of the docked pose
        theta = np.deg2rad(123.0)
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        shift = np.array([10.0, -5.0, 3.0])
        conf = docked.GetConformer()
        for i in range(docked.GetNumAtoms()):
            p = np.array(conf.GetAtomPosition(i)) @ R + shift
            conf.SetAtomPosition(i, p.tolist())

        match = minimization._coordinate_assignment_match(template, docked)
        assert match is not None

        # The match is defined up to a rigid transform: verify the max
        # residual AFTER optimal proper-rotation alignment, not raw distance.
        t_coords = np.asarray(template.GetConformer().GetPositions(), dtype=float)
        d_coords = np.asarray(docked.GetConformer().GetPositions(), dtype=float)
        t_sel = t_coords[np.array(match)]
        t_cen = t_sel.mean(axis=0)
        d_cen = d_coords.mean(axis=0)
        H = (d_coords - d_cen).T @ (t_sel - t_cen)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        resid = np.linalg.norm((t_sel - t_cen) @ R + d_cen - d_coords, axis=1)
        assert resid.max() < 0.5

    @pytest.mark.skipif(not _have_scipy(), reason="scipy not installed")
    def test_match_heavy_atoms_embeds_smiles_template_when_needed(self):
        """SMILES template has no conformer; fallback must embed it first."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
        AllChem.EmbedMolecule(mol, randomSeed=3)
        docked = Chem.RemoveHs(Chem.Mol(mol))
        template = Chem.RemoveHs(Chem.MolFromSmiles("c1ccccc1O"))  # no conformer

        match = minimization._match_heavy_atoms(template, docked)
        assert match is not None
        # _match_heavy_atoms embeds an internal copy; docked indices must map
        # onto valid template indices with matching elements
        for d_idx, t_idx in enumerate(match):
            assert 0 <= t_idx < template.GetNumAtoms()
            assert (
                template.GetAtomWithIdx(t_idx).GetAtomicNum()
                == docked.GetAtomWithIdx(d_idx).GetAtomicNum()
            )
