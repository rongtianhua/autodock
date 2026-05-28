"""
autodock — Publication-grade molecular docking automation.
==========================================================

Quick start:
    from autodock import prepare_receptor, prepare_ligand, find_top_pockets, dock_ligand
    from autodock.core import print_environment_status

    print_environment_status()

    receptor = prepare_receptor("6LU7.pdb", "receptor.pdbqt")
    ligand = prepare_ligand("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "ligand.pdbqt")
    pockets = find_top_pockets("6LU7.pdb")
    center, box = pockets[0]["center"], pockets[0]["box_size"]
    result = dock_ligand(receptor, ligand, center, box)
"""

from autodock.config import load_config, write_default_config
from autodock.core import (
    ConfigurationError,
    DataSourceError,
    DockingCalculationError,
    DockingError,
    DockingResult,
    PreparationError,
    StructureFetchError,
    ValidationError,
    VisualizationError,
    get_environment_status,
    logger,
    print_environment_status,
    set_log_level,
)
from autodock.docking import batch_dock, dock_ligand, dock_ligand_multi_conformer, virtual_screen
from autodock.interactions import (
    detect_interactions,
    detect_interactions_plip,
    detect_interactions_prolif,
)
from autodock.preparation import (
    find_top_pockets,
    prepare_ligand,
    prepare_ligand_adaptive,
    prepare_ligand_conformers,
    prepare_ligand_multi,
    prepare_receptor,
)
from autodock.rendering import composite_summary, render_interactions_2d, render_scene_pymol
from autodock.reporting import generate_csv_report, generate_excel_report, generate_pdf_report
from autodock.analysis import analyze_scoring_bias
from autodock.utils import (
    StructureCache,
    compute_bounding_box,
    compute_bounding_box_from_pdb,
    compute_bounding_box_from_pdbqt,
    download_ligand_sdf_from_pdb,
    download_pdb,
    ensure_dir,
    extract_ligand_from_pdb,
    filter_pdb_lines,
    obabel_convert,
    read_pdb_atoms,
    rmsd_matrix,
    write_temp_file,
)
from autodock.validation import (
    compute_clash_score,
    compute_rmsd,
    compute_rmsd_to_crystal,
    run_redocking_validation,
    validate_pose_with_posebusters,
)

__version__ = "1.0.0"
__all__ = [
    "DockingResult",
    "DockingError",
    "DockingCalculationError",
    "PreparationError",
    "StructureFetchError",
    "ValidationError",
    "VisualizationError",
    "ConfigurationError",
    "DataSourceError",
    "logger",
    "set_log_level",
    "print_environment_status",
    "get_environment_status",
    "load_config",
    "write_default_config",
    "prepare_receptor",
    "prepare_ligand",
    "prepare_ligand_multi",
    "prepare_ligand_adaptive",
    "prepare_ligand_conformers",
    "find_top_pockets",
    "dock_ligand",
    "dock_ligand_multi_conformer",
    "virtual_screen",
    "batch_dock",
    "validate_pose_with_posebusters",
    "compute_clash_score",
    "compute_rmsd",
    "compute_rmsd_to_crystal",
    "run_redocking_validation",
    "detect_interactions",
    "detect_interactions_plip",
    "detect_interactions_prolif",
    "render_scene_pymol",
    "render_interactions_2d",
    "composite_summary",
    "generate_pdf_report",
    "generate_excel_report",
    "generate_csv_report",
    "ensure_dir",
    "write_temp_file",
    "read_pdb_atoms",
    "compute_bounding_box",
    "compute_bounding_box_from_pdb",
    "compute_bounding_box_from_pdbqt",
    "filter_pdb_lines",
    "obabel_convert",
    "extract_ligand_from_pdb",
    "rmsd_matrix",
    "download_pdb",
    "download_ligand_sdf_from_pdb",
    "StructureCache",
    "analyze_scoring_bias",
]
