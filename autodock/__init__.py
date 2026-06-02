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

from autodock.alphafold_tools import assess_alphafold_quality, relax_alphafold_structure
from autodock.analysis import analyze_scoring_bias
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
from autodock.fetchers import fetch_protein_structure, find_best_pdb_structure, search_pdb_by_name
from autodock.heatmap import plot_energy_heatmap
from autodock.interactions import (
    detect_interactions,
    detect_interactions_plip,
    detect_interactions_prolif,
)
from autodock.pipeline import build_pair_dir, post_process_docking, read_docking_results
from autodock.preparation import (
    find_top_pockets,
    prepare_ligand,
    prepare_ligand_adaptive,
    prepare_ligand_conformers,
    prepare_ligand_multi,
    prepare_receptor,
)
from autodock.rendering import (
    composite_summary,
    render_interactions_2d,
    render_scene_pymol,
)
from autodock.reporting import generate_csv_report, generate_excel_report, generate_pdf_report
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
    strip_model_headers,
    write_temp_file,
)
from autodock.validation import (
    compute_clash_score,
    compute_rmsd,
    compute_rmsd_to_crystal,
    run_redocking_validation,
    validate_pose_with_posebusters,
)
from autodock.workflow import DockingWorkflowResult, run_docking_workflow

__version__ = "1.0.0"
__all__ = [
    "ConfigurationError",
    "DataSourceError",
    "DockingCalculationError",
    "DockingError",
    "DockingResult",
    "PreparationError",
    "StructureCache",
    "StructureFetchError",
    "ValidationError",
    "VisualizationError",
    "analyze_scoring_bias",
    "assess_alphafold_quality",
    "batch_dock",
    "build_pair_dir",
    "composite_summary",
    "compute_bounding_box",
    "compute_bounding_box_from_pdb",
    "compute_bounding_box_from_pdbqt",
    "compute_clash_score",
    "compute_rmsd",
    "compute_rmsd_to_crystal",
    "detect_interactions",
    "detect_interactions_plip",
    "detect_interactions_prolif",
    "DockingWorkflowResult",
    "dock_ligand",
    "dock_ligand_multi_conformer",
    "run_docking_workflow",
    "download_ligand_sdf_from_pdb",
    "download_pdb",
    "ensure_dir",
    "extract_ligand_from_pdb",
    "fetch_protein_structure",
    "filter_pdb_lines",
    "find_best_pdb_structure",
    "find_top_pockets",
    "generate_csv_report",
    "generate_excel_report",
    "generate_pdf_report",
    "get_environment_status",
    "load_config",
    "logger",
    "obabel_convert",
    "plot_energy_heatmap",
    "post_process_docking",
    "prepare_ligand",
    "prepare_ligand_adaptive",
    "prepare_ligand_conformers",
    "prepare_ligand_multi",
    "prepare_receptor",
    "print_environment_status",
    "read_docking_results",
    "read_pdb_atoms",
    "relax_alphafold_structure",
    "render_interactions_2d",
    "render_scene_pymol",
    "rmsd_matrix",
    "run_redocking_validation",
    "search_pdb_by_name",
    "strip_model_headers",
    "set_log_level",
    "validate_pose_with_posebusters",
    "virtual_screen",
    "write_default_config",
    "write_temp_file",
]
