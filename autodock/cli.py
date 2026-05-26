"""
autodock.cli — Command-line interface for the docking pipeline.
===============================================================
"""

from __future__ import annotations

import argparse
import logging
import os

from autodock.config import write_default_config
from autodock.core import (
    logger,
    print_environment_status,
    set_log_level,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-q", "--quiet", action="store_true", help="Only warnings and errors")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug output")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")


def _setup_logging(args: argparse.Namespace) -> None:
    if args.quiet:
        set_log_level(logging.WARNING)
    elif args.verbose:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.INFO)

    if args.log_file:
        fh = logging.FileHandler(args.log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)


def cmd_status(args: argparse.Namespace) -> int:
    """Print environment status."""
    print_environment_status()
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a default config file."""
    path = write_default_config(args.config)
    print(f"✅ Default config written to: {path}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch a PDB structure or ligand."""
    from autodock.utils import download_ligand_sdf_from_pdb, download_pdb, ensure_dir

    outdir = ensure_dir(args.outdir)
    if args.type == "pdb":
        path = download_pdb(args.id, str(outdir))
        print(f"✅ Downloaded: {path}")
    elif args.type == "ligand":
        path = download_ligand_sdf_from_pdb(args.id, str(outdir))
        print(f"✅ Downloaded: {path}")
    return 0


def cmd_prepare_receptor(args: argparse.Namespace) -> int:
    """Prepare receptor PDB → PDBQT."""
    from autodock.preparation import prepare_receptor

    output = args.output or args.pdb.replace(".pdb", ".pdbqt")
    prepare_receptor(
        args.pdb,
        output,
        remove_water=not args.keep_waters,
        remove_hetatms=args.remove_hetatms,
    )
    print(f"✅ Receptor prepared: {output}")
    return 0


def cmd_prepare_ligand(args: argparse.Namespace) -> int:
    """Prepare ligand SMILES → PDBQT."""
    from autodock.preparation import prepare_ligand

    output = args.output or "ligand.pdbqt"
    prepare_ligand(args.smiles, output, name=args.name, seed=args.seed)
    print(f"✅ Ligand prepared: {output}")
    return 0


def cmd_find_pockets(args: argparse.Namespace) -> int:
    """Find binding pockets."""
    from autodock.preparation import find_top_pockets

    pockets = find_top_pockets(
        args.receptor,
        ligand_pdb=args.ligand,
        padding=args.padding,
        max_pockets=args.max_pockets,
    )
    print(f"\nFound {len(pockets)} pocket(s):\n")
    for i, p in enumerate(pockets, 1):
        print(f"  Pocket {i} (#{p['pocket_num']}):")
        print(f"    Center:  {p['center']}")
        print(f"    Box:     {p['box_size']}")
        print(f"    Druggability: {p['druggability']:.3f}")
        prob = p.get("p2rank_prob")
        print(f"    P2Rank prob:  {prob:.3f}" if prob is not None else "    P2Rank prob:  N/A")
    return 0


def cmd_dock(args: argparse.Namespace) -> int:
    """Run molecular docking."""
    from autodock.docking import dock_ligand
    from autodock.preparation import find_top_pockets

    center = tuple(args.center) if args.center else None
    box_size = tuple(args.box_size) if args.box_size else None

    # Auto-detect pocket if center not provided
    if center is None:
        pockets = find_top_pockets(args.receptor.replace(".pdbqt", ".pdb"))
        center = pockets[0]["center"]
        box_size = pockets[0]["box_size"]
        print(f"Auto-detected pocket: center={center}, box={box_size}")

    result = dock_ligand(
        args.receptor,
        args.ligand,
        center=center,
        box_size=box_size,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
        output_dir=args.output_dir,
        compound_name=args.name,
    )

    print(f"\n{'='*50}")
    print(f"🧬  Docking Complete: {result.compound_name}")
    print(f"{'='*50}")
    print(f"  Best affinity:       {result.best_affinity:.3f} kcal/mol")
    if result.consensus_affinity is not None:
        print(f"  Consensus affinity:  {result.consensus_affinity:.3f} kcal/mol")
    print(f"  Best pose:           {result.best_pose_pdbqt}")
    print(f"  Output dir:          {result.output_dir}")
    print(f"{'='*50}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate docking protocol via redocking."""
    from autodock.validation import run_redocking_validation

    result = run_redocking_validation(
        args.holo_pdb,
        ligand_resname=args.ligand_resname,
        chain_id=args.chain_id,
        ligand_smiles=args.ligand_smiles,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
        output_dir=args.output_dir,
        box_padding=args.box_padding,
    )

    rmsd = result["rmsd"]
    rmsd_str = f"{rmsd:.2f} Å" if rmsd is not None else "N/A"
    success = result["success"]

    print(f"\n{'='*50}")
    print("🔬  Redocking Validation")
    print(f"{'='*50}")
    print(f"  RMSD:          {rmsd_str}")
    print(f"  Threshold:     {result['threshold']} Å")
    print(f"  Result:        {'✅ PASS' if success else '❌ FAIL'}")
    print(f"  Best affinity: {result['best_affinity']:.3f} kcal/mol")
    print(f"{'='*50}")
    return 0 if success else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Detect interactions and render figures."""
    from autodock.interactions import detect_interactions
    from autodock.rendering import render_interactions_2d, render_scene_pymol

    intx = detect_interactions(args.receptor, args.ligand, method="plip")

    print(f"\nDetected {len(intx)} interactions:")
    for i in intx[:20]:
        print(f"  {i['description']}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        scene_png = os.path.join(args.output_dir, "scene_3d.png")
        diagram_png = os.path.join(args.output_dir, "interactions_2d.png")

        render_scene_pymol(
            args.receptor,
            args.ligand,
            scene_png,
            scene="interaction",
            interactions=intx,
        )
        render_interactions_2d(
            args.receptor,
            args.ligand,
            intx,
            diagram_png,
        )
        print(f"\n🖼️  Figures saved to {args.output_dir}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate PDF/Excel report."""

    # For now, a simplified report from a config file
    # In full pipeline, this reads result pickles / JSON
    print("Report generation requires a completed docking run.")
    print("Use 'autodock run' for end-to-end pipeline with auto-reporting.")
    return 0


def cmd_posebusters_eval(args: argparse.Namespace) -> int:
    """Run PoseBusters benchmark evaluation."""
    from autodock.posebusters_eval import run_posebusters_evaluation

    summary = run_posebusters_evaluation(
        args.id_list,
        output_dir=args.outdir,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
        n_workers=args.workers,
        max_targets=args.max_targets,
    )

    print(f"\n{'='*60}")
    print("📊  PoseBusters Evaluation Complete")
    print(f"{'='*60}")
    print(f"  Total targets:      {summary['n_total']}")
    print(f"  Successful:         {summary['n_success']}")
    print(f"  Success rate:       {summary['success_rate']*100:.1f}%")
    if summary["median_rmsd"] is not None:
        print(f"  Median RMSD:        {summary['median_rmsd']:.2f} Å")
    print(f"  PoseBusters pass:   {summary['posebusters_pass_count']}/{summary['n_success']}")
    print(f"  PoseBusters rate:   {summary['posebusters_pass_rate']*100:.1f}%")
    print(f"{'='*60}")
    print(f"  Output directory: {args.outdir}")
    print(f"{'='*60}")
    return 0


def cmd_benchmark_redock(args: argparse.Namespace) -> int:
    """Run redocking benchmark on a standard target set."""
    import json

    from autodock.benchmark import run_redocking_benchmark

    targets = None
    if args.targets:
        with open(args.targets) as fh:
            targets = json.load(fh)

    summary = run_redocking_benchmark(
        targets=targets,
        output_dir=args.outdir,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
        n_workers=args.workers,
    )

    print(f"\n{'='*55}")
    print("📊  Redocking Benchmark Results")
    print(f"{'='*55}")
    print(f"  Total targets:     {summary['n_total']}")
    print(f"  Successful:        {summary['n_success']}")
    print(f"  Success rate:      {summary['success_rate']*100:.1f}%")
    if summary["median_rmsd"] is not None:
        print(f"  Median RMSD:       {summary['median_rmsd']:.2f} Å")
        print(f"  Mean RMSD:         {summary['mean_rmsd']:.2f} Å ± {summary['rmsd_std']:.2f}")
    print("\n  By family:")
    for fam, stats in summary.get("by_family", {}).items():
        print(
            f"    {fam:20s}  {stats['n_success']}/{stats['n_total']}  ({stats['success_rate']*100:.1f}%)  mean={stats['mean_rmsd']:.2f}Å"
            if stats["mean_rmsd"]
            else f"    {fam:20s}  {stats['n_success']}/{stats['n_total']}"
        )
    print(f"{'='*55}")
    print(f"  JSON: {summary['json_path']}")
    if summary.get("csv_path"):
        print(f"  CSV:  {summary['csv_path']}")
    print(f"{'='*55}")
    return 0


def cmd_batch_dock(args: argparse.Namespace) -> int:
    """Run batch docking across multiple receptors and ligands."""
    import json

    from autodock.docking import batch_dock

    # Load pocket definitions
    with open(args.pockets) as fh:
        pockets_raw = json.load(fh)

    pockets: dict[str, dict[str, tuple[float, float, float]]] = {}
    for rec_name, pdef in pockets_raw.items():
        pockets[rec_name] = {
            "center": tuple(pdef["center"]),
            "box_size": tuple(pdef["box_size"]),
        }

    # Build receptor/ligand dicts from file paths
    receptors: dict[str, str] = {}
    for path in args.receptors:
        name = os.path.splitext(os.path.basename(path))[0]
        receptors[name] = path

    ligands: dict[str, str] = {}
    for path in args.ligands:
        name = os.path.splitext(os.path.basename(path))[0]
        ligands[name] = path

    results = batch_dock(
        receptors,
        ligands,
        pockets,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
        output_dir=args.outdir,
        n_workers=args.workers,
    )

    print(f"\n{'='*55}")
    print("🧬  Batch Docking Complete")
    print(f"{'='*55}")
    for rec_name, res_list in results.items():
        successes = sum(1 for r in res_list if r.best_affinity is not None)
        print(f"  {rec_name}: {successes}/{len(res_list)} ligands docked successfully")
        for r in res_list:
            aff = r.best_affinity
            aff_str = f"{aff:.2f} kcal/mol" if aff is not None else "FAILED"
            print(f"    {r.compound_name:20s}  {aff_str}")
    print(f"{'='*55}")
    print(f"  Output directory: {args.outdir}")
    print(f"{'='*55}")
    return 0


def cmd_ensemble_dock(args: argparse.Namespace) -> int:
    """Run repeated docking with ensemble statistics."""
    from autodock.docking import dock_ensemble
    from autodock.preparation import find_top_pockets

    center = tuple(args.center) if args.center else None
    box_size = tuple(args.box_size) if args.box_size else None

    if center is None:
        pockets = find_top_pockets(args.receptor.replace(".pdbqt", ".pdb"))
        center = pockets[0]["center"]
        box_size = pockets[0]["box_size"]
        print(f"Auto-detected pocket: center={center}, box={box_size}")

    summary = dock_ensemble(
        args.receptor,
        args.ligand,
        center=center,
        box_size=box_size,
        n_repeats=args.n_repeats,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
        output_dir=args.outdir,
        compound_name=args.name,
    )

    print(f"\n{'='*60}")
    print(f"🧬  Ensemble Docking Complete: {args.name or args.ligand}")
    print(f"{'='*60}")
    print(f"  Repeats:          {summary['n_repeats']} ({summary['n_successful']} successful)")
    print(
        f"  Best affinity:    {summary['ensemble_best_affinity_mean']:.3f} ± {summary['ensemble_best_affinity_std']:.3f} kcal/mol"
    )
    print(
        f"  Range:            {summary['ensemble_best_affinity_min']:.3f} → {summary['ensemble_best_affinity_max']:.3f} kcal/mol"
    )
    print(f"  CV:               {summary['ensemble_best_affinity_cv']:.3f}")
    if summary["pose_stability_rmsd_mean"] is not None:
        print(
            f"  Pose RMSD:        {summary['pose_stability_rmsd_mean']:.2f} ± {summary['pose_stability_rmsd_std']:.2f} Å"
        )
    print(f"  Clusters:         {summary['n_clusters']}")
    print(f"  Confidence:       {summary['confidence'].upper()}")
    print(f"  Recommendation:   {summary['recommendation']}")
    print(f"{'='*60}")
    print(f"  Output directory: {args.outdir}")
    print(f"{'='*60}")
    return 0


def cmd_virtual_screen(args: argparse.Namespace) -> int:
    """Run virtual screening against a receptor."""
    from autodock.docking import virtual_screen
    from autodock.preparation import find_top_pockets, prepare_receptor
    from autodock.utils import download_pdb, ensure_dir

    outdir = ensure_dir(args.outdir)
    receptor_name = args.receptor.upper()

    # Fetch / use cached receptor
    receptor_pdb = os.path.join(outdir, f"{receptor_name}.pdb")
    if not os.path.exists(receptor_pdb):
        download_pdb(receptor_name, outdir)

    # Prepare receptor
    receptor_pdbqt = os.path.join(outdir, f"{receptor_name}.pdbqt")
    prepare_receptor(receptor_pdb, receptor_pdbqt)

    # Detect pocket
    pockets = find_top_pockets(receptor_pdb, max_pockets=3, use_p2rank=False)
    center = pockets[0]["center"]
    box_size = pockets[0]["box_size"]

    # Read compound library
    library = {}
    with open(args.library) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                library[parts[0]] = parts[1]

    print(f"Screening {len(library)} compounds against {receptor_name}...")
    results, csv_path = virtual_screen(
        receptor_pdbqt,
        library,
        center,
        box_size,
        output_dir=outdir,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        n_workers=args.workers,
    )

    print(f"\n{'='*50}")
    print("🧪  Virtual Screening Results")
    print(f"{'='*50}")
    print(f"  Compounds screened: {len(results)}")
    print(f"  CSV report: {csv_path}")

    # Top 10 hits
    sorted_results = sorted(
        [r for r in results if r.best_affinity is not None],
        key=lambda r: r.best_affinity or 999,
    )
    print("\n  Top 10 hits:")
    for r in sorted_results[:10]:
        print(f"    {r.compound_name:20s}  {r.best_affinity:8.3f} kcal/mol")
    print(f"{'='*50}")
    return 0


def cmd_md(args: argparse.Namespace) -> int:
    """Run short MD simulation on a docked complex."""
    from autodock.md_simulation import run_md_stability

    result = run_md_stability(
        receptor_pdb=args.receptor,
        ligand_pdbqt=args.ligand,
        output_dir=args.outdir,
        n_steps=args.steps,
        dt_fs=args.dt,
        temperature_k=args.temperature,
        solvent_model=args.solvent,
        platform_name=args.platform,
    )

    print(f"\n{'='*50}")
    print("🌊  MD Simulation Results")
    print(f"{'='*50}")
    print(f"  Trajectory:     {result.get('trajectory')}")
    print(f"  Final structure: {result.get('final_structure')}")
    if "ligand_rmsd_mean" in result:
        print(
            f"  Ligand RMSD:    {result['ligand_rmsd_mean']:.2f} ± {result.get('ligand_rmsd_std', 0):.2f} Å"
        )
    if "receptor_ca_rmsd_mean" in result:
        print(f"  Receptor RMSD:  {result['receptor_ca_rmsd_mean']:.2f} Å")
    if "n_hbonds_mean" in result:
        print(f"  Avg H-bonds:    {result['n_hbonds_mean']:.1f}")
    print(f"{'='*50}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Full end-to-end pipeline: fetch → prepare → dock → analyze → report."""
    from autodock.docking import dock_ligand
    from autodock.interactions import detect_interactions
    from autodock.preparation import find_top_pockets, prepare_ligand, prepare_receptor
    from autodock.rendering import composite_summary, render_interactions_2d, render_scene_pymol
    from autodock.reporting import generate_csv_report, generate_pdf_report
    from autodock.utils import download_pdb, ensure_dir

    outdir = ensure_dir(args.outdir)
    receptor_name = args.receptor.upper()
    ligand_name = args.ligand

    print("\n" + "=" * 55)
    print("📥  Step 1: Fetching structures")
    print("=" * 55)
    receptor_pdb = os.path.join(outdir, f"{receptor_name}.pdb")
    if not os.path.exists(receptor_pdb):
        download_pdb(receptor_name, outdir)
    else:
        print(f"  Using cached: {receptor_pdb}")

    print("\n" + "=" * 55)
    print("🔧  Step 2: Preparing structures")
    print("=" * 55)
    receptor_pdbqt = os.path.join(outdir, f"{receptor_name}.pdbqt")
    ligand_pdbqt = os.path.join(outdir, f"{ligand_name}.pdbqt")

    prepare_receptor(receptor_pdb, receptor_pdbqt)

    # Try to get ligand SMILES from PubChem
    try:
        import pubchempy as pcp

        compounds = pcp.get_compounds(ligand_name, "name")
        if compounds:
            smiles = compounds[0].canonical_smiles
            print(f"  Ligand SMILES (PubChem): {smiles}")
        else:
            # Fallback: treat as raw SMILES
            smiles = ligand_name
            print(f"  Treating ligand as raw SMILES: {smiles}")
    except Exception:
        smiles = ligand_name
        print(f"  Treating ligand as raw SMILES: {smiles}")

    prepare_ligand(smiles, ligand_pdbqt, name="LIG")

    print("\n" + "=" * 55)
    print("🔍  Step 3: Detecting binding pocket")
    print("=" * 55)
    pockets = find_top_pockets(receptor_pdb, max_pockets=3)
    center = pockets[0]["center"]
    box_size = pockets[0]["box_size"]
    print(f"  Best pocket: center={center}, box={box_size}")

    print("\n" + "=" * 55)
    print("🧬  Step 4: Docking")
    print("=" * 55)
    result = dock_ligand(
        receptor_pdbqt,
        ligand_pdbqt,
        center,
        box_size,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        output_dir=outdir,
        compound_name=ligand_name,
        receptor_pdb=receptor_pdb,
    )
    print(f"  Best affinity: {result.best_affinity:.3f} kcal/mol")

    print("\n" + "=" * 55)
    print("🧪  Step 5: Interaction analysis")
    print("=" * 55)
    intx = detect_interactions(receptor_pdb, result.best_pose_pdbqt, method="plip")
    print(f"  Detected {len(intx)} interactions")
    for i in intx[:10]:
        print(f"    {i['description']}")

    print("\n" + "=" * 55)
    print("🎨  Step 6: Rendering")
    print("=" * 55)
    try:
        scene_complex = os.path.join(outdir, "fig_complex.png")
        scene_pocket = os.path.join(outdir, "fig_pocket.png")
        scene_intx = os.path.join(outdir, "fig_interactions.png")
        diagram_2d = os.path.join(outdir, "fig_2d.png")

        render_scene_pymol(receptor_pdb, result.best_pose_pdbqt, scene_complex, scene="complex")
        render_scene_pymol(
            receptor_pdb, result.best_pose_pdbqt, scene_pocket, scene="pocket", center=center
        )
        render_scene_pymol(
            receptor_pdb,
            result.best_pose_pdbqt,
            scene_intx,
            scene="interaction",
            center=center,
            interactions=intx,
        )
        render_interactions_2d(receptor_pdb, result.best_pose_pdbqt, intx, diagram_2d)

        composite = os.path.join(outdir, "fig_composite.png")
        composite_summary(
            [scene_complex, scene_pocket, scene_intx],
            composite,
            ncols=2,
            panel_titles=["A. Complex", "B. Pocket", "C. Interactions"],
            figure_title=f"Docking: {ligand_name} ↔ {receptor_name}",
        )
        print(f"  Figures saved to {outdir}")
    except Exception as exc:
        logger.warning(f"Rendering failed: {exc}")

    print("\n" + "=" * 55)
    print("📊  Step 7: Reporting")
    print("=" * 55)
    result.interactions = intx
    pdf_path = os.path.join(outdir, "report.pdf")
    try:
        figs = [scene_complex, scene_pocket, scene_intx, diagram_2d]
        figs = [f for f in figs if os.path.exists(f)]
        generate_pdf_report(result, pdf_path, figure_paths=figs)
        print(f"  PDF report: {pdf_path}")
    except Exception as exc:
        logger.warning(f"PDF report failed: {exc}")

    csv_path = os.path.join(outdir, "report.csv")
    generate_csv_report([result], csv_path)
    print(f"  CSV report: {csv_path}")

    print("\n" + "=" * 55)
    print("✅  Pipeline Complete!")
    print("=" * 55)
    print(f"  Output directory: {outdir}")
    print(f"  Best affinity:    {result.best_affinity:.3f} kcal/mol")
    if result.consensus_affinity:
        print(f"  Consensus:        {result.consensus_affinity:.3f} kcal/mol")
    print(f"  Best pose:        {result.best_pose_pdbqt}")
    print(f"  PDF report:       {pdf_path}")
    print("=" * 55)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autodock",
        description="Publication-grade molecular docking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    p_status = subparsers.add_parser("status", help="Check environment")
    p_status.set_defaults(func=cmd_status)

    # init
    p_init = subparsers.add_parser("init", help="Create default config")
    p_init.add_argument("--config", default="docking_config.yaml")
    p_init.set_defaults(func=cmd_init)

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="Download structure")
    p_fetch.add_argument("type", choices=["pdb", "ligand"])
    p_fetch.add_argument("id", help="PDB ID or ligand code")
    p_fetch.add_argument("-o", "--outdir", default=".")
    p_fetch.set_defaults(func=cmd_fetch)

    # prepare-receptor
    p_prep_rec = subparsers.add_parser("prepare-receptor", help="PDB → PDBQT")
    p_prep_rec.add_argument("pdb", help="Input PDB file")
    p_prep_rec.add_argument("-o", "--output", help="Output PDBQT file")
    p_prep_rec.add_argument("--keep-waters", action="store_true")
    p_prep_rec.add_argument("--remove-hetatms", action="store_true", default=True)
    p_prep_rec.set_defaults(func=cmd_prepare_receptor)

    # prepare-ligand
    p_prep_lig = subparsers.add_parser("prepare-ligand", help="SMILES → PDBQT")
    p_prep_lig.add_argument("smiles", help="SMILES string or compound name")
    p_prep_lig.add_argument("-o", "--output", default="ligand.pdbqt")
    p_prep_lig.add_argument("--name", default="LIG")
    p_prep_lig.add_argument("--seed", type=int, default=42)
    p_prep_lig.set_defaults(func=cmd_prepare_ligand)

    # find-pockets
    p_pocket = subparsers.add_parser("find-pockets", help="Detect binding pockets")
    p_pocket.add_argument("receptor", help="Receptor PDB file")
    p_pocket.add_argument("--ligand", help="Optional co-crystal ligand PDB")
    p_pocket.add_argument("--padding", type=float, default=5.0)
    p_pocket.add_argument("--max-pockets", type=int, default=3)
    p_pocket.set_defaults(func=cmd_find_pockets)

    # dock
    p_dock = subparsers.add_parser("dock", help="Run Vina docking")
    p_dock.add_argument("receptor", help="Receptor PDBQT")
    p_dock.add_argument("ligand", help="Ligand PDBQT")
    p_dock.add_argument("--center", nargs=3, type=float, help="Box center (x y z)")
    p_dock.add_argument("--box-size", nargs=3, type=float, help="Box size (sx sy sz)")
    p_dock.add_argument("--exhaustiveness", type=int, default=32)
    p_dock.add_argument("--n-poses", type=int, default=20)
    p_dock.add_argument("--seed", type=int, default=42)
    p_dock.add_argument("--output-dir", default="./docking_results")
    p_dock.add_argument("--name", help="Compound name")
    p_dock.set_defaults(func=cmd_dock)

    # validate
    p_val = subparsers.add_parser("validate", help="Redocking validation")
    p_val.add_argument("holo_pdb", help="Holo PDB with co-crystal ligand")
    p_val.add_argument(
        "--ligand-resname", default=None, help="Residue name of ligand (HETATM mode)"
    )
    p_val.add_argument(
        "--chain-id", default=None, help="Chain ID to extract (e.g., 'C' for 6LU7 N3)"
    )
    p_val.add_argument(
        "--ligand-smiles", default=None, help="Optional SMILES for ligand preparation"
    )
    p_val.add_argument("--exhaustiveness", type=int, default=32)
    p_val.add_argument("--n-poses", type=int, default=20)
    p_val.add_argument("--seed", type=int, default=42)
    p_val.add_argument("--box-padding", type=float, default=5.0)
    p_val.add_argument("--output-dir", default="./redock_validation")
    p_val.set_defaults(func=cmd_validate)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Detect interactions")
    p_analyze.add_argument("receptor", help="Receptor PDB")
    p_analyze.add_argument("ligand", help="Docked ligand PDBQT")
    p_analyze.add_argument("--output-dir", help="Directory for figures")
    p_analyze.set_defaults(func=cmd_analyze)

    # report
    p_report = subparsers.add_parser("report", help="Generate reports")
    p_report.add_argument("result_dir", help="Directory with docking results")
    p_report.set_defaults(func=cmd_report)

    # benchmark-redock
    p_bench = subparsers.add_parser(
        "benchmark-redock", help="Run redocking benchmark on standard target set"
    )
    p_bench.add_argument("--outdir", default="./benchmark_results")
    p_bench.add_argument("--exhaustiveness", type=int, default=32)
    p_bench.add_argument("--n-poses", type=int, default=20)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--workers", type=int, default=1)
    p_bench.add_argument(
        "--targets", type=str, default=None, help="JSON file with custom target list"
    )
    p_bench.set_defaults(func=cmd_benchmark_redock)

    # posebusters-eval
    p_pb = subparsers.add_parser("posebusters-eval", help="Run PoseBusters benchmark evaluation")
    p_pb.add_argument("id_list", help="Text file with PoseBusters IDs (PDBID_CCD per line)")
    p_pb.add_argument("--outdir", default="./posebusters_results")
    p_pb.add_argument("--exhaustiveness", type=int, default=32)
    p_pb.add_argument("--n-poses", type=int, default=20)
    p_pb.add_argument("--seed", type=int, default=42)
    p_pb.add_argument("--workers", type=int, default=1)
    p_pb.add_argument(
        "--max-targets", type=int, default=None, help="Limit to first N targets for quick tests"
    )
    p_pb.set_defaults(func=cmd_posebusters_eval)

    # batch-dock
    p_batch = subparsers.add_parser(
        "batch-dock", help="Multi-receptor × multi-ligand batch docking"
    )
    p_batch.add_argument("--receptors", nargs="+", required=True, help="Receptor PDBQT files")
    p_batch.add_argument("--ligands", nargs="+", required=True, help="Ligand PDBQT files")
    p_batch.add_argument(
        "--pockets", required=True, help="JSON file with pocket definitions per receptor"
    )
    p_batch.add_argument("--exhaustiveness", type=int, default=32)
    p_batch.add_argument("--n-poses", type=int, default=20)
    p_batch.add_argument("--seed", type=int, default=42)
    p_batch.add_argument("--workers", type=int, default=1, help="Parallel workers (-1 = all cores)")
    p_batch.add_argument("--outdir", default="./batch_docking_results")
    p_batch.set_defaults(func=cmd_batch_dock)

    # ensemble-dock
    p_ensemble = subparsers.add_parser(
        "ensemble-dock", help="Repeated docking with ensemble statistics"
    )
    p_ensemble.add_argument("receptor", help="Receptor PDBQT")
    p_ensemble.add_argument("ligand", help="Ligand PDBQT")
    p_ensemble.add_argument(
        "--center", type=float, nargs=3, default=None, help="Box center (x y z)"
    )
    p_ensemble.add_argument(
        "--box-size", type=float, nargs=3, default=[20.0, 20.0, 20.0], help="Box dimensions (x y z)"
    )
    p_ensemble.add_argument("--n-repeats", type=int, default=10, help="Number of independent runs")
    p_ensemble.add_argument("--exhaustiveness", type=int, default=32)
    p_ensemble.add_argument("--n-poses", type=int, default=20)
    p_ensemble.add_argument("--seed", type=int, default=42)
    p_ensemble.add_argument("--outdir", default="./ensemble_docking_results")
    p_ensemble.add_argument("--name", default=None, help="Compound name")
    p_ensemble.set_defaults(func=cmd_ensemble_dock)

    # virtual-screen
    p_vs = subparsers.add_parser("virtual-screen", help="Screen compound library")
    p_vs.add_argument("--receptor", required=True, help="PDB ID (e.g. 6LU7)")
    p_vs.add_argument("--library", required=True, help="TSV/txt file: 'name SMILES' per line")
    p_vs.add_argument("--outdir", default="./vs_results")
    p_vs.add_argument("--exhaustiveness", type=int, default=16)
    p_vs.add_argument("--n-poses", type=int, default=3)
    p_vs.add_argument("--seed", type=int, default=42)
    p_vs.add_argument("--workers", type=int, default=1, help="Parallel workers (-1 = all cores)")
    p_vs.set_defaults(func=cmd_virtual_screen)

    # md
    p_md = subparsers.add_parser("md", help="Short MD stability simulation")
    p_md.add_argument("--receptor", required=True, help="Receptor PDB file")
    p_md.add_argument("--ligand", required=True, help="Docked ligand PDBQT file")
    p_md.add_argument("--outdir", default="./md_results")
    p_md.add_argument(
        "--steps", type=int, default=500_000, help="Production steps (default 500k = 1 ns)"
    )
    p_md.add_argument("--dt", type=float, default=2.0, help="Timestep (fs)")
    p_md.add_argument("--temperature", type=float, default=300.0, help="Temperature (K)")
    p_md.add_argument("--solvent", choices=["implicit", "explicit"], default="implicit")
    p_md.add_argument("--platform", default=None, help="OpenMM platform (CPU/OpenCL/CUDA/Metal)")
    p_md.set_defaults(func=cmd_md)

    # run (full pipeline)
    p_run = subparsers.add_parser("run", help="Full pipeline")
    p_run.add_argument("--receptor", required=True, help="PDB ID (e.g. 6LU7)")
    p_run.add_argument("--ligand", required=True, help="Ligand name or SMILES")
    p_run.add_argument("--outdir", default="./docking_results")
    p_run.add_argument("--exhaustiveness", type=int, default=32)
    p_run.add_argument("--n-poses", type=int, default=20)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    _setup_logging(args)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130
    except Exception as exc:
        logger.error(f"Command failed: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
