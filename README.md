# Autodock — Publication-Grade Molecular Docking Pipeline

[![CI](https://github.com/yourorg/autodock/actions/workflows/ci.yml/badge.svg)](https://github.com/yourorg/autodock/actions)
[![Coverage](https://codecov.io/gh/yourorg/autodock/branch/main/graph/badge.svg)](https://codecov.io/gh/yourorg/autodock)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Autodock** is an open-source, end-to-end molecular docking automation pipeline designed for reproducible, publication-quality results. It integrates modern cheminformatics tools (RDKit, Meeko), authoritative interaction analysis (PLIP, ProLIF), pose validation (PoseBusters), and molecular dynamics stability checking (OpenMM) into a single coherent workflow.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/autodock.git
cd autodock

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate autodock

# Install package
pip install -e ".[all]"
```

### 5-Minute Example

```bash
# Check environment
autodock status

# Single-ligand docking from PDB ID
autodock run --receptor 6LU7 --ligand "CC(C)Cc1ccc(C(C)C(=O)O)cc1" --outdir ./demo

# Virtual screening
autodock virtual-screen --receptor 6LU7 --library compounds.txt --workers -1
```

### Python API

```python
from autodock import (
    prepare_receptor, prepare_ligand, find_top_pockets, dock_ligand
)
from autodock.core import print_environment_status

print_environment_status()

# Prepare structures
receptor = prepare_receptor("6LU7.pdb", "receptor.pdbqt")
ligand = prepare_ligand("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "ligand.pdbqt")

# Detect pocket
pockets = find_top_pockets("6LU7.pdb")
center, box = pockets[0]["center"], pockets[0]["box_size"]

# Dock (deterministic by default)
result = dock_ligand(
    receptor, ligand, center, box,
    exhaustiveness=32, n_poses=20, seed=42
)
print(f"Best affinity: {result.best_affinity:.2f} kcal/mol")
```

---

## 📋 Features

| Feature | Description |
|---------|-------------|
| **🔬 Docking Engines** | AutoDock Vina with consensus scoring (Vina + Vinardo) |
| **🎯 Pocket Detection** | fpocket geometric + P2Rank ML rescoring |
| **🧬 Structure Prep** | Meeko-based PDBQT generation (modern replacement for MGLTools) |
| **🤝 Interactions** | PLIP (primary) + ProLIF (cross-validation) for 8 interaction types |
| **✅ Validation** | PoseBusters geometric checks + custom clash detection + RMSD |
| **🌊 MD Stability** | OpenMM short MD with ligand RMSD and H-bond analysis |
| **🎨 Visualization** | PyMOL 3D rendering + RDKit 2D interaction diagrams |
| **📊 Reporting** | PDF, Excel, and CSV reports with publication-ready figures (300 dpi) |
| **🧪 Virtual Screening** | Parallel compound library screening with CSV ranking |

---

## 🏗️ Architecture

```
Input (PDB ID / SMILES / File)
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Preparation    │───▶│    Docking      │───▶│   Validation    │
│  (preparation)  │    │   (docking)     │    │  (validation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │                         │                      │
    ▼                         ▼                      ▼
Receptor PDBQT           Pose PDBQT            PoseBusters Pass?
Ligand PDBQT             Affinity Scores       Clash Score
Pocket Center/Box        Consensus Score       RMSD vs Crystal
    │                         │                      │
    ▼                         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Analysis     │◀───│   Rendering     │◀───│      MD         │
│ (interactions)  │    │  (rendering)    │    │ (md_simulation) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
Output (PDF / CSV / PNG / PDBQT)
```

---

## 📖 Usage Scenarios

### 1. Single-Ligand Docking

Dock a known ligand into a known receptor with automatic pocket detection:

```bash
autodock dock receptor.pdbqt ligand.pdbqt \
    --center 10.0 20.0 30.0 \
    --box-size 20 20 20 \
    --exhaustiveness 32 \
    --n-poses 20 \
    --seed 42 \
    --output-dir ./results
```

### 2. Multi-Ligand Multi-Receptor Batch Docking

Perform pairwise docking across multiple receptors and ligands:

```bash
autodock batch-dock \
    --receptors rec1.pdbqt rec2.pdbqt \
    --ligands lig1.pdbqt lig2.pdbqt lig3.pdbqt \
    --pockets pockets.json \
    --seed 42 \
    --workers -1 \
    --output-dir ./batch_results
```

Or via Python API:

```python
from autodock.docking import batch_dock

results = batch_dock(
    receptors={"6LU7": "6LU7.pdbqt", "3CLP": "3CLP.pdbqt"},
    ligands={"aspirin": "aspirin.pdbqt", "ibu": "ibuprofen.pdbqt"},
    pockets={"6LU7": ({"center": (x,y,z), "box_size": (sx,sy,sz)}), ...},
    seed=42,
    n_workers=-1,
)
```

### 3. Virtual Screening

Screen a compound library against a single target:

```bash
# Library file format: one compound per line
# name SMILES
autodock virtual-screen \
    --receptor 6LU7 \
    --library library.txt \
    --exhaustiveness 16 \
    --n-poses 3 \
    --workers -1 \
    --outdir ./vs_results
```

### 4. Redocking Validation

Validate protocol accuracy by re-docking a co-crystallized ligand:

```bash
autodock validate 6LU7_holo.pdb --chain-id C --output-dir ./validation
```

---

## ⚙️ Configuration

Generate a default configuration file:

```bash
autodock init --config docking_config.yaml
```

Key publication-grade defaults:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `exhaustiveness` | 32 | Vina publication standard for reliable pose prediction |
| `num_modes` | 20 | Sufficient for clustering and validation |
| `energy_range` | 3.0 | Standard energy window above best pose |
| `seed` | 42 | Deterministic by default for reproducibility |
| `dpi` | 300 | Publication-ready figure resolution |

---

## 🧪 Development

```bash
# Install in development mode
pip install -e ".[dev,all]"

# Run tests
pytest

# Run with coverage
pytest --cov=autodock --cov-report=html

# Lint and format
ruff check autodock
black autodock

# Type check
mypy autodock
```

---

## 📚 Documentation

- **API Reference**: https://autodock.readthedocs.io
- **Tutorials**: See `docs/tutorials/`
- **Methodology**: See `METHODS.md`

---

## 🙏 Citation

If you use Autodock in your research, please cite:

```bibtex
@software{autodock2024,
  title = {Autodock: A Publication-Grade Molecular Docking Pipeline},
  author = {Autodock Contributors},
  year = {2024},
  url = {https://github.com/yourorg/autodock}
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built for reproducible science. Every docking run records its random seed, software version, and all parameters — because results you can't reproduce are results you can't trust.*
