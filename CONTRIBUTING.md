# Contributing to Autodock

Thank you for your interest in improving Autodock! This document provides
guidelines for contributing code, reporting issues, and submitting pull requests.

## Code of Conduct

Be respectful, constructive, and inclusive. Scientific software thrives on
collaborative, evidence-based discussion.

## How to Contribute

### Reporting Bugs

1. Search existing [issues](https://github.com/tianhuarong/autodock/issues) to
   avoid duplicates.
2. Open a new issue with:
   - A minimal reproducible example (MRE)
   - Expected vs. actual behavior
   - Environment details: OS, Python version, and key dependency versions
     (`pip freeze | grep -E "rdkit|openmm|vina|meeko"`)
   - Full traceback or error message

### Suggesting Features

Open an issue labelled `enhancement` with:
- Clear description of the scientific use case
- Reference to published method or benchmark, if applicable
- Proposed API or CLI interface

### Pull Requests

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Write tests** for new functionality or bug fixes.
3. **Run the test suite** locally:
   ```bash
   pytest autodock/tests/ --cov=autodock --cov-fail-under=60 -x
   ```
4. **Lint and format** your code:
   ```bash
   ruff check autodock/
   ruff format autodock/
   ```
5. **Update documentation** if you change public APIs or add new modules.
6. **Commit** using clear, imperative messages:
   ```
   Add hydrogen-bond directionality filter to interaction analysis
   ```
7. **Push** and open a pull request against `main`.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/autodock.git
cd autodock

# Create environment
conda env create -f environment.yml
conda activate autodock

# Editable install with all extras
pip install -e ".[all]"
```

## Coding Standards

- **Python 3.10+** syntax (union types with `|`, match statements where appropriate)
- **Type hints** on all public functions; `Any` is acceptable for optional
  dependency objects (e.g. OpenFF `Molecule`)
- **Docstrings** follow Google style with Args/Returns sections
- **Scientific rigor**: cite methods, document assumptions, and include
  uncertainty estimates where applicable

## Testing Guidelines

- Unit tests go in `autodock/tests/`
- Use `pytest` fixtures for shared test data
- Mock external network calls (PDB, PubChem, AlphaFold) to avoid flaky tests
- Integration tests requiring Vina or RDKit should be marked:
  ```python
  @pytest.mark.requires_vina
  @pytest.mark.requires_rdkit
  ```

## Release Process

1. Update `CHANGELOG.md` with user-facing changes.
2. Bump version in `pyproject.toml` following [SemVer](https://semver.org/).
3. Tag the release: `git tag -a v1.x.x -m "Release v1.x.x"`
4. Push tags: `git push origin --tags`

## Questions?

Open a [discussion](https://github.com/tianhuarong/autodock/discussions) or
reach out via the issue tracker.
