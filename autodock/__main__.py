"""
autodock.__main__ — Entry point for `python -m autodock`.
"""
from autodock.cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
