"""
autodock.__main__ — Entry point for `python -m autodock`.
"""

import sys

from autodock.cli import main

if __name__ == "__main__":
    sys.exit(main())
