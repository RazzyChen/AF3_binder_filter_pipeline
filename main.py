"""Compatibility wrapper for the Aerith CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from af3_binder_filter.cli import main

if __name__ == "__main__":
    main()
