#!/usr/bin/env python3
"""Run the optional ASCII map viewer from a checkout, without installing it."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep the launcher usable from any working directory with this checkout's Python.
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from davinci_monet.terminal_map.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
