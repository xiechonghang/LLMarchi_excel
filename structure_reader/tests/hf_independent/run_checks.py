#!/usr/bin/env python3
"""Self-contained runner for the independent HF decoder expectations.

Usage:
    <engineering-python> structure_reader/tests/hf_independent/run_checks.py

Runs only the tests in this directory. It never imports torch/transformers and
never touches product output, .work coordination state or generated figures.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    os.chdir(HERE)
    loader = unittest.TestLoader()
    suite = loader.discover(str(HERE), pattern="test_*.py", top_level_dir=str(HERE))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
