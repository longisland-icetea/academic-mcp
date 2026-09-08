#!/usr/bin/env python3
"""Console entrypoint: academic-mcp"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from academic_mcp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
