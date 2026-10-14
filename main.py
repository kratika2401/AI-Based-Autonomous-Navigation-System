"""Top-level executable entry point for AI-Based Autonomous Navigation System.

Delegates execution to the packaged entry point autonomous_nav.main.
"""

import sys
from pathlib import Path

# Ensure src directory is in sys.path when running main.py directly
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from autonomous_nav.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
