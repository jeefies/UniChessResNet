"""Compatibility entry point for the standard AlphaZero loop.

The previous actor/learner implementation is kept as worker.legacy.py.
All managed services must use --mode loop.
"""
from __future__ import annotations

import sys

from autoloop.alphazero import main as alphazero_main


def main() -> None:
    if "--mode" in sys.argv:
        index = sys.argv.index("--mode")
        if index + 1 < len(sys.argv) and sys.argv[index + 1] != "loop":
            raise SystemExit(
                "deprecated worker mode; use --mode loop for the standard AlphaZero loop"
            )
    alphazero_main()


if __name__ == "__main__":
    main()
