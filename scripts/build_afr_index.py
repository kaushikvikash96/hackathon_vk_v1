"""Build the AFR inverted index.

    python scripts/build_afr_index.py

Reads DATA_ROOT/AFR/*.jsonl and writes ARTIFACTS_DIR/afr/. Takes a few minutes
and only needs to be re-run when the corpus changes. The agent builds it
automatically on first start if it is missing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import afr_index  # noqa: E402
from src.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    if not settings.afr_dir.exists():
        print(f"AFR directory not found: {settings.afr_dir}", file=sys.stderr)
        return 1

    started = time.time()
    out_dir = afr_index.build(verbose=True)
    print(f"done in {time.time() - started:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
