"""Module entrypoint allowing `python -m ral ...`."""

from __future__ import annotations

from ral.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
