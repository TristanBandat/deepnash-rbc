"""Project version discovery.

The version is the single label that ties a checkpoint to the architecture and
hyperparameters that produced it (see ``checkpoints.py``). We read it from the
live ``pyproject.toml`` when running out of the source tree -- the thesis/dev
workflow bumps the version frequently without reinstalling -- and fall back to
the installed package metadata when running from a built wheel.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the project version, e.g. ``"0.2.0"`` (no leading ``v``)."""
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            import tomllib

            data = tomllib.loads(pyproject.read_text())
            version = data.get("project", {}).get("version")
            if version:
                return str(version)

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("deepnash-rbc")
    except PackageNotFoundError:
        return "0.0.0"
