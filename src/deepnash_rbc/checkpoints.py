"""Versioned checkpoint layout.

Checkpoints live under ``<checkpoint_dir>/v<version>/`` and are named
``<prefix>_v<version>_<step>.pt``. Each project version gets its own folder
because the network architecture (``NetworkConfig``) may change between
versions, and a ``state_dict`` is shape-locked to the architecture that
produced it -- mixing versions would fail to load. Isolating by version keeps
``--resume auto`` and the play UI from ever picking an incompatible file.

The per-checkpoint dict already embeds ``net_cfg``/``enc_cfg`` so a single file
is self-describing (``play_session.load_net`` rebuilds the exact net). This
module adds the directory convention on top so a version's models are grouped
without cracking open every ``.pt``.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional

from .version import get_version


def version_dir(checkpoint_dir: str, version: Optional[str] = None) -> str:
    """Return ``<checkpoint_dir>/v<version>/`` for the given (or current) version."""
    version = version or get_version()
    return os.path.join(checkpoint_dir, f"v{version}")


def checkpoint_path(
    checkpoint_dir: str,
    step: int,
    prefix: str = "deepnash_async",
    version: Optional[str] = None,
) -> str:
    """Full path for a checkpoint at ``step`` in the version's folder.

    e.g. ``checkpoints/v0.2.0/deepnash_async_v0.2.0_100000.pt``.
    """
    version = version or get_version()
    return os.path.join(
        version_dir(checkpoint_dir, version), f"{prefix}_v{version}_{step}.pt"
    )


def find_latest_checkpoint(
    checkpoint_dir: str,
    prefix: str = "deepnash_async",
    version: Optional[str] = None,
) -> Optional[str]:
    """Latest checkpoint in the version's folder, by trailing step (not mtime)."""
    d = version_dir(checkpoint_dir, version)
    paths = glob.glob(os.path.join(d, f"{prefix}_v*_*.pt"))
    best, best_step = None, -1
    for p in paths:
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        if m and int(m.group(1)) > best_step:
            best, best_step = p, int(m.group(1))
    return best
