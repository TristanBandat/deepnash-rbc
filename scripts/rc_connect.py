"""Connect the DeepNash bot to the JHU APL RBC server.

Game/model settings come from rbc_connect.json in the repo root (git-tracked,
so the config history records which checkpoint was played when). Credentials
come from a gitignored .env file in the repo root:

    RBC_USERNAME=mybot
    RBC_PASSWORD=secret

Real environment variables of the same name take precedence over .env. If
RBC_PASSWORD is empty everywhere, rc-connect prompts for it at launch.

    uv run python scripts/rc_connect.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from reconchess.scripts.rc_connect import main as rc_connect_main

ROOT = Path(__file__).resolve().parent.parent


def _read_env_file(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip("'\"")
    return env


def main() -> None:
    cfg = json.loads((ROOT / "rbc_connect.json").read_text())
    env = _read_env_file(ROOT / ".env")

    username = os.environ.get("RBC_USERNAME") or env.get("RBC_USERNAME")
    password = os.environ.get("RBC_PASSWORD") or env.get("RBC_PASSWORD")
    if not username:
        sys.exit("set RBC_USERNAME in .env (repo root) or the environment first")

    argv = [
        "rc-connect",
        str(ROOT / "scripts" / "deepnash_bot.py"),
        "--username", username,
        "--server-url", cfg.get("server_url", "https://rbc.jhuapl.edu"),
        "--max-concurrent-games", str(cfg.get("max_concurrent_games", 1)),
    ]
    if password:  # else rc-connect prompts for it
        argv += ["--password", password]
    if cfg.get("ranked"):
        argv.append("--ranked")
        if cfg.get("keep_version", True):
            argv.append("--keep-version")

    sys.argv = argv
    rc_connect_main()


if __name__ == "__main__":
    main()
