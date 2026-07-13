"""Download and featurize our games from the JHU APL RBC server.

The server's ``/api/games/?page=N`` endpoint paginates *all* server games
(newest first, 10 per page) with no user filter, so we scan pages back to a
cutoff date, keep the games involving our bot, and fetch the full
:class:`reconchess.GameHistory` for each finished one. Everything is cached
under ``results/rbc_games/`` so re-runs only fetch new games:

    results/rbc_games/index.jsonl        one metadata row per game (server schema)
    results/rbc_games/histories/<id>.json    GameHistory as saved by reconchess

Credentials come from ``.env`` in the repo root (same convention as
``scripts/rc_connect.py``). Run a sync from the CLI:

    uv run python -m deepnash_rbc.analysis.rbc_games --since 2026-07-06

The bot's ranked identity on the server is versioned (V1, V2, ...); each
game row records our ``version_id``, which :data:`VERSION_CHECKPOINTS` maps
back to the checkpoint named in ``rbc_connect.json`` at the time (kept by
hand -- extend it when a new version is benched).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import chess
import requests
from reconchess.history import GameHistory, GameHistoryDecoder

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = REPO_ROOT / "results" / "rbc_games"
SERVER_URL = "https://rbc.jhuapl.edu"
USERNAME = "deepnash-rbc"

# Server bot-version id -> checkpoint played (see rbc_connect.json history and
# bench commits). Extend by hand whenever a new version is put on the ladder.
VERSION_CHECKPOINTS = {
    1: "v0.8.0_70000 (unranked smoke test)",
    2: "v0.8.0_70000",
    3: "v0.14.0_80000",
    4: "v0.27.0_80000",
    5: "v0.13.0_70000",
    6: "v0.33.0_240000",
}

WIN_REASONS = {1: "KING_CAPTURE", 2: "TIMEOUT", 3: "RESIGN", 4: "TURN_LIMIT", None: "UNFINISHED"}

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


# --------------------------------------------------------------- downloading

def read_env(path: Path = REPO_ROOT / ".env") -> dict:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip("'\"")
    return env


def make_session() -> requests.Session:
    env = read_env()
    session = requests.Session()
    session.auth = (env["RBC_USERNAME"], env["RBC_PASSWORD"])
    return session


def scan_games(session: requests.Session, since: str, username: str = USERNAME,
               max_pages: int = 5000) -> list[dict]:
    """Collect metadata for all games involving ``username`` since ``since``.

    ``since`` is an ISO date/datetime string compared against the server's
    ``time_created`` (UTC, ``YYYY-MM-DD HH:MM:SS``); string comparison works
    because both are zero-padded ISO-ordered.
    """
    ours = []
    for page in range(1, max_pages + 1):
        resp = session.get(f"{SERVER_URL}/api/games/?page={page}", timeout=30)
        resp.raise_for_status()
        games = resp.json()["games"]
        if not games:
            break
        for g in games:
            if username in (g["white_username"], g["black_username"]):
                ours.append(g)
        if min(g["time_created"] for g in games) < since:
            break
        time.sleep(0.05)  # be polite; the endpoint is unthrottled
    return [g for g in ours if g["time_created"] >= since]


def sync(cache_dir: Path = DEFAULT_CACHE, since: str = "2026-07-06",
         progress: bool = True) -> list[dict]:
    """Refresh the local cache and return the merged game index (newest first)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = cache_dir / "histories"
    hist_dir.mkdir(exist_ok=True)
    index_path = cache_dir / "index.jsonl"

    known = {}
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            row = json.loads(line)
            known[row["game_id"]] = row

    session = make_session()
    for meta in scan_games(session, since):
        # refresh unfinished rows too: they gain winner/win_reason once done
        if not (known.get(meta["game_id"], {}).get("finished")):
            known[meta["game_id"]] = meta

    missing = [g for g in known.values()
               if g["finished"] and not (hist_dir / f"{g['game_id']}.json").exists()]
    for i, meta in enumerate(missing):
        gid = meta["game_id"]
        resp = session.get(f"{SERVER_URL}/api/games/{gid}/game_history", timeout=60)
        if resp.status_code != 200:  # very old games may be pruned server-side
            if progress:
                print(f"  history {gid}: HTTP {resp.status_code}, skipped")
            continue
        (hist_dir / f"{gid}.json").write_text(json.dumps(resp.json()["game_history"]))
        if progress:
            print(f"  history {gid} ({i + 1}/{len(missing)})")
        time.sleep(0.05)

    rows = sorted(known.values(), key=lambda g: g["game_id"], reverse=True)
    index_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def load_index(cache_dir: Path = DEFAULT_CACHE) -> list[dict]:
    index_path = cache_dir / "index.jsonl"
    if not index_path.exists():
        return []
    return [json.loads(l) for l in index_path.read_text().splitlines()]


def load_history(game_id: int, cache_dir: Path = DEFAULT_CACHE) -> Optional[GameHistory]:
    path = cache_dir / "histories" / f"{game_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(), cls=GameHistoryDecoder)


# --------------------------------------------------------------- featurizing

def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(len(board.pieces(pt, color)) * v for pt, v in PIECE_VALUES.items())


def game_features(meta: dict, history: Optional[GameHistory]) -> dict:
    """Flatten one game into a stats row (metadata always, history when cached)."""
    us_white = meta["white_username"] == USERNAME
    us = chess.WHITE if us_white else chess.BLACK
    row = {
        "game_id": meta["game_id"],
        "time_created": meta["time_created"],
        "version_id": meta["white_version_id"] if us_white else meta["black_version_id"],
        "opponent": meta["black_username"] if us_white else meta["white_username"],
        "opponent_version": meta["black_version_id"] if us_white else meta["white_version_id"],
        "color": "white" if us_white else "black",
        "finished": meta["finished"],
        "won": None if meta.get("winner_color") is None else meta["winner_color"] == us_white,
        "win_reason": WIN_REASONS.get(meta.get("win_reason"), str(meta.get("win_reason"))),
    }
    row["checkpoint"] = VERSION_CHECKPOINTS.get(row["version_id"], f"V{row['version_id']}")
    if meta.get("time_finished") and meta.get("time_created"):
        t0, t1 = meta["time_created"], meta["time_finished"]
        fmt = "%Y-%m-%d %H:%M:%S"
        from datetime import datetime
        row["duration_s"] = (datetime.strptime(t1, fmt) - datetime.strptime(t0, fmt)).total_seconds()

    if history is None:
        return row

    them = not us
    our_turns = [t for t in history.turns(us) if history.has_move(t)]
    their_turns = [t for t in history.turns(them) if history.has_move(t)]
    row["our_moves"] = len(our_turns)

    requested = [history.requested_move(t) for t in our_turns]
    taken = [history.taken_move(t) for t in our_turns]
    row["pass_rate"] = sum(m is None for m in requested) / max(len(requested), 1)
    # requested a move but something else happened (slide/blocked/illegal)
    row["bumped_rate"] = sum(r is not None and r != t for r, t in zip(requested, taken)) / max(len(requested), 1)

    row["captures_made"] = sum(history.capture_square(t) is not None for t in our_turns)
    row["captures_suffered"] = sum(history.capture_square(t) is not None for t in their_turns)

    # sense usefulness: fraction of our senses that saw >= 1 opponent piece
    senses = list(history.turns(us))
    seen = 0
    n_sense = 0
    for t in senses:
        try:
            result = history.sense_result(t)
        except (KeyError, IndexError):
            continue
        if history.sense(t) is None:
            continue
        n_sense += 1
        if any(p is not None and p.color == them for _, p in result):
            seen += 1
    row["sense_hit_rate"] = seen / n_sense if n_sense else None

    # material balance (ours - theirs) at the end, from the last truth FEN
    try:
        last_turn = max(our_turns + their_turns)
        board = history.truth_board_after_move(last_turn)
        row["material_diff_end"] = _material(board, us) - _material(board, them)
    except Exception:
        row["material_diff_end"] = None

    return row


def features_frame(cache_dir: Path = DEFAULT_CACHE, rows: Optional[Iterable[dict]] = None):
    """All cached games as a pandas DataFrame of feature rows (needs pandas)."""
    import pandas as pd

    rows = list(rows) if rows is not None else load_index(cache_dir)
    feats = [game_features(m, load_history(m["game_id"], cache_dir)) for m in rows]
    return pd.DataFrame(feats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="2026-07-06",
                    help="scan server games back to this ISO date (default: first bench)")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()
    rows = sync(args.cache, since=args.since)
    finished = sum(r["finished"] for r in rows)
    print(f"index: {len(rows)} games ({finished} finished) -> {args.cache / 'index.jsonl'}")


if __name__ == "__main__":
    main()
