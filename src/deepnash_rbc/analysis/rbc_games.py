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
import os
import time
from pathlib import Path
from typing import Iterable, Optional

try:  # optional: ~10x faster JSON parsing for the archive scan
    import orjson as _fast_json
except ImportError:
    _fast_json = json

import chess
import requests
from reconchess.history import GameHistory, GameHistoryDecoder

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = REPO_ROOT / "results" / "rbc_games"
SERVER_URL = "https://rbc.jhuapl.edu"
USERNAME = "deepnash-rbc"

# The server publishes every game as one periodically-updated zip of
# ``<game_id>.json`` GameHistory files (public, no auth). This is the right
# source for population-wide statistics; the paginated API above is for
# incrementally tracking our own recent games.
ARCHIVE_URL = f"{SERVER_URL}/static/RBC_game_logs.zip"
DEFAULT_ARCHIVE = REPO_ROOT / "results" / "RBC_game_logs.zip"

# Server bot-version id -> checkpoint played (see rbc_connect.json history and
# bench commits). Extend by hand whenever a new version is put on the ladder.
VERSION_CHECKPOINTS = {
    1: "v0.12.0_80000",
    2: "v0.8.0_70000",
    3: "v0.14.0_80000",
    4: "v0.27.0_80000",
    5: "v0.13.0_70000",
    6: "v0.33.0_240000",
    7: "v0.40.0_80000",
    8: "v0.42.0_80000",
}

WIN_REASONS = {
    1: "KING_CAPTURE",
    2: "TIMEOUT",
    3: "RESIGN",
    4: "TURN_LIMIT",
    None: "UNFINISHED",
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
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


def scan_games(
    session: requests.Session,
    since: str,
    username: Optional[str] = USERNAME,
    max_pages: int = 5000,
) -> list[dict]:
    """Collect metadata for all games since ``since``.

    ``username`` filters to games involving that bot; pass ``username=None`` to
    keep *every* server game (for aggregate, cross-bot statistics).

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
            if username is None or username in (
                g["white_username"],
                g["black_username"],
            ):
                ours.append(g)
        if min(g["time_created"] for g in games) < since:
            break
        time.sleep(0.05)  # be polite; the endpoint is unthrottled
    return [g for g in ours if g["time_created"] >= since]


def sync(
    cache_dir: Path = DEFAULT_CACHE,
    since: str = "2026-07-06",
    username: Optional[str] = USERNAME,
    max_histories: Optional[int] = None,
    progress: bool = True,
) -> list[dict]:
    """Refresh the local cache and return the merged game index (newest first).

    ``username=None`` caches *all* server games (see :func:`scan_games`).
    ``max_histories`` caps how many missing game histories are fetched in one
    call -- a safety valve when caching the whole server, whose history
    downloads are one request each.
    """
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
    for meta in scan_games(session, since, username=username):
        # refresh unfinished rows too: they gain winner/win_reason once done
        if not (known.get(meta["game_id"], {}).get("finished")):
            known[meta["game_id"]] = meta

    missing = [
        g
        for g in known.values()
        if g["finished"] and not (hist_dir / f"{g['game_id']}.json").exists()
    ]
    if max_histories is not None:
        missing = missing[:max_histories]
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


def load_history(
    game_id: int, cache_dir: Path = DEFAULT_CACHE
) -> Optional[GameHistory]:
    path = cache_dir / "histories" / f"{game_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(), cls=GameHistoryDecoder)


# ------------------------------------------------------- full-corpus archive


def download_archive(
    dest: Path = DEFAULT_ARCHIVE,
    url: str = ARCHIVE_URL,
    progress: bool = True,
    chunk: int = 1 << 20,
) -> Path:
    """Download the server's full game-log zip, resuming a partial file.

    The archive is ~1.3 GB (hundreds of thousands of ``<game_id>.json``
    histories). Re-running resumes via an HTTP range request when the server
    still reports the same total size, and is a no-op once complete.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers.get("content-length", 0))

    # The server drops long connections partway through the ~1.3 GB transfer,
    # so download in a resume loop: each attempt continues from the bytes on
    # disk via a range request. It honours ranges (206) but does not advertise
    # Accept-Ranges on HEAD, so trust the response status, not the header.
    stalled = 0
    while True:
        got = dest.stat().st_size if dest.exists() else 0
        if total and got >= total:
            break
        headers, mode = {}, "wb"
        if got and total:
            headers["Range"] = f"bytes={got}-"
            mode = "ab"
        else:
            got = 0
        before, next_mark = got, got + (64 << 20)
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                if mode == "ab" and resp.status_code != 206:
                    # server ignored the range and is sending the whole file
                    got, mode = 0, "wb"
                with open(dest, mode) as fh:
                    for block in resp.iter_content(chunk):
                        fh.write(block)
                        got += len(block)
                        if progress and got >= next_mark:
                            pct = f" ({got / total:.0%})" if total else ""
                            print(f"  downloaded {got / 1e9:.2f} GB{pct}")
                            next_mark += 64 << 20
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as err:
            if not total:
                raise  # cannot tell progress from truncation without a size
            stalled = stalled + 1 if got <= before else 0
            if stalled >= 5:
                raise IOError(f"download stuck at {got}/{total} bytes") from err
            if progress:
                print(f"  connection dropped at {got / 1e9:.2f} GB, resuming…")
            time.sleep(2)
            continue
        if not total:
            break  # no content-length: assume the single pass was complete
    if progress:
        print(f"archive saved: {dest} ({got / 1e9:.2f} GB)")
    return dest


def iter_archive(
    zip_path: Path = DEFAULT_ARCHIVE,
    limit: Optional[int] = None,
    sample: Optional[int] = None,
    seed: int = 0,
):
    """Yield ``(game_id, history_dict)`` for games in the archive zip.

    Reads members directly from the zip -- no extraction of the ~800k files.
    ``sample`` draws a random subset (reproducible via ``seed``); ``limit``
    takes the first N. Each ``history_dict`` is the raw reconchess
    ``GameHistory`` JSON (keys ``taken_moves``, ``requested_moves``,
    ``capture_squares``, ``winner_color``, ``win_reason``, ``white_name`` ...).
    """
    import random
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if sample is not None and sample < len(names):
            names = random.Random(seed).sample(names, sample)
        if limit is not None:
            names = names[:limit]
        for name in names:
            stem = name.rsplit("/", 1)[-1][:-5]  # drop dir + ".json"
            gid = int(stem) if stem.isdigit() else None
            with zf.open(name) as fh:
                yield gid, json.load(fh)


def _archive_row(name: str, raw: bytes) -> dict:
    """Flatten one archived GameHistory JSON into a flat stats row."""
    h = _fast_json.loads(raw)
    stem = name.rsplit("/", 1)[-1][:-5]  # drop dir + ".json"
    wc = h.get("winner_color")
    wr = h.get("win_reason")
    tm = h.get("taken_moves") or {}
    rm = h.get("requested_moves") or {}
    cs = h.get("capture_squares") or {}
    white = tm.get("true") or []
    black = tm.get("false") or []
    requested = (rm.get("true") or []) + (rm.get("false") or [])
    captures = (cs.get("true") or []) + (cs.get("false") or [])
    return {
        "game_id": int(stem) if stem.isdigit() else None,
        "white": h.get("white_name"),
        "black": h.get("black_name"),
        "winner_color": wc,
        "win_reason": wr.get("value") if isinstance(wr, dict) else (wr or "UNKNOWN"),
        "outcome": "draw" if wc is None else "white" if wc else "black",
        "white_moves": len(white),
        "black_moves": len(black),
        "total_plies": len(white) + len(black),
        "passes": sum(m is None for m in requested),
        "captures": sum(s is not None for s in captures),
    }


_worker_zip = None  # per-process ZipFile, set by _archive_worker_init


def _archive_worker_init(zip_path: str) -> None:
    global _worker_zip
    import zipfile

    _worker_zip = zipfile.ZipFile(zip_path)


def _archive_scan_chunk(names: list[str]) -> list[dict]:
    return [_archive_row(n, _worker_zip.read(n)) for n in names]


def archive_frame(
    zip_path: Path = DEFAULT_ARCHIVE,
    sample: Optional[int] = None,
    seed: int = 0,
    workers: Optional[int] = None,
    refresh: bool = False,
    progress: bool = False,
):
    """One flat stats row per archived game, as a polars DataFrame.

    Scans the zip once in parallel (JSON parsing dominates, so worker
    processes each open the zip themselves) and caches a *full* scan as
    parquet next to the zip -- later calls load that in well under a second.
    ``sample`` draws a reproducible random subset; sampled scans skip the
    zip only when the parquet cache already exists, and never write it.
    ``refresh`` forces a re-scan (e.g. after re-downloading the archive).
    """
    import random
    import zipfile
    from concurrent.futures import ProcessPoolExecutor

    import polars as pl

    zip_path = Path(zip_path)
    cache = zip_path.with_suffix(".parquet")
    if (
        not refresh
        and cache.exists()
        and cache.stat().st_mtime >= zip_path.stat().st_mtime
    ):
        df = pl.read_parquet(cache)
        if sample is not None and sample < df.height:
            df = df.sample(sample, seed=seed)
        return df

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
    full = sample is None or sample >= len(names)
    if not full:
        names = random.Random(seed).sample(names, sample)

    workers = workers or min(16, os.cpu_count() or 1)
    chunks = [names[i : i + 2000] for i in range(0, len(names), 2000)]
    rows: list[dict] = []
    with ProcessPoolExecutor(
        workers, initializer=_archive_worker_init, initargs=(str(zip_path),)
    ) as ex:
        for part in ex.map(_archive_scan_chunk, chunks):
            rows.extend(part)
            if progress and len(rows) % 100_000 < 2000:
                print(f"  scanned {len(rows):,}/{len(names):,} games")
    df = pl.DataFrame(rows).sort("game_id")
    if full:
        df.write_parquet(cache)
    return df


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
        "version_id": meta["white_version_id"]
        if us_white
        else meta["black_version_id"],
        "opponent": meta["black_username"] if us_white else meta["white_username"],
        "opponent_version": meta["black_version_id"]
        if us_white
        else meta["white_version_id"],
        "color": "white" if us_white else "black",
        "finished": meta["finished"],
        "won": None
        if meta.get("winner_color") is None
        else meta["winner_color"] == us_white,
        "win_reason": WIN_REASONS.get(
            meta.get("win_reason"), str(meta.get("win_reason"))
        ),
    }
    row["checkpoint"] = VERSION_CHECKPOINTS.get(
        row["version_id"], f"V{row['version_id']}"
    )
    if meta.get("time_finished") and meta.get("time_created"):
        t0, t1 = meta["time_created"], meta["time_finished"]
        fmt = "%Y-%m-%d %H:%M:%S"
        from datetime import datetime

        row["duration_s"] = (
            datetime.strptime(t1, fmt) - datetime.strptime(t0, fmt)
        ).total_seconds()

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
    row["bumped_rate"] = sum(
        r is not None and r != t for r, t in zip(requested, taken)
    ) / max(len(requested), 1)

    row["captures_made"] = sum(history.capture_square(t) is not None for t in our_turns)
    row["captures_suffered"] = sum(
        history.capture_square(t) is not None for t in their_turns
    )

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


def features_frame(
    cache_dir: Path = DEFAULT_CACHE, rows: Optional[Iterable[dict]] = None
):
    """All cached games as a pandas DataFrame of feature rows (needs pandas)."""
    import pandas as pd

    rows = list(rows) if rows is not None else load_index(cache_dir)
    feats = [game_features(m, load_history(m["game_id"], cache_dir)) for m in rows]
    return pd.DataFrame(feats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--since",
        default="2026-07-06",
        help="scan server games back to this ISO date (default: first bench)",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="cache dir (defaults to results/rbc_games, or "
        "results/rbc_games_all with --all)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="cache EVERY server game, not just ours (cross-bot stats)",
    )
    ap.add_argument(
        "--max-histories",
        type=int,
        default=None,
        help="cap game-history downloads per run (safety valve for --all)",
    )
    ap.add_argument(
        "--archive",
        action="store_true",
        help="download the full-corpus game-log zip instead of scanning the API",
    )
    args = ap.parse_args()
    if args.archive:
        path = download_archive(args.cache or DEFAULT_ARCHIVE)
        print(f"archive: {path}")
        return
    cache = args.cache or (
        REPO_ROOT / "results" / "rbc_games_all" if args.all else DEFAULT_CACHE
    )
    rows = sync(
        cache,
        since=args.since,
        username=None if args.all else USERNAME,
        max_histories=args.max_histories,
    )
    finished = sum(r["finished"] for r in rows)
    print(f"index: {len(rows)} games ({finished} finished) -> {cache / 'index.jsonl'}")


if __name__ == "__main__":
    main()
