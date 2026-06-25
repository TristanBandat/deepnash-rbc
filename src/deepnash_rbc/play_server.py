"""Local web server to play against a trained checkpoint.

A thin Flask wrapper over GameSession. Single active game at a time (it's a local
testing tool, not a service). Serves the board UI at / and a small JSON API.

Run:  uv run deepnash-play --checkpoint-dir checkpoints
      then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import chess
import torch
from flask import Flask, jsonify, request, send_file

from .config import EncodingConfig, NetworkConfig
from .network import DeepNashNet
from .play_session import GameSession, load_net

app = Flask(__name__)

_HTML = Path(__file__).parent / "play.html"
_lock = threading.Lock()
_session: Optional[GameSession] = None
_net_cache: Dict[str, Tuple[DeepNashNet, EncodingConfig]] = {}
_args = argparse.Namespace(checkpoint_dir="checkpoints", device="cpu")


def _get_net(checkpoint: str) -> Tuple[DeepNashNet, EncodingConfig]:
    if checkpoint in _net_cache:
        return _net_cache[checkpoint]
    device = torch.device(_args.device if torch.cuda.is_available() or _args.device == "cpu" else "cpu")
    if checkpoint == "__random__":
        enc = EncodingConfig()
        net = DeepNashNet(enc, NetworkConfig()).to(device).eval()
    else:
        path = str(Path(_args.checkpoint_dir) / checkpoint)
        net, enc = load_net(path, device)
    _net_cache[checkpoint] = (net, enc)
    return net, enc


@app.get("/")
def index():
    return send_file(_HTML)


@app.get("/api/checkpoints")
def checkpoints():
    d = Path(_args.checkpoint_dir)
    files = sorted((f.name for f in d.glob("*.pt")), reverse=True) if d.exists() else []
    return jsonify({"checkpoints": files})


@app.post("/api/new")
def new_game():
    global _session
    data = request.get_json(force=True)
    checkpoint = data.get("checkpoint", "__random__")
    human_color = chess.WHITE if data.get("color", "white") == "white" else chess.BLACK
    sample = bool(data.get("sample", False))
    try:
        net, enc = _get_net(checkpoint)
    except Exception as e:
        return jsonify({"error": f"could not load checkpoint: {e}"}), 400
    with _lock:
        _session = GameSession(net, human_color, history=enc.history, sample=sample)
        return jsonify(_session.state())


@app.get("/api/state")
def state():
    if _session is None:
        return jsonify({"error": "no active game"}), 400
    return jsonify(_session.state())


@app.post("/api/sense")
def sense():
    if _session is None:
        return jsonify({"error": "no active game"}), 400
    square = request.get_json(force=True).get("square")
    with _lock:
        try:
            return jsonify(_session.do_sense(square))
        except Exception as e:
            return jsonify({"error": str(e)}), 400


@app.post("/api/move")
def move():
    if _session is None:
        return jsonify({"error": "no active game"}), 400
    uci = request.get_json(force=True).get("uci")  # None => pass
    with _lock:
        try:
            return jsonify(_session.do_move(uci))
        except Exception as e:
            return jsonify({"error": str(e)}), 400


def main() -> None:
    global _args
    p = argparse.ArgumentParser(description="Play against a DeepNash-RBC checkpoint.")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    _args = p.parse_args()
    print(f"Open http://{_args.host}:{_args.port}  (checkpoints: {_args.checkpoint_dir})")
    app.run(host=_args.host, port=_args.port, threaded=True)


if __name__ == "__main__":
    main()
