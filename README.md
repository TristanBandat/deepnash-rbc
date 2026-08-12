[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Pull Requests][pulls-shield]][pulls-url]
[![MIT License][license-shield]][license-url]
[![closed Pull Requests][closed_pulls-shield]][closed_pulls-url]
[![closed Issues][closed_issues-shield]][closed_issues-url]

<!-- PROJECT LOGO -->
<p align="center">
  <h3 align="center">DeepNash for RBC</h3>
  <p align="center">
    A model-free multiagend reinforcement learning model for Reconnaissance Blind Chess (RBC)<br/>
    <a href="https://github.com/TristanBandat/deepnash-rbc"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/TristanBandat/deepnash-rbc/issues">Report Bug</a>
    ·
    <a href="https://github.com/TristanBandat/deepnash-rbc/issues">Request Feature</a>
  </p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#training">Training</a></li>
        <li><a href="#sweeps-and-campaigns">Sweeps and campaigns</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
        <li><a href="#playing-against-a-checkpoint">Playing against a checkpoint</a></li>
        <li><a href="#playing-on-the-official-ladder">Playing on the official ladder</a></li>
      </ul>
    </li>
    <li><a href="#repository-layout">Repository Layout</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

The project is being carried out by Tristan Bandat as his bachelor
thesis at the Johannes Kepler University in the Bachelor AI program.

The article [Mastering the game of Stratego with model-free multiagent reinforcement
learning][https://www.science.org/doi/10.1126/science.add4679]([1)](#1) serves as the basis.

> Reconnaissance Blind Chess (RBC) is a chess variant designed for new research in artificial intelligence (AI).
> RBC includes imperfect information, long-term strategy, explicit observations, and almost no common knowledge.
> These features appear in real-world scenarios, and challenge even state of the art algorithms.[[2]](#2)

### Built With

* [PyCharm](https://www.jetbrains.com/pycharm/)
* [Pytorch](https://pytorch.org/)
* [Vim](https://www.vim.org/)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* **Python 3.11–3.13** (developed and run on 3.12).
* **[uv](https://docs.astral.sh/uv/)** for dependency management — the lockfile
  (`uv.lock`) is what makes an environment reproducible.
* **An NVIDIA GPU** for training. Everything else (self-play, evaluation, playing
  against a checkpoint) runs on CPU as well, just slower — pass `--device cpu`
  or `--set train.device=cpu`.
* **Stockfish** is *not* a separate install: a Linux build ships in
  `tools/stockfish/` and is found automatically. Set `STOCKFISH_EXECUTABLE` to
  use your own binary instead. It backs the `trout` and StrangeFish2 opponents
  and the move-quality grading.

### Installation

```sh
git clone --recurse-submodules https://github.com/TristanBandat/deepnash-rbc.git
cd deepnash-rbc
uv sync --all-extras     # locked environment, incl. the local play UI (Flask)
uv run deepnash-smoke    # end-to-end check on CPU
```

`--recurse-submodules` pulls in `third_party/strangefish2`, the vendored
StrangeFish2 bot used as an evaluation opponent. If you already cloned without
it:

```sh
git submodule update --init third_party/strangefish2
```

`uv sync` alone is enough if you do not need the local play UI. The test suite
runs with `uv run pytest`.

<!-- USAGE EXAMPLES -->
## Usage

Every entry point is a `uv run` command, so there is no virtualenv to activate.
All of them accept `--help`.

### Training

The asynchronous trainer is the one used for every result in the thesis: a pool
of CPU self-play actors feeds a single GPU learner.

```sh
uv run deepnash-train-async                       # default configuration
uv run deepnash-train-async --set rnad.eta=0.5 --seed 0
uv run deepnash-train-async --resume              # continue the latest checkpoint
```

Any configuration field can be overridden by dotted path with `--set`, and the
common ones have their own flags:

```sh
uv run deepnash-train-async \
    --set rnad.eta=0.5 --set train.total_iters=160000 \
    --history 16 --channels 128 --blocks 6 \
    --async-actors 16 --device cuda
```

Sequence models replace the stacked observation window with a learned memory
over the whole game and ignore `--history`:

```sh
uv run deepnash-train-async --arch gru --set network.mixer_dim=256
uv run deepnash-train-async --arch transformer --set network.mixer_layers=2
```

`--arch` takes `resnet` (default), `gru`, `lstm`, `transformer`, or `xlstm`.

Checkpoints, metrics, and a pinned `config.json` land in
`checkpoints/v<version>/`, where `<version>` is the project version from
`pyproject.toml`. That version is the label tying a checkpoint to the
architecture and hyperparameters that produced it, so **bump it in
`pyproject.toml` before starting a run with a different architecture** —
existing checkpoints are shape-locked to theirs.

`deepnash-train` is the synchronous single-process trainer. It is simpler to
debug but much slower; prefer `deepnash-train-async`.

### Sweeps and campaigns

`scripts/train_campaign.py` runs a list of configurations back-to-back,
auto-versioning each so every run gets its own checkpoint folder. Runs are
described in a JSON manifest (see `sweeps/` for the ones used in the thesis).

```sh
uv run python scripts/train_campaign.py --write-template my-sweep.json
uv run python scripts/train_campaign.py --sweep sweeps/sweep.json --dry-run
uv run python scripts/train_campaign.py --sweep sweeps/sweep.json
```

A manifest entry either starts fresh, `"resume"`s an existing version to a
longer horizon, or `"from"`-forks a specific checkpoint into a new run. An
optional `"tournament"` block rates each finished run against the baselines
right away.

### Evaluation

**Internal tournament.** `tools/tournament.py` plays checkpoints and baseline
bots against each other, appends every game to a JSONL log, and fits a
Bradley–Terry/Elo leaderboard over it. Re-running the same command only plays
the games still missing, so the ladder grows incrementally.

```sh
uv run python tools/tournament.py --model v0.14.0 --dry-run
uv run python tools/tournament.py --model v0.14.0 --pair-games 8 --workers 8
uv run python tools/tournament.py 'checkpoints/v0.14.0/*.pt' --vs-top 20
uv run python tools/tournament.py --leaderboard-only
```

Baselines are `random`, `attacker`, `trout` (Stockfish-backed) and `mht`
(multi-hypothesis tracking).

**Stockfish move quality.** Play a checkpoint against an opponent, then replay
the finished games against the arbiter's ground-truth board and grade every move
— separating chess skill from information skill.

```sh
uv run deepnash-stockfish-eval \
    -c checkpoints/v0.14.0/deepnash_async_v0.14.0_70000.pt \
    --num-games 100 --mq-opponent trout mht \
    --json results/eval.json
```

The observation history is read from the checkpoint itself. Add `--ladder` for a
win-rate curve against TroutBot at a sweep of Stockfish skill levels.

### Playing against a checkpoint

```sh
uv run deepnash-play --checkpoint-dir checkpoints
# then open http://127.0.0.1:8000
```

A local board UI to play a game yourself against a trained net (requires the
`ui` extra, included in `uv sync --all-extras`).

### Playing on the official ladder

Deployment settings live in `rbc_connect.json` (git-tracked, so its history
records which checkpoint was played when); credentials live in a git-ignored
`.env` in the repository root.

```sh
cat > .env <<'EOF'
RBC_USERNAME=<your bot account>
RBC_PASSWORD=<your password>
EOF

uv run python scripts/rc_connect.py
```

```jsonc
{
  "checkpoint": "checkpoints/v0.14.0/deepnash_async_v0.14.0_70000.pt",
  "device": null,              // null = cuda if available
  "greedy": false,             // sample the policy; it is a mixed strategy
  "sample_threshold": 0.05,    // drop actions below this, then renormalize
  "server_url": "https://rbc.jhuapl.edu",
  "ranked": true,
  "keep_version": false,
  "max_concurrent_games": 8
}
```

Real environment variables take precedence over `.env`, and an empty password is
prompted for at launch. `DEEPNASH_CKPT`, `DEEPNASH_DEVICE`, `DEEPNASH_GREEDY`
and `DEEPNASH_SAMPLE_THRESHOLD` override the config file for one-off runs.

<!-- REPOSITORY LAYOUT -->
## Repository Layout

| Path | Contents |
| --- | --- |
| `src/deepnash_rbc/` | The package: encodings, network, R-NaD learner, actors, evaluation |
| `scripts/` | Campaign driver, ladder connector, the reconchess bot entry point |
| `tools/` | Tournament / Elo tooling, analysis helpers, bundled Stockfish |
| `sweeps/` | Campaign manifests, one per experiment round |
| `checkpoints/` | Per-run `v<version>/` folders: weights, metrics, pinned `config.json` |
| `results/` | Tournament game log and evaluation reports |
| `third_party/` | Vendored opponent bots (StrangeFish2, as a submodule) |
| `tests/` | pytest suite |

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/TristanBandat/deepnash-rbc/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create.<br>
Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Tristan Bandat - [@TBandat](https://twitter.com/TBandat)

Project Link: [https://github.com/TristanBandat/deepnash-rbc](https://github.com/TristanBandat/deepnash-rbc)

<!-- ACKNOWLEDGEMENTS 
## Acknowledgements

* []()
* []()
* []()

-->

<!-- References -->

## References

<a id="1">[1]</a>
Julien Perolat et al. ,Mastering the game of Stratego with model-free multiagent reinforcement learning.<br>
Science378,990-996(2022).DOI: [10.1126/science.add4679](https://www.science.org/doi/10.1126/science.add4679)

<a id="2">[2]</a>
Reconnaissance Blind Chess (RBC) by The Johns Hopkins University Applied Physics Laboratory LLC.<br>
More information: [https://rbc.jhuapl.edu/](https://rbc.jhuapl.edu/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TristanBandat/deepnash-rbc.svg?style=for-the-badge
[contributors-url]: https://github.com/TristanBandat/deepnash-rbc/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TristanBandat/deepnash-rbc.svg?style=for-the-badge
[forks-url]: https://github.com/TristanBandat/deepnash-rbc/network/members
[stars-shield]: https://img.shields.io/github/stars/TristanBandat/deepnash-rbc.svg?style=for-the-badge
[stars-url]: https://github.com/TristanBandat/deepnash-rbc/stargazers
[issues-shield]: https://img.shields.io/github/issues/TristanBandat/deepnash-rbc.svg?style=for-the-badge
[issues-url]: https://github.com/TristanBandat/deepnash-rbc/issues
[pulls-shield]: https://img.shields.io/github/issues-pr/TristanBandat/deepnash-rbc.svg?style=for-the-badge
[pulls-url]: https://github.com/TristanBandat/deepnash-rbc/pulls
[license-shield]: https://img.shields.io/github/license/TristanBandat/deepnash-rbc.svg?style=for-the-badge
[license-url]: https://github.com/TristanBandat/deepnash-rbc/blob/master/LICENSE.txt
[closed_pulls-shield]: https://img.shields.io/github/issues-pr-closed/TristanBandat/deepnash-rbc?style=for-the-badge
[closed_pulls-url]: https://github.com/TristanBandat/deepnash-rbc/pulls?q=is%3Apr+is%3Aclosed
[closed_issues-shield]: https://img.shields.io/github/issues-closed/TristanBandat/deepnash-rbc?style=for-the-badge
[closed_issues-url]: https://github.com/TristanBandat/deepnash-rbc/issues?q=is%3Aissue+is%3Aclosed
