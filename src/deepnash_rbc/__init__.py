"""deepnash_rbc package.

Import-time bootstrap: export ``STOCKFISH_EXECUTABLE`` for every ``deepnash-*``
console script. pyproject's ``[project.scripts]`` can only declare entry points,
not set environment variables -- but every entry point is ``deepnash_rbc.<mod>:main``,
so importing this package runs first and is the one place a var is guaranteed to be
set for all of them. We resolve the binary the usual way (explicit env, PATH, then
the bundled ``tools/stockfish/``) and use ``setdefault`` so a value the user already
exported always wins. Anything that reads the raw env var -- notably reconchess's
stock ``TroutBot`` -- then just works without a manual export. Wrapped in a bare
``except`` because importing the package must never fail just because no engine is
installed.
"""

import os as _os

try:
    from .analysis.engine import STOCKFISH_ENV_VAR as _SF_VAR, resolve_engine_path as _resolve

    _sf_path = _resolve()
    if _sf_path:
        _os.environ.setdefault(_SF_VAR, _sf_path)
except Exception:  # never let engine resolution break `import deepnash_rbc`
    pass
