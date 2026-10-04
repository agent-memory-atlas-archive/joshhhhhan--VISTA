"""Load the pinned public runtime, never a benchmark model or API client."""

from __future__ import annotations

import importlib
import logging
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GAMEWORLD_REVISION = "f2883c34e377c8b73fdb503a4a8e87be12310df7"
GAMES_REVISION = "a23b4370c3f99df6fa4ee01ea0299fdab9668b86"
COMPUTER_USE_PROTOCOL = "gameworld-computer-use"
PRIMARY_STEPS = 100
_IMPORT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Upstream:
    root: Path
    runtime: Any
    catalog: Any
    launcher: Any

    def task(self, game_id: str, task_id: str) -> tuple[Any, Any]:
        if not all(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", x) for x in (game_id, task_id)
        ):
            raise ValueError("Use exact catalog game and task identifiers")
        game = self.catalog.load_game(game_id)
        task = self.catalog.load_task(game_id, task_id)
        if not game.game_roles:
            raise ValueError("A GameWorld game must define a role")
        if any(not role.semantic_controls for role in game.game_roles):
            raise ValueError("Every role must define its controls")
        return game, task


def load_upstream(root: Path) -> Upstream:
    root = root.expanduser().resolve(strict=True)
    with _IMPORT_LOCK:
        for name in ("agents", "catalog", "env", "runtime", "tools"):
            existing = sys.modules.get(name)
            if existing is not None:
                file = getattr(existing, "__file__", None)
                if not file or not Path(file).resolve().is_relative_to(root):
                    raise ImportError(
                        f"Module {name!r} is already bound outside {root}"
                    )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        for name in ("env", "model", "game", "task"):
            if not hasattr(logging.Logger, name):
                setattr(logging.Logger, name, logging.Logger.debug)
        runtime = importlib.import_module("runtime")
        catalog = importlib.import_module("catalog")
        return Upstream(
            root,
            runtime,
            catalog,
            importlib.import_module("env.game_launcher"),
        )
