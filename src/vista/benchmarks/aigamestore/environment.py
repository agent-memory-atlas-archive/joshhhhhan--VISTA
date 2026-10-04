"""Host-only browser environment for the ten public AI GameStore games.

The benchmark's decision protocol: one agent decision controls one game second
as five chronological 0.2-second input segments, each a set of simultaneous tap
or hold inputs. The environment captures one canvas frame per segment, pauses
the game (Esc) before returning, and keeps score, phase, and level
evaluator-private.

Descriptions, controls, and human median scores are bundled in public_games.json.
Game pages come from a local checkout or https://aigamestore.org.
"""

from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import http.server
import io
import json
import socket
import threading
import time
from base64 import b64decode
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from vista.core.json import canonical

PROTOCOL = "aigamestore-segmented-input"
SEGMENTS_PER_DECISION = 5
SEGMENT_SECONDS = 0.2
MAX_DECISIONS = 120
PUBLIC_GAMES = tuple(range(1, 11))
PAUSE_SETTLE_SECONDS = 0.15
START_SETTLE_SECONDS = 1.0
LIVE_GAME_URL = "https://aigamestore.org/api/games/games/game{n}"

# Escape pauses the game and R/Enter restart it; those keys belong to the host
# protocol and are never part of the model's vocabulary.
KEY_VALUES: dict[str, str] = {
    "UP": "ArrowUp",
    "DOWN": "ArrowDown",
    "LEFT": "ArrowLeft",
    "RIGHT": "ArrowRight",
    "SPACE": " ",
    "Z": "z",
    "SHIFT": "Shift",
    "W": "w",
    "A": "a",
    "S": "s",
    "D": "d",
}
TAP_ACTIONS = frozenset(KEY_VALUES)
HOLD_ACTIONS = frozenset(f"HOLD_{action}" for action in KEY_VALUES)
MODEL_ACTIONS = frozenset({"NOOP", *TAP_ACTIONS, *HOLD_ACTIONS})

PHASE_START = "START"
PHASE_PLAYING = "PLAYING"
PHASE_PAUSED = "PAUSED"
WIN_PHASES = frozenset({"GAME_OVER_WIN", "GAME_WON", "WIN"})
LOSE_PHASES = frozenset({"GAME_OVER", "GAME_OVER_LOSE", "ENDED"})
ENDED_PHASES = WIN_PHASES | LOSE_PHASES

_GAME_STATE_JS = """
() => {
    const candidates = [];
    const canvas = document.querySelector('canvas');
    if (canvas && canvas.ownerDocument && canvas.ownerDocument.defaultView) {
        candidates.push(canvas.ownerDocument.defaultView);
    }
    candidates.push(window);
    for (const iframe of document.querySelectorAll('iframe')) {
        try {
            if (iframe.contentWindow) candidates.push(iframe.contentWindow);
        } catch (e) {}
    }
    let gameWin = window;
    for (const win of candidates) {
        try {
            if (typeof win.getGameState === 'function' || win.gameState ||
                (win.gameInstance && win.gameInstance.gameState)) {
                gameWin = win;
                break;
            }
        } catch (e) {}
    }
    let state = null;
    try {
        if (typeof gameWin.getGameState === 'function') {
            state = gameWin.getGameState();
        } else if (gameWin.gameInstance && gameWin.gameInstance.gameState) {
            state = gameWin.gameInstance.gameState;
        } else if (gameWin.gameState) {
            state = gameWin.gameState;
        }
    } catch (e) {
        state = null;
    }
    const out = {score: null, highScore: null, gamePhase: null, level: null};
    if (state && typeof state === 'object') {
        if (typeof state.score !== 'undefined') out.score = state.score;
        if (typeof state.highScore !== 'undefined') out.highScore = state.highScore;
        if (typeof state.gamePhase !== 'undefined') out.gamePhase = state.gamePhase;
        if (typeof state.currentLevel !== 'undefined') out.level = state.currentLevel;
    }
    return out;
}
"""

_SEED_INIT_JS = """
(() => {
    // mulberry32: well distributed from any seed, unlike a raw xorshift32,
    // whose first draws stay near zero for small seeds and would hand the
    // games a degenerate world at startup.
    let s = (%d >>> 0) || 0x9e3779b9;
    Math.random = function () {
        s = (s + 0x6D2B79F5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
})();
"""

_CANVAS_PNG_JS = """
() => {
    const canvas = document.querySelector('#defaultCanvas0') ||
        document.querySelector('canvas');
    if (!canvas) return null;
    try {
        return canvas.toDataURL('image/png');
    } catch (e) {
        return null;
    }
}
"""


@dataclass(frozen=True)
class PublicGame:
    """The public game description and controls."""

    number: int
    game_id: str
    description: str
    controls: str
    original_name: str | None = None

    def __post_init__(self) -> None:
        if self.number not in PUBLIC_GAMES:
            raise ValueError("public AI GameStore game must be in 1..10")
        for text in (self.game_id, self.description, self.controls):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("A public game needs an id, description and controls")

    @property
    def revision(self) -> str:
        """Fingerprint of the public entry; the live site has no version tag."""
        digest = hashlib.sha256(canonical(self.manifest()).encode("utf-8"))
        return f"public-manifest-{digest.hexdigest()[:16]}"

    def manifest(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "game_id": self.game_id,
            "description": self.description,
            "controls": self.controls,
            "original_name": self.original_name,
        }


@dataclass(frozen=True)
class Frame:
    png: bytes
    width: int
    height: int

    @classmethod
    def from_png(cls, png: bytes) -> Frame:
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG":
                raise ValueError("Canvas captures must be PNG")
            width, height = image.size
        return cls(png, width, height)


@dataclass(frozen=True)
class BrowserStep:
    """Frames are public evidence; ``state`` is evaluator-private."""

    frames: tuple[Frame, ...]
    state: Mapping[str, Any]

    @property
    def phase(self) -> str | None:
        phase = self.state.get("gamePhase")
        return phase if isinstance(phase, str) else None

    @property
    def ended(self) -> bool:
        return self.phase in ENDED_PHASES

    @property
    def won(self) -> bool:
        return self.phase in WIN_PHASES


class GameStoreRuntime(Protocol):
    """One browser-backed game; the adapter owns decisions and budgets."""

    max_score: float

    def start(self) -> BrowserStep: ...

    def run_second(self, segments: tuple[tuple[str, ...], ...]) -> BrowserStep: ...

    def restart(self) -> BrowserStep: ...

    def close(self) -> None: ...


def validate_segments(segments: Any) -> tuple[tuple[str, ...], ...]:
    """Validate five non-empty action segments and normalize simultaneous inputs.

    ``NOOP`` beside other inputs is dropped, and a key that is both tapped and
    held in one segment is held. Repeated inputs are collapsed. Invalid
    payload structure and unknown inputs are rejected.
    """
    if (
        not isinstance(segments, (list, tuple))
        or len(segments) != SEGMENTS_PER_DECISION
    ):
        raise ValueError("exactly five action segments are required")
    normalized: list[tuple[str, ...]] = []
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or not segment:
            raise ValueError("action segments must be non-empty arrays")
        if any(not isinstance(action, str) for action in segment):
            raise ValueError("actions must be strings")
        unknown = [action for action in segment if action not in MODEL_ACTIONS]
        if unknown:
            raise ValueError(f"unknown game action: {unknown[0]}")
        held = {
            action.removeprefix("HOLD_")
            for action in segment
            if action.startswith("HOLD_")
        }
        ordered: list[str] = []
        for action in segment:
            if action == "NOOP" or action in held or action in ordered:
                continue
            ordered.append(action)
        normalized.append(tuple(ordered) if ordered else ("NOOP",))
    return tuple(normalized)


PUBLIC_GAMES_FILE = Path(__file__).with_name("public_games.json")


@functools.cache
def _public_entries() -> dict[int, dict[str, Any]]:
    """The ten games' own text and the human median a score is relative to.

    Read once from the site and kept here: the median is what a raw score is
    normalised against, so a run must not depend on what the site serves that
    day. The games themselves come from the pinned harness checkout.
    """
    payload = json.loads(PUBLIC_GAMES_FILE.read_text(encoding="utf-8"))
    entries = {int(game["number"]): game for game in payload["games"]}
    if sorted(entries) != list(range(1, 11)):
        raise RuntimeError("The public game table must hold games 1 to 10")
    return entries


def load_public_game(game_number: int) -> PublicGame:
    try:
        raw = _public_entries()[int(game_number)]
    except KeyError:
        raise RuntimeError(f"AI GameStore has no public game {game_number}") from None
    return PublicGame(
        game_number,
        raw["game_id"],
        raw["description"],
        raw["controls"],
        raw.get("original_name"),
    )


def load_human_median(game_id: str) -> float:
    for raw in _public_entries().values():
        if raw["game_id"] == game_id:
            median = float(raw["human_median"])
            if median > 0:
                return median
            break
    raise RuntimeError(f"AI GameStore has no human median for {game_id}")


def normalized_score(raw_score: float, human_median: float) -> float:
    if human_median <= 0:
        raise ValueError("human median must be positive")
    return min(10_000.0, max(1.0, 100.0 * raw_score / human_median))


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class GameStoreBrowser:
    """Playwright Chromium driving one game; canvas pixels are never rescaled.

    Playwright's sync API is bound to the thread that started it, and the
    transports call tool handlers from their own threads (the Claude MCP host
    does; Codex happens not to). Every browser operation therefore runs on one
    private worker thread, and the public methods only marshal calls to it.
    """

    def __init__(
        self,
        game: PublicGame,
        *,
        games_root: Path | None = None,
        headless: bool = True,
        seed: int | None = None,
    ) -> None:
        self.game = game
        self.games_root = games_root.resolve() if games_root else None
        self.headless = headless
        self.seed = seed
        self.max_score = 0.0
        self._worker: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="aigamestore-browser"
            )
        )
        # Uncaught page errors, kept host-side: a crashed game is an
        # infrastructure fact about the run, not a model score.
        self.errors: list[str] = []
        self.browser_version: str | None = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._canvas = None
        self._capture_composited = False
        self._httpd: http.server.ThreadingHTTPServer | None = None
        try:
            self._run(self._launch)
        except BaseException:
            self.close()
            raise

    def _run(self, operation, *args):
        if self._worker is None:
            raise RuntimeError("The game browser has been closed")
        return self._worker.submit(operation, *args).result()

    # -- lifecycle ---------------------------------------------------------

    def _game_url(self) -> str:
        if self.games_root is None:
            return LIVE_GAME_URL.format(n=self.game.number)
        port = _free_port()
        serve_root = self.games_root.parent
        handler = partial(_QuietHandler, directory=str(serve_root))
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return (
            f"http://127.0.0.1:{port}/"
            f"{self.games_root.name}/game{self.game.number}/index.html"
        )

    def _launch(self) -> None:
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self.browser_version = self._browser.version
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            device_scale_factor=1,
            service_workers="block",
        )
        self._page = self._context.new_page()
        self._page.on("pageerror", lambda error: self.errors.append(str(error)))
        if self.seed is not None:
            # Runs before any game script, so seeded generators built at module
            # load time (game2/4/5 draw straight from Math.random) are fixed.
            self._page.add_init_script(_SEED_INIT_JS % self.seed)
        self._page.goto(self._game_url(), wait_until="domcontentloaded")
        self._canvas = self._find_canvas()
        # toDataURL on a WebGL canvas without preserveDrawingBuffer returns
        # black frames; capture those through the compositor instead.
        try:
            self._capture_composited = bool(
                self._canvas.evaluate(
                    "el => Boolean(el.getContext('webgl2') || el.getContext('webgl'))"
                )
            )
        except Exception:
            self._capture_composited = False
        try:
            self._canvas.evaluate(
                "el => { if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '0'); }"
            )
        except Exception:
            pass
        self._canvas.click()

    def _find_canvas(self):
        page = self._page
        try:
            canvas = page.locator("#defaultCanvas0").first
            canvas.wait_for(state="visible", timeout=15000)
            return canvas
        except Exception:
            pass
        for frame in page.frames:
            try:
                canvas = frame.locator("#defaultCanvas0").first
                canvas.wait_for(state="visible", timeout=3000)
                return canvas
            except Exception:
                continue
        canvas = page.locator("canvas").first
        canvas.wait_for(state="visible", timeout=15000)
        return canvas

    def close(self) -> None:
        if self._worker is None:
            return
        try:
            self._run(self._close)
        finally:
            self._worker.shutdown(wait=True)
            self._worker = None

    def _close(self) -> None:
        for closer in (
            lambda: self._context.close() if self._context else None,
            lambda: self._browser.close() if self._browser else None,
            lambda: self._playwright.stop() if self._playwright else None,
            lambda: self._httpd.shutdown() if self._httpd else None,
        ):
            try:
                closer()
            except Exception:
                pass
        self._context = None
        self._browser = None
        self._playwright = None
        self._httpd = None

    # -- game IO -----------------------------------------------------------

    def _game_state(self) -> dict[str, Any]:
        try:
            raw = self._page.evaluate(_GAME_STATE_JS)
        except Exception:
            raw = None
        if not isinstance(raw, dict):
            return {"score": None, "highScore": None, "gamePhase": None, "level": None}
        return raw

    def _observe_score(self, state: Mapping[str, Any]) -> None:
        for key in ("score", "highScore"):
            value = state.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.max_score = max(self.max_score, float(value))

    def _press(self, key: str) -> None:
        self._page.keyboard.press(key)

    def _pause(self) -> None:
        for _ in range(2):
            if self._game_state().get("gamePhase") != PHASE_PLAYING:
                return
            self._press("Escape")
            time.sleep(PAUSE_SETTLE_SECONDS)

    def _resume(self) -> None:
        if self._game_state().get("gamePhase") == PHASE_PAUSED:
            self._press("Escape")
            time.sleep(0.05)

    def _focus(self) -> None:
        try:
            self._canvas.click()
        except Exception:
            self._canvas = self._find_canvas()
            self._canvas.click()

    def _capture(self) -> Frame:
        if self._capture_composited:
            return Frame.from_png(self._canvas.screenshot())
        try:
            data_url = self._page.evaluate(_CANVAS_PNG_JS)
        except Exception:
            data_url = None
        if isinstance(data_url, str) and data_url.startswith("data:image/png;base64,"):
            return Frame.from_png(b64decode(data_url.split(",", 1)[1]))
        return Frame.from_png(self._canvas.screenshot())

    def _step(self, frames: list[Frame]) -> BrowserStep:
        state = self._game_state()
        self._observe_score(state)
        return BrowserStep(tuple(frames), {**state, "runtime_errors": len(self.errors)})

    # -- protocol interface --------------------------------------------------

    def start(self) -> BrowserStep:
        """Start (Enter) the game and pause it at its first frame."""
        return self._run(self._restart)

    def restart(self) -> BrowserStep:
        """R to the start screen when needed, Enter to play, then pause."""
        return self._run(self._restart)

    def run_second(self, segments: tuple[tuple[str, ...], ...]) -> BrowserStep:
        """Run one game second as five 0.2-second input segments."""
        return self._run(self._run_second, validate_segments(segments))

    def _restart(self) -> BrowserStep:
        self._focus()
        self._resume()
        if self._game_state().get("gamePhase") not in (None, PHASE_START):
            self._press("r")
            time.sleep(0.4)
        self._press("Enter")
        time.sleep(START_SETTLE_SECONDS)
        if self._game_state().get("gamePhase") == PHASE_START:
            self._focus()
            self._press("Enter")
            time.sleep(START_SETTLE_SECONDS)
        self._pause()
        return self._step([self._capture()])

    def _run_second(self, segments: tuple[tuple[str, ...], ...]) -> BrowserStep:
        self._focus()
        self._resume()
        keyboard = self._page.keyboard
        frames: list[Frame] = []
        boundary = time.monotonic()
        held: list[str] = []
        try:
            for index, segment in enumerate(segments):
                held = []
                if segment != ("NOOP",):
                    for action in segment:
                        if action.startswith("HOLD_"):
                            key = KEY_VALUES[action.removeprefix("HOLD_")]
                            keyboard.down(key)
                            held.append(key)
                    for action in segment:
                        if not action.startswith("HOLD_"):
                            keyboard.press(KEY_VALUES[action])
                boundary += SEGMENT_SECONDS
                delay = boundary - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if index == len(segments) - 1:
                    for key in held:
                        keyboard.up(key)
                    held = []
                    self._pause()
                frames.append(self._capture())
                for key in held:
                    keyboard.up(key)
                held = []
        finally:
            for key in held:
                keyboard.up(key)
        return self._step(frames)
