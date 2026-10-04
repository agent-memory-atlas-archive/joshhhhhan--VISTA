"""Host-only browser worker delegating each transition to the upstream coordinator."""

from __future__ import annotations

import asyncio
import io
import math
import secrets
import socket
import threading
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from vista.core.contracts import EvidenceKind, Observation, ObservationBundle, Visual

from .upstream import (
    PRIMARY_STEPS,
    Upstream,
)


def artifact_value(value: Any) -> Any:
    """Keep malformed parser numbers representable without changing execution."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_number": repr(value)}
    if isinstance(value, dict):
        return {key: artifact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [artifact_value(item) for item in value]
    return value


@dataclass(frozen=True)
class Transition:
    role: int
    before: ObservationBundle
    after: ObservationBundle
    action: dict[str, Any] | None
    validity: dict[str, Any]
    evaluation: dict[str, Any] | None
    finished: bool


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class PaperDriver:
    """Own only the outer role cursor; all single-step semantics remain upstream."""

    def __init__(
        self,
        upstream: Upstream,
        env: Any,
        agents: list[Any],
        evaluator: Any,
        *,
        before_reset: Callable[[], Awaitable[None]] | None = None,
    ):
        class Coordinator(upstream.runtime.Coordinator):
            def _init_agent_loggers(self):
                pass

            async def _get_raw_action(self, agent):
                return self.supplied_action

            def _log_model_interaction(self, agent, agent_logger):
                return {"tool_call": self.supplied_action, "error": self.parse_error}

            def _build_action_validity_record(self, *args, **kwargs):
                result = super()._build_action_validity_record(*args, **kwargs)
                self.last_validity = result
                return result

            async def _handle_eval_controls(self, agent, result):
                self.last_evaluation = asdict(result) if result is not None else None
                if result is not None and result.should_reset and before_reset:
                    await before_reset()
                await super()._handle_eval_controls(agent, result)

        self.coordinator = Coordinator(env, agents, evaluator)
        self.coordinator.last_evaluation = None
        self.role = 0

    @property
    def finished(self) -> bool:
        return self.coordinator._stop_event.is_set()

    async def step(self, action: Any, parse_error: str | None = None) -> None:
        if self.finished:
            raise RuntimeError("The GameWorld task has ended")
        coordinator = self.coordinator
        coordinator.supplied_action = action
        coordinator.parse_error = parse_error
        try:
            await coordinator._run_agent_step(coordinator.agents[self.role], None)
        except Exception:
            coordinator._stop_event.set()
            raise
        if self.finished:
            return
        # The official loop has a frozen inter-role gap and checks the budget
        # after the entire cycle, not immediately after the primary role acts.
        await asyncio.sleep(0.05)
        self.role = (self.role + 1) % len(coordinator.agents)
        if (
            self.role == 0
            and coordinator.agents[0].step_index >= coordinator.env.config.max_steps
        ):
            coordinator._stop_event.set()


class GameWorldEngine:
    def __init__(
        self,
        upstream: Upstream,
        *,
        games_root: Path,
        game_id: str,
        task_id: str,
        directory: Path,
        seed: int | None = None,
        headless: bool = True,
        port: int | None = None,
    ) -> None:
        self.upstream = upstream
        self.games_root = games_root.expanduser().resolve(strict=True)
        self.game, self.task = upstream.task(game_id, task_id)
        self.directory = directory.resolve()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.seed = seed
        self.headless = headless
        self.port = port or available_port()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="gameworld-browser"
        )
        self._thread.start()
        self._closed = False
        self._reset_evidence: tuple[Observation, ...] = ()
        self.latest_evaluation: dict[str, Any] | None = None
        self.transitions: list[Transition] = []
        try:
            self._call(self._start())
        except BaseException:
            self.close()
            raise

    def _call(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    async def _start(self) -> None:
        owner = self
        upstream = self.upstream

        class Environment(upstream.runtime.GameEnv):
            def _build_game_url(self):
                self.game_launcher = upstream.launcher.GameLauncher(
                    owner.game.game_id,
                    port=owner.port,
                    base_dir=owner.games_root / "benchmark",
                )
                return upstream.launcher.append_url_suffix(
                    self.game_launcher.start(),
                    self.config.game_url_suffix,
                )

            def _build_browser_config(self):
                config = super()._build_browser_config()
                config.screenshot_dir = owner.directory / "screenshots"
                return config

        self.config = upstream.runtime.RuntimeConfig(
            game_id=self.game.game_id,
            task_id=self.task.task_id,
            game_url_suffix=self.task.game_url_suffix,
            evaluator_config=self.task.evaluator_config,
            width=self.game.width,
            height=self.game.height,
            speed_multiplier=self.game.speed_multiplier,
            pause_during_inference=True,
            random_seed=self.seed,
            max_steps=PRIMARY_STEPS,
            agent_count=self.game.role_count,
        )
        self.env = Environment(self.config, headless=self.headless, port=self.port)
        self.agents = [
            upstream.runtime.Agent(
                agent_id=f"role-{index}",
                agent_type="computer_use",
                client=None,
                controls=role.controls.copy(),
                semantic_controls_map=None,
            )
            for index, role in enumerate(self.game.game_roles)
        ]
        self.driver = PaperDriver(
            upstream,
            self.env,
            self.agents,
            upstream.runtime.Evaluator(self.config),
            before_reset=self._capture_before_reset,
        )
        await self.env.start()
        self.browser_version = self.env.game_manager.browser.version
        await self.env.pause_game()
        self.current = await self._capture()

    async def _capture_before_reset(self) -> None:
        evidence = await self._capture()
        self._reset_evidence = tuple(
            replace(observation, kind=EvidenceKind.HISTORICAL)
            for observation in evidence.observations
        )

    async def _capture(self) -> ObservationBundle:
        path = await self.env.capture_screenshot("shared")
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        if (width, height) != (self.game.width, self.game.height):
            raise RuntimeError("GameWorld returned unexpected screenshot dimensions")
        identity = f"frame-{secrets.token_hex(8)}"
        return ObservationBundle(
            (Observation(identity, (Visual("screen", identity, data, width, height),)),)
        )

    @property
    def role(self) -> int:
        return self.driver.role

    @property
    def finished(self) -> bool:
        return self.driver.finished

    def step(self, role: int, name: str | None, arguments: Any = None) -> Transition:
        if self._closed or role != self.role:
            raise RuntimeError("An inactive role cannot act")
        return self._call(self._step(role, name, arguments))

    async def _step(self, role: int, name: str | None, arguments: Any) -> Transition:
        before = self.current
        self._reset_evidence = ()
        error = None
        try:
            action = (
                dict(arguments)
                if name == "act" and isinstance(arguments, dict)
                else None
            )
            if action is None:
                error = "No function call parsed"
        except Exception:
            action, error = None, "Failed to parse action"
        await self.driver.step(action, error)
        self.current = await self._capture()
        # Archive the pre-reset frame, but deliver only the actual current screen.
        after = (
            ObservationBundle(
                (*self._reset_evidence, *self.current.observations),
                display=(len(self._reset_evidence),),
            )
            if self._reset_evidence
            else self.current
        )
        coordinator = self.driver.coordinator
        if coordinator.last_evaluation is not None:
            self.latest_evaluation = coordinator.last_evaluation
        transition = Transition(
            role,
            before,
            after,
            action,
            coordinator.last_validity,
            coordinator.last_evaluation,
            self.finished,
        )
        self.transitions.append(transition)
        return transition

    def result(self) -> dict[str, Any]:
        return {
            "game_id": self.game.game_id,
            "task_id": self.task.task_id,
            "finished": self.finished,
            "role_steps": [agent.step_index for agent in self.agents],
            "evaluation": self.latest_evaluation,
            "metrics": dict(self.agents[0].eval_metrics),
            "browser_version": self.browser_version,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            env = getattr(self, "env", None)
            if env is not None:
                self._call(env.close_game())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("GameWorld browser worker did not stop")
            self._loop.close()
