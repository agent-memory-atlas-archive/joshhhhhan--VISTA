"""AI GameStore decisions and evidence; VISTA tools live in the core."""

from __future__ import annotations

from typing import Any

from vista.core.contracts import (
    Effect,
    EvidenceKind,
    Observation,
    ObservationBundle,
    PublicTask,
    SubmissionSpec,
    ToolBinding,
    ToolReply,
    Visual,
)

from .environment import (
    MAX_DECISIONS,
    SEGMENT_SECONDS,
    SEGMENTS_PER_DECISION,
    BrowserStep,
    Frame,
    GameStoreRuntime,
    PublicGame,
    validate_segments,
)
from .profiles import action_specs, objective

VIEW = "screen"
SCREEN_PROMPT = "Game screen (paused):\n"


class GameStoreSession:
    """One game run: decisions, frames, and restarts; score never leaves the host."""

    def __init__(
        self,
        game: PublicGame,
        runtime: GameStoreRuntime,
        *,
        max_decisions: int = MAX_DECISIONS,
    ) -> None:
        if type(max_decisions) is not int or not 1 <= max_decisions <= MAX_DECISIONS:
            raise ValueError(f"max_decisions must be in 1..{MAX_DECISIONS}")
        self.game = game
        self.runtime = runtime
        self.max_decisions = max_decisions
        self.ended = False
        self.decisions = 0
        self.episodes = 1
        self.finished = False
        self.won = False
        self.closed = False
        # Host-only evaluator states, one per decision plus the start.
        self.private: list[dict[str, Any]] = []
        step = runtime.start()
        self.width, self.height = step.frames[-1].width, step.frames[-1].height
        bundle = self._bundle(step.frames, 0)
        self.latest = bundle
        self.current = self._current_of(bundle)
        self._record(None, step)
        self.task = PublicTask(
            f"aigamestore/game{game.number}",
            objective(),
            {
                "game_number": game.number,
                "game_id": game.game_id,
                "description": game.description,
                "controls": game.controls,
                "segments_per_decision": SEGMENTS_PER_DECISION,
                "segment_seconds": SEGMENT_SECONDS,
                "game_seconds": max_decisions,
            },
            bundle,
        )
        # A final response is a redirection, not a game second.
        self.submission = SubmissionSpec(Effect.SUBMISSION)
        handlers = {"play": self._play, "restart": self._restart}
        self.tools = tuple(
            ToolBinding(spec, handlers[spec.name]) for spec in action_specs()
        )

    # -- evidence --------------------------------------------------------------

    def _bundle(self, frames: tuple[Frame, ...], decision: int) -> ObservationBundle:
        observations = []
        for index, frame in enumerate(frames):
            if (frame.width, frame.height) != (self.width, self.height):
                raise RuntimeError("The game canvas changed size")
            moment = f"decision-{decision:03d}-frame-{index}"
            final = index == len(frames) - 1
            observations.append(
                Observation(
                    moment,
                    (Visual(VIEW, moment, frame.png, frame.width, frame.height),),
                    EvidenceKind.CURRENT if final else EvidenceKind.HISTORICAL,
                )
            )
        display = None
        return ObservationBundle(tuple(observations), display=display)

    @staticmethod
    def _current_of(bundle: ObservationBundle) -> ObservationBundle:
        return ObservationBundle(
            tuple(
                item
                for item in bundle.observations
                if item.kind == EvidenceKind.CURRENT
            )
        )

    def _record(self, action: dict[str, Any] | None, step: BrowserStep) -> None:
        self.private.append(
            {
                "decision": self.decisions,
                "episode": self.episodes,
                "action": action,
                "state": dict(step.state),
                "ended": step.ended,
                "won": step.won,
                "max_score": self.runtime.max_score,
            }
        )

    def _advance(self, action: dict[str, Any], step: BrowserStep) -> ObservationBundle:
        self.decisions += 1
        bundle = self._bundle(step.frames, self.decisions)
        self.latest = bundle
        self.current = self._current_of(bundle)
        self.ended = step.ended
        self._record(action, step)
        if step.won:
            self.won = True
        self.finished = self.decisions >= self.max_decisions
        return bundle

    def _reply(self, event: str, guidance: str, bundle: ObservationBundle) -> ToolReply:
        if self.decisions >= self.max_decisions:
            event += " The run's game seconds are spent."
        if self.finished:
            guidance = ""
        text = event + (" " + guidance if guidance else "")
        return ToolReply(
            text, observation=bundle, finished=self.finished, event_text=event
        )

    # -- actions ---------------------------------------------------------------

    def _play(self, arguments: Any) -> ToolReply:
        if self.finished:
            return ToolReply("The run has ended.", success=False)
        try:
            segments = validate_segments(
                arguments.get("segments") if isinstance(arguments, dict) else None
            )
        except ValueError as exc:
            return ToolReply(f"Invalid segments: {exc}", success=False)
        step = self.runtime.run_second(segments)
        bundle = self._advance(
            {"name": "play", "segments": [list(segment) for segment in segments]},
            step,
        )
        event = "The game ran for one game second and is paused."
        guidance = ""
        if step.ended:
            event += " The game has ended."
            guidance = "Call restart to begin a new attempt."
        return self._reply(event, guidance, bundle)

    def _restart(self, arguments: Any) -> ToolReply:
        if self.finished:
            return ToolReply("The run has ended.", success=False)
        step = self.runtime.restart()
        self.episodes += 1
        bundle = self._advance({"name": "restart"}, step)
        return self._reply(
            "The game restarted and is paused at its first frame.", "", bundle
        )

    def submit(self, answer: str) -> ToolReply:
        """A final response is a redirection: the game advances through tools."""
        return ToolReply(
            "No game action was executed. Continue with the play or restart tool.",
            success=False,
        )

    # -- host results -------------------------------------------------------------

    def result(self) -> dict[str, Any]:
        last = self.private[-1]
        return {
            "decisions": self.decisions,
            "episodes": self.episodes,
            "finished": self.finished,
            "won": self.won,
            "raw_score": self.runtime.max_score,
            "final_phase": last["state"].get("gamePhase"),
            "canvas": {"width": self.width, "height": self.height},
            "browser_errors": list(getattr(self.runtime, "errors", ())),
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.runtime.close()
