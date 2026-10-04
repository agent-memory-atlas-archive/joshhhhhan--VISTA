"""The public AI GameStore protocol: one decision is one game second of input."""

from .environment import (
    MAX_DECISIONS,
    PROTOCOL,
    SEGMENT_SECONDS,
    SEGMENTS_PER_DECISION,
)

__all__ = ["MAX_DECISIONS", "PROTOCOL", "SEGMENT_SECONDS", "SEGMENTS_PER_DECISION"]
