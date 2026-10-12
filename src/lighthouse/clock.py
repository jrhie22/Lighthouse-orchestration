"""A tiny injectable clock.

Freshness checks ("is this evidence stale?") depend on *now*. Tests and scenarios freeze the
clock so results are reproducible; production code just calls ``now()``.
"""
from __future__ import annotations

from datetime import datetime, timezone

_frozen: datetime | None = None


def now() -> datetime:
    return _frozen or datetime.now(timezone.utc)


def freeze(dt: datetime) -> None:
    global _frozen
    _frozen = dt


def unfreeze() -> None:
    global _frozen
    _frozen = None


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse(s: str) -> datetime:
    return datetime.fromisoformat(s)
