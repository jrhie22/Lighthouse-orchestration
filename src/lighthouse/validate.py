"""Step 'validate evidence' of the activity diagram.

An agent output is checked *after* the agent produced it and *before* Lighthouse reasons about it:

  * evidence IDs must exist and must have been issued to THIS agent for THIS case
    (otherwise: status=INVALID, flag UNSUPPORTED_CLAIM)  -> hallucination control
  * evidence must be fresh: age <= data source's max_age_hours; stale evidence does not make an
    output invalid, but its confidence is multiplied by ``stale_confidence_multiplier``
  * confidence below ``low_confidence_threshold`` is flagged
  * the recommendation must be in the vocabulary allowed for that agent

The original output is never edited in place: an invalid one is *copied* with status INVALID, and
governance judgements (effective confidence, flags) live in ``AgentView`` next to it.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from . import clock
from .schema import AgentName, AgentOutput, AgentView, OutputStatus, Rec, RiskFlag

ANALYTIC_RECS = {Rec.PROCEED, Rec.PROCEED_WITH_CONDITIONS, Rec.HOLD}


def check(conn: sqlite3.Connection, out: AgentOutput, cfg: dict, now: datetime) -> tuple[AgentOutput, AgentView]:
    if out.status != OutputStatus.OK:
        return out, AgentView(output=out, effective_confidence=0.0, flags=[out.status.value], usable=False)

    problems: set[RiskFlag] = set()
    allowed = {Rec.REVIEW_REQUIRED} if out.agent == AgentName.LEGAL else ANALYTIC_RECS
    if out.recommendation not in allowed:
        problems.add(RiskFlag.INVALID_RECOMMENDATION)
    if not out.evidence_ids:
        problems.add(RiskFlag.MISSING_EVIDENCE)

    ages: list[float] = []
    stale_sources: list[str] = []
    for eid in out.evidence_ids:
        row = conn.execute(
            "SELECT e.decision_id, e.agent, e.source, e.as_of, d.max_age_hours "
            "FROM evidence e LEFT JOIN data_sources d ON d.name = e.source WHERE e.evidence_id=?",
            (eid,),
        ).fetchone()
        if row is None or row["decision_id"] != out.decision_id or row["agent"] != out.agent.value:
            problems.add(RiskFlag.UNSUPPORTED_CLAIM)
            continue
        age_h = (now - clock.parse(row["as_of"])).total_seconds() / 3600
        ages.append(age_h)
        if age_h > (row["max_age_hours"] if row["max_age_hours"] is not None else float("inf")):
            stale_sources.append(row["source"])

    if problems:
        names = ", ".join(sorted(p.value for p in problems))
        bad = out.model_copy(update={
            "status": OutputStatus.INVALID,
            "risk_flags": list(dict.fromkeys([*out.risk_flags, *sorted(problems, key=lambda p: p.value)])),
            "rationale": f"[INVALID: {names}] {out.rationale}",
        })
        return bad, AgentView(output=bad, effective_confidence=0.0, flags=sorted(p.value for p in problems), usable=False)

    flags: list[str] = []
    effective = out.confidence
    if stale_sources:
        effective *= cfg["stale_confidence_multiplier"]
        flags.append("STALE_EVIDENCE")
    if effective < cfg["low_confidence_threshold"]:
        flags.append("LOW_CONFIDENCE")
    return out, AgentView(
        output=out, effective_confidence=round(effective, 3), flags=flags,
        oldest_evidence_age_hours=round(max(ages), 1) if ages else None, usable=True,
    )
