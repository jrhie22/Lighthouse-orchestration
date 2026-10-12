"""Reconcile, Govern & Recommend (process 6.0 in the Level-0 DFD).

Pure functions: validated agent views in, one Packet out. No database, no clock, no LLM - so it is
trivially unit-testable and behaves identically every time (OBJ-02, OBJ-03, OBJ-06).

Principles from Deliverable 1 that are enforced here:
  * no blind averaging - dissent is turned into explicit ``Conflict`` objects;
  * ``confidence_floor`` is the *lowest* usable confidence, never a mean;
  * a missing or invalid required agent makes the packet incomplete (never silently skipped);
  * Legal REVIEW_REQUIRED locks approval regardless of what the other agents say;
  * if the Legal agent itself is unavailable, the gate fails CLOSED (human review still required).
"""
from __future__ import annotations

from datetime import datetime

from .schema import (
    ALL_AGENTS, REQUIRED_AGENTS, AgentName, AgentView, CaseInput, Conflict, ConflictType, Escalation,
    GateStatus, OutputStatus, Packet, Rec,
)

ORDER = [AgentName.FINANCE, AgentName.SALES, AgentName.OPERATIONS, AgentName.LEGAL]
LEGAL_UNAVAILABLE = (OutputStatus.TIMEOUT, OutputStatus.INVALID, OutputStatus.DISABLED)


def _names(agents) -> str:
    return ", ".join(a.value for a in agents)


def build_packet(decision_id: str, case: CaseInput, views: list[AgentView], cfg: dict, built_at: datetime) -> Packet:
    by = {v.output.agent: v for v in views}
    views = sorted(views, key=lambda v: ORDER.index(v.output.agent))

    # --- stances of the three analytical agents -----------------------------------------------
    stance = {
        a: by[a].output.recommendation
        for a in REQUIRED_AGENTS
        if a in by and by[a].usable
    }
    missing = [a for a in REQUIRED_AGENTS if a not in stance]
    incomplete = bool(missing)
    holders = [a for a, r in stance.items() if r == Rec.HOLD]
    unconditional = [a for a, r in stance.items() if r == Rec.PROCEED]
    conditional = [a for a, r in stance.items() if r == Rec.PROCEED_WITH_CONDITIONS]
    proceeders = unconditional + conditional

    # --- conflicts: surfaced, never averaged (OBJ-03) -------------------------------------------
    conflicts: list[Conflict] = []
    if holders and proceeders:
        conflicts.append(Conflict(
            type=ConflictType.HARD_DISAGREEMENT, severity="HARD", agents=holders + proceeders,
            description=f"{_names(holders)} recommend HOLD while {_names(proceeders)} recommend proceeding.",
        ))
    if unconditional and conditional:
        conds = "; ".join(c for a in conditional for c in by[a].output.conditions) or "conditions apply"
        conflicts.append(Conflict(
            type=ConflictType.CONDITION_TENSION, severity="SOFT", agents=unconditional + conditional,
            description=f"{_names(unconditional)} recommend PROCEED, but {_names(conditional)} require conditions: {conds}",
        ))

    # --- legal gate -----------------------------------------------------------------------------
    legal = by.get(AgentName.LEGAL)
    legal_required = bool(legal and legal.usable and legal.output.recommendation == Rec.REVIEW_REQUIRED)
    legal_failed = legal is None or legal.output.status in LEGAL_UNAVAILABLE
    gate = GateStatus.PENDING if (legal_required or legal_failed) else GateStatus.NOT_REQUIRED

    # --- flags raised by validation --------------------------------------------------------------
    stale = [v for v in views if "STALE_EVIDENCE" in v.flags]
    low_conf = [v for v in views if "LOW_CONFIDENCE" in v.flags and v.usable]

    # --- escalations: *why* this needs a written human reason (success condition in Deliverable 1) -
    esc: list[Escalation] = []
    if gate != GateStatus.NOT_REQUIRED:
        esc.append(Escalation(
            code="LEGAL_GATE",
            detail="Legal agent unavailable - human legal review required" if legal_failed and not legal_required
            else "Legal flagged REVIEW_REQUIRED - approval is locked until a human legal disposition",
        ))
    fin = by.get(AgentName.FINANCE)
    if fin and fin.usable and float(fin.output.metrics.get("total_cost_usd", 0)) >= cfg["financial_threshold_usd"]:
        esc.append(Escalation(code="FINANCIAL_THRESHOLD", detail=f"Total cost ${float(fin.output.metrics['total_cost_usd']):,.0f} is at or above ${cfg['financial_threshold_usd']:,.0f}"))
    ops = by.get(AgentName.OPERATIONS)
    if ops and ops.usable and float(ops.output.metrics.get("util_bulk", 0)) >= cfg["ops_util_escalate"]:
        esc.append(Escalation(code="OPERATIONAL_THRESHOLD", detail=f"Bulk delivery would reach {float(ops.output.metrics['util_bulk']):.0%} DC utilization"))
    if any(c.severity == "HARD" for c in conflicts):
        esc.append(Escalation(code="HARD_CONFLICT", detail="Agents disagree on whether to proceed"))
    if incomplete:
        esc.append(Escalation(code="INCOMPLETE", detail="No usable output from " + ", ".join(f"{a.value} ({by[a].output.status.value if a in by else 'MISSING'})" for a in missing)))
    if stale:
        esc.append(Escalation(code="STALE_EVIDENCE", detail="Stale evidence from " + _names(v.output.agent for v in stale)))
    if low_conf:
        esc.append(Escalation(code="LOW_CONFIDENCE", detail="Low confidence from " + ", ".join(f"{v.output.agent.value} ({v.effective_confidence:.2f})" for v in low_conf)))

    # --- recommendation (one of PROCEED / PROCEED_WITH_CONDITIONS / HOLD) ------------------------
    if incomplete or holders:
        rec = Rec.HOLD
    elif conditional or conflicts or gate != GateStatus.NOT_REQUIRED or stale or low_conf:
        rec = Rec.PROCEED_WITH_CONDITIONS  # flagged evidence downgrades PROCEED (Deliverable 1 risk table)
    else:
        rec = Rec.PROCEED

    # --- conditions --------------------------------------------------------------------------------
    conditions: list[str] = []
    for v in views:
        if v.usable:
            conditions += [f"{v.output.agent.value}: {c}" for c in v.output.conditions]
    if gate != GateStatus.NOT_REQUIRED and not any(c.startswith("LEGAL:") for c in conditions):
        conditions.append("LEGAL: human legal / compliance review before approval")
    for v in stale:
        age = f" ({v.oldest_evidence_age_hours:.0f}h old)" if v.oldest_evidence_age_hours is not None else ""
        conditions.append(f"{v.output.agent.value}: refresh stale evidence{age} and re-run")
    for v in low_conf:
        conditions.append(f"{v.output.agent.value}: confidence {v.effective_confidence:.2f} is low - confirm the assessment before approving")

    # --- unresolved risks ------------------------------------------------------------------------
    risks: list[str] = []
    for v in views:
        risks += [f"{v.output.agent.value}: {f.value}" for f in v.output.risk_flags]
        risks += [f"{v.output.agent.value}: {f}" for f in v.flags]
    risks += [f"Conflict: {c.description}" for c in conflicts if c.severity == "HARD"]
    risks = list(dict.fromkeys(risks))

    usable_conf = [v.effective_confidence for v in views if v.usable and v.output.agent != AgentName.LEGAL]
    floor = min(usable_conf) if usable_conf else None

    summary_parts = [
        f"Recommendation: {rec.value.replace('_', ' ')}.",
        f"{len(proceeders)} of {len(stance)} analytical agents support proceeding ({len(conditional)} with conditions); {len(holders)} recommend holding.",
    ]
    if incomplete:
        summary_parts.append(f"Incomplete: no usable output from {_names(missing)}.")
    if conflicts:
        summary_parts.append(f"{len(conflicts)} conflict(s) surfaced and not averaged.")
    if gate != GateStatus.NOT_REQUIRED:
        summary_parts.append("Legal review is required before approval.")
    if floor is not None:
        summary_parts.append(f"Lowest usable confidence: {floor:.2f}.")

    evidence_ids = sorted({e for v in views for e in v.output.evidence_ids})
    return Packet(
        decision_id=decision_id, built_at=built_at, recommendation=rec, summary=" ".join(summary_parts),
        confidence_floor=floor, agents=views, conflicts=conflicts, escalations=esc, conditions=conditions,
        unresolved_risks=risks, legal_gate=gate, requires_reason=bool(esc), incomplete=incomplete,
        evidence_ids=evidence_ids,
    )


assert set(ORDER) == set(ALL_AGENTS)
