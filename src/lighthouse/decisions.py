"""Human-in-the-loop rules: who may do what, and what blocks what.

Three actions change a case after the packet exists:
  open_case          Decision Owner opens a case (also used for revisions)
  legal_disposition  Legal Reviewer clears or blocks a REVIEW_REQUIRED case
  record_decision    Decision Owner approves / rejects / requests a revision

Hard rules enforced here (each raises DomainError -> HTTP 4xx, and nothing is written):
  * APPROVE is impossible while the legal gate is not CLEARED                (US-03, OBJ-02)
  * a case with escalations (or an override of a HOLD) needs a written reason (automation-bias control)
  * only cases in an open state can be decided; decisions are never overwritten
Everything that does happen is appended to the audit log together with the packet's evidence IDs (US-04).
"""
from __future__ import annotations

from . import audit, db, store
from .schema import (
    CaseInput, DecisionAction, DecisionRequest, GateStatus, LegalDisposition, LegalDispositionRequest, Packet, Rec, Status,
)

MIN_REASON_CHARS = 10


class DomainError(Exception):
    def __init__(self, http_status: int, code: str, message: str):
        super().__init__(message)
        self.http_status, self.code, self.message = http_status, code, message


def _new_id(conn, parent_id: str | None) -> str:
    if parent_id:
        k = conn.execute("SELECT COUNT(*) n FROM cases WHERE parent_id=?", (parent_id,)).fetchone()["n"] + 1
        return f"{parent_id}.r{k}"
    return f"DC-{db.next_number(conn, 'case'):04d}"


def open_case(case: CaseInput, mode: str, scenario: str | None, actor: str) -> str:
    with db.tx() as conn:
        decision_id = _new_id(conn, None)
        store.insert_case(conn, decision_id, parent_id=None, opened_by=actor, mode=mode, scenario=scenario, case=case)
        audit.append(conn, actor=actor, action="CASE_OPENED", decision_id=decision_id,
                     payload={"input": case.model_dump(), "mode": mode, "scenario": scenario})
    return decision_id


def _get_open_case(conn, decision_id: str) -> tuple[dict, Packet]:
    case = store.get_case(conn, decision_id)
    if case is None:
        raise DomainError(404, "NOT_FOUND", f"No such case: {decision_id}")
    packet = store.load_packet(conn, decision_id)
    if packet is None or case["status"] not in Status.OPEN:
        raise DomainError(409, "CASE_NOT_OPEN", f"Case {decision_id} is {case['status']}; it cannot be changed now")
    return case, packet


def legal_disposition(decision_id: str, req: LegalDispositionRequest, actor: str) -> dict:
    with db.tx() as conn:
        case, packet = _get_open_case(conn, decision_id)
        if case["legal_gate"] != GateStatus.PENDING.value:
            raise DomainError(409, "GATE_NOT_PENDING", f"Legal gate is {case['legal_gate']}, nothing to dispose")
        blocked = req.disposition == LegalDisposition.BLOCKED
        gate = GateStatus.BLOCKED if blocked else GateStatus.CLEARED
        status = Status.LEGAL_BLOCKED if blocked else Status.READY_FOR_DECISION
        store.set_status(conn, decision_id, status, legal_gate=gate.value)
        audit.append(
            conn, actor=actor, action="LEGAL_DISPOSITION", decision_id=decision_id,
            payload={"disposition": req.disposition.value, "comment": req.comment, "gate": gate.value},
            evidence_ids=packet.evidence_ids,
        )
    return {"decision_id": decision_id, "legal_gate": gate.value, "status": status}


def record_decision(decision_id: str, req: DecisionRequest, actor: str) -> dict:
    with db.tx() as conn:
        case, packet = _get_open_case(conn, decision_id)

        if req.action == DecisionAction.APPROVE and case["legal_gate"] in (GateStatus.PENDING.value, GateStatus.BLOCKED.value):
            raise DomainError(409, "LEGAL_GATE_LOCKED", f"Approval is locked: legal gate is {case['legal_gate']}")

        override = req.action == DecisionAction.APPROVE and packet.recommendation == Rec.HOLD
        if (packet.requires_reason or override) and len(req.reason.strip()) < MIN_REASON_CHARS:
            raise DomainError(422, "REASON_REQUIRED", f"A written reason (>= {MIN_REASON_CHARS} characters) is required for this case")

        new_case_id = None
        if req.action == DecisionAction.APPROVE:
            status = Status.APPROVED
        elif req.action == DecisionAction.REJECT:
            status = Status.REJECTED
        else:  # REVISE: loop back to re-ingestion as a child case
            merged = {**case["input"].model_dump(), **(req.revised_input or {})}
            try:
                revised = CaseInput(**merged)
            except Exception as e:
                raise DomainError(422, "INVALID_REVISION", f"Revised input is invalid: {e}")
            new_case_id = _new_id(conn, decision_id)
            # a replayed scenario cannot answer a changed question, so revisions run on real (rules) agents
            new_mode = "rules" if case["mode"] == "replay" else case["mode"]
            store.insert_case(conn, new_case_id, parent_id=decision_id, opened_by=actor, mode=new_mode, scenario=None, case=revised)
            audit.append(conn, actor=actor, action="CASE_OPENED", decision_id=new_case_id,
                         payload={"input": revised.model_dump(), "mode": new_mode, "parent": decision_id, "revision": True})
            status = Status.REVISION_REQUESTED

        store.set_status(conn, decision_id, status)
        audit.append(
            conn, actor=actor, action="DECISION_RECORDED", decision_id=decision_id,
            payload={
                "action": req.action.value, "reason": req.reason, "override_of_hold": override,
                "packet_recommendation": packet.recommendation.value, "legal_gate": case["legal_gate"],
                "escalations": [e.code for e in packet.escalations], "revision_case": new_case_id,
            },
            evidence_ids=packet.evidence_ids,
        )
    return {"decision_id": decision_id, "status": status, "revision_case_id": new_case_id}
