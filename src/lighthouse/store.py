"""Reads and writes for cases, agent outputs, packets and evidence (no business rules here)."""
from __future__ import annotations

import json
import sqlite3

from . import clock, db
from .schema import AgentOutput, CaseInput, Evidence, GateStatus, Packet, Status


def insert_case(conn: sqlite3.Connection, decision_id: str, *, parent_id: str | None, opened_by: str, mode: str,
                scenario: str | None, case: CaseInput) -> None:
    conn.execute(
        "INSERT INTO cases(decision_id,parent_id,created_at,opened_by,mode,scenario,input_json,status,legal_gate) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (decision_id, parent_id, clock.iso(clock.now()), opened_by, mode, scenario, db.dumps(case.model_dump()), Status.RUNNING, GateStatus.NOT_REQUIRED.value),
    )


def set_status(conn: sqlite3.Connection, decision_id: str, status: str, legal_gate: str | None = None) -> None:
    if legal_gate is None:
        conn.execute("UPDATE cases SET status=? WHERE decision_id=?", (status, decision_id))
    else:
        conn.execute("UPDATE cases SET status=?, legal_gate=? WHERE decision_id=?", (status, legal_gate, decision_id))


def get_case(conn: sqlite3.Connection, decision_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM cases WHERE decision_id=?", (decision_id,)).fetchone()
    if r is None:
        return None
    return {
        "decision_id": r["decision_id"], "parent_id": r["parent_id"], "created_at": r["created_at"],
        "opened_by": r["opened_by"], "mode": r["mode"], "scenario": r["scenario"],
        "input": CaseInput(**json.loads(r["input_json"])), "status": r["status"], "legal_gate": r["legal_gate"],
    }


def list_cases(conn: sqlite3.Connection, limit: int = 50, status: str | None = None) -> list[dict]:
    q = "SELECT decision_id, parent_id, created_at, mode, scenario, status, legal_gate, input_json FROM cases"
    args: list = []
    if status:
        q += " WHERE status=?"
        args.append(status)
    q += " ORDER BY created_at DESC, decision_id DESC LIMIT ?"
    args.append(limit)
    return [
        {**{k: r[k] for k in ("decision_id", "parent_id", "created_at", "mode", "scenario", "status", "legal_gate")},
         "title": json.loads(r["input_json"])["title"]}
        for r in conn.execute(q, args)
    ]


def save_output(conn: sqlite3.Connection, o: AgentOutput) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO agent_outputs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            o.decision_id, o.agent.value, clock.iso(o.timestamp), o.status.value,
            o.recommendation.value if o.recommendation else None, o.confidence, o.rationale,
            db.dumps(o.conditions), db.dumps(o.evidence_ids), db.dumps([f.value for f in o.risk_flags]),
            db.dumps(o.metrics), o.mode,
        ),
    )


def load_outputs(conn: sqlite3.Connection, decision_id: str) -> list[AgentOutput]:
    return [
        AgentOutput(
            decision_id=r["decision_id"], agent=r["agent"], timestamp=r["timestamp"], status=r["status"],
            recommendation=r["recommendation"], confidence=r["confidence"], rationale=r["rationale"],
            conditions=json.loads(r["conditions_json"]), evidence_ids=json.loads(r["evidence_ids_json"]),
            risk_flags=json.loads(r["risk_flags_json"]), metrics=json.loads(r["metrics_json"]), mode=r["mode"],
        )
        for r in conn.execute("SELECT * FROM agent_outputs WHERE decision_id=? ORDER BY agent", (decision_id,))
    ]


def save_packet(conn: sqlite3.Connection, p: Packet) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO packets(decision_id, built_at, packet_ms, json) VALUES(?,?,?,?)",
        (p.decision_id, clock.iso(p.built_at), p.packet_ms, p.model_dump_json()),
    )


def load_packet(conn: sqlite3.Connection, decision_id: str) -> Packet | None:
    """The stored packet is immutable. Things that change afterwards (legal gate, case status)
    are read from the case row and merged in here."""
    r = conn.execute("SELECT json FROM packets WHERE decision_id=?", (decision_id,)).fetchone()
    if r is None:
        return None
    packet = Packet.model_validate_json(r["json"])
    case = conn.execute("SELECT status, legal_gate FROM cases WHERE decision_id=?", (decision_id,)).fetchone()
    packet.status = case["status"]
    packet.legal_gate = GateStatus(case["legal_gate"])
    return packet


def get_evidence(conn: sqlite3.Connection, evidence_id: str) -> Evidence | None:
    r = conn.execute("SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
    if r is None:
        return None
    return Evidence(
        evidence_id=r["evidence_id"], decision_id=r["decision_id"], agent=r["agent"], source=r["source"],
        as_of=r["as_of"], summary=r["summary"], payload=json.loads(r["payload_json"]),
    )
