"""Append-only, hash-chained audit log (OBJ-04: evidence-to-decision trail).

How it works
------------
Every entry stores ``prev_hash`` (the previous entry's hash) and its own ``hash``:

    hash = sha256( prev_hash + canonical_json({ts, actor, action, decision_id, payload, evidence_ids}) )

* SQLite triggers (see db.DDL) reject UPDATE and DELETE on the table.
* ``verify()`` recomputes the whole chain. If anyone edits or removes a row in the middle
  (for example after dropping the triggers), the chain breaks at that row.

Honest limits (say this in the report): the chain makes tampering *detectable*, not impossible.
Someone with write access to the file could rewrite the tail and recompute every hash. A real
deployment would anchor the head hash somewhere external (a ledger service or a signed daily
export) - copy ``head`` from ``/audit/verify`` into your slides as the prototype's version of that.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Iterable

from . import clock
from .db import dumps

GENESIS = "0" * 64


def _hash(prev_hash: str, body: dict) -> str:
    return hashlib.sha256((prev_hash + dumps(body)).encode("utf-8")).hexdigest()


def append(
    conn: sqlite3.Connection,
    *,
    actor: str,
    action: str,
    decision_id: str | None,
    payload: dict,
    evidence_ids: Iterable[str] = (),
) -> dict:
    """Add one entry. Must be called inside ``db.tx()`` so the read-last-hash / insert pair is atomic."""
    assert conn.in_transaction, "audit.append must run inside db.tx()"
    last = conn.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
    prev_hash = last["hash"] if last else GENESIS
    body = {
        "ts": clock.iso(clock.now()),
        "actor": actor,
        "action": action,
        "decision_id": decision_id,
        "payload": payload,
        "evidence_ids": sorted(set(evidence_ids)),
    }
    entry_hash = _hash(prev_hash, body)
    cur = conn.execute(
        "INSERT INTO audit_log(ts, actor, action, decision_id, payload_json, evidence_ids_json, prev_hash, hash) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (
            body["ts"], actor, action, decision_id, dumps(payload), dumps(body["evidence_ids"]),
            prev_hash, entry_hash,
        ),
    )
    return {"seq": cur.lastrowid, "hash": entry_hash}


def verify(conn: sqlite3.Connection) -> dict:
    """Recompute the chain from the first entry. Returns ok / first_bad_seq / head hash."""
    prev = GENESIS
    n = 0
    for row in conn.execute("SELECT * FROM audit_log ORDER BY seq"):
        body = {
            "ts": row["ts"],
            "actor": row["actor"],
            "action": row["action"],
            "decision_id": row["decision_id"],
            "payload": json.loads(row["payload_json"]),
            "evidence_ids": json.loads(row["evidence_ids_json"]),
        }
        if row["prev_hash"] != prev:
            return {"ok": False, "first_bad_seq": row["seq"], "reason": "prev_hash does not match", "entries": n}
        if _hash(prev, body) != row["hash"]:
            return {"ok": False, "first_bad_seq": row["seq"], "reason": "entry content does not match its hash", "entries": n}
        prev = row["hash"]
        n += 1
    return {"ok": True, "entries": n, "head": prev}


def entries(conn: sqlite3.Connection, decision_id: str | None = None, limit: int = 500) -> list[dict]:
    if decision_id:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE decision_id=? ORDER BY seq LIMIT ?", (decision_id, limit)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
    return [
        {
            "seq": r["seq"], "ts": r["ts"], "actor": r["actor"], "action": r["action"],
            "decision_id": r["decision_id"], "payload": json.loads(r["payload_json"]),
            "evidence_ids": json.loads(r["evidence_ids_json"]), "hash": r["hash"], "prev_hash": r["prev_hash"],
        }
        for r in rows
    ]
