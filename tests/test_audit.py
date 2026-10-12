import sqlite3

import pytest

from lighthouse import audit, db
from lighthouse.settings import settings


def _add(n):
    with db.tx() as c:
        for i in range(n):
            audit.append(c, actor="t", action="X", decision_id="DC-1", payload={"i": i}, evidence_ids=["EV-2", "EV-1"])


def test_chain_verifies(fresh_db):
    _add(5)
    with db.read() as c:
        r = audit.verify(c)
    assert r["ok"] and r["entries"] == 5 and len(r["head"]) == 64


def test_update_and_delete_are_blocked(fresh_db):
    _add(2)
    with db.read() as c:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            c.execute("UPDATE audit_log SET actor='evil'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            c.execute("DELETE FROM audit_log")


def test_tampering_is_detected_even_if_triggers_are_dropped(fresh_db):
    _add(4)
    raw = sqlite3.connect(settings.db_path)
    raw.execute("DROP TRIGGER audit_no_update")
    raw.execute("UPDATE audit_log SET payload_json='{\"i\":99}' WHERE seq=2")
    raw.commit()
    raw.close()
    with db.read() as c:
        r = audit.verify(c)
    assert not r["ok"] and r["first_bad_seq"] == 2


def test_deleting_a_middle_row_breaks_the_chain(fresh_db):
    _add(4)
    raw = sqlite3.connect(settings.db_path)
    raw.execute("DROP TRIGGER audit_no_delete")
    raw.execute("DELETE FROM audit_log WHERE seq=2")
    raw.commit()
    raw.close()
    with db.read() as c:
        r = audit.verify(c)
    assert not r["ok"] and r["first_bad_seq"] == 3


def test_append_requires_a_transaction(fresh_db):
    with db.read() as c:
        with pytest.raises(AssertionError):
            audit.append(c, actor="t", action="X", decision_id=None, payload={})
