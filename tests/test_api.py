"""HTTP contract: roles, validation errors, SSE, evidence lookup, admin changes, audit, CSV ingestion."""
import json
import sqlite3

import pytest

from conftest import as_role
from lighthouse import data, db

OWNER, LEGAL, ADMIN = as_role("decision_owner", "owner"), as_role("legal_reviewer", "counsel"), as_role("admin", "root")
HEALTH = {"input": {"title": "Health promo, 50 stores", "category": "Health & Wellness", "stores": 50, "weeks": 6, "discount_pct": 2.0,
                    "budget_usd": 500000, "claims": ["supports immune health"]}, "mode": "rules"}


def open_health(client):
    return client.post("/cases?wait=true", json=HEALTH, headers=OWNER).json()["decision_id"]


# ---- roles ---------------------------------------------------------------------------------------------


def test_unknown_role_is_400(client):
    assert client.get("/cases", headers={"X-Role": "ceo"}).status_code == 200  # reads need no principal
    assert client.post("/cases", json=HEALTH, headers={"X-Role": "ceo"}).status_code == 400


@pytest.mark.parametrize("role", ["legal_reviewer", "finance_reviewer", "sales_reviewer", "operations_reviewer"])
def test_only_owner_or_admin_can_open_a_case(client, role):
    assert client.post("/cases", json=HEALTH, headers=as_role(role)).status_code == 403


def test_only_legal_reviewer_can_dispose(client):
    did = open_health(client)
    for h in (OWNER, ADMIN, as_role("finance_reviewer")):
        assert client.post(f"/cases/{did}/legal-disposition", json={"disposition": "CLEARED", "comment": "Reviewed, acceptable."}, headers=h).status_code == 403
    assert client.post(f"/cases/{did}/legal-disposition", json={"disposition": "CLEARED", "comment": "Reviewed, acceptable."}, headers=LEGAL).status_code == 200


def test_only_owner_can_decide(client):
    did = open_health(client)
    client.post(f"/cases/{did}/legal-disposition", json={"disposition": "CLEARED", "comment": "Reviewed, acceptable."}, headers=LEGAL)
    r = client.post(f"/cases/{did}/decision", json={"action": "REJECT", "reason": "not now, thanks"}, headers=LEGAL)
    assert r.status_code == 403
    assert client.get(f"/cases/{did}").json()["status"] == "READY_FOR_DECISION"


def test_admin_endpoints_are_admin_only(client):
    assert client.put("/admin/config", json={"financial_threshold_usd": 1}, headers=OWNER).status_code == 403
    assert client.put("/admin/agents/LEGAL", json={"enabled": False}, headers=OWNER).status_code == 403
    assert client.post("/admin/refresh", headers=OWNER).status_code == 403
    assert client.put("/admin/rules/R-X", json={"name": "x"}, headers=OWNER).status_code == 403


# ---- domain rules -------------------------------------------------------------------------------------------


def test_legal_disposition_cannot_be_given_twice_or_when_not_needed(client):
    did = open_health(client)
    ok = {"disposition": "CLEARED", "comment": "Reviewed, acceptable."}
    assert client.post(f"/cases/{did}/legal-disposition", json=ok, headers=LEGAL).status_code == 200
    r = client.post(f"/cases/{did}/legal-disposition", json=ok, headers=LEGAL)
    assert r.status_code == 409 and r.json()["code"] == "GATE_NOT_PENDING"


def test_a_decided_case_cannot_be_changed(client):
    did = open_health(client)
    client.post(f"/cases/{did}/legal-disposition", json={"disposition": "CLEARED", "comment": "Reviewed, acceptable."}, headers=LEGAL)
    assert client.post(f"/cases/{did}/decision", json={"action": "REJECT", "reason": "changed my mind"}, headers=OWNER).status_code == 200
    r = client.post(f"/cases/{did}/decision", json={"action": "APPROVE", "reason": "now I want it approved"}, headers=OWNER)
    assert r.status_code == 409 and r.json()["code"] == "CASE_NOT_OPEN"


def test_failed_attempts_write_nothing_to_the_audit_log(client):
    did = open_health(client)
    n0 = len(client.get("/audit", params={"decision_id": did}).json())
    for _ in range(3):
        assert client.post(f"/cases/{did}/decision", json={"action": "APPROVE", "reason": "locked, but trying"}, headers=OWNER).status_code == 409
    assert len(client.get("/audit", params={"decision_id": did}).json()) == n0


def test_short_reason_is_rejected(client):
    did = open_health(client)
    client.post(f"/cases/{did}/legal-disposition", json={"disposition": "CLEARED", "comment": "Reviewed, acceptable."}, headers=LEGAL)
    r = client.post(f"/cases/{did}/decision", json={"action": "APPROVE", "reason": "ok"}, headers=OWNER)
    assert r.status_code == 422 and r.json()["code"] == "REASON_REQUIRED"


def test_invalid_input_is_422(client):
    bad = {"input": {**HEALTH["input"], "stores": 500}, "mode": "rules"}
    assert client.post("/cases", json=bad, headers=OWNER).status_code == 422
    assert client.post("/cases", json={"mode": "rules"}, headers=OWNER).status_code == 422
    assert client.post("/cases", json={"input": HEALTH["input"], "mode": "replay"}, headers=OWNER).status_code == 422
    assert client.post("/cases", json={"scenario": "nope"}, headers=OWNER).status_code == 404


def test_invalid_revision_is_422_and_leaves_the_case_open(client):
    did = open_health(client)
    client.post(f"/cases/{did}/legal-disposition", json={"disposition": "CLEARED", "comment": "Reviewed, acceptable."}, headers=LEGAL)
    r = client.post(f"/cases/{did}/decision", json={"action": "REVISE", "reason": "try again with other numbers", "revised_input": {"stores": 999}}, headers=OWNER)
    assert r.status_code == 422 and r.json()["code"] == "INVALID_REVISION"
    assert client.get(f"/cases/{did}").json()["status"] == "READY_FOR_DECISION"


def test_missing_case_is_404(client):
    assert client.get("/cases/DC-9999").status_code == 404
    assert client.get("/cases/DC-9999/packet").status_code == 404
    assert client.post("/cases/DC-9999/decision", json={"action": "REJECT", "reason": "long enough reason"}, headers=OWNER).status_code == 404


# ---- async path & SSE -----------------------------------------------------------------------------------------


def test_async_open_then_sse_stream_ends_with_done(client):
    r = client.post("/cases", json=HEALTH, headers=OWNER)
    assert r.status_code == 202
    did = r.json()["decision_id"]
    seen = []
    with client.stream("GET", f"/cases/{did}/events") as s:
        for line in s.iter_lines():
            if line.startswith("event:"):
                seen.append(line.split(":", 1)[1].strip())
    assert seen[0] == "case_opened" and seen[-1] == "done" and "packet_ready" in seen
    assert client.get(f"/cases/{did}/packet").status_code == 200


def test_sse_for_an_unknown_case_reports_an_error(client):
    with client.stream("GET", "/cases/DC-9999/events") as s:
        text = "".join(s.iter_text())
    assert "event: error" in text


def test_sse_for_a_finished_case_after_restart_says_done(client):
    did = open_health(client)
    from lighthouse import orchestrator

    orchestrator.bus._events.clear()  # as if the server had restarted
    with client.stream("GET", f"/cases/{did}/events") as s:
        text = "".join(s.iter_text())
    assert "event: done" in text


# ---- evidence, audit -------------------------------------------------------------------------------------------


def test_every_evidence_id_in_a_packet_resolves(client):
    did = open_health(client)
    packet = client.get(f"/cases/{did}/packet").json()
    assert packet["evidence_ids"]
    for eid in packet["evidence_ids"]:
        ev = client.get(f"/evidence/{eid}").json()
        assert ev["decision_id"] == did and ev["as_of"] and ev["summary"]
    assert client.get("/evidence/EV-999999").status_code == 404


def test_audit_log_is_chained_and_tamper_is_detected(client, tmp_path):
    open_health(client)
    assert client.get("/audit/verify").json()["ok"] is True
    from lighthouse.settings import settings

    # tampering through the application's own connection is blocked by triggers ...
    with pytest.raises(sqlite3.DatabaseError):
        with db.tx() as conn:
            conn.execute("UPDATE audit_log SET actor='mallory' WHERE seq=1")
    # ... and a tamper that bypasses them (someone edits the file with the triggers dropped) is caught by the hash chain
    raw = sqlite3.connect(settings.db_path, isolation_level=None)
    raw.execute("DROP TRIGGER IF EXISTS audit_no_update")
    raw.execute("UPDATE audit_log SET actor='mallory' WHERE seq=2")
    raw.close()
    v = client.get("/audit/verify").json()
    assert v["ok"] is False and v["first_bad_seq"] == 2


# ---- admin changes take effect ------------------------------------------------------------------------------------------


def test_threshold_change_via_api_changes_the_next_packet(client):
    body = {"input": {**HEALTH["input"], "category": "Grocery", "claims": [], "budget_usd": 5_000_000}, "mode": "rules"}
    p1 = client.post("/cases?wait=true", json=body, headers=OWNER).json()["packet"]
    assert "FINANCIAL_THRESHOLD" in {e["code"] for e in p1["escalations"]}
    r = client.put("/admin/config", json={"financial_threshold_usd": 50_000_000}, headers=ADMIN)
    assert r.status_code == 200 and client.get("/admin/config").json()["financial_threshold_usd"] == 50_000_000
    p2 = client.post("/cases?wait=true", json=body, headers=OWNER).json()["packet"]
    assert "FINANCIAL_THRESHOLD" not in {e["code"] for e in p2["escalations"]}
    changes = [e for e in client.get("/audit").json() if e["action"] == "CONFIG_CHANGED"]
    assert changes and changes[0]["actor"] == "root" and changes[0]["payload"]["after"] == {"financial_threshold_usd": 50_000_000}


def test_unknown_config_key_is_rejected(client):
    assert client.put("/admin/config", json={"nonsense": 1}, headers=ADMIN).status_code == 422


def test_legal_rule_edit_is_audited_and_changes_the_gate(client):
    grocery = {"input": {**HEALTH["input"], "category": "Grocery", "claims": ["certified organic produce"]}, "mode": "rules"}
    assert client.post("/cases?wait=true", json=grocery, headers=OWNER).json()["packet"]["legal_gate"] == "NOT_REQUIRED"
    r = client.put("/admin/rules/R-ORGANIC", json={"name": "Organic label claims", "category_match": "", "keyword_match": "organic", "severity": "HIGH"}, headers=ADMIN)
    assert r.status_code == 200
    assert client.post("/cases?wait=true", json=grocery, headers=OWNER).json()["packet"]["legal_gate"] == "PENDING"
    assert any(e["action"] == "RULE_CHANGED" for e in client.get("/audit").json())


def test_stats_and_scenarios_endpoints(client):
    assert client.get("/admin/stats").json()["packet_ms"] is None
    open_health(client)
    s = client.get("/admin/stats").json()
    assert s["cases"] == 1 and s["agent_outputs"] == 4 and s["packets"] == 1 and s["packet_ms"]["p95"] < 60_000
    assert len(client.get("/scenarios").json()) >= 10
    assert client.get("/health").json()["ok"] is True
    assert client.get("/").status_code == 200


# ---- CSV ingestion ---------------------------------------------------------------------------------------------------------------


def test_csv_ingestion_replaces_generated_sales_and_agents_use_it(fresh_db, tmp_path):
    csv_path = tmp_path / "walmart_like.csv"
    lines = ["Store,Date,Weekly_Sales,Holiday_Flag"]
    for store in range(1, 4):
        for week in range(20):
            promo = 1 if week % 5 == 0 else 0
            sales = 1000.0 * (1.3 if promo else 1.0)
            lines.append(f"{store},{2024 + week // 52}-{(week % 12) + 1:02d}-{(week % 28) + 1:02d},{sales},{promo}")
    csv_path.write_text("\n".join(lines))
    out = data.ingest_sales_csv(str(csv_path), store_col="Store", date_col="Date", sales_col="Weekly_Sales", promo_col="Holiday_Flag",
                                category="Grocery", replace_generated=True, origin="test")
    assert out["rows"] == 60
    from lighthouse import tools
    from lighthouse.agents import rules
    from lighthouse.schema import AgentName, CaseInput

    rules_out = rules.run(AgentName.SALES, "DC-0001", CaseInput(title="CSV test case", category="Grocery", stores=3, weeks=4, discount_pct=2, budget_usd=100_000))
    assert rules_out.metrics["uplift_pct"] == pytest.approx(30.0, abs=0.5)
    with db.read() as conn:
        assert conn.execute("SELECT origin FROM data_sources WHERE name='sales_weekly'").fetchone()["origin"] == "test"


# ---- bulk generator & CLI -------------------------------------------------------------------------------------------------------------


def test_bulk_generator_goes_through_the_real_pipeline(fresh_db):
    from lighthouse import audit, bulk

    s = bulk.generate(12, seed=3)
    assert s["cases"] == 12 and s["agent_outputs_total"] == 48
    assert sum(s["recommendations"].values()) == 12
    assert s["packet_ms"]["p95"] < 60_000
    with db.read() as conn:
        assert audit.verify(conn)["ok"]
        assert conn.execute("SELECT COUNT(*) n FROM audit_log WHERE action='PACKET_CREATED'").fetchone()["n"] == 12


def test_bulk_generator_is_reproducible(fresh_db):
    import random

    from lighthouse import bulk

    a = [bulk.random_case(random.Random(5), i).model_dump() for i in range(5)]
    b = [bulk.random_case(random.Random(5), i).model_dump() for i in range(5)]
    assert a == b


def test_cli_verify_and_init(tmp_path, monkeypatch, capsys):
    from lighthouse import cli
    from lighthouse.settings import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "cli.db"))
    assert cli.main(["init"]) == 0
    assert cli.main(["verify-audit"]) == 0
    assert '"ok": true' in capsys.readouterr().out
