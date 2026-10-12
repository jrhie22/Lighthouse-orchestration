"""Every scripted scenario is run through the real HTTP API and checked against its own `expect` dict.

This is the OBJ-07 evidence: >= 10 scripted scenarios, each with a stated expected outcome that is verified
by code, covering agreement, disagreement, missing output, hallucination control, stale data, low confidence,
legal block, and the revision loop.
"""
import time

import pytest

from conftest import as_role
from lighthouse import scenarios

IDS = list(scenarios.SCENARIOS)


def test_there_are_at_least_ten_scenarios():
    assert len(IDS) >= 10


def _flags(packet, agent):
    v = next(a for a in packet["agents"] if a["output"]["agent"] == agent)
    return set(v["flags"]) | set(v["output"]["risk_flags"])


@pytest.mark.parametrize("sid", IDS)
def test_scenario_packet_matches_expectation(client, sid):
    sc = scenarios.SCENARIOS[sid]
    r = client.post("/cases?wait=true", json={"scenario": sid}, headers=as_role("decision_owner", "tester"))
    assert r.status_code == 201, r.text
    body = r.json()
    p = body["packet"]
    exp = sc.expect

    assert p["recommendation"] == exp["rec"], p["summary"]
    if "gate" in exp:
        assert p["legal_gate"] == exp["gate"]
    assert {c["type"] for c in p["conflicts"]} == exp["conflicts"]
    assert {e["code"] for e in p["escalations"]} == exp["escalations"], p["escalations"]
    assert p["requires_reason"] is exp["requires_reason"]
    assert p["incomplete"] is exp["incomplete"]
    for agent, status in exp.get("status", {}).items():
        a = next(a for a in p["agents"] if a["output"]["agent"] == agent)
        assert a["output"]["status"] == status
    for agent, flags in exp.get("flags", {}).items():
        assert flags <= _flags(p, agent), (agent, flags, _flags(p, agent))

    # the audit trail of the case must contain: opened, 4 agent outputs, packet
    entries = client.get("/audit", params={"decision_id": body["decision_id"]}).json()
    actions = [e["action"] for e in entries]
    assert actions.count("CASE_OPENED") == 1
    assert actions.count("AGENT_OUTPUT_RECORDED") == 4
    assert actions.count("PACKET_CREATED") == 1
    assert client.get("/audit/verify").json()["ok"] is True


@pytest.mark.parametrize("sid", [s for s in IDS if scenarios.SCENARIOS[s].steps])
def test_scenario_human_steps(client, sid):
    sc = scenarios.SCENARIOS[sid]
    case = client.post("/cases?wait=true", json={"scenario": sid}, headers=as_role("decision_owner", "owner")).json()
    did = case["decision_id"]

    for i, step in enumerate(sc.steps):
        url = f"/cases/{did}/legal-disposition" if step.kind == "legal" else f"/cases/{did}/decision"
        r = client.post(url, json=step.body, headers=as_role(step.role, step.role))
        assert r.status_code == step.expect_http, f"step {i}: {r.status_code} {r.text}"
        if step.expect_status:
            assert client.get(f"/cases/{did}").json()["status"] == step.expect_status, f"step {i}"

    assert client.get("/audit/verify").json()["ok"] is True


def test_revision_loop_creates_child_case_with_new_packet(client):
    sid = "s11_revision_loop"
    first = client.post("/cases?wait=true", json={"scenario": sid}, headers=as_role("decision_owner")).json()
    did = first["decision_id"]
    assert first["packet"]["recommendation"] == "HOLD"

    step = scenarios.SCENARIOS[sid].steps[0]
    r = client.post(f"/cases/{did}/decision", json=step.body, headers=as_role("decision_owner"))
    assert r.status_code == 200
    child = r.json()["revision_case_id"]
    assert child == f"{did}.r1"

    # the child case runs in the background; wait for its packet
    deadline = time.time() + 10
    while time.time() < deadline:
        st = client.get(f"/cases/{child}").json()["status"]
        if st not in ("RUNNING", "OPENED"):
            break
        time.sleep(0.05)
    packet = client.get(f"/cases/{child}/packet").json()
    assert packet["recommendation"] != "HOLD", packet["summary"]  # the revised, smaller promotion is acceptable
    assert client.get(f"/cases/{did}").json()["status"] == "REVISION_REQUESTED"
    assert child in client.get(f"/cases/{did}").json()["revisions"]
    assert client.get(f"/cases/{child}").json()["input"]["discount_pct"] == 1.0
