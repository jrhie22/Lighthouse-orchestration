"""The orchestrator: parallelism, timeouts, crashes, disabled agents, stale data, LLM mode end-to-end."""
import time
from datetime import timedelta

from conftest import as_role
from lighthouse import clock, orchestrator
from lighthouse.agents import rules as rules_mod
from lighthouse.schema import CaseInput

from test_agents import FakeModel

BODY = {"input": {"title": "Grocery weekend promo", "category": "Grocery", "stores": 20, "weeks": 4, "discount_pct": 1.5, "budget_usd": 400000}, "mode": "rules"}
OWNER = as_role("decision_owner", "owner")


def _agent(packet, name):
    return next(a for a in packet["agents"] if a["output"]["agent"] == name)


def _open(client, body=BODY):
    r = client.post("/cases?wait=true", json=body, headers=OWNER)
    assert r.status_code == 201, r.text
    return r.json()


def test_agents_run_in_parallel(client, monkeypatch):
    real = rules_mod.run

    def slow(agent, decision_id, case, record=True):
        time.sleep(0.5)
        return real(agent, decision_id, case, record)

    monkeypatch.setattr(rules_mod, "run", slow)
    t0 = time.perf_counter()
    case = _open(client)
    elapsed = time.perf_counter() - t0
    assert case["packet"]["agents"] and elapsed < 1.4, f"four 0.5 s agents took {elapsed:.2f}s - they ran one after another"


def test_slow_agent_times_out_but_the_packet_is_still_built(client, monkeypatch):
    from lighthouse.settings import settings

    monkeypatch.setattr(settings, "agent_timeout_s", 0.3)
    real = rules_mod.run

    def maybe_slow(agent, decision_id, case, record=True):
        if agent.value == "OPERATIONS":
            time.sleep(1.2)
        return real(agent, decision_id, case, record)

    monkeypatch.setattr(rules_mod, "run", maybe_slow)
    t0 = time.perf_counter()
    p = _open(client)["packet"]
    assert time.perf_counter() - t0 < 1.1  # we did not wait for the slow agent
    assert _agent(p, "OPERATIONS")["output"]["status"] == "TIMEOUT"
    assert p["incomplete"] is True and p["recommendation"] == "HOLD"
    assert "INCOMPLETE" in {e["code"] for e in p["escalations"]}
    time.sleep(1.2)  # let the abandoned worker finish: its late answer must not corrupt anything
    assert client.get("/audit/verify").json()["ok"] is True


def test_a_crashing_agent_becomes_invalid_not_a_failed_case(client, monkeypatch):
    real = rules_mod.run

    def boom(agent, decision_id, case, record=True):
        if agent.value == "SALES":
            raise RuntimeError("database exploded")
        return real(agent, decision_id, case, record)

    monkeypatch.setattr(rules_mod, "run", boom)
    body = _open(client)
    out = _agent(body["packet"], "SALES")["output"]
    assert out["status"] == "INVALID" and "database exploded" in out["rationale"]
    assert body["status"] != "FAILED" and body["packet"]["incomplete"] is True


def test_disabled_legal_agent_fails_closed(client):
    r = client.put("/admin/agents/LEGAL", json={"enabled": False}, headers=as_role("admin"))
    assert r.status_code == 200
    body = _open(client)
    p = body["packet"]
    assert _agent(p, "LEGAL")["output"]["status"] == "DISABLED"
    assert p["legal_gate"] == "PENDING" and "LEGAL_GATE" in {e["code"] for e in p["escalations"]}
    # and approval really is locked
    r = client.post(f"/cases/{body['decision_id']}/decision", json={"action": "APPROVE", "reason": "ship it anyway please"}, headers=OWNER)
    assert r.status_code == 409 and r.json()["code"] == "LEGAL_GATE_LOCKED"


def test_disabled_analytic_agent_makes_the_packet_incomplete(client):
    client.put("/admin/agents/FINANCE", json={"enabled": False}, headers=as_role("admin"))
    p = _open(client)["packet"]
    assert p["incomplete"] is True and p["recommendation"] == "HOLD"


def test_stale_inventory_downgrades_operations(client):
    # move the clock 48 h past the data refresh: inventory (36 h limit) is stale, weekly sales (192 h) are not
    clock.freeze(clock.now() + timedelta(hours=48))
    p = _open(client)["packet"]
    ops, sales = _agent(p, "OPERATIONS"), _agent(p, "SALES")
    assert "STALE_EVIDENCE" in ops["flags"] and "STALE_EVIDENCE" not in sales["flags"]
    assert ops["effective_confidence"] < ops["output"]["confidence"]  # x0.6
    assert "STALE_EVIDENCE" in {e["code"] for e in p["escalations"]}
    assert any("refresh stale evidence" in c for c in p["conditions"])


def test_admin_refresh_makes_stale_data_fresh_again(client):
    clock.freeze(clock.now() + timedelta(hours=48))
    assert client.post("/admin/refresh", headers=as_role("admin")).status_code == 200
    p = _open(client)["packet"]
    assert "STALE_EVIDENCE" not in {e["code"] for e in p["escalations"]}


def test_events_are_published_in_order(client):
    body = _open(client)
    types = [e["type"] for e in orchestrator.bus.since(body["decision_id"], 0)]
    assert types[0] == "case_opened" and types[-1] == "done"
    assert types.count("agent_started") == 4 and types.count("agent_done") == 4
    assert types.index("packet_ready") > max(i for i, t in enumerate(types) if t == "agent_done")


def test_llm_mode_end_to_end_with_a_fake_model(client, monkeypatch):
    import lighthouse.api as api

    class Up:
        model = "fake"

        def available(self):
            return True

    monkeypatch.setattr(api, "OllamaClient", Up)

    class Factory:
        def chat(self, messages, tools=None, format=None):
            # route by system prompt: each agent run builds its own FakeModel with the right first tool
            system = messages[0]["content"] if messages else ""
            key = "check_dc_capacity" if "Operations" in system else "get_demand_uplift" if "Sales" in system else "evaluate_financials"
            return FakeModel(tool=key).chat(messages, tools, format)

    orchestrator.set_llm_factory(lambda: Factory())
    body = _open(client, {**BODY, "mode": "agent"})
    p = body["packet"]
    modes = {a["output"]["agent"]: a["output"]["mode"] for a in p["agents"]}
    assert modes["FINANCE"] == modes["SALES"] == modes["OPERATIONS"] == "agent"
    assert all(a["output"]["status"] in ("OK", "NOT_TRIGGERED") for a in p["agents"])
    fin = _agent(p, "FINANCE")["output"]
    assert "baseline_recommendation" in fin["metrics"]  # rules shadow baseline stored next to the model's answer


def test_agent_mode_returns_503_when_ollama_is_unreachable(client, monkeypatch):
    import lighthouse.api as api

    class Down:
        model = "fake"

        def available(self):
            return False

    monkeypatch.setattr(api, "OllamaClient", Down)
    r = client.post("/cases", json={**BODY, "mode": "agent"}, headers=OWNER)
    assert r.status_code == 503 and "mode=rules" in r.json()["detail"]
