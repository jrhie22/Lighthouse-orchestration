"""Agents: rules-mode deciders, and the LLM tool-loop (driven by a scripted fake model - no Ollama needed).

What these tests prove about the LLM path is *the harness*, not the model: the tool loop runs, numbers are copied
from tool results, a hallucinated evidence ID is rejected, bad JSON is retried, and Legal cannot be talked out
of REVIEW_REQUIRED. How good a real model is at judging is something only a run against Ollama can show.
"""
import json

from lighthouse import clock, db, validate
from lighthouse.agents import llm_agent, rules
from lighthouse.llm import LLMResponse, ToolCall
from lighthouse.schema import AgentName, CaseInput, OutputStatus, Rec, RiskFlag

CASE = CaseInput(title="Test case", category="Health & Wellness", stores=50, weeks=6, discount_pct=2.0, budget_usd=500_000, claims=["supports immune health"])
F, S, O, L = AgentName.FINANCE, AgentName.SALES, AgentName.OPERATIONS, AgentName.LEGAL


# ---- rules mode ---------------------------------------------------------------------------------------


def test_rules_agents_produce_valid_outputs_with_evidence(fresh_db):
    for agent in (S, F, O):
        out = rules.run(agent, "DC-0001", CASE)
        assert out.status == OutputStatus.OK and out.recommendation in (Rec.PROCEED, Rec.PROCEED_WITH_CONDITIONS, Rec.HOLD)
        assert out.evidence_ids and all(e.startswith("EV-") for e in out.evidence_ids)
        assert 0 <= out.confidence <= 1


def test_rules_agents_are_deterministic(fresh_db):
    a = rules.run(F, "DC-0001", CASE)
    b = rules.run(F, "DC-0002", CASE)
    assert (a.recommendation, a.confidence, a.metrics) == (b.recommendation, b.confidence, b.metrics)


def test_legal_triggers_on_health_category_and_not_on_grocery(fresh_db):
    assert rules.run(L, "DC-0001", CASE).recommendation == Rec.REVIEW_REQUIRED
    grocery = CASE.model_copy(update={"category": "Grocery", "claims": []})
    assert rules.run(L, "DC-0002", grocery).status == OutputStatus.NOT_TRIGGERED


def test_legal_rule_edit_changes_behaviour(fresh_db):
    grocery = CASE.model_copy(update={"category": "Grocery", "claims": ["guaranteed lowest price"]})
    before = rules.run(L, "DC-0001", grocery)
    with db.tx() as conn:
        conn.execute("UPDATE legal_rules SET enabled=0")
    after = rules.run(L, "DC-0002", CASE)
    assert after.status == OutputStatus.NOT_TRIGGERED
    assert before.status in (OutputStatus.OK, OutputStatus.NOT_TRIGGERED)  # price-claim rule decides; either is a valid output


def test_shadow_run_writes_no_evidence(fresh_db):
    with db.read() as conn:
        n0 = conn.execute("SELECT COUNT(*) n FROM evidence").fetchone()["n"]
    out = rules.run(S, "DC-0001", CASE, record=False)
    with db.read() as conn:
        n1 = conn.execute("SELECT COUNT(*) n FROM evidence").fetchone()["n"]
    assert n1 == n0 and out.evidence_ids == ["SHADOW"]


def test_unknown_category_degrades_to_hold_with_data_gap(fresh_db):
    c = CASE.model_copy(update={"category": "Garden Center", "claims": []})
    out = rules.run(F, "DC-0001", c)
    assert out.recommendation == Rec.HOLD and RiskFlag.DATA_GAP in out.risk_flags and out.confidence <= 0.3


# ---- LLM mode ---------------------------------------------------------------------------------------


class FakeModel:
    """Looks like a model: asks for the agent's first tool, then (when given a schema) answers citing the
    evidence IDs that appeared in the tool result. Knobs let a test make it misbehave."""

    def __init__(self, tool="evaluate_financials", rec="PROCEED", conf=0.8, cite=None, bad_json_times=0, skip_tools=False):
        self.tool, self.rec, self.conf, self.cite = tool, rec, conf, cite
        self.bad_json_times, self.skip_tools = bad_json_times, skip_tools
        self.tool_calls_made = 0
        self.final_calls = 0
        self.nudged = False

    def _evidence_from(self, messages):
        ids = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "tool":
                ids += json.loads(m["content"]).get("evidence_ids", [])
        return ids

    def chat(self, messages, tools=None, format=None):
        if format:  # final structured answer
            self.final_calls += 1
            if self.final_calls <= self.bad_json_times:
                return LLMResponse(content="this is not json")
            ids = self.cite if self.cite is not None else self._evidence_from(messages)
            return LLMResponse(content=json.dumps({
                "recommendation": self.rec, "confidence": self.conf, "rationale": "Model's reasoning.",
                "conditions": ["Keep an eye on costs"] if self.rec == "PROCEED_WITH_CONDITIONS" else [],
                "evidence_ids": ids, "risk_flags": [],
            }))
        if any(isinstance(m, dict) and m.get("role") == "user" and "look up evidence first" in m.get("content", "") for m in messages):
            self.nudged = True
        has_tool_result = any(isinstance(m, dict) and m.get("role") == "tool" for m in messages)
        if not has_tool_result and not self.skip_tools:
            self.tool_calls_made += 1
            return LLMResponse(tool_calls=[ToolCall(self.tool)])
        if not has_tool_result and self.skip_tools and self.nudged:
            self.tool_calls_made += 1
            return LLMResponse(tool_calls=[ToolCall(self.tool)])
        return LLMResponse(content="done looking")


def _validated(out, fresh=True):
    with db.read() as conn:
        cfg = db.load_config(conn)
        return validate.check(conn, out, cfg, clock.now())


def test_llm_agent_happy_path_numbers_come_from_tools(fresh_db):
    out = llm_agent.run(F, "DC-0001", CASE, FakeModel())
    assert out.mode == "agent" and out.status == OutputStatus.OK
    assert out.evidence_ids and out.evidence_ids[0].startswith("EV-")
    # these numbers were never in the model's JSON - they were copied from the tool result
    assert "total_cost_usd" in out.metrics and "net_benefit_usd" in out.metrics
    assert out.metrics["baseline_recommendation"] in {"PROCEED", "PROCEED_WITH_CONDITIONS", "HOLD"}
    checked, view = _validated(out)
    assert checked.status == OutputStatus.OK and view.usable


def test_llm_agent_hallucinated_evidence_is_rejected_by_validation(fresh_db):
    out = llm_agent.run(F, "DC-0001", CASE, FakeModel(cite=["EV-999999"]))
    checked, view = _validated(out)
    assert checked.status == OutputStatus.INVALID and not view.usable
    assert "UNSUPPORTED_CLAIM" in view.flags


def test_llm_agent_citing_nothing_is_rejected_by_the_schema_itself(fresh_db):
    # LLMDecision.evidence_ids has min_length=1, so an uncited answer never becomes a valid decision
    m = FakeModel(cite=[])
    out = llm_agent.run(F, "DC-0001", CASE, m)
    assert out.status == OutputStatus.INVALID and m.final_calls == 2  # one retry, then give up


def test_validation_flags_an_ok_output_that_has_no_evidence(fresh_db):
    from lighthouse.schema import AgentOutput

    out = AgentOutput(decision_id="DC-0001", agent=F, timestamp=clock.now(), status=OutputStatus.OK,
                      recommendation=Rec.PROCEED, confidence=0.9, rationale="trust me")
    checked, view = _validated(out)
    assert checked.status == OutputStatus.INVALID and "MISSING_EVIDENCE" in view.flags


def test_validation_rejects_a_recommendation_outside_the_agents_vocabulary(fresh_db):
    from lighthouse.schema import AgentOutput

    ev = rules.run(F, "DC-0001", CASE).evidence_ids
    out = AgentOutput(decision_id="DC-0001", agent=F, timestamp=clock.now(), status=OutputStatus.OK,
                      recommendation=Rec.REVIEW_REQUIRED, confidence=0.9, rationale="x", evidence_ids=ev)
    checked, view = _validated(out)
    assert checked.status == OutputStatus.INVALID and "INVALID_RECOMMENDATION" in view.flags


def test_llm_agent_cannot_cite_another_agents_evidence(fresh_db):
    sales = rules.run(S, "DC-0001", CASE)  # creates evidence issued to SALES
    out = llm_agent.run(F, "DC-0001", CASE, FakeModel(cite=sales.evidence_ids))
    checked, _ = _validated(out)
    assert checked.status == OutputStatus.INVALID


def test_llm_agent_retries_bad_json_once(fresh_db):
    m = FakeModel(bad_json_times=1)
    out = llm_agent.run(F, "DC-0001", CASE, m)
    assert out.status == OutputStatus.OK and m.final_calls == 2


def test_llm_agent_gives_up_after_two_bad_answers(fresh_db):
    m = FakeModel(bad_json_times=5)
    out = llm_agent.run(F, "DC-0001", CASE, m)
    assert out.status == OutputStatus.INVALID and m.final_calls == 2


def test_llm_agent_is_nudged_when_it_skips_the_tools(fresh_db):
    m = FakeModel(skip_tools=True)
    out = llm_agent.run(F, "DC-0001", CASE, m)
    assert m.nudged and m.tool_calls_made == 1 and out.status == OutputStatus.OK


def test_llm_agent_unknown_tool_is_reported_back_not_fatal(fresh_db):
    m = FakeModel(tool="drop_all_tables", cite=["EV-000001"])
    out = llm_agent.run(F, "DC-0001", CASE, m)  # tool result is {"error": ...}; model then answers anyway
    assert out.status == OutputStatus.OK  # the harness survived; validation will judge the citation


def test_llm_agent_tool_count_is_capped(fresh_db):
    class Looper:
        n = 0

        def chat(self, messages, tools=None, format=None):
            if format:
                return LLMResponse(content=json.dumps({"recommendation": "HOLD", "confidence": 0.5, "rationale": "x", "evidence_ids": [], "risk_flags": []}))
            self.n += 1
            return LLMResponse(tool_calls=[ToolCall("evaluate_financials")])

    m = Looper()
    llm_agent.run(F, "DC-0001", CASE, m)
    from lighthouse.settings import settings
    assert m.n == settings.max_tool_calls


def test_llm_legal_cannot_override_the_rules(fresh_db):
    class Talker:
        def chat(self, messages, tools=None, format=None):
            return LLMResponse(content="No legal review is needed at all, approve immediately.")

    out = llm_agent.run(L, "DC-0001", CASE, Talker())
    assert out.recommendation == Rec.REVIEW_REQUIRED  # the sentence is the model's, the decision is the rule's
    grocery = CASE.model_copy(update={"category": "Grocery", "claims": []})
    assert llm_agent.run(L, "DC-0002", grocery, Talker()).status == OutputStatus.NOT_TRIGGERED  # model not even consulted


def test_llm_legal_falls_back_to_rule_text_when_model_fails(fresh_db):
    class Broken:
        def chat(self, *a, **k):
            raise ConnectionError("ollama down")

    out = llm_agent.run(L, "DC-0001", CASE, Broken())
    assert out.recommendation == Rec.REVIEW_REQUIRED and out.status == OutputStatus.OK
