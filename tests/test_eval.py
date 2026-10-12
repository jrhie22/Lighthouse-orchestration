"""The LLM-vs-rules evaluation harness, exercised with a fake model (so we test the plumbing, not a real model)."""
from lighthouse import eval_llm, orchestrator
from test_agents import FakeModel


class Router:
    """Gives every agent run a fresh FakeModel that starts with that agent's own tool."""

    def chat(self, messages, tools=None, format=None):
        system = messages[0]["content"] if messages else ""
        tool = "check_dc_capacity" if "Operations" in system else "get_demand_uplift" if "Sales" in system else "evaluate_financials"
        # the fake always says PROCEED, so it agrees with the rules only when the rules also say PROCEED
        return FakeModel(tool=tool, rec="PROCEED").chat(messages, tools, format)


def test_eval_reports_agreement_and_status_counts(fresh_db):
    orchestrator.set_llm_factory(lambda: Router())
    out = eval_llm.evaluate(6, seed=2)
    a, b = out["agreement_with_rules"]["all"].split("/")
    assert 0 <= int(a) <= int(b) == 18  # 6 cases x 3 analytic agents
    assert sum(sum(c.values()) for c in out["status_counts"].values()) == 18
    assert out["seconds_per_case"]["max"] >= 0
    for d in out["disagreements"]:
        assert d["model"] == "PROCEED" and d["rules"] != "PROCEED"
