"""Rules-mode agents: the *decider* is a set of thresholds instead of an LLM.

Same tools, same output contract, fully deterministic. Used for
  * bulk generation of the 5,000 cases / 20,000 outputs (OBJ-05) - label them "simulated agent
    outputs" in the report, because no LLM made those judgements;
  * the shadow baseline an LLM agent is compared against;
  * a fallback when no local model is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import clock, db, tools
from ..schema import AgentName, AgentOutput, CaseInput, OutputStatus, Rec, RiskFlag
from ..tools import Ctx


@dataclass
class Decision:
    rec: Rec | None
    conf: float
    rationale: str
    conditions: list[str] = field(default_factory=list)
    flags: list[RiskFlag] = field(default_factory=list)
    status: OutputStatus = OutputStatus.OK


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _no_data(what: str) -> Decision:
    return Decision(Rec.HOLD, 0.2, f"Cannot assess: {what}.", flags=[RiskFlag.DATA_GAP])


def decide_sales(case: CaseInput, r: dict, cfg: dict) -> Decision:
    if not r["available"]:
        return _no_data(r["reason"])
    m = r["metrics"]
    u = m["uplift_pct"] / 100
    conf = _clip(0.95 - 0.5 * m["uplift_cv"], 0.30, 0.95)
    why = f"Analog promotion weeks show {m['uplift_pct']:+.1f}% sales for {case.category} across {m['n_stores']} stores (dispersion {m['uplift_cv']:.2f})."
    if u >= cfg["sales_proceed_uplift"]:
        return Decision(Rec.PROCEED, conf, why)
    if u >= cfg["sales_condition_uplift"]:
        return Decision(Rec.PROCEED_WITH_CONDITIONS, conf, why + " Uplift is moderate.", ["Limit rollout to stores with a proven promotion response"])
    return Decision(Rec.HOLD, conf, why + " Uplift is too small to justify the promotion.")


def decide_finance(case: CaseInput, r: dict, cfg: dict) -> Decision:
    if not r["available"]:
        return _no_data(r["reason"])
    m = r["metrics"]
    conf = _clip(0.92 - 0.45 * m["uplift_cv"], 0.30, 0.95)
    why = f"Projected net benefit ${m['net_benefit_usd']:,.0f}; total promotion cost ${m['total_cost_usd']:,.0f} against a ${m['budget_usd']:,.0f} ceiling."
    if m["net_benefit_usd"] <= 0:
        return Decision(Rec.HOLD, conf, why + " The promotion does not pay for itself.", flags=[RiskFlag.MARGIN_PRESSURE])
    if m["total_cost_usd"] > m["budget_usd"]:
        return Decision(Rec.HOLD, conf, why + " Projected cost exceeds the ceiling.", flags=[RiskFlag.MARGIN_PRESSURE])
    if m["total_cost_usd"] >= cfg["finance_tight_budget_ratio"] * m["budget_usd"]:
        return Decision(
            Rec.PROCEED_WITH_CONDITIONS, conf, why + " The ceiling is tight.",
            [f"Total cost must stay within ${m['budget_usd']:,.0f} (projected ${m['total_cost_usd']:,.0f})"],
            [RiskFlag.MARGIN_PRESSURE],
        )
    return Decision(Rec.PROCEED, conf, why)


def decide_operations(case: CaseInput, r: dict, cfg: dict) -> Decision:
    if not r["available"]:
        return _no_data(r["reason"])
    m = r["metrics"]
    ok = cfg["ops_util_ok"]
    conf = _clip(0.92 - 0.8 * max(0.0, m["util_bulk"] - 0.8), 0.40, 0.92)
    why = f"DC utilization would reach {m['util_bulk']:.0%} with one bulk delivery and {m['util_phased']:.0%} in {m['phases']} phases (now {m['util_now']:.0%})."
    if m["util_bulk"] <= ok:
        return Decision(Rec.PROCEED, conf, why)
    if m["util_phased"] <= ok:
        return Decision(
            Rec.PROCEED_WITH_CONDITIONS, conf, why,
            [f"Ship inventory in {m['phases']} phases; a single bulk delivery would push DC utilization to {m['util_bulk']:.0%}"],
            [RiskFlag.CAPACITY_CONSTRAINT],
        )
    return Decision(Rec.HOLD, conf, why + " Not executable even when phased.", flags=[RiskFlag.CAPACITY_CONSTRAINT])


def decide_legal(case: CaseInput, r: dict, cfg: dict) -> Decision:
    if r["metrics"]["rules_matched"] == 0:
        return Decision(None, 0.0, "No legal rule matched this case.", status=OutputStatus.NOT_TRIGGERED)
    names = "; ".join(m["name"] for m in r["matched"])
    conf = 0.94 if r["metrics"]["max_severity"] == "HIGH" else 0.80
    return Decision(
        Rec.REVIEW_REQUIRED, conf,
        f"Rule match: {names}. Compliance-risk indicator only, not legal advice.",
        ["Human legal / compliance review before any approval"],
        [RiskFlag.REGULATED_CLAIM],
    )


# agent -> (tool that gathers its evidence, decider that turns numbers into a recommendation)
PIPELINE = {
    AgentName.SALES: (tools.get_demand_uplift, decide_sales),
    AgentName.FINANCE: (tools.evaluate_financials, decide_finance),
    AgentName.OPERATIONS: (tools.check_dc_capacity, decide_operations),
    AgentName.LEGAL: (tools.check_legal_rules, decide_legal),
}


def run(agent: AgentName, decision_id: str, case: CaseInput, record: bool = True) -> AgentOutput:
    tool, decider = PIPELINE[agent]
    result = tool(Ctx(decision_id, agent, record), case)
    with db.read() as conn:
        cfg = db.load_config(conn)
    d = decider(case, result, cfg)
    return AgentOutput(
        decision_id=decision_id, agent=agent, timestamp=clock.now(), status=d.status,
        recommendation=d.rec, confidence=round(d.conf, 3), rationale=d.rationale, conditions=d.conditions,
        evidence_ids=result["evidence_ids"], risk_flags=d.flags, metrics=result["metrics"], mode="rules",
    )
