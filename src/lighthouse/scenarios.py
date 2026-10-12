"""Scripted scenarios (OBJ-07) - the same material drives three things:

  1. the live demo ("scenario picker" in the UI),
  2. `replay` mode: agents answer from a script, but EVERYTHING after the agents (validation,
     conflict detection, legal gate, synthesis, decisions, audit) is the real code,
  3. the regression tests: every scenario states what the packet must look like (`expect`) and,
     optionally, a script of human actions (`steps`) with the HTTP result each must produce.

Two scenarios (s09, s11) run on `rules` agents against the generated data instead of a script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from . import clock, db
from .schema import AgentName, AgentOutput, CaseInput, OutputStatus, Rec, RiskFlag

F, S, O, L = AgentName.FINANCE, AgentName.SALES, AgentName.OPERATIONS, AgentName.LEGAL


@dataclass
class EvSpec:
    source: str
    summary: str
    age_hours: float = 0.0


@dataclass
class OutSpec:
    status: OutputStatus = OutputStatus.OK
    rec: Rec | None = None
    conf: float = 0.0
    rationale: str = ""
    conditions: list[str] = field(default_factory=list)
    flags: list[RiskFlag] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    evidence: list[EvSpec] = field(default_factory=list)
    never_issued_ids: list[str] = field(default_factory=list)  # evidence IDs the agent "made up"


@dataclass
class Step:
    role: str  # decision_owner | legal_reviewer
    kind: str  # decision | legal
    body: dict
    expect_http: int = 200
    expect_status: str | None = None  # case status after this step


@dataclass
class Scenario:
    id: str
    title: str
    description: str
    case: CaseInput
    outputs: dict[AgentName, OutSpec] = field(default_factory=dict)
    mode: str = "replay"
    expect: dict = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)


def _ok(rec, conf, why, *, cond=(), flags=(), metrics=None, ev=()) -> OutSpec:
    return OutSpec(rec=rec, conf=conf, rationale=why, conditions=list(cond), flags=list(flags),
                   metrics=metrics or {}, evidence=list(ev))


def _not_triggered(why="No legal rule matched this case.") -> OutSpec:
    return OutSpec(status=OutputStatus.NOT_TRIGGERED, rationale=why)


SALES_EV = EvSpec("sales_weekly", "Analog promotion weeks")
COST_EV = EvSpec("cost_assumptions", "Margin and promotion cost assumptions")
INV_EV = EvSpec("inventory_position", "DC inventory and capacity")
LEGAL_EV = EvSpec("legal_rules", "Legal rule check")

P, PC, H, RR = Rec.PROCEED, Rec.PROCEED_WITH_CONDITIONS, Rec.HOLD, Rec.REVIEW_REQUIRED

SCENARIOS: dict[str, Scenario] = {}


def _add(s: Scenario) -> None:
    SCENARIOS[s.id] = s


_add(Scenario(
    id="s01_all_agree", title="All agents agree",
    description="Three PROCEED, Legal not triggered. Clean path: no conflict, no gate, no reason needed.",
    case=CaseInput(title="Grocery weekend promotion, 20 stores", category="Grocery", stores=20, weeks=4, discount_pct=1.5, budget_usd=400_000),
    outputs={
        S: _ok(P, 0.88, "Grocery promotions lift sales about 6%.", metrics={"uplift_pct": 6.1}, ev=[SALES_EV]),
        F: _ok(P, 0.85, "Positive net benefit, well inside the cost ceiling.", metrics={"net_benefit_usd": 62_000, "total_cost_usd": 140_000, "budget_usd": 400_000}, ev=[SALES_EV, COST_EV]),
        O: _ok(P, 0.83, "DCs absorb a single delivery comfortably.", metrics={"util_bulk": 0.71}, ev=[INV_EV]),
        L: _not_triggered(),
    },
    expect=dict(rec="PROCEED", gate="NOT_REQUIRED", conflicts=set(), escalations=set(), requires_reason=False, incomplete=False),
    steps=[Step("decision_owner", "decision", {"action": "APPROVE"}, 200, "APPROVED")],
))

_add(Scenario(
    id="s02_case1_walkthrough", title="Decision Case #1: Health & Wellness (the deck's walkthrough)",
    description="Sales PROCEED 0.86, Finance and Operations PROCEED with conditions, Legal REVIEW_REQUIRED 0.94. "
                "Approval is locked until Legal clears it; then a written reason is required.",
    case=CaseInput(title="Health & Wellness six-week promotion, 50 stores", category="Health & Wellness", stores=50, weeks=6, discount_pct=2.0, budget_usd=315_000, claims=["supports immune health"]),
    outputs={
        S: _ok(P, 0.86, "Demand uplift +13.2%. Promotion analogs are supportive.", metrics={"uplift_pct": 13.2}, ev=[SALES_EV]),
        F: _ok(PC, 0.82, "Net benefit $95K, only if total costs stay within $315K.", cond=["Total costs must stay within $315K"], flags=[RiskFlag.MARGIN_PRESSURE],
               metrics={"net_benefit_usd": 95_000, "total_cost_usd": 298_000, "budget_usd": 315_000}, ev=[SALES_EV, COST_EV]),
        O: _ok(PC, 0.79, "A single bulk delivery would push the distribution center to 92% capacity.",
               cond=["Ship inventory in phases"], flags=[RiskFlag.CAPACITY_CONSTRAINT],
               metrics={"util_bulk": 0.92, "util_phased": 0.71, "phases": 3}, ev=[INV_EV]),
        L: _ok(RR, 0.94, "Health category marketing claims require human review. Compliance-risk indicator, not legal advice.",
               cond=["Human legal / compliance review before any approval"], flags=[RiskFlag.REGULATED_CLAIM], metrics={"rules_matched": 1}, ev=[LEGAL_EV]),
    },
    expect=dict(rec="PROCEED_WITH_CONDITIONS", gate="PENDING", conflicts={"CONDITION_TENSION"},
                escalations={"LEGAL_GATE", "FINANCIAL_THRESHOLD", "OPERATIONAL_THRESHOLD"}, requires_reason=True, incomplete=False),
    steps=[
        Step("decision_owner", "decision", {"action": "APPROVE", "reason": "Trying to approve before legal review."}, 409, "AWAITING_LEGAL"),
        Step("legal_reviewer", "legal", {"disposition": "CLEARED_WITH_EDITS", "comment": "Remove disease-prevention wording; 'supports immune health' is acceptable."}, 200, "READY_FOR_DECISION"),
        Step("decision_owner", "decision", {"action": "APPROVE"}, 422, "READY_FOR_DECISION"),
        Step("decision_owner", "decision", {"action": "APPROVE", "reason": "Approved with phased shipment and the $315K ceiling."}, 200, "APPROVED"),
    ],
))

_add(Scenario(
    id="s03_finance_hold_vs_sales", title="Disagreement: Finance HOLD vs Sales PROCEED",
    description="Strong demand but the promotion loses money. A hard conflict is surfaced (not averaged); approving anyway is an override that needs a reason.",
    case=CaseInput(title="Electronics deep-discount event", category="Electronics", stores=30, weeks=4, discount_pct=8.0, budget_usd=500_000),
    outputs={
        S: _ok(P, 0.90, "Electronics promotions lift sales about 22%.", metrics={"uplift_pct": 22.0}, ev=[SALES_EV]),
        F: _ok(H, 0.81, "Net benefit is negative once the discount is paid.", flags=[RiskFlag.MARGIN_PRESSURE],
               metrics={"net_benefit_usd": -40_000, "total_cost_usd": 210_000, "budget_usd": 500_000}, ev=[SALES_EV, COST_EV]),
        O: _ok(P, 0.80, "Capacity is fine.", metrics={"util_bulk": 0.70}, ev=[INV_EV]),
        L: _not_triggered(),
    },
    expect=dict(rec="HOLD", gate="NOT_REQUIRED", conflicts={"HARD_DISAGREEMENT"}, escalations={"HARD_CONFLICT"}, requires_reason=True, incomplete=False),
    steps=[
        Step("decision_owner", "decision", {"action": "APPROVE"}, 422, "READY_FOR_DECISION"),
        Step("decision_owner", "decision", {"action": "APPROVE", "reason": "Override: strategic traffic driver, finance accepts the loss."}, 200, "APPROVED"),
    ],
))

_add(Scenario(
    id="s04_ops_capacity_hold", title="Disagreement: Operations cannot execute",
    description="Demand and margin are fine but the DCs cannot absorb the inventory even when phased.",
    case=CaseInput(title="Apparel clearance across 50 stores", category="Apparel", stores=50, weeks=8, discount_pct=2.0, budget_usd=600_000),
    outputs={
        S: _ok(P, 0.86, "Apparel promotions lift sales about 18%.", metrics={"uplift_pct": 18.0}, ev=[SALES_EV]),
        F: _ok(P, 0.84, "Healthy margin inside the ceiling.", metrics={"net_benefit_usd": 120_000, "total_cost_usd": 180_000, "budget_usd": 600_000}, ev=[SALES_EV, COST_EV]),
        O: _ok(H, 0.74, "DC utilization would exceed capacity even when phased.", flags=[RiskFlag.CAPACITY_CONSTRAINT],
               metrics={"util_bulk": 1.08, "util_phased": 0.93, "phases": 3}, ev=[INV_EV]),
        L: _not_triggered(),
    },
    expect=dict(rec="HOLD", gate="NOT_REQUIRED", conflicts={"HARD_DISAGREEMENT"}, escalations={"HARD_CONFLICT", "OPERATIONAL_THRESHOLD"}, requires_reason=True, incomplete=False),
))

_add(Scenario(
    id="s05_agent_timeout", title="Missing output: Operations times out",
    description="One required agent never answers. The packet is built anyway, marked incomplete, and defaults to HOLD.",
    case=CaseInput(title="Home refresh promotion", category="Home", stores=25, weeks=4, discount_pct=2.0, budget_usd=300_000),
    outputs={
        S: _ok(P, 0.84, "Home promotions lift sales about 12%.", metrics={"uplift_pct": 12.0}, ev=[SALES_EV]),
        F: _ok(P, 0.83, "Positive net benefit.", metrics={"net_benefit_usd": 50_000, "total_cost_usd": 110_000, "budget_usd": 300_000}, ev=[SALES_EV, COST_EV]),
        O: OutSpec(status=OutputStatus.TIMEOUT, rationale="No answer within the time budget (scripted)."),
        L: _not_triggered(),
    },
    expect=dict(rec="HOLD", gate="NOT_REQUIRED", conflicts=set(), escalations={"INCOMPLETE"}, requires_reason=True, incomplete=True,
                status={"OPERATIONS": "TIMEOUT"}),
))

_add(Scenario(
    id="s06_unsupported_evidence", title="Hallucination control: cited evidence does not exist",
    description="Finance cites an evidence ID that was never issued. Validation marks the output INVALID; the packet is incomplete.",
    case=CaseInput(title="Apparel spring promotion", category="Apparel", stores=20, weeks=4, discount_pct=2.0, budget_usd=300_000),
    outputs={
        S: _ok(P, 0.85, "Apparel promotions lift sales about 18%.", metrics={"uplift_pct": 18.0}, ev=[SALES_EV]),
        F: OutSpec(rec=P, conf=0.92, rationale="Net benefit is large.", metrics={"net_benefit_usd": 500_000}, never_issued_ids=["EV-999999"]),
        O: _ok(P, 0.82, "Capacity is fine.", metrics={"util_bulk": 0.68}, ev=[INV_EV]),
        L: _not_triggered(),
    },
    expect=dict(rec="HOLD", gate="NOT_REQUIRED", conflicts=set(), escalations={"INCOMPLETE"}, requires_reason=True, incomplete=True,
                status={"FINANCE": "INVALID"}, flags={"FINANCE": {"UNSUPPORTED_CLAIM"}}),
))

_add(Scenario(
    id="s07_stale_inventory", title="Stale data: inventory evidence is 60 hours old",
    description="Operations' evidence is older than its 36-hour limit. Confidence is downgraded and the PROCEED is downgraded to PROCEED_WITH_CONDITIONS.",
    case=CaseInput(title="Grocery midweek promotion", category="Grocery", stores=30, weeks=3, discount_pct=1.5, budget_usd=500_000),
    outputs={
        S: _ok(P, 0.87, "Grocery promotions lift sales about 6%.", metrics={"uplift_pct": 6.0}, ev=[SALES_EV]),
        F: _ok(P, 0.84, "Positive net benefit.", metrics={"net_benefit_usd": 40_000, "total_cost_usd": 150_000, "budget_usd": 500_000}, ev=[SALES_EV, COST_EV]),
        O: _ok(P, 0.79, "DCs absorb a single delivery.", metrics={"util_bulk": 0.72}, ev=[EvSpec("inventory_position", "DC inventory snapshot", age_hours=60)]),
        L: _not_triggered(),
    },
    expect=dict(rec="PROCEED_WITH_CONDITIONS", gate="NOT_REQUIRED", conflicts=set(), escalations={"STALE_EVIDENCE", "LOW_CONFIDENCE"}, requires_reason=True, incomplete=False,
                flags={"OPERATIONS": {"STALE_EVIDENCE", "LOW_CONFIDENCE"}}),
))

_add(Scenario(
    id="s08_low_confidence", title="Low confidence: Sales is unsure",
    description="Sales says PROCEED but with confidence 0.41. It is flagged and the packet asks the human to confirm.",
    case=CaseInput(title="Electronics new-brand launch", category="Electronics", stores=10, weeks=2, discount_pct=2.0, budget_usd=200_000),
    outputs={
        S: _ok(P, 0.41, "Few analog promotions exist for this brand.", flags=[RiskFlag.LOW_CONFIDENCE], metrics={"uplift_pct": 9.0}, ev=[SALES_EV]),
        F: _ok(P, 0.82, "Positive net benefit.", metrics={"net_benefit_usd": 30_000, "total_cost_usd": 90_000, "budget_usd": 200_000}, ev=[SALES_EV, COST_EV]),
        O: _ok(P, 0.80, "Capacity is fine.", metrics={"util_bulk": 0.66}, ev=[INV_EV]),
        L: _not_triggered(),
    },
    expect=dict(rec="PROCEED_WITH_CONDITIONS", gate="NOT_REQUIRED", conflicts=set(), escalations={"LOW_CONFIDENCE"}, requires_reason=True, incomplete=False,
                flags={"SALES": {"LOW_CONFIDENCE"}}),
))

_add(Scenario(
    id="s09_missing_data", title="Missing data: no history for this category (real rules agents)",
    description="Runs the real rules agents on a category that has no sales, cost or inventory data. Each agent says HOLD with a DATA_GAP flag.",
    mode="rules",
    case=CaseInput(title="Garden center spring event", category="Garden Center", stores=20, weeks=4, discount_pct=2.0, budget_usd=200_000),
    expect=dict(rec="HOLD", gate="NOT_REQUIRED", conflicts=set(), escalations={"LOW_CONFIDENCE"}, requires_reason=True, incomplete=False,
                flags={"SALES": {"LOW_CONFIDENCE"}}),
))

_add(Scenario(
    id="s10_legal_blocked_then_reject", title="Rejected decision: Legal blocks the marketing claim",
    description="Legal blocks the claim; approval is impossible; the owner rejects with a written reason.",
    case=CaseInput(title="Health & Wellness energy campaign", category="Health & Wellness", stores=20, weeks=4, discount_pct=2.0, budget_usd=300_000, claims=["clinically proven to boost energy"]),
    outputs={
        S: _ok(P, 0.85, "Health promotions lift sales about 14%.", metrics={"uplift_pct": 14.0}, ev=[SALES_EV]),
        F: _ok(P, 0.83, "Positive net benefit.", metrics={"net_benefit_usd": 55_000, "total_cost_usd": 150_000, "budget_usd": 300_000}, ev=[SALES_EV, COST_EV]),
        O: _ok(P, 0.80, "Capacity is fine.", metrics={"util_bulk": 0.74}, ev=[INV_EV]),
        L: _ok(RR, 0.94, "The claim 'clinically proven' needs substantiation. Compliance-risk indicator, not legal advice.", flags=[RiskFlag.REGULATED_CLAIM], metrics={"rules_matched": 2}, ev=[LEGAL_EV]),
    },
    expect=dict(rec="PROCEED_WITH_CONDITIONS", gate="PENDING", conflicts=set(), escalations={"LEGAL_GATE"}, requires_reason=True, incomplete=False),
    steps=[
        Step("legal_reviewer", "legal", {"disposition": "BLOCKED", "comment": "The claim cannot be substantiated; do not run as written."}, 200, "LEGAL_BLOCKED"),
        Step("decision_owner", "decision", {"action": "APPROVE", "reason": "Trying to approve a blocked case."}, 409, "LEGAL_BLOCKED"),
        Step("decision_owner", "decision", {"action": "REJECT"}, 422, "LEGAL_BLOCKED"),
        Step("decision_owner", "decision", {"action": "REJECT", "reason": "Rejected: legal blocked the marketing claim."}, 200, "REJECTED"),
    ],
))

_add(Scenario(
    id="s11_revision_loop", title="Revision loop: change the promotion and re-run (real rules agents)",
    description="A deep discount makes Finance say HOLD. The owner requests a revision with a smaller, shorter promotion; the case loops back to ingestion as DC-xxxx.r1 and gets a new packet.",
    mode="rules",
    case=CaseInput(title="Electronics deep discount", category="Electronics", stores=50, weeks=8, discount_pct=5.0, budget_usd=60_000),
    expect=dict(rec="HOLD", gate="NOT_REQUIRED", conflicts={"HARD_DISAGREEMENT", "CONDITION_TENSION"},
                escalations={"HARD_CONFLICT", "FINANCIAL_THRESHOLD", "OPERATIONAL_THRESHOLD"}, requires_reason=True, incomplete=False,
                status={"FINANCE": "OK"}),
    steps=[
        Step("decision_owner", "decision",
             {"action": "REVISE", "reason": "Reduce the discount, shorten the promotion and raise the ceiling.",
              "revised_input": {"discount_pct": 1.0, "weeks": 4, "budget_usd": 5_000_000}}, 200, "REVISION_REQUESTED"),
    ],
))


def list_scenarios() -> list[dict]:
    return [{"id": s.id, "title": s.title, "description": s.description, "mode": s.mode} for s in SCENARIOS.values()]


def produce(scenario_id: str, agent: AgentName, decision_id: str) -> AgentOutput:
    """Build an agent's scripted answer, creating its evidence rows (with the scripted ages) first."""
    spec = SCENARIOS[scenario_id].outputs[agent]
    ids: list[str] = []
    if spec.status == OutputStatus.OK:
        with db.tx() as conn:
            for ev in spec.evidence:
                eid = f"EV-{db.next_number(conn, 'evidence'):06d}"
                as_of = clock.iso(clock.now() - timedelta(hours=ev.age_hours))
                conn.execute(
                    "INSERT INTO evidence(evidence_id, decision_id, agent, source, as_of, summary, payload_json) VALUES(?,?,?,?,?,?,?)",
                    (eid, decision_id, agent.value, ev.source, as_of, ev.summary, db.dumps({"scripted": True, "scenario": scenario_id})),
                )
                ids.append(eid)
    return AgentOutput(
        decision_id=decision_id, agent=agent, timestamp=clock.now(), status=spec.status, recommendation=spec.rec,
        confidence=spec.conf, rationale=spec.rationale, conditions=spec.conditions,
        evidence_ids=ids + spec.never_issued_ids, risk_flags=spec.flags, metrics=spec.metrics, mode="replay",
    )
