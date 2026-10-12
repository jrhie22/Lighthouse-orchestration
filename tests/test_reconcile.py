"""reconcile.build_packet is a pure function, so we test it with hand-made agent views."""
from datetime import datetime, timezone

from lighthouse import db, reconcile
from lighthouse.schema import (
    AgentName as A, AgentOutput, AgentView, CaseInput, GateStatus, OutputStatus as OS, Rec, RiskFlag,
)

CFG = {**db.DEFAULT_CONFIG}
CASE = CaseInput(title="t t t", category="Home", stores=5, weeks=2, budget_usd=1000)
NOW = datetime(2026, 10, 12, tzinfo=timezone.utc)


def view(agent, rec=None, conf=0.8, status=OS.OK, conditions=(), flags=(), metrics=None, eff=None, vflags=()):
    out = AgentOutput(decision_id="DC-1", agent=agent, timestamp=NOW, status=status, recommendation=rec, confidence=conf,
                      conditions=list(conditions), evidence_ids=["EV-1"], risk_flags=list(flags), metrics=metrics or {})
    usable = status == OS.OK
    return AgentView(output=out, effective_confidence=(eff if eff is not None else conf) if usable else 0.0, flags=list(vflags), usable=usable)


def legal_none():
    return view(A.LEGAL, status=OS.NOT_TRIGGERED)


def pack(*views):
    return reconcile.build_packet("DC-1", CASE, list(views), CFG, NOW)


def test_all_agree_is_clean():
    p = pack(view(A.FINANCE, Rec.PROCEED), view(A.SALES, Rec.PROCEED), view(A.OPERATIONS, Rec.PROCEED), legal_none())
    assert p.recommendation == Rec.PROCEED and not p.conflicts and not p.escalations and not p.requires_reason
    assert p.legal_gate == GateStatus.NOT_REQUIRED


def test_confidence_floor_is_the_minimum_not_the_mean():
    p = pack(view(A.FINANCE, Rec.PROCEED, 0.9), view(A.SALES, Rec.PROCEED, 0.5, eff=0.5), view(A.OPERATIONS, Rec.PROCEED, 0.7), legal_none())
    assert p.confidence_floor == 0.5


def test_hold_versus_proceed_is_a_hard_conflict_and_the_packet_holds():
    p = pack(view(A.FINANCE, Rec.HOLD), view(A.SALES, Rec.PROCEED), view(A.OPERATIONS, Rec.PROCEED), legal_none())
    assert p.recommendation == Rec.HOLD
    assert [c.type.value for c in p.conflicts] == ["HARD_DISAGREEMENT"]
    assert "HARD_CONFLICT" in {e.code for e in p.escalations}


def test_unconditional_vs_conditional_is_a_soft_conflict():
    p = pack(view(A.FINANCE, Rec.PROCEED), view(A.SALES, Rec.PROCEED),
             view(A.OPERATIONS, Rec.PROCEED_WITH_CONDITIONS, conditions=["Ship in phases"]), legal_none())
    assert p.recommendation == Rec.PROCEED_WITH_CONDITIONS
    assert [(c.type.value, c.severity) for c in p.conflicts] == [("CONDITION_TENSION", "SOFT")]
    assert "OPERATIONS: Ship in phases" in p.conditions


def test_legal_review_required_locks_the_gate_whatever_the_others_say():
    p = pack(view(A.FINANCE, Rec.PROCEED), view(A.SALES, Rec.PROCEED), view(A.OPERATIONS, Rec.PROCEED),
             view(A.LEGAL, Rec.REVIEW_REQUIRED, 0.94))
    assert p.legal_gate == GateStatus.PENDING and p.recommendation == Rec.PROCEED_WITH_CONDITIONS
    assert "LEGAL_GATE" in {e.code for e in p.escalations}


def test_legal_agent_failure_fails_closed():
    for st in (OS.TIMEOUT, OS.INVALID, OS.DISABLED):
        p = pack(view(A.FINANCE, Rec.PROCEED), view(A.SALES, Rec.PROCEED), view(A.OPERATIONS, Rec.PROCEED), view(A.LEGAL, status=st))
        assert p.legal_gate == GateStatus.PENDING, st


def test_missing_required_agent_makes_the_packet_incomplete_and_holds():
    p = pack(view(A.FINANCE, Rec.PROCEED), view(A.SALES, Rec.PROCEED), view(A.OPERATIONS, status=OS.TIMEOUT), legal_none())
    assert p.incomplete and p.recommendation == Rec.HOLD and "INCOMPLETE" in {e.code for e in p.escalations}


def test_financial_and_operational_thresholds_escalate():
    p = pack(view(A.FINANCE, Rec.PROCEED, metrics={"total_cost_usd": CFG["financial_threshold_usd"]}), view(A.SALES, Rec.PROCEED),
             view(A.OPERATIONS, Rec.PROCEED, metrics={"util_bulk": CFG["ops_util_escalate"]}), legal_none())
    assert {"FINANCIAL_THRESHOLD", "OPERATIONAL_THRESHOLD"} <= {e.code for e in p.escalations} and p.requires_reason


def test_stale_and_low_confidence_downgrade_proceed():
    p = pack(view(A.FINANCE, Rec.PROCEED), view(A.SALES, Rec.PROCEED, 0.9, eff=0.54, vflags=["STALE_EVIDENCE", "LOW_CONFIDENCE"]),
             view(A.OPERATIONS, Rec.PROCEED), legal_none())
    assert p.recommendation == Rec.PROCEED_WITH_CONDITIONS
    assert {"STALE_EVIDENCE", "LOW_CONFIDENCE"} <= {e.code for e in p.escalations}


def test_evidence_ids_are_collected_and_risk_flags_listed():
    p = pack(view(A.FINANCE, Rec.PROCEED, flags=[RiskFlag.MARGIN_PRESSURE]), view(A.SALES, Rec.PROCEED), view(A.OPERATIONS, Rec.PROCEED), legal_none())
    assert p.evidence_ids == ["EV-1"] and "FINANCE: MARGIN_PRESSURE" in p.unresolved_risks
