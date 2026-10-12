"""Bulk case generator (OBJ-05: 5,000 cases / 20,000 agent outputs) and the numbers for OBJ-06 (packet time).

It does NOT use a shortcut: every case goes through ``decisions.open_case`` and ``orchestrator.run_case`` -
the same code path the API uses - so each case really gets 4 agent outputs, validation, a packet and audit rows.
The only difference from a live case is that the deciders are the *rules* agents, so in the report these must be
called **simulated agent outputs** (no language model made those judgements).

Cases are generated from a seeded random generator, so the same seed gives the same 5,000 cases.
"""
from __future__ import annotations

import asyncio
import random
import statistics
import time
from collections import Counter

from . import db, decisions, orchestrator, tools
from .data import CATEGORIES
from .schema import AgentName, CaseInput
from .tools import Ctx

HEALTH_CLAIMS = ["supports immune health", "clinically proven to boost energy", "helps you sleep better", "natural vitamin blend"]
PRICE_CLAIMS = ["lowest price guaranteed", "price match promise", "weekend savings event", "limited time offer"]


def random_case(rng: random.Random, i: int) -> CaseInput:
    """A plausible promotion proposal. The budget ceiling is set relative to the model's own cost estimate (a planner
    would not propose a $50k ceiling for a multi-million-dollar promotion), so outcomes are a realistic mix
    instead of 'Finance says HOLD' for almost every case."""
    category = rng.choice(list(CATEGORIES))
    stores = rng.choice([5, 10, 20, 30, 40, 50])
    weeks = rng.randint(1, 10)
    discount = round(rng.triangular(0.5, 5.0, 1.5), 1)  # most proposals are small discounts; a few are aggressive
    claims: list[str] = []
    if category == "Health & Wellness" and rng.random() < 0.5:
        claims = [rng.choice(HEALTH_CLAIMS)]
    elif rng.random() < 0.1:
        claims = [rng.choice(PRICE_CLAIMS)]
    draft = dict(title=f"Simulated #{i:04d}: {category}, {stores} stores, {weeks} wks", category=category, stores=stores,
                 weeks=weeks, discount_pct=discount, claims=claims)
    est = tools.evaluate_financials(Ctx("ESTIMATE", AgentName.FINANCE, record=False), CaseInput(**draft, budget_usd=1_000_000))
    est_cost = est["metrics"]["total_cost_usd"] if est["available"] else 200_000
    budget = round(max(est_cost * rng.uniform(0.9, 2.0), 10_000), -3)
    return CaseInput(**draft, budget_usd=budget)


async def _run(n: int, seed: int, progress) -> list[str]:
    rng = random.Random(seed)
    ids = []
    for i in range(1, n + 1):
        case = random_case(rng, i)
        did = await asyncio.to_thread(decisions.open_case, case, "rules", None, "bulk-generator")
        await orchestrator.run_case(did)
        ids.append(did)
        if progress and i % 250 == 0:
            progress(i, n)
    return ids


def generate(n: int = 5000, seed: int = 42, progress=None) -> dict:
    """Create ``n`` cases in the current database and return a summary (also what the report quotes)."""
    t0 = time.perf_counter()
    ids = asyncio.run(_run(n, seed, progress))
    wall = time.perf_counter() - t0
    return summarize(ids, wall)


def summarize(ids: list[str] | None = None, wall_s: float | None = None) -> dict:
    with db.read() as conn:
        q = "SELECT decision_id FROM cases WHERE opened_by='bulk-generator'"
        mine = {r["decision_id"] for r in conn.execute(q)} if ids is None else set(ids)
        pk = {r["decision_id"]: r for r in conn.execute("SELECT decision_id, packet_ms, json FROM packets")}
        n_out = conn.execute("SELECT COUNT(*) n FROM agent_outputs").fetchone()["n"]
        n_audit = conn.execute("SELECT COUNT(*) n FROM audit_log").fetchone()["n"]
        statuses = Counter(r["status"] for r in conn.execute("SELECT status FROM cases WHERE opened_by='bulk-generator'"))
    import json

    recs, gates, escs, confl = Counter(), Counter(), Counter(), Counter()
    needs_reason = incomplete = 0
    ms: list[float] = []
    for did in mine:
        row = pk.get(did)
        if not row:
            continue
        p = json.loads(row["json"])
        recs[p["recommendation"]] += 1
        gates[p["legal_gate"]] += 1
        escs.update(e["code"] for e in p["escalations"])
        confl.update(c["type"] for c in p["conflicts"])
        needs_reason += bool(p["requires_reason"])
        incomplete += bool(p["incomplete"])
        ms.append(row["packet_ms"])
    ms.sort()
    k = max(len(mine), 1)
    return {
        "cases": len(mine), "agent_outputs_total": n_out, "audit_entries_total": n_audit, "wall_seconds": round(wall_s, 1) if wall_s else None,
        "recommendations": dict(recs), "legal_gate": dict(gates), "escalations": dict(escs), "conflicts": dict(confl),
        "share_requiring_written_reason": round(needs_reason / k, 3), "incomplete": incomplete, "case_status": dict(statuses),
        "packet_ms": {"p50": round(statistics.median(ms), 3), "p95": round(ms[int(0.95 * (len(ms) - 1))], 3), "max": round(ms[-1], 3)} if ms else None,
    }
