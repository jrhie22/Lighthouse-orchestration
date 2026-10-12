"""Tools: the only place where agents touch data.

Design rule that makes the whole system explainable: **tools compute, deciders decide.**

* A *tool* reads the pilot data, does the arithmetic, writes an ``evidence`` row (with an ID and an
  as-of timestamp), and returns the numbers plus the evidence ID(s).
* A *decider* (rules thresholds, or an LLM) looks at those numbers and chooses PROCEED / HOLD / ...

So numbers always come from code, and every claim can be traced to an evidence ID (OBJ-04, US-05).
Rules agents and LLM agents call exactly the same tools.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from . import db
from .schema import AgentName, CaseInput


@dataclass(frozen=True)
class Ctx:
    decision_id: str
    agent: AgentName
    record: bool = True  # False = "shadow" run: compute, but do not write evidence rows


def _record(conn, ctx: Ctx, source: str, summary: str, payload: dict) -> str:
    if not ctx.record:
        return "SHADOW"
    eid = f"EV-{db.next_number(conn, 'evidence'):06d}"
    as_of = conn.execute("SELECT refreshed_at FROM data_sources WHERE name=?", (source,)).fetchone()["refreshed_at"]
    conn.execute(
        "INSERT INTO evidence(evidence_id, decision_id, agent, source, as_of, summary, payload_json) VALUES(?,?,?,?,?,?,?)",
        (eid, ctx.decision_id, ctx.agent.value, source, as_of, summary, db.dumps(payload)),
    )
    return eid


# ---- small data helpers (each used by several tools) -------------------------------------------


def _sales_stats(conn, category: str, stores: int) -> dict | None:
    """Analog-promotion uplift: for each store, mean sales in promo weeks vs. non-promo weeks."""
    rows = conn.execute(
        """SELECT store,
                  AVG(CASE WHEN promo_flag=0 THEN sales_usd END) AS base,
                  AVG(CASE WHEN promo_flag=1 THEN sales_usd END) AS promo,
                  SUM(promo_flag) AS n_promo
           FROM sales_weekly WHERE category=? AND store<=? GROUP BY store""",
        (category, stores),
    ).fetchall()
    rows = [r for r in rows if r["base"]]
    if not rows:
        return None
    ratios = [r["promo"] / r["base"] - 1 for r in rows if r["promo"] is not None]
    if not ratios:
        return None
    mean = statistics.fmean(ratios)
    sd = statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
    cv = min(2.0, sd / abs(mean)) if abs(mean) > 1e-9 else 2.0
    return {
        "uplift": mean,
        "cv": cv,
        "base_week_total": sum(r["base"] for r in rows),  # USD per week across the stores in scope
        "n_stores": len(rows),
        "n_promo_weeks": int(sum(r["n_promo"] for r in rows)),
    }


def _sales_evidence(conn, ctx: Ctx, category: str, stores: int):
    s = _sales_stats(conn, category, stores)
    if s is None:
        eid = _record(conn, ctx, "sales_weekly", f"No sales history found for category '{category}'", {"category": category, "rows": 0})
    else:
        eid = _record(
            conn, ctx, "sales_weekly",
            f"Analog promo weeks: {category} uplift {s['uplift']:+.1%} across {s['n_stores']} stores",
            {"category": category, "stores_in_scope": stores, **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}},
        )
    return s, eid


def _cost_row(conn, ctx: Ctx, category: str):
    row = conn.execute("SELECT * FROM cost_assumptions WHERE category=?", (category,)).fetchone()
    if row is None:
        return None, _record(conn, ctx, "cost_assumptions", f"No cost assumptions for '{category}'", {"category": category})
    payload = {k: row[k] for k in row.keys()}
    return row, _record(
        conn, ctx, "cost_assumptions",
        f"Margin {row['gross_margin']:.0%}, unit price ${row['unit_price']:.2f}, promo cost ${row['promo_cost_per_store_week']:.0f}/store-week",
        payload,
    )


def _inventory(conn, ctx: Ctx, category: str):
    rows = conn.execute("SELECT * FROM inventory_position WHERE category=?", (category,)).fetchall()
    if not rows:
        return None, _record(conn, ctx, "inventory_position", f"No inventory position for '{category}'", {"category": category})
    inv = {
        "on_hand": sum(r["on_hand_units"] for r in rows),
        "inbound": sum(r["inbound_units"] for r in rows),
        "capacity": sum(r["capacity_units"] for r in rows),
        "dcs": [r["dc_id"] for r in rows],
    }
    return inv, _record(
        conn, ctx, "inventory_position",
        f"{len(rows)} DCs hold {inv['on_hand']:,.0f} units (+{inv['inbound']:,.0f} inbound) of {inv['capacity']:,.0f} capacity",
        inv,
    )


def _r(x: float, n: int = 2) -> float:
    return round(float(x), n)


def _unavailable(tool: str, ids: list[str], why: str) -> dict:
    return {"tool": tool, "available": False, "reason": why, "evidence_ids": ids, "metrics": {}}


# ---- the tools ----------------------------------------------------------------------------------


def get_demand_uplift(ctx: Ctx, case: CaseInput) -> dict:
    with db.tx() as conn:
        s, eid = _sales_evidence(conn, ctx, case.category, case.stores)
    if s is None:
        return _unavailable("get_demand_uplift", [eid], f"no sales history for {case.category}")
    return {
        "tool": "get_demand_uplift", "available": True, "evidence_ids": [eid],
        "metrics": {
            "uplift_pct": _r(s["uplift"] * 100, 1), "uplift_cv": _r(s["cv"]),
            "baseline_weekly_sales_usd": round(s["base_week_total"]), "n_promo_weeks": s["n_promo_weeks"], "n_stores": s["n_stores"],
        },
    }


def get_cost_assumptions(ctx: Ctx, case: CaseInput) -> dict:
    with db.tx() as conn:
        row, eid = _cost_row(conn, ctx, case.category)
    if row is None:
        return _unavailable("get_cost_assumptions", [eid], f"no cost assumptions for {case.category}")
    return {
        "tool": "get_cost_assumptions", "available": True, "evidence_ids": [eid],
        "metrics": {"gross_margin": row["gross_margin"], "unit_price_usd": row["unit_price"], "promo_cost_per_store_week_usd": row["promo_cost_per_store_week"]},
    }


def get_inventory_position(ctx: Ctx, case: CaseInput) -> dict:
    with db.tx() as conn:
        inv, eid = _inventory(conn, ctx, case.category)
    if inv is None:
        return _unavailable("get_inventory_position", [eid], f"no inventory for {case.category}")
    return {
        "tool": "get_inventory_position", "available": True, "evidence_ids": [eid],
        "metrics": {
            "on_hand_units": round(inv["on_hand"]), "inbound_units": round(inv["inbound"]), "capacity_units": round(inv["capacity"]),
            "current_utilization": _r((inv["on_hand"] + inv["inbound"]) / inv["capacity"], 3),
        },
    }


def evaluate_financials(ctx: Ctx, case: CaseInput) -> dict:
    """Finance model: incremental margin of the promotion vs. its cost, using analog uplift."""
    with db.tx() as conn:
        s, e1 = _sales_evidence(conn, ctx, case.category, case.stores)
        cost, e2 = _cost_row(conn, ctx, case.category)
    if s is None or cost is None:
        return _unavailable("evaluate_financials", [e1, e2], "missing sales history or cost assumptions")
    d = case.discount_pct / 100
    m = cost["gross_margin"]
    baseline_sales = s["base_week_total"] * case.weeks
    promo_sales = baseline_sales * (1 + s["uplift"])
    incremental_margin = promo_sales * max(m - d, 0) - baseline_sales * m
    fixed_cost = cost["promo_cost_per_store_week"] * case.stores * case.weeks
    markdown_cost = promo_sales * d
    total_cost = fixed_cost + markdown_cost
    return {
        "tool": "evaluate_financials", "available": True, "evidence_ids": [e1, e2],
        "metrics": {
            "uplift_pct": _r(s["uplift"] * 100, 1), "uplift_cv": _r(s["cv"]),
            "baseline_sales_usd": round(baseline_sales), "promo_sales_usd": round(promo_sales),
            "incremental_margin_usd": round(incremental_margin), "promo_fixed_cost_usd": round(fixed_cost),
            "markdown_cost_usd": round(markdown_cost), "total_cost_usd": round(total_cost),
            "budget_usd": round(case.budget_usd), "net_benefit_usd": round(incremental_margin - fixed_cost),
        },
    }


def check_dc_capacity(ctx: Ctx, case: CaseInput) -> dict:
    """Operations model: can the DCs absorb the promotion's inventory in one delivery, or in phases?"""
    with db.tx() as conn:
        s, e1 = _sales_evidence(conn, ctx, case.category, case.stores)
        cost, e2 = _cost_row(conn, ctx, case.category)
        inv, e3 = _inventory(conn, ctx, case.category)
        phases = int(db.load_config(conn)["ops_phases"])
    if s is None or cost is None or inv is None:
        return _unavailable("check_dc_capacity", [e1, e2, e3], "missing sales, cost or inventory data")
    promo_sales = s["base_week_total"] * case.weeks * (1 + s["uplift"])
    units_required = promo_sales / cost["unit_price"] * 1.10  # 10% safety stock
    held = inv["on_hand"] + inv["inbound"]
    return {
        "tool": "check_dc_capacity", "available": True, "evidence_ids": [e1, e2, e3],
        "metrics": {
            "uplift_pct": _r(s["uplift"] * 100, 1), "units_required": round(units_required),
            "util_now": _r(held / inv["capacity"], 3),
            "util_bulk": _r((held + units_required) / inv["capacity"], 3),
            "util_phased": _r((held + units_required / phases) / inv["capacity"], 3),
            "phases": phases,
        },
    }


def check_legal_rules(ctx: Ctx, case: CaseInput) -> dict:
    """Legal model: which enabled legal rules does this case match? (compliance-risk indicator only)"""
    cat = case.category.lower()
    claims = [c.lower() for c in case.claims]
    with db.tx() as conn:
        rules = conn.execute("SELECT * FROM legal_rules WHERE enabled=1").fetchall()
        matched = []
        for r in rules:
            cats = [x.strip() for x in r["category_match"].split(",") if x.strip()]
            kws = [x.strip() for x in r["keyword_match"].split(",") if x.strip()]
            hit_cat = cat in cats
            hit_kw = [k for k in kws if any(k in c for c in claims)]
            if hit_cat or hit_kw:
                matched.append({"rule_id": r["rule_id"], "name": r["name"], "severity": r["severity"], "why": "category" if hit_cat else f"claim keyword(s): {', '.join(hit_kw)}"})
        eid = _record(
            conn, ctx, "legal_rules",
            f"{len(matched)} legal rule(s) matched" + (": " + "; ".join(m["name"] for m in matched) if matched else ""),
            {"category": case.category, "claims": case.claims, "matched": matched},
        )
    sev_rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    top = max((m["severity"] for m in matched), key=lambda s: sev_rank.get(s, 0), default="NONE")
    return {
        "tool": "check_legal_rules", "available": True, "evidence_ids": [eid],
        "metrics": {"rules_matched": len(matched), "max_severity": top},
        "matched": matched,
    }
