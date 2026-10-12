"""Pilot data (D1): load a real CSV if you have one, otherwise generate a seeded stand-in.

Per Deliverable 1, store identifiers, inventory, cost assumptions and legal rules are *generated*
anyway; only demand/sales is meant to be calibrated on the Kaggle retail dataset. This module gives
you both paths:

* ``generate_pilot_data()``  - seeded synthetic weekly sales for 50 stores x 5 categories,
                                plus cost assumptions and two distribution centers.
* ``ingest_sales_csv()``     - load a real weekly-sales CSV (e.g. the Kaggle file) into the same
                                ``sales_weekly`` table. Column names are passed in, because the
                                column layout of the Kaggle file is not assumed anywhere.

Everything written here is labelled in ``data_sources.origin`` so the report can say exactly which
numbers are real and which are generated.
"""
from __future__ import annotations

import csv
import math
import random
from datetime import date, datetime, timedelta

from . import clock, db

ANCHOR = date(2026, 9, 27)  # last week of generated history

# base = typical weekly sales per store (USD); uplift = mean sales lift in promo weeks
CATEGORIES = {
    "Health & Wellness": dict(base=40_000, uplift=0.14, margin=0.32, price=14.0, promo_cost=600),
    "Grocery": dict(base=400_000, uplift=0.06, margin=0.22, price=6.0, promo_cost=900),
    "Electronics": dict(base=120_000, uplift=0.22, margin=0.18, price=85.0, promo_cost=1500),
    "Apparel": dict(base=70_000, uplift=0.18, margin=0.38, price=28.0, promo_cost=700),
    "Home": dict(base=60_000, uplift=0.12, margin=0.30, price=22.0, promo_cost=650),
}
DCS = ("DC-EAST", "DC-WEST")  # "two generated distribution nodes" (Deliverable 1 charter)


def has_data() -> bool:
    with db.read() as conn:
        return conn.execute("SELECT COUNT(*) n FROM sales_weekly").fetchone()["n"] > 0


def generate_pilot_data(seed: int = 7, n_stores: int = 50, n_weeks: int = 104, force: bool = False) -> dict:
    if has_data() and not force:
        return {"skipped": True}
    rng = random.Random(seed)
    store_factor = {s: rng.uniform(0.7, 1.4) for s in range(1, n_stores + 1)}
    sales_rows = []
    for cat, p in CATEGORIES.items():
        for s in range(1, n_stores + 1):
            for k in range(n_weeks):
                week = (ANCHOR - timedelta(weeks=k)).isoformat()
                seasonal = 1 + 0.08 * math.sin(2 * math.pi * k / 52)
                noise = rng.gauss(1.0, 0.04)
                promo = 1 if rng.random() < 0.12 else 0
                lift = max(-0.05, rng.gauss(p["uplift"], 0.05)) if promo else 0.0
                sales_rows.append((s, week, cat, round(p["base"] * store_factor[s] * seasonal * noise * (1 + lift), 2), promo))

    now = clock.iso(clock.now())
    with db.tx() as conn:
        if force:
            for t in ("sales_weekly", "cost_assumptions", "inventory_position"):
                conn.execute(f"DELETE FROM {t}")
        conn.executemany("INSERT INTO sales_weekly VALUES(?,?,?,?,?)", sales_rows)
        for cat, p in CATEGORIES.items():
            conn.execute(
                "INSERT INTO cost_assumptions VALUES(?,?,?,?)", (cat, p["margin"], p["price"], p["promo_cost"])
            )
            ref_units = p["base"] * 50 * 6 / p["price"]  # units a typical 50-store, 6-week promotion needs
            for dc in DCS:
                capacity = ref_units * rng.uniform(1.7, 2.1)
                conn.execute(
                    "INSERT INTO inventory_position VALUES(?,?,?,?,?)",
                    (dc, cat, round(capacity * rng.uniform(0.55, 0.72)), round(capacity * rng.uniform(0.02, 0.06)), round(capacity)),
                )
        for src in ("sales_weekly", "cost_assumptions", "inventory_position"):
            conn.execute("UPDATE data_sources SET refreshed_at=?, origin='generated' WHERE name=?", (now, src))
    return {"sales_rows": len(sales_rows), "categories": list(CATEGORIES), "stores": n_stores, "weeks": n_weeks}


def _parse_date(s: str) -> str:
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {s!r}")


def ingest_sales_csv(
    path: str,
    *,
    store_col: str,
    date_col: str,
    sales_col: str,
    promo_col: str | None = None,
    category: str = "ALL",
    replace_generated: bool = False,
    origin: str | None = None,
) -> dict:
    """Load a weekly-sales CSV into ``sales_weekly``.

    ``promo_col`` is optional. If you pass a holiday flag it is used as the 'promotion' analog -
    say so in the report; it is a proxy, not a real promotion flag.
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            promo = int(float(r[promo_col])) if promo_col else 0
            rows.append((int(float(r[store_col])), _parse_date(r[date_col]), category, float(r[sales_col]), 1 if promo else 0))
    now = clock.iso(clock.now())
    with db.tx() as conn:
        if replace_generated:
            conn.execute("DELETE FROM sales_weekly")
        conn.executemany("INSERT INTO sales_weekly VALUES(?,?,?,?,?)", rows)
        conn.execute(
            "UPDATE data_sources SET refreshed_at=?, origin=? WHERE name='sales_weekly'",
            (now, origin or f"csv:{path}"),
        )
    return {"rows": len(rows), "category": category}


def refresh_sources(names: list[str] | None = None, actor: str = "system") -> list[str]:
    """Simulate the nightly / daily refresh jobs from the agent triggering model: bump refreshed_at."""
    from . import audit  # local import: audit depends on db only, keep module import order simple

    now = clock.iso(clock.now())
    with db.tx() as conn:
        targets = names or [r["name"] for r in conn.execute("SELECT name FROM data_sources")]
        for n in targets:
            conn.execute("UPDATE data_sources SET refreshed_at=? WHERE name=?", (now, n))
            conn.execute("UPDATE agents_registry SET last_refresh=? WHERE refresh_source=?", (now, n))
        audit.append(conn, actor=actor, action="DATA_REFRESHED", decision_id=None, payload={"sources": targets})
    return targets
