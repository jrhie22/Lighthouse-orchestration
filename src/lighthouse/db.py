"""SQLite access. One file holds everything, which keeps the whole system inspectable with a
single tool (`sqlite3 data/lighthouse.db`). It maps onto the Level-0 DFD like this:

    D1 Pilot Data Store  -> sales_weekly, cost_assumptions, inventory_position (+ data_sources)
    D2 Policy / Risk Rules -> legal_rules, config
    D3 Decision / Audit Log -> agent_outputs, packets, audit_log (hash-chained)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import clock
from .settings import settings

DDL = """
CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, value INTEGER NOT NULL);

-- ---- cases and what agents / Lighthouse produced for them --------------------------------------
CREATE TABLE IF NOT EXISTS cases(
  decision_id TEXT PRIMARY KEY,
  parent_id   TEXT,
  created_at  TEXT NOT NULL,
  opened_by   TEXT NOT NULL,
  mode        TEXT NOT NULL,
  scenario    TEXT,
  input_json  TEXT NOT NULL,
  status      TEXT NOT NULL,
  legal_gate  TEXT NOT NULL DEFAULT 'NOT_REQUIRED'
);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);

CREATE TABLE IF NOT EXISTS evidence(
  evidence_id  TEXT PRIMARY KEY,
  decision_id  TEXT NOT NULL,
  agent        TEXT NOT NULL,
  source       TEXT NOT NULL,
  as_of        TEXT NOT NULL,
  summary      TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_case ON evidence(decision_id);

CREATE TABLE IF NOT EXISTS agent_outputs(
  decision_id       TEXT NOT NULL,
  agent             TEXT NOT NULL,
  timestamp         TEXT NOT NULL,
  status            TEXT NOT NULL,
  recommendation    TEXT,
  confidence        REAL NOT NULL,
  rationale         TEXT NOT NULL,
  conditions_json   TEXT NOT NULL,
  evidence_ids_json TEXT NOT NULL,
  risk_flags_json   TEXT NOT NULL,
  metrics_json      TEXT NOT NULL,
  mode              TEXT NOT NULL,
  PRIMARY KEY(decision_id, agent)
);

CREATE TABLE IF NOT EXISTS packets(
  decision_id TEXT PRIMARY KEY,
  built_at    TEXT NOT NULL,
  packet_ms   REAL NOT NULL,
  json        TEXT NOT NULL
);

-- ---- append-only, hash-chained audit log (OBJ-04) -----------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log(
  seq               INTEGER PRIMARY KEY AUTOINCREMENT,
  ts                TEXT NOT NULL,
  actor             TEXT NOT NULL,
  action            TEXT NOT NULL,
  decision_id       TEXT,
  payload_json      TEXT NOT NULL,
  evidence_ids_json TEXT NOT NULL,
  prev_hash         TEXT NOT NULL,
  hash              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_log(decision_id);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

-- ---- policy / configuration (D2) ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS legal_rules(
  rule_id        TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  category_match TEXT NOT NULL DEFAULT '',   -- comma separated categories
  keyword_match  TEXT NOT NULL DEFAULT '',   -- comma separated keywords searched in marketing claims
  severity       TEXT NOT NULL DEFAULT 'HIGH',
  enabled        INTEGER NOT NULL DEFAULT 1,
  updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS agents_registry(
  name           TEXT PRIMARY KEY,
  enabled        INTEGER NOT NULL DEFAULT 1,
  trigger_type   TEXT NOT NULL,
  refresh_source TEXT,
  last_refresh   TEXT
);

-- ---- pilot data (D1) ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_sources(
  name          TEXT PRIMARY KEY,
  refreshed_at  TEXT NOT NULL,
  max_age_hours REAL NOT NULL,
  origin        TEXT NOT NULL DEFAULT 'generated'
);
CREATE TABLE IF NOT EXISTS sales_weekly(
  store      INTEGER NOT NULL,
  week_date  TEXT NOT NULL,
  category   TEXT NOT NULL,
  sales_usd  REAL NOT NULL,
  promo_flag INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sales_cat ON sales_weekly(category, store);
CREATE TABLE IF NOT EXISTS cost_assumptions(
  category                  TEXT PRIMARY KEY,
  gross_margin              REAL NOT NULL,
  unit_price                REAL NOT NULL,
  promo_cost_per_store_week REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS inventory_position(
  dc_id          TEXT NOT NULL,
  category       TEXT NOT NULL,
  on_hand_units  REAL NOT NULL,
  inbound_units  REAL NOT NULL,
  capacity_units REAL NOT NULL,
  PRIMARY KEY(dc_id, category)
);
"""

DEFAULT_CONFIG = {
    # --- validation ---
    "stale_confidence_multiplier": 0.6,  # stale evidence downgrades confidence (Deliverable 1 risk table)
    "low_confidence_threshold": 0.6,
    # --- escalation thresholds (when a human must give a written reason) ---
    "financial_threshold_usd": 250_000,
    "ops_util_escalate": 0.90,
    # --- rules-agent decision thresholds ---
    "sales_proceed_uplift": 0.10,
    "sales_condition_uplift": 0.05,
    "finance_tight_budget_ratio": 0.90,
    "ops_util_ok": 0.85,
    "ops_phases": 3,
}

DEFAULT_RULES = [
    ("R-HEALTH-CATEGORY", "Health category marketing review", "health & wellness,pharmacy", "", "HIGH"),
    ("R-HEALTH-CLAIMS", "Regulated health claim language", "", "cure,treat,prevent,clinically proven,fda", "HIGH"),
    ("R-PRICE-CLAIMS", "Advertising / price claim language", "", "price match,lowest price,guarantee", "MEDIUM"),
    ("R-RESTRICTED-CATEGORY", "Jurisdiction-restricted category", "alcohol,tobacco", "", "HIGH"),
]

DEFAULT_SOURCES = {  # name: max_age_hours
    "sales_weekly": 192.0,  # weekly data: 8 days
    "cost_assumptions": 24.0 * 35,
    "inventory_position": 36.0,
    "legal_rules": 1e9,  # rules do not go stale
}

DEFAULT_REGISTRY = [
    ("FINANCE", "case-based + nightly refresh", "cost_assumptions"),
    ("SALES", "case-based + daily refresh", "sales_weekly"),
    ("OPERATIONS", "case-based + daily refresh", "inventory_position"),
    ("LEGAL", "case-based only (rule match)", None),
]


def connect() -> sqlite3.Connection:
    path = settings.db_path
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # autocommit; tx() manages BEGIN/COMMIT
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    """Write transaction. BEGIN IMMEDIATE takes the write lock up front, so two requests can never
    interleave inside one transaction - this is what makes the audit hash chain safe."""
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


@contextmanager
def read() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def next_number(conn: sqlite3.Connection, name: str) -> int:
    conn.execute(
        "INSERT INTO counters(name, value) VALUES(?, 1) "
        "ON CONFLICT(name) DO UPDATE SET value = value + 1",
        (name,),
    )
    return conn.execute("SELECT value FROM counters WHERE name=?", (name,)).fetchone()["value"]


def dumps(obj) -> str:
    """Canonical JSON: sorted keys, no spaces. Used for storage *and* hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def init_db() -> None:
    """Create tables and seed defaults. Safe to call on every start."""
    now = clock.iso(clock.now())
    conn = connect()
    try:
        conn.executescript(DDL)  # executescript manages its own transaction
    finally:
        conn.close()
    with tx() as conn:
        for key, value in DEFAULT_CONFIG.items():
            conn.execute("INSERT OR IGNORE INTO config(key, value_json) VALUES(?,?)", (key, json.dumps(value)))
        for rule_id, name, cats, kws, sev in DEFAULT_RULES:
            conn.execute(
                "INSERT OR IGNORE INTO legal_rules(rule_id,name,category_match,keyword_match,severity,enabled,updated_at) "
                "VALUES(?,?,?,?,?,1,?)",
                (rule_id, name, cats, kws, sev, now),
            )
        for name, max_age in DEFAULT_SOURCES.items():
            conn.execute(
                "INSERT OR IGNORE INTO data_sources(name, refreshed_at, max_age_hours) VALUES(?,?,?)",
                (name, now, max_age),
            )
        for name, trigger, source in DEFAULT_REGISTRY:
            conn.execute(
                "INSERT OR IGNORE INTO agents_registry(name, enabled, trigger_type, refresh_source, last_refresh) "
                "VALUES(?,?,?,?,?)",
                (name, 1, trigger, source, now),
            )


def load_config(conn: sqlite3.Connection) -> dict:
    return {r["key"]: json.loads(r["value_json"]) for r in conn.execute("SELECT key, value_json FROM config")}
