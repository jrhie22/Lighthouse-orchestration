"""HTTP API. This is the *only* surface the front end talks to - which is what lets you start with a
one-page HTML file today and swap in React later without touching anything behind this file.

Roles: there is no login in the prototype. Send ``X-Role`` and ``X-Actor`` headers (the UI's "viewing as"
dropdown does this). Reads are open to every role; each write is limited to the role that owns it.

    decision_owner   open cases, record decisions
    legal_reviewer   legal dispositions
    admin            rules, config, agent registry, data refresh
    finance_reviewer / sales_reviewer / operations_reviewer   read-only (they review the packet)

Interactive docs (Swagger UI) are served at /docs - a free API explorer, handy while you learn the system.
"""
from __future__ import annotations

import asyncio
import json
import statistics
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import audit, data, db, decisions, orchestrator, scenarios, store
from .decisions import DomainError
from .llm import OllamaClient
from .schema import AgentName, DecisionRequest, LegalDispositionRequest, OpenCaseRequest, Packet, Status
from .settings import settings

STATIC_DIR = Path(__file__).parent / "static"
ROLES = {"decision_owner", "legal_reviewer", "finance_reviewer", "sales_reviewer", "operations_reviewer", "admin"}
_tasks: set[asyncio.Task] = set()  # keep references so background runs are not garbage-collected


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    if not data.has_data():
        data.generate_pilot_data()
    yield


app = FastAPI(title="Lighthouse", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(DomainError)
async def _domain_error(_: Request, e: DomainError):
    return JSONResponse(status_code=e.http_status, content={"code": e.code, "message": e.message})


# ---- identity (no auth: a header says who you are) --------------------------------------------------


class Principal(BaseModel):
    role: str
    actor: str


def principal(x_role: str = Header(default="decision_owner"), x_actor: str = Header(default="demo-user")) -> Principal:
    if x_role not in ROLES:
        raise HTTPException(400, f"Unknown role {x_role!r}; use one of {sorted(ROLES)}")
    return Principal(role=x_role, actor=x_actor)


def require(*roles: str):
    def dep(p: Principal = Depends(principal)) -> Principal:
        if p.role not in roles:
            raise HTTPException(403, f"Role '{p.role}' may not do this; allowed: {', '.join(roles)}")
        return p

    return dep


def _start(decision_id: str) -> None:
    task = asyncio.create_task(orchestrator.run_case(decision_id))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


# ---- pages / health ------------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    try:
        c = OllamaClient()
        ollama = {"available": c.available(), "model": c.model, "host": settings.ollama_host}
    except Exception as e:  # package missing or server down
        ollama = {"available": False, "error": f"{type(e).__name__}"}
    return {"ok": True, "ollama": ollama}


@app.get("/scenarios")
def list_scenarios():
    return scenarios.list_scenarios()


# ---- cases -----------------------------------------------------------------------------------------


@app.post("/cases", status_code=202)
async def open_case(req: OpenCaseRequest, response: Response,
                    wait: bool = Query(False, description="Block until the packet is built (returns 201 with the packet)"),
                    who: Principal = Depends(require("decision_owner", "admin"))):
    if req.scenario:
        sc = scenarios.SCENARIOS.get(req.scenario)
        if sc is None:
            raise HTTPException(404, f"Unknown scenario {req.scenario!r}")
        case_input, mode = sc.case, sc.mode
    else:
        if req.input is None:
            raise HTTPException(422, "Provide `input` or `scenario`")
        if req.mode == "replay":
            raise HTTPException(422, "mode=replay needs a scenario")
        case_input, mode = req.input, req.mode
    if mode == "agent" and not OllamaClient().available():
        raise HTTPException(503, f"Ollama model '{settings.ollama_model}' is not reachable at {settings.ollama_host}. "
                                 "Start Ollama and pull the model, or use mode=rules.")
    decision_id = await asyncio.to_thread(decisions.open_case, case_input, mode, req.scenario, who.actor)
    if wait:
        await orchestrator.run_case(decision_id)
        response.status_code = 201
        return await asyncio.to_thread(_case_with_packet, decision_id)
    _start(decision_id)
    return {"decision_id": decision_id, "status": Status.RUNNING, "events": f"/cases/{decision_id}/events"}


def _case_with_packet(decision_id: str) -> dict:
    with db.read() as conn:
        case = store.get_case(conn, decision_id)
        packet = store.load_packet(conn, decision_id)
    return {"decision_id": decision_id, "status": case["status"], "packet": packet.model_dump(mode="json") if packet else None}


@app.get("/cases")
def list_cases(status: str | None = None, limit: int = Query(50, le=500)):
    with db.read() as conn:
        return store.list_cases(conn, limit, status)


@app.get("/cases/{decision_id}")
def get_case(decision_id: str):
    with db.read() as conn:
        case = store.get_case(conn, decision_id)
        if case is None:
            raise HTTPException(404, "No such case")
        children = [r["decision_id"] for r in conn.execute("SELECT decision_id FROM cases WHERE parent_id=? ORDER BY decision_id", (decision_id,))]
    return {**case, "input": case["input"].model_dump(), "revisions": children}


@app.get("/cases/{decision_id}/packet", response_model=Packet)
def get_packet(decision_id: str):
    with db.read() as conn:
        case = store.get_case(conn, decision_id)
        if case is None:
            raise HTTPException(404, "No such case")
        packet = store.load_packet(conn, decision_id)
    if packet is None:
        raise HTTPException(409, f"Packet not ready (case is {case['status']})")
    return packet


@app.get("/cases/{decision_id}/events")
async def case_events(decision_id: str, request: Request):
    """Server-Sent Events: one event per step of the pipeline (agent_started, agent_done, packet_ready, done)."""

    async def gen():
        idx = 0
        while True:
            if await request.is_disconnected():
                return
            for ev in orchestrator.bus.since(decision_id, idx):
                idx += 1
                yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                if ev["type"] in ("done", "error"):
                    return
            if not orchestrator.bus.known(decision_id):  # e.g. a case from before a server restart
                with db.read() as conn:
                    case = store.get_case(conn, decision_id)
                if case is None:
                    yield 'event: error\ndata: {"type":"error","data":{"message":"no such case"}}\n\n'
                    return
                if case["status"] != Status.RUNNING:
                    yield 'event: done\ndata: {"type":"done","data":{}}\n\n'
                    return
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/cases/{decision_id}/legal-disposition")
def legal_disposition(decision_id: str, req: LegalDispositionRequest, who: Principal = Depends(require("legal_reviewer"))):
    return decisions.legal_disposition(decision_id, req, who.actor)


@app.post("/cases/{decision_id}/decision")
async def decide(decision_id: str, req: DecisionRequest, who: Principal = Depends(require("decision_owner"))):
    result = await asyncio.to_thread(decisions.record_decision, decision_id, req, who.actor)
    if result["revision_case_id"]:
        _start(result["revision_case_id"])  # loop back: the revised case goes through the whole pipeline again
    return result


# ---- evidence & audit ------------------------------------------------------------------------------


@app.get("/evidence/{evidence_id}")
def get_evidence(evidence_id: str):
    with db.read() as conn:
        ev = store.get_evidence(conn, evidence_id)
    if ev is None:
        raise HTTPException(404, "No such evidence")
    return ev


@app.get("/audit")
def get_audit(decision_id: str | None = None, limit: int = Query(200, le=2000)):
    with db.read() as conn:
        return audit.entries(conn, decision_id, limit)


@app.get("/audit/verify")
def verify_audit():
    with db.read() as conn:
        return audit.verify(conn)


# ---- admin -------------------------------------------------------------------------------------------


class RuleBody(BaseModel):
    name: str
    category_match: str = ""
    keyword_match: str = ""
    severity: str = "HIGH"
    enabled: bool = True


@app.get("/admin/rules")
def get_rules():
    with db.read() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM legal_rules ORDER BY rule_id")]


@app.put("/admin/rules/{rule_id}")
def put_rule(rule_id: str, body: RuleBody, who: Principal = Depends(require("admin"))):
    from . import clock

    now = clock.iso(clock.now())
    with db.tx() as conn:
        before = conn.execute("SELECT * FROM legal_rules WHERE rule_id=?", (rule_id,)).fetchone()
        conn.execute(
            "INSERT INTO legal_rules(rule_id,name,category_match,keyword_match,severity,enabled,updated_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(rule_id) DO UPDATE SET name=excluded.name, category_match=excluded.category_match, "
            "keyword_match=excluded.keyword_match, severity=excluded.severity, enabled=excluded.enabled, updated_at=excluded.updated_at",
            (rule_id, body.name, body.category_match.lower(), body.keyword_match.lower(), body.severity, int(body.enabled), now),
        )
        conn.execute("UPDATE data_sources SET refreshed_at=? WHERE name='legal_rules'", (now,))
        audit.append(conn, actor=who.actor, action="RULE_CHANGED", decision_id=None,
                     payload={"rule_id": rule_id, "before": dict(before) if before else None, "after": body.model_dump()})
    return {"rule_id": rule_id, **body.model_dump()}


@app.get("/admin/config")
def get_config():
    with db.read() as conn:
        return db.load_config(conn)


@app.put("/admin/config")
def put_config(changes: dict[str, float], who: Principal = Depends(require("admin"))):
    with db.tx() as conn:
        current = db.load_config(conn)
        unknown = [k for k in changes if k not in current]
        if unknown:
            raise HTTPException(422, f"Unknown config keys: {unknown}")
        for k, v in changes.items():
            conn.execute("UPDATE config SET value_json=? WHERE key=?", (json.dumps(v), k))
        audit.append(conn, actor=who.actor, action="CONFIG_CHANGED", decision_id=None,
                     payload={"before": {k: current[k] for k in changes}, "after": changes})
    return {**current, **changes}


class AgentToggle(BaseModel):
    enabled: bool


@app.get("/admin/agents")
def get_agents():
    with db.read() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM agents_registry ORDER BY name")]


@app.put("/admin/agents/{name}")
def put_agent(name: AgentName, body: AgentToggle, who: Principal = Depends(require("admin"))):
    with db.tx() as conn:
        conn.execute("UPDATE agents_registry SET enabled=? WHERE name=?", (int(body.enabled), name.value))
        audit.append(conn, actor=who.actor, action="AGENT_REGISTRY_CHANGED", decision_id=None,
                     payload={"agent": name.value, "enabled": body.enabled})
    return {"name": name.value, "enabled": body.enabled}


@app.post("/admin/refresh")
def refresh(who: Principal = Depends(require("admin"))):
    return {"refreshed": data.refresh_sources(actor=who.actor)}


@app.get("/admin/data-sources")
def data_sources():
    with db.read() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM data_sources ORDER BY name")]


@app.get("/admin/stats")
def stats():
    """Numbers for the report: OBJ-05 (volume) and OBJ-06 (packet time)."""
    with db.read() as conn:
        n = lambda t: conn.execute(f"SELECT COUNT(*) n FROM {t}").fetchone()["n"]
        ms = sorted(r["packet_ms"] for r in conn.execute("SELECT packet_ms FROM packets"))
        by_status = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) n FROM cases GROUP BY status")}
        counts = {"cases": n("cases"), "agent_outputs": n("agent_outputs"), "packets": n("packets"), "audit_entries": n("audit_log")}
    return {
        **counts,
        "cases_by_status": by_status,
        "packet_ms": {"p50": statistics.median(ms), "p95": ms[int(0.95 * (len(ms) - 1))], "max": ms[-1]} if ms else None,
    }
