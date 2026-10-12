"""Fork -> run four agents in parallel -> join -> validate -> reconcile -> packet.

This file is the executable version of the UML activity diagram:

    open case --fork--> [Finance] [Sales] [Operations] [Legal] --join--> validate --> conflicts
                    --> legal gate --> synthesize --> packet --> (human decides, see decisions.py)

Concurrency model (the part worth understanding):
  * the API process runs ONE asyncio event loop;
  * each agent is a *blocking* function (SQL queries, or an HTTP call to Ollama), so it runs in a
    worker thread via ``asyncio.to_thread``;
  * ``asyncio.wait_for`` puts a time budget on each; on timeout we record status=TIMEOUT and move on
    (the worker thread cannot be killed - its late result is simply ignored);
  * each finished agent is validated and written to SQLite + the audit log immediately, and an event
    is pushed to ``bus`` so the browser (Server-Sent Events) sees progress live.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Callable

from . import audit, clock, db, reconcile, store, validate
from .agents import llm_agent, rules
from .llm import LLMClient, default_client
from .schema import ALL_AGENTS, AgentName, AgentOutput, AgentView, CaseInput, OutputStatus, Status
from .settings import settings


class EventBus:
    """In-memory progress events per case, replayed to late subscribers. (Durable history is the audit log.)"""

    def __init__(self) -> None:
        self._events: dict[str, list[dict]] = defaultdict(list)

    def publish(self, decision_id: str, type_: str, data: dict | None = None) -> None:
        self._events[decision_id].append({"type": type_, "data": data or {}, "ts": clock.iso(clock.now())})

    def since(self, decision_id: str, index: int) -> list[dict]:
        return self._events.get(decision_id, [])[index:]

    def known(self, decision_id: str) -> bool:
        return decision_id in self._events


bus = EventBus()

_llm: LLMClient | None = None
_llm_factory: Callable[[], LLMClient] = default_client


def set_llm_factory(factory: Callable[[], LLMClient]) -> None:
    """Tests (or you) can plug in a different model client."""
    global _llm, _llm_factory
    _llm, _llm_factory = None, factory


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = _llm_factory()
    return _llm


# ---- one agent ------------------------------------------------------------------------------------


def _produce(agent: AgentName, decision_id: str, case: CaseInput, mode: str, scenario: str | None) -> AgentOutput:
    """Runs in a worker thread. Returns the agent's raw output (not yet validated)."""
    if mode == "rules":
        return rules.run(agent, decision_id, case)
    if mode == "agent":
        return llm_agent.run(agent, decision_id, case, get_llm())
    if mode == "replay":
        from . import scenarios  # local import: scenarios is optional demo/test material

        return scenarios.produce(scenario or "", agent, decision_id)
    raise ValueError(f"unknown mode {mode!r}")


def _placeholder(decision_id: str, agent: AgentName, status: OutputStatus, why: str, mode: str) -> AgentOutput:
    return AgentOutput(decision_id=decision_id, agent=agent, timestamp=clock.now(), status=status, rationale=why, mode=mode)


def _validate_and_store(out: AgentOutput) -> AgentView:
    """Validate, then persist the output and its audit entry in ONE transaction."""
    with db.tx() as conn:
        cfg = db.load_config(conn)
        checked, view = validate.check(conn, out, cfg, clock.now())
        store.save_output(conn, checked)
        audit.append(
            conn, actor=f"agent:{checked.agent.value}", action="AGENT_OUTPUT_RECORDED", decision_id=checked.decision_id,
            payload={
                "agent": checked.agent.value, "status": checked.status.value,
                "recommendation": checked.recommendation.value if checked.recommendation else None,
                "confidence": checked.confidence, "risk_flags": [f.value for f in checked.risk_flags],
                "mode": checked.mode, "governance_flags": view.flags,
            },
            evidence_ids=checked.evidence_ids,
        )
    return view


async def _run_agent(decision_id: str, agent: AgentName, case: CaseInput, mode: str, scenario: str | None,
                     enabled: bool, timeout: float) -> AgentView:
    bus.publish(decision_id, "agent_started", {"agent": agent.value})
    if not enabled:
        out = _placeholder(decision_id, agent, OutputStatus.DISABLED, "Disabled in the agent registry.", mode)
    else:
        try:
            out = await asyncio.wait_for(asyncio.to_thread(_produce, agent, decision_id, case, mode, scenario), timeout)
        except asyncio.TimeoutError:
            out = _placeholder(decision_id, agent, OutputStatus.TIMEOUT, f"No answer within {timeout:g}s.", mode)
        except Exception as e:  # a crashing agent must not take the whole case down
            out = _placeholder(decision_id, agent, OutputStatus.INVALID, f"Agent error: {type(e).__name__}: {e}", mode)
    view = await asyncio.to_thread(_validate_and_store, out)
    o = view.output
    bus.publish(decision_id, "agent_done", {
        "agent": agent.value, "status": o.status.value,
        "recommendation": o.recommendation.value if o.recommendation else None,
        "confidence": view.effective_confidence, "flags": view.flags,
    })
    return view


# ---- the whole case -------------------------------------------------------------------------------


def _finalize(decision_id: str, case: CaseInput, views: list[AgentView]):
    """Join step: build the packet, persist it, move the case to its next state, audit it."""
    with db.tx() as conn:
        cfg = db.load_config(conn)
        t0 = time.perf_counter()
        packet = reconcile.build_packet(decision_id, case, views, cfg, clock.now())
        packet.packet_ms = round((time.perf_counter() - t0) * 1000, 3)  # OBJ-06: time to assemble the packet
        store.save_packet(conn, packet)
        status = Status.AWAITING_LEGAL if packet.legal_gate.value == "PENDING" else Status.READY_FOR_DECISION
        store.set_status(conn, decision_id, status, legal_gate=packet.legal_gate.value)
        audit.append(
            conn, actor="lighthouse", action="PACKET_CREATED", decision_id=decision_id,
            payload={
                "recommendation": packet.recommendation.value, "legal_gate": packet.legal_gate.value,
                "conflicts": [c.type.value for c in packet.conflicts], "escalations": [e.code for e in packet.escalations],
                "incomplete": packet.incomplete, "confidence_floor": packet.confidence_floor, "packet_ms": packet.packet_ms,
            },
            evidence_ids=packet.evidence_ids,
        )
    return packet


async def run_case(decision_id: str):
    """Run the full pipeline for an already-opened case. Returns the Packet (or None if it failed)."""
    with db.read() as conn:
        case = store.get_case(conn, decision_id)
        registry = {r["name"]: bool(r["enabled"]) for r in conn.execute("SELECT name, enabled FROM agents_registry")}
    mode, scenario = case["mode"], case["scenario"]
    timeout = settings.llm_agent_timeout_s if mode == "agent" else settings.agent_timeout_s
    bus.publish(decision_id, "case_opened", {"mode": mode, "title": case["input"].title})
    try:
        views = await asyncio.gather(*[
            _run_agent(decision_id, a, case["input"], mode, scenario, registry.get(a.value, True), timeout)
            for a in ALL_AGENTS
        ])
        packet = await asyncio.to_thread(_finalize, decision_id, case["input"], list(views))
    except Exception as e:
        def _fail():
            with db.tx() as conn:
                store.set_status(conn, decision_id, Status.FAILED)
                audit.append(conn, actor="lighthouse", action="CASE_FAILED", decision_id=decision_id, payload={"error": f"{type(e).__name__}: {e}"})
        await asyncio.to_thread(_fail)
        bus.publish(decision_id, "error", {"message": str(e)})
        return None
    bus.publish(decision_id, "packet_ready", {"recommendation": packet.recommendation.value, "legal_gate": packet.legal_gate.value})
    bus.publish(decision_id, "done", {})
    return packet
