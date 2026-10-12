"""LLM-mode agents: a model that *chooses which tools to call* and then makes a judgement.

This is the part that is an "agent" in the strict sense (the model directs its own tool use):

    1. TOOL LOOP    model <-> tools, up to ``settings.max_tool_calls`` calls
    2. FINAL ANSWER one more call constrained by ``LLMDecision``'s JSON schema (structured output)

Guardrails that keep the model honest (these are what you explain in the report):
  * Numbers are NEVER taken from the model. ``metrics`` is copied from tool results.
  * The model must cite evidence IDs; ``validate.py`` rejects IDs that were never issued to this
    agent for this case (status=INVALID, flag UNSUPPORTED_CLAIM).
  * The deterministic rules decision is computed on the side ("shadow baseline") and stored in
    ``metrics.baseline_recommendation`` so you can show where the LLM and the rules disagree.
  * Legal is special: whether the legal gate is required is decided by rules; the model may only
    draft the explanation. A model can never talk its way out of REVIEW_REQUIRED.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from .. import clock, tools
from ..llm import LLMClient
from ..schema import AgentName, AgentOutput, CaseInput, LLMDecision, OutputStatus
from ..settings import settings
from ..tools import Ctx
from . import rules

SYSTEM = (
    "You are the {role} agent in a retail decision-support system. A human makes the final decision; "
    "you only recommend.\n"
    "{focus}\n"
    "Call your tools to gather evidence before answering. Never invent numbers or evidence IDs: every "
    "evidence_id you cite must appear in a tool result.\n"
    "Recommendation is one of PROCEED, PROCEED_WITH_CONDITIONS (list the conditions), or HOLD.\n"
    "Confidence is between 0 and 1; use a lower value when the evidence is noisy or close to a limit."
)

FOCUS = {
    AgentName.SALES: "Judge whether the expected demand uplift justifies running the promotion. Start with get_demand_uplift.",
    AgentName.FINANCE: "Judge whether the promotion pays for itself and stays within the cost ceiling. Start with evaluate_financials.",
    AgentName.OPERATIONS: "Judge whether the distribution centers can execute the promotion, in one delivery or in phases. Start with check_dc_capacity.",
}

FINAL_PROMPT = (
    "Now give your final decision as JSON matching the required schema. "
    "Cite only evidence IDs that appeared in your tool results."
)


def make_toolbox(agent: AgentName, ctx: Ctx, case: CaseInput, log: list[dict]) -> list:
    """Zero-argument tools bound to this case. (The harness knows the case; the model only decides
    *which* tool to call and *when*. Zero arguments also removes a whole class of small-model errors.)"""

    def _call(fn):
        result = fn(ctx, case)
        log.append(result)
        return result

    def get_demand_uplift() -> dict:
        """Estimate sales uplift (percent) from past promotion weeks for this case's category and stores."""
        return _call(tools.get_demand_uplift)

    def get_cost_assumptions() -> dict:
        """Look up gross margin, unit price and promotion cost assumptions for this case's category."""
        return _call(tools.get_cost_assumptions)

    def get_inventory_position() -> dict:
        """Look up current distribution-center inventory and capacity for this case's category."""
        return _call(tools.get_inventory_position)

    def evaluate_financials() -> dict:
        """Run the finance model: incremental margin, total promotion cost and net benefit versus the budget."""
        return _call(tools.evaluate_financials)

    def check_dc_capacity() -> dict:
        """Run the capacity model: DC utilization after a single bulk delivery versus phased deliveries."""
        return _call(tools.check_dc_capacity)

    box = {
        AgentName.SALES: [get_demand_uplift],
        AgentName.FINANCE: [evaluate_financials, get_cost_assumptions, get_demand_uplift],
        AgentName.OPERATIONS: [check_dc_capacity, get_inventory_position, get_demand_uplift],
    }
    return box[agent]


def _case_prompt(case: CaseInput) -> str:
    return "Decision case:\n" + json.dumps(case.model_dump(), indent=2)


def _run_legal(decision_id: str, case: CaseInput, llm: LLMClient, record: bool) -> AgentOutput:
    base = rules.run(AgentName.LEGAL, decision_id, case, record)
    if base.status != OutputStatus.OK:
        return base  # rules say no legal rule applies -> the model is not even asked
    try:
        resp = llm.chat([
            {"role": "system", "content": "You are a compliance analyst. Write two plain sentences explaining why a human legal review is needed. Do not give legal advice."},
            {"role": "user", "content": f"{base.rationale}\nMarketing claims: {case.claims or 'none'}\nCategory: {case.category}"},
        ])
        text = resp.content.strip()
        if text:
            return base.model_copy(update={"rationale": text[:450] + " (Compliance-risk indicator, not legal advice.)", "mode": "agent"})
    except Exception:
        pass  # the rule-based explanation is already a valid output
    return base


def run(agent: AgentName, decision_id: str, case: CaseInput, llm: LLMClient, record: bool = True) -> AgentOutput:
    if agent == AgentName.LEGAL:
        return _run_legal(decision_id, case, llm, record)

    ctx = Ctx(decision_id, agent, record)
    log: list[dict] = []
    toolbox = make_toolbox(agent, ctx, case, log)
    by_name = {f.__name__: f for f in toolbox}
    messages: list = [
        {"role": "system", "content": SYSTEM.format(role=agent.value.title(), focus=FOCUS[agent])},
        {"role": "user", "content": _case_prompt(case)},
    ]

    # ---- phase 1: tool loop -------------------------------------------------------------------
    n_calls, nudged = 0, False
    while n_calls < settings.max_tool_calls:
        resp = llm.chat(messages, tools=toolbox)
        messages.append(resp.raw_message)
        if not resp.tool_calls:
            if n_calls == 0 and not nudged:  # small models sometimes answer without looking anything up
                nudged = True
                messages.append({"role": "user", "content": "Use your tools to look up evidence first."})
                continue
            break
        for call in resp.tool_calls:
            n_calls += 1
            fn = by_name.get(call.name)
            try:
                result = fn() if fn else {"error": f"unknown tool '{call.name}'"}
            except Exception as e:  # tool failures are information for the model, not crashes
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_name": call.name, "content": json.dumps(result)})

    # ---- phase 2: constrained final answer (one retry if the JSON does not validate) ----------
    messages.append({"role": "user", "content": FINAL_PROMPT})
    decision: LLMDecision | None = None
    last_error = ""
    for _ in range(2):
        final = llm.chat(messages, format=LLMDecision.model_json_schema())
        try:
            decision = LLMDecision.model_validate_json(final.content)
            break
        except ValidationError as e:
            last_error = str(e)
            messages.append(final.raw_message)
            messages.append({"role": "user", "content": f"That did not match the schema ({e.error_count()} errors). Answer again with valid JSON."})
    if decision is None:
        return AgentOutput(
            decision_id=decision_id, agent=agent, timestamp=clock.now(), status=OutputStatus.INVALID,
            rationale=f"Model did not return a valid decision: {last_error[:200]}", mode="agent",
        )

    metrics: dict = {}
    for r in log:
        metrics.update(r.get("metrics", {}))  # numbers come from code, not from the model
    baseline = rules.run(agent, decision_id, case, record=False)
    metrics["baseline_recommendation"] = baseline.recommendation.value if baseline.recommendation else baseline.status.value

    return AgentOutput(
        decision_id=decision_id, agent=agent, timestamp=clock.now(), status=OutputStatus.OK,
        recommendation=decision.recommendation, confidence=decision.confidence, rationale=decision.rationale,
        conditions=decision.conditions, evidence_ids=decision.evidence_ids, risk_flags=decision.risk_flags,
        metrics=metrics, mode="agent",
    )
