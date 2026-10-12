"""How good is the local model, compared with the deterministic rules?  (`lighthouse eval-llm --n 20`)

Runs N random cases through the real pipeline with ``mode="agent"`` and compares each analytic agent's
recommendation with ``metrics.baseline_recommendation`` - the rules agent's answer for the same tools and numbers
(computed on the side as a "shadow baseline"). It reports

  * agreement with the baseline, per agent         (does the model reach the same judgement as the thresholds?)
  * how often the output was INVALID / TIMEOUT      (format failures, hallucinated evidence IDs, too slow)
  * wall-clock seconds per case                     (your OBJ-06 number for the LLM path)

Disagreement is NOT automatically "wrong": the model may weigh a close call differently. Read a few of the
disagreeing cases in the UI (the "Numbers" drawer shows the baseline) before drawing conclusions.
Use a separate database for it:  lighthouse eval-llm --db data/eval.db --n 20
"""
from __future__ import annotations

import asyncio
import random
import time
from collections import Counter, defaultdict

from . import db, decisions, orchestrator, store
from .bulk import random_case
from .schema import AgentName, OutputStatus

ANALYTIC = (AgentName.FINANCE, AgentName.SALES, AgentName.OPERATIONS)


async def _one(case) -> tuple[str, float]:
    did = await asyncio.to_thread(decisions.open_case, case, "agent", None, "eval-llm")
    t0 = time.perf_counter()
    await orchestrator.run_case(did)
    return did, time.perf_counter() - t0


def evaluate(n: int = 20, seed: int = 11, progress=None) -> dict:
    rng = random.Random(seed)
    cases = [random_case(rng, i) for i in range(1, n + 1)]
    ids, secs = [], []

    async def run_all():
        for i, c in enumerate(cases, 1):
            did, s = await _one(c)
            ids.append(did), secs.append(s)
            if progress:
                progress(i, n)

    asyncio.run(run_all())

    agree, total, status = defaultdict(int), defaultdict(int), defaultdict(Counter)
    confusions: list[dict] = []
    with db.read() as conn:
        for did in ids:
            for o in store.load_outputs(conn, did):
                if o.agent not in ANALYTIC:
                    continue
                status[o.agent.value][o.status.value] += 1
                if o.status != OutputStatus.OK:
                    continue
                base = o.metrics.get("baseline_recommendation")
                total[o.agent.value] += 1
                if base == o.recommendation.value:
                    agree[o.agent.value] += 1
                else:
                    confusions.append({"case": did, "agent": o.agent.value, "model": o.recommendation.value, "rules": base})
    secs_sorted = sorted(secs)
    return {
        "cases": n,
        "agreement_with_rules": {a: f"{agree[a]}/{total[a]}" for a in total} | {"all": f"{sum(agree.values())}/{sum(total.values())}"},
        "status_counts": {a: dict(c) for a, c in status.items()},
        "seconds_per_case": {"median": round(secs_sorted[len(secs_sorted) // 2], 1), "p95": round(secs_sorted[int(0.95 * (len(secs_sorted) - 1))], 1), "max": round(secs_sorted[-1], 1)} if secs_sorted else None,
        "disagreements": confusions[:25],
    }
