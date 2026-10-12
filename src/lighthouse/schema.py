"""The common contracts. Everything else in the system speaks these types.

Sprint 1 backlog item #1 ("common agent output schema") lives here. The same ``AgentOutput`` model
is (a) the row format in SQLite, (b) the JSON the API returns, and (c) - via ``LLMDecision`` - the
JSON schema we hand to the local LLM so it can only answer in this shape.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AgentName(str, Enum):
    FINANCE = "FINANCE"
    SALES = "SALES"
    OPERATIONS = "OPERATIONS"
    LEGAL = "LEGAL"


REQUIRED_AGENTS = (AgentName.FINANCE, AgentName.SALES, AgentName.OPERATIONS)
ALL_AGENTS = (AgentName.FINANCE, AgentName.SALES, AgentName.OPERATIONS, AgentName.LEGAL)


class Rec(str, Enum):
    """One enum for every agent (the Deliverable 2 deck used 'APPROVE' in one place and
    PROCEED/HOLD in another - pick one vocabulary; this is it)."""

    PROCEED = "PROCEED"
    PROCEED_WITH_CONDITIONS = "PROCEED_WITH_CONDITIONS"
    HOLD = "HOLD"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"  # Legal only: a human must review before any approval


class OutputStatus(str, Enum):
    OK = "OK"
    NOT_TRIGGERED = "NOT_TRIGGERED"  # e.g. Legal runs only when a legal rule matches the case
    TIMEOUT = "TIMEOUT"  # the agent did not answer within its time budget
    INVALID = "INVALID"  # the agent answered, but the answer failed validation
    DISABLED = "DISABLED"  # switched off in the agent registry (Legal disabled -> gate stays closed)


class RiskFlag(str, Enum):
    STALE_EVIDENCE = "STALE_EVIDENCE"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"  # cites an evidence ID that does not exist
    INVALID_RECOMMENDATION = "INVALID_RECOMMENDATION"  # e.g. a non-Legal agent says REVIEW_REQUIRED
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CAPACITY_CONSTRAINT = "CAPACITY_CONSTRAINT"
    MARGIN_PRESSURE = "MARGIN_PRESSURE"
    REGULATED_CLAIM = "REGULATED_CLAIM"
    DATA_GAP = "DATA_GAP"


class CaseInput(BaseModel):
    """What a decision owner types when opening a case (regional promotion / allocation decision)."""

    title: str = Field(min_length=3, max_length=120)
    category: str
    stores: int = Field(ge=1, le=50)
    weeks: int = Field(ge=1, le=26)
    discount_pct: float = Field(default=2.0, ge=0, le=50, description="Promo discount in percent")
    budget_usd: float = Field(gt=0, description="Cost ceiling for the promotion")
    claims: list[str] = Field(default_factory=list, description="Marketing claims to be used")


class AgentOutput(BaseModel):
    """The common output contract (OBJ-01): agent ID, decision ID, timestamp, evidence IDs,
    recommendation, confidence, risk flags - for every agent, every time."""

    decision_id: str
    agent: AgentName
    timestamp: datetime
    status: OutputStatus = OutputStatus.OK
    recommendation: Rec | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    rationale: str = ""
    conditions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    # Numbers are computed by code (tools), never typed in by an LLM.
    metrics: dict[str, float | str] = Field(default_factory=dict)
    mode: str = "rules"  # how it was produced: rules | agent | replay

    @model_validator(mode="after")
    def _ok_needs_recommendation(self) -> "AgentOutput":
        if self.status == OutputStatus.OK and self.recommendation is None:
            raise ValueError("status OK requires a recommendation")
        return self


class LLMDecision(BaseModel):
    """The *only* thing an LLM agent is allowed to produce. This model's JSON schema is passed to
    Ollama's `format=` parameter, so the model cannot answer in any other shape."""

    recommendation: Rec
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(max_length=600)
    conditions: list[str] = Field(default_factory=list, max_length=4)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1, description="Only IDs returned by your tools")


class Evidence(BaseModel):
    evidence_id: str
    decision_id: str
    agent: AgentName
    source: str
    as_of: datetime
    summary: str
    payload: dict = Field(default_factory=dict)


# ----- reconciliation (what Lighthouse itself produces) ------------------------------------------


class AgentView(BaseModel):
    """An agent output *after* validation: the original output is never edited; governance adds
    its own judgement (effective confidence, flags) next to it."""

    output: AgentOutput
    effective_confidence: float
    flags: list[str] = Field(default_factory=list)
    oldest_evidence_age_hours: float | None = None
    usable: bool = True


class ConflictType(str, Enum):
    HARD_DISAGREEMENT = "HARD_DISAGREEMENT"  # someone says HOLD while someone says PROCEED
    CONDITION_TENSION = "CONDITION_TENSION"  # unconditional PROCEED vs PROCEED_WITH_CONDITIONS


class Conflict(BaseModel):
    type: ConflictType
    severity: Literal["HARD", "SOFT"]
    agents: list[AgentName]
    description: str


class Escalation(BaseModel):
    code: str  # LEGAL_GATE | FINANCIAL_THRESHOLD | OPERATIONAL_THRESHOLD | LOW_CONFIDENCE | ...
    detail: str


class GateStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"  # Legal said REVIEW_REQUIRED, no human legal disposition yet -> approve is locked
    CLEARED = "CLEARED"
    BLOCKED = "BLOCKED"


class Packet(BaseModel):
    """The recommendation packet: one recommendation + every dissent, reason, and open risk."""

    decision_id: str
    built_at: datetime
    recommendation: Rec  # PROCEED | PROCEED_WITH_CONDITIONS | HOLD  (never REVIEW_REQUIRED)
    summary: str
    confidence_floor: float | None = None  # the *lowest* usable confidence - never an average
    agents: list[AgentView]
    conflicts: list[Conflict] = Field(default_factory=list)
    escalations: list[Escalation] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)
    legal_gate: GateStatus = GateStatus.NOT_REQUIRED
    requires_reason: bool = False
    incomplete: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    packet_ms: float = 0.0
    # Filled in when the packet is read (they change after the packet is built):
    status: str = "READY_FOR_DECISION"


# ----- human actions ----------------------------------------------------------------------------


class DecisionAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REVISE = "REVISE"


class DecisionRequest(BaseModel):
    action: DecisionAction
    reason: str = ""
    revised_input: dict | None = Field(default=None, description="Fields to change when action=REVISE")


class LegalDisposition(str, Enum):
    CLEARED = "CLEARED"
    CLEARED_WITH_EDITS = "CLEARED_WITH_EDITS"
    BLOCKED = "BLOCKED"


class LegalDispositionRequest(BaseModel):
    disposition: LegalDisposition
    comment: str = Field(min_length=10)


class OpenCaseRequest(BaseModel):
    input: CaseInput | None = None  # required unless scenario is given
    mode: Literal["rules", "agent", "replay"] = "rules"
    scenario: str | None = None


class Status:
    """Case lifecycle (plain strings keep the SQLite rows readable)."""

    RUNNING = "RUNNING"
    AWAITING_LEGAL = "AWAITING_LEGAL"  # packet ready, legal gate pending
    READY_FOR_DECISION = "READY_FOR_DECISION"
    LEGAL_BLOCKED = "LEGAL_BLOCKED"  # legal reviewer blocked it: only REJECT / REVISE remain
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVISION_REQUESTED = "REVISION_REQUESTED"
    FAILED = "FAILED"

    OPEN = (AWAITING_LEGAL, READY_FOR_DECISION, LEGAL_BLOCKED)
    FINAL = (APPROVED, REJECTED, REVISION_REQUESTED)
