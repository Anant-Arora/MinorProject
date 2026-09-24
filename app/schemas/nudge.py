from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class NudgeDecisionRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    trigger: str | None = Field(
        default=None,
        max_length=100,
    )

    source_event_id: str | None = Field(
        default=None,
        max_length=128,
    )


class NudgeEventRequest(BaseModel):
    decision_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
    )

    event_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    event_value: dict[str, Any] | None = None


class RenderedNudgeResponse(BaseModel):
    arm_id: int
    arm_name: str
    title: str
    message: str
    reason_codes: list[str]
    cta_label: str | None = None
    facts: dict[str, Any] | None = None


class NudgeDecisionResponse(BaseModel):
    decision_id: str
    user_id: str
    selected_arm_id: int
    selected_arm_name: str
    propensity: float
    eligible_arms: list[int]
    action_scores: dict[str, float]
    context_snapshot: dict[str, Any]
    nudge: RenderedNudgeResponse
    reason_codes: list[str]
    policy_name: str
    policy_version: str
    model_version: str
    feature_schema_version: str
    created_at: datetime


class NudgeEventResponse(BaseModel):
    event_id: str
    status: str