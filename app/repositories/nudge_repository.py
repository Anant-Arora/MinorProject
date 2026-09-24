from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.models import (
    NudgeDecision,
    NudgeFeedbackEvent,
)


ALLOWED_EVENT_TYPES = {
    "displayed",
    "opened",
    "clicked",
    "viewed_explanation",
    "snoozed",
    "dismissed",
    "marked_irrelevant",
    "target_action",
}


def save_decision(
    db: Session,
    *,
    decision: Any,
    source_event_id: str | None = None,
    reward_window_days: int = 7,
) -> NudgeDecision:
    """
    Persist one contextual-bandit decision.

    The caller controls commit or rollback so this function does not commit.
    """

    reward_due_at = (
        decision.created_at
        + timedelta(days=reward_window_days)
    )

    row = NudgeDecision(
        id=decision.decision_id,
        user_id=decision.user_id,
        source_event_id=source_event_id,
        selected_arm_id=decision.selected_arm_id,
        selected_arm_name=decision.selected_arm_name,
        propensity=float(decision.propensity),
        context_snapshot=decision.context_snapshot,
        eligible_arms=decision.eligible_arms,
        action_scores=decision.action_scores,
        reason_codes=decision.reason_codes,
        policy_name=decision.policy_name,
        policy_version=decision.policy_version,
        model_version=decision.model_version,
        feature_schema_version=decision.feature_schema_version,
        reward_status="pending",
        reward_due_at=reward_due_at,
    )

    db.add(row)
    db.flush()

    return row


def get_decision(
    db: Session,
    decision_id: str,
) -> NudgeDecision | None:
    return (
        db.query(NudgeDecision)
        .filter(NudgeDecision.id == decision_id)
        .first()
    )


def save_feedback_event(
    db: Session,
    *,
    decision_id: str,
    user_id: str,
    event_id: str,
    event_type: str,
    event_value: dict[str, Any] | None = None,
) -> NudgeFeedbackEvent:
    """
    Store one user feedback event.

    Duplicate event types for the same decision return the existing event.
    """

    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(
            f"Unsupported nudge event type: {event_type}"
        )

    existing = (
        db.query(NudgeFeedbackEvent)
        .filter(
            NudgeFeedbackEvent.decision_id == decision_id,
            NudgeFeedbackEvent.event_type == event_type,
        )
        .first()
    )

    if existing is not None:
        return existing

    event = NudgeFeedbackEvent(
        id=event_id,
        decision_id=decision_id,
        user_id=user_id,
        event_type=event_type,
        event_value=event_value,
    )

    db.add(event)
    db.flush()

    return event


def count_user_nudges_today(
    db: Session,
    user_id: str,
    now: datetime | None = None,
) -> int:
    now = now or datetime.now(timezone.utc)
    start_time = now - timedelta(hours=24)

    return int(
        db.query(func.count(NudgeDecision.id))
        .filter(
            NudgeDecision.user_id == user_id,
            NudgeDecision.created_at >= start_time,
            NudgeDecision.selected_arm_id != 0,
        )
        .scalar()
        or 0
    )


def count_user_nudges_last_7d(
    db: Session,
    user_id: str,
    now: datetime | None = None,
) -> int:
    now = now or datetime.now(timezone.utc)
    start_time = now - timedelta(days=7)

    return int(
        db.query(func.count(NudgeDecision.id))
        .filter(
            NudgeDecision.user_id == user_id,
            NudgeDecision.created_at >= start_time,
            NudgeDecision.selected_arm_id != 0,
        )
        .scalar()
        or 0
    )


def get_recently_dismissed_arm_ids(
    db: Session,
    user_id: str,
    now: datetime | None = None,
    cooldown_days: int = 3,
) -> list[int]:
    """
    Return arms dismissed during the configured cooldown period.
    """

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=cooldown_days)

    rows = (
        db.query(NudgeDecision.selected_arm_id)
        .join(
            NudgeFeedbackEvent,
            NudgeFeedbackEvent.decision_id == NudgeDecision.id,
        )
        .filter(
            NudgeDecision.user_id == user_id,
            NudgeFeedbackEvent.event_type.in_(
                ["dismissed", "marked_irrelevant"]
            ),
            NudgeFeedbackEvent.created_at >= cutoff,
        )
        .distinct()
        .all()
    )

    return [int(row[0]) for row in rows]