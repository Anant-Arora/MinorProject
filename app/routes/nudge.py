from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import User
from app.ml.nudge.context import NudgeContext
from app.ml.nudge.decision_service import NudgeDecisionService
from app.ml.nudge.linucb import DisjointLinUCB
from app.repositories.nudge_repository import (
    count_user_nudges_last_7d,
    count_user_nudges_today,
    get_decision,
    get_recently_dismissed_arm_ids,
    save_decision,
    save_feedback_event,
)
from app.schemas.nudge import (
    NudgeDecisionRequest,
    NudgeDecisionResponse,
    NudgeEventRequest,
    NudgeEventResponse,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/nudge",
    tags=["Adaptive Nudge Engine"],
)


ARM_NAMES = [
    "no_nudge",
    "savings_milestone",
    "subscription_warning",
    "price_creep_review",
    "impulse_spend_interception",
    "budget_threshold",
    "cashflow_warning",
    "positive_reinforcement",
]


_ENGINE = DisjointLinUCB(
    arm_names=ARM_NAMES,
    feature_dim=10,
    alpha=0.8,
    epsilon=0.10,
)


def get_engine() -> DisjointLinUCB:
    """
    Return the shared in-memory LinUCB engine.

    The current prototype keeps the engine in memory while the FastAPI
    process runs. Database-backed state persistence can be added later.
    """

    return _ENGINE

def ensure_user_exists(
    db: Session,
    user_id: str,
) -> User:
    """
    Return an existing user or create a minimal local user record.

    This supports the current CSV-upload and prototype nudge workflow.
    In a production system, user creation should normally happen during
    authentication or onboarding.
    """

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if user is None:
        user = User(id=user_id)
        db.add(user)
        db.flush()

    return user


def build_context_for_user(
    *,
    user_id: str,
    db: Session,
    nudges_last_7d: int,
) -> NudgeContext:
    """
    Temporary context adapter.

    Replace static values with real features built from:
    - transactions
    - subscription detector
    - price-creep detector
    - anomaly detector
    - spending velocity
    - budgets
    - nudge history
    """

    return NudgeContext(
        spending_velocity=1.0,
        anomaly_count=0,
        subscription_ratio=0.0,
        net_cashflow_health=0.5,
        budget_utilization=0.0,
        days_to_month_end=15,
        price_creep_count=0,
        days_since_last_nudge=30,
        nudges_last_7d=nudges_last_7d,
    )


def get_decision_service(
    engine: DisjointLinUCB = Depends(get_engine),
) -> NudgeDecisionService:
    """Create a request-level decision service using the shared engine."""

    return NudgeDecisionService(
        engine=engine,
        decision_id_factory=lambda: str(uuid4()),
        rng=np.random.default_rng(),
    )


@router.post(
    "/decide",
    response_model=NudgeDecisionResponse,
)
def decide_nudge(
    payload: NudgeDecisionRequest,
    db: Session = Depends(get_db),
    service: NudgeDecisionService = Depends(
        get_decision_service
    ),
):
    """
    Build context, enforce safety constraints, select a nudge,
    persist the decision, and return a deterministic response.
    """

    now = datetime.now(timezone.utc)

    try:
        # Ensure the foreign-key target exists before storing a decision.
        ensure_user_exists(
            db,
            payload.user_id,
        )

        nudges_today = count_user_nudges_today(
            db,
            payload.user_id,
            now,
        )

        nudges_last_7d = count_user_nudges_last_7d(
            db,
            payload.user_id,
            now,
        )

        recently_dismissed_arm_ids = (
            get_recently_dismissed_arm_ids(
                db,
                payload.user_id,
                now,
            )
        )

        context = build_context_for_user(
            user_id=payload.user_id,
            db=db,
            nudges_last_7d=nudges_last_7d,
        )

        decision = service.decide(
            user_id=payload.user_id,
            context=context,
            now=now,
            nudges_today=nudges_today,
            nudges_last_7d=nudges_last_7d,
            recently_dismissed_arms=(
                recently_dismissed_arm_ids
            ),
        )

        save_decision(
            db,
            decision=decision,
            source_event_id=payload.source_event_id,
        )

        # Commit the newly-created user and decision together.
        db.commit()

        return {
            "decision_id": decision.decision_id,
            "user_id": decision.user_id,
            "selected_arm_id": decision.selected_arm_id,
            "selected_arm_name": decision.selected_arm_name,
            "propensity": decision.propensity,
            "eligible_arms": decision.eligible_arms,
            "action_scores": decision.action_scores,
            "context_snapshot": decision.context_snapshot,
            "nudge": {
                "arm_id": decision.nudge.arm_id,
                "arm_name": decision.nudge.arm_name,
                "title": decision.nudge.title,
                "message": decision.nudge.message,
                "reason_codes": decision.nudge.reason_codes,
                "cta_label": decision.nudge.cta_label,
                "facts": decision.nudge.facts,
            },
            "reason_codes": decision.reason_codes,
            "policy_name": decision.policy_name,
            "policy_version": decision.policy_version,
            "model_version": decision.model_version,
            "feature_schema_version": (
                decision.feature_schema_version
            ),
            "created_at": decision.created_at,
        }

    except HTTPException:
        db.rollback()
        raise

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Nudge decision creation failed for user_id=%s",
            payload.user_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to create nudge decision. "
                f"Error type: {type(exc).__name__}"
            ),
        ) from exc

@router.post(
    "/events",
    response_model=NudgeEventResponse,
)
def record_nudge_event(
    payload: NudgeEventRequest,
    db: Session = Depends(get_db),
):
    """
    Store a user feedback event for an existing nudge decision.
    """

    decision = get_decision(
        db,
        payload.decision_id,
    )

    if decision is None:
        raise HTTPException(
            status_code=404,
            detail="Nudge decision not found.",
        )

    try:
        event = save_feedback_event(
            db,
            decision_id=decision.id,
            user_id=decision.user_id,
            event_id=str(uuid4()),
            event_type=payload.event_type,
            event_value=payload.event_value,
        )

        db.commit()

        return {
            "event_id": event.id,
            "status": "recorded",
        }

    except ValueError as exc:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        db.rollback()

        logger.exception(
            "Nudge feedback event creation failed "
            "for decision_id=%s",
            payload.decision_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to record nudge event.",
        )


@router.get("/health")
def nudge_health():
    """Return non-sensitive health and policy metadata."""

    return {
        "status": "healthy",
        "policy_name": "disjoint_linucb",
        "policy_version": "v1",
        "model_version": "nudge-model-v1",
        "feature_schema_version": "1.0",
        "kill_switch": False,
    }