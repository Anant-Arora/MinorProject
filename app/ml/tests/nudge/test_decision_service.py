from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from app.ml.nudge.context import NudgeContext
from app.ml.nudge.decision_service import (
    NudgeDecisionService,
)
from app.ml.nudge.linucb import DisjointLinUCB


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


def build_context() -> NudgeContext:
    return NudgeContext(
        spending_velocity=1.5,
        anomaly_count=2,
        subscription_ratio=0.25,
        net_cashflow_health=0.40,
        budget_utilization=0.85,
        days_to_month_end=10,
        price_creep_count=1,
        days_since_last_nudge=10,
        nudges_last_7d=0,
    )


def build_service() -> NudgeDecisionService:
    engine = DisjointLinUCB(
        arm_names=ARM_NAMES,
        feature_dim=10,
        alpha=0.8,
        epsilon=0.10,
    )

    return NudgeDecisionService(
        engine,
        decision_id_factory=lambda: str(uuid4()),
        rng=np.random.default_rng(42),
    )


def test_decision_returns_valid_eligible_arm():
    service = build_service()

    decision = service.decide(
        user_id="test_user",
        context=build_context(),
        now=datetime(
            2026,
            9,
            22,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert decision.selected_arm_id in decision.eligible_arms
    assert decision.selected_arm_name
    assert 0.0 < decision.propensity <= 1.0
    assert decision.nudge.message
    assert decision.context_snapshot["schema_version"] == "1.0"
    assert str(decision.selected_arm_id) in decision.action_scores


def test_daily_safety_limit_returns_no_nudge():
    service = build_service()

    decision = service.decide(
        user_id="test_user",
        context=build_context(),
        nudges_today=1,
        now=datetime(
            2026,
            9,
            22,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert decision.selected_arm_name == "no_nudge"
    assert decision.selected_arm_id == 0
    assert decision.propensity == 1.0
    assert decision.eligible_arms == [0]
    assert "daily_limit_reached" in decision.reason_codes


def test_weekly_safety_limit_returns_no_nudge():
    service = build_service()

    decision = service.decide(
        user_id="test_user",
        context=build_context(),
        nudges_last_7d=3,
        now=datetime(
            2026,
            9,
            22,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert decision.selected_arm_name == "no_nudge"
    assert decision.propensity == 1.0
    assert "weekly_limit_reached" in decision.reason_codes


def test_disabled_subscription_arm_is_not_eligible():
    service = build_service()

    decision = service.decide(
        user_id="test_user",
        context=build_context(),
        disabled_arms=[2],
        now=datetime(
            2026,
            9,
            22,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert 2 not in decision.eligible_arms