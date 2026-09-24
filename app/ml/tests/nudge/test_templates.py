from app.ml.nudge import NudgeArm
from app.ml.nudge.context import NudgeContext
from app.ml.nudge.templates import render_nudge


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


def test_subscription_warning_template():
    result = render_nudge(
        arm_id=int(NudgeArm.SUBSCRIPTION_WARNING),
        context=build_context(),
        reason_codes=[],
    )

    assert result.arm_name == "subscription_warning"
    assert "25%" in result.message
    assert result.cta_label == "Review payments"


def test_budget_threshold_template():
    result = render_nudge(
        arm_id=int(NudgeArm.BUDGET_THRESHOLD),
        context=build_context(),
        reason_codes=[],
    )

    assert result.arm_name == "budget_threshold"
    assert "85%" in result.message
    assert result.cta_label == "Review budget"


def test_no_nudge_template():
    result = render_nudge(
        arm_id=int(NudgeArm.NO_NUDGE),
        context=build_context(),
        reason_codes=["daily_limit_reached"],
    )

    assert result.arm_name == "no_nudge"
    assert result.reason_codes == ["daily_limit_reached"]