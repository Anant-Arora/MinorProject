from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ml.nudge import NudgeArm


@dataclass(frozen=True)
class RenderedNudge:
    arm_id: int
    arm_name: str
    title: str
    message: str
    reason_codes: list[str]
    cta_label: str | None = None
    facts: dict[str, Any] | None = None


def _percent(value: float) -> str:
    return f"{value:.0%}"


def render_nudge(
    *,
    arm_id: int,
    context: Any,
    reason_codes: list[str],
) -> RenderedNudge:
    """
    Render a deterministic, template-based nudge.

    Numerical values must come from the validated context object.
    No LLM is used to create financial claims.
    """

    arm_name = {
        int(arm): name
        for arm, name in {
            NudgeArm.NO_NUDGE: "no_nudge",
            NudgeArm.SAVINGS_MILESTONE: "savings_milestone",
            NudgeArm.SUBSCRIPTION_WARNING: "subscription_warning",
            NudgeArm.PRICE_CREEP_REVIEW: "price_creep_review",
            NudgeArm.IMPULSE_SPEND_INTERCEPTION: (
                "impulse_spend_interception"
            ),
            NudgeArm.BUDGET_THRESHOLD: "budget_threshold",
            NudgeArm.CASHFLOW_WARNING: "cashflow_warning",
            NudgeArm.POSITIVE_REINFORCEMENT: (
                "positive_reinforcement"
            ),
        }.items()
    }[arm_id]

    if arm_id == int(NudgeArm.NO_NUDGE):
        return RenderedNudge(
            arm_id=arm_id,
            arm_name=arm_name,
            title="No action needed",
            message=(
                "There is no financial alert for you right now."
            ),
            reason_codes=reason_codes,
        )

    if arm_id == int(NudgeArm.SAVINGS_MILESTONE):
        return RenderedNudge(
            arm_id=arm_id,
            arm_name=arm_name,
            title="Review your savings progress",
            message=(
                f"There are {context.days_to_month_end} days "
                "left in the current month. Review your savings "
                "progress when convenient."
            ),
            reason_codes=reason_codes + [
                "savings_review_opportunity"
            ],
            cta_label="Review savings",
            facts={
                "days_to_month_end": context.days_to_month_end,
            },
        )

    if arm_id == int(NudgeArm.SUBSCRIPTION_WARNING):
        return RenderedNudge(
            arm_id=arm_id,
            arm_name=arm_name,
            title="Review recurring payments",
            message=(
                "Recurring payments make up "
                f"{_percent(context.subscription_ratio)} "
                "of the tracked spending. Would you like to "
                "review them?"
            ),
            reason_codes=reason_codes + [
                "subscription_ratio_high"
            ],
            cta_label="Review payments",
            facts={
                "subscription_ratio": context.subscription_ratio,
            },
        )

    if arm_id == int(NudgeArm.PRICE_CREEP_REVIEW):
        return RenderedNudge(
            arm_id=arm_id,
            arm_name=arm_name,
            title="Review a recurring price change",
            message=(
                f"{context.price_creep_count} recurring expense"
                f"{'' if context.price_creep_count == 1 else 's'} "
                "may have increased. Would you like to review it?"
            ),
            reason_codes=reason_codes + [
                "price_creep_detected"
            ],
            cta_label="Review expenses",
            facts={
                "price_creep_count": context.price_creep_count,
            },
        )

    if arm_id == int(NudgeArm.IMPULSE_SPEND_INTERCEPTION):
        return RenderedNudge(
            arm_id=arm_id,
            arm_name=arm_name,
            title="Review recent spending",
            message=(
                "Your recent spending is above your usual "
                "pattern. Would you like to review the recent "
                "transactions?"
            ),
            reason_codes=reason_codes + [
                "recent_anomaly_detected"
            ],
            cta_label="Review transactions",
            facts={
                "anomaly_count": context.anomaly_count,
                "spending_velocity": context.spending_velocity,
            },
        )

    if arm_id == int(NudgeArm.BUDGET_THRESHOLD):
        return RenderedNudge(
            arm_id=arm_id,
            arm_name=arm_name,
            title="Budget threshold reached",
            message=(
                "You have used "
                f"{_percent(context.budget_utilization)} "
                "of the tracked budget."
            ),
            reason_codes=reason_codes + [
                "budget_threshold_reached"
            ],
            cta_label="Review budget",
            facts={
                "budget_utilization": context.budget_utilization,
            },
        )

    if arm_id == int(NudgeArm.CASHFLOW_WARNING):
        return RenderedNudge(
            arm_id=arm_id,
            arm_name=arm_name,
            title="Review upcoming cashflow",
            message=(
                "Your recent spending pattern may create "
                "cashflow pressure before the current period ends."
            ),
            reason_codes=reason_codes + [
                "cashflow_health_low"
            ],
            cta_label="Review spending",
            facts={
                "net_cashflow_health": context.net_cashflow_health,
            },
        )

    return RenderedNudge(
        arm_id=arm_id,
        arm_name=arm_name,
        title="Positive spending pattern",
        message=(
            "Your recent spending is within the tracked pattern. "
            "Keep reviewing your financial goals regularly."
        ),
        reason_codes=reason_codes + [
            "positive_pattern"
        ],
        cta_label="View progress",
        facts={
            "budget_utilization": context.budget_utilization,
        },
    )