from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Iterable

from app.ml.nudge import NudgeArm
from app.ml.nudge.context import NudgeContext


@dataclass(frozen=True)
class SafetyConfig:
    """
    Hard safety and notification-frequency limits.

    These rules are evaluated before the bandit is called.
    """

    max_daily_nudges: int = 1
    max_weekly_nudges: int = 3

    quiet_start: time = time(22, 0)
    quiet_end: time = time(7, 0)

    anomaly_threshold: int = 1
    budget_threshold: float = 0.80
    price_creep_threshold: int = 1

    kill_switch: bool = False


def is_quiet_hours(
    current_time: datetime,
    config: SafetyConfig,
) -> bool:
    """
    Return True when current_time falls inside configured quiet hours.

    This also supports an overnight interval such as 22:00 to 07:00.
    """

    current = current_time.time()

    if config.quiet_start < config.quiet_end:
        return config.quiet_start <= current < config.quiet_end

    return (
        current >= config.quiet_start
        or current < config.quiet_end
    )


def get_eligible_arms(
    context: NudgeContext,
    *,
    now: datetime,
    nudges_today: int = 0,
    nudges_last_7d: int = 0,
    disabled_arms: Iterable[int] = (),
    recently_dismissed_arms: Iterable[int] = (),
    config: SafetyConfig | None = None,
) -> tuple[list[int], list[str]]:
    """
    Return eligible arms and safety reason codes.

    no_nudge is always returned when the policy is allowed to operate.
    If a hard safety rule blocks notifications, only no_nudge is returned.
    """

    config = config or SafetyConfig()

    disabled = {
        int(arm)
        for arm in disabled_arms
    }

    recently_dismissed = {
        int(arm)
        for arm in recently_dismissed_arms
    }

    no_nudge = int(NudgeArm.NO_NUDGE)

    if config.kill_switch:
        return [no_nudge], ["kill_switch_active"]

    if is_quiet_hours(now, config):
        return [no_nudge], ["quiet_hours"]

    if nudges_today >= config.max_daily_nudges:
        return [no_nudge], ["daily_limit_reached"]

    if nudges_last_7d >= config.max_weekly_nudges:
        return [no_nudge], ["weekly_limit_reached"]

    eligible = [no_nudge]
    reason_codes: list[str] = []

    subscription_arm = int(NudgeArm.SUBSCRIPTION_WARNING)

    if (
        context.subscription_ratio > 0.10
        and subscription_arm not in disabled
        and subscription_arm not in recently_dismissed
    ):
        eligible.append(subscription_arm)

    price_creep_arm = int(NudgeArm.PRICE_CREEP_REVIEW)

    if (
        context.price_creep_count
        >= config.price_creep_threshold
        and price_creep_arm not in disabled
        and price_creep_arm not in recently_dismissed
    ):
        eligible.append(price_creep_arm)

    impulse_arm = int(
        NudgeArm.IMPULSE_SPEND_INTERCEPTION
    )

    if (
        context.anomaly_count
        >= config.anomaly_threshold
        and impulse_arm not in disabled
        and impulse_arm not in recently_dismissed
    ):
        eligible.append(impulse_arm)

    budget_arm = int(NudgeArm.BUDGET_THRESHOLD)

    if (
        context.budget_utilization
        >= config.budget_threshold
        and budget_arm not in disabled
        and budget_arm not in recently_dismissed
    ):
        eligible.append(budget_arm)

    cashflow_arm = int(NudgeArm.CASHFLOW_WARNING)

    if (
        context.net_cashflow_health < 0.25
        and cashflow_arm not in disabled
        and cashflow_arm not in recently_dismissed
    ):
        eligible.append(cashflow_arm)

    savings_arm = int(NudgeArm.SAVINGS_MILESTONE)

    if (
        context.days_to_month_end <= 7
        and context.net_cashflow_health >= 0.25
        and savings_arm not in disabled
        and savings_arm not in recently_dismissed
    ):
        eligible.append(savings_arm)

    positive_arm = int(NudgeArm.POSITIVE_REINFORCEMENT)

    if (
        context.budget_utilization < 0.80
        and context.net_cashflow_health >= 0.50
        and positive_arm not in disabled
        and positive_arm not in recently_dismissed
    ):
        eligible.append(positive_arm)

    return sorted(set(eligible)), reason_codes