from enum import IntEnum


class NudgeArm(IntEnum):
    NO_NUDGE = 0
    SAVINGS_MILESTONE = 1
    SUBSCRIPTION_WARNING = 2
    PRICE_CREEP_REVIEW = 3
    IMPULSE_SPEND_INTERCEPTION = 4
    BUDGET_THRESHOLD = 5
    CASHFLOW_WARNING = 6
    POSITIVE_REINFORCEMENT = 7


ARM_NAMES: dict[int, str] = {
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
}

NAME_TO_ARMS: dict[str, int] = {
    name: int(arm)
    for arm, name in ARM_NAMES.items()
}