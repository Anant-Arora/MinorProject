from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardResult:
    reward: float
    components: dict[str, float]


def calculate_reward(
    *,
    opened: bool = False,
    meaningful_click: bool = False,
    target_action: bool = False,
    seven_day_improvement: bool = False,
    dismissed: bool = False,
    marked_irrelevant: bool = False,
) -> RewardResult:
    components = {
        "opened": 0.10 if opened else 0.0,
        "meaningful_click": 0.20 if meaningful_click else 0.0,
        "target_action": 0.40 if target_action else 0.0,
        "seven_day_improvement": (
            0.50 if seven_day_improvement else 0.0
        ),
        "dismissed": -0.25 if dismissed else 0.0,
        "marked_irrelevant": (
            -0.50 if marked_irrelevant else 0.0
        ),
    }

    raw_reward = sum(components.values())
    clipped_reward = max(-1.0, min(1.0, raw_reward))

    return RewardResult(
        reward=clipped_reward,
        components={
            **components,
            "raw_reward": raw_reward,
            "clipped_reward": clipped_reward,
        },
    )