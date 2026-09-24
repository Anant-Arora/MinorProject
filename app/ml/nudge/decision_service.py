from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import numpy as np

from app.ml.nudge import ARM_NAMES, NudgeArm
from app.ml.nudge.context import NudgeContext
from app.ml.nudge.eligibility import (
    SafetyConfig,
    get_eligible_arms,
)
from app.ml.nudge.linucb import AdaptiveNudgeEngine
from app.ml.nudge.templates import (
    RenderedNudge,
    render_nudge,
)


@dataclass(frozen=True)
class NudgeDecision:
    """Complete result of one adaptive nudge decision."""

    decision_id: str
    user_id: str
    selected_arm_id: int
    selected_arm_name: str
    propensity: float
    eligible_arms: list[int]
    action_scores: dict[str, float]
    context_snapshot: dict[str, Any]
    nudge: RenderedNudge
    reason_codes: list[str]
    policy_name: str
    policy_version: str
    model_version: str
    feature_schema_version: str
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""

        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


class NudgeDecisionService:
    """
    Connects validated context, deterministic safety rules,
    eligibility filtering, LinUCB action selection, and deterministic
    nudge template rendering.

    Safety rules are always evaluated before LinUCB.
    If safety allows only no_nudge, the bandit is never called.
    """

    def __init__(
        self,
        engine: AdaptiveNudgeEngine,
        *,
        decision_id_factory: Callable[[], str],
        safety_config: SafetyConfig | None = None,
        policy_name: str = "disjoint_linucb",
        policy_version: str = "v1",
        model_version: str = "nudge-model-v1",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.engine = engine
        self.decision_id_factory = decision_id_factory
        self.safety_config = safety_config or SafetyConfig()
        self.policy_name = policy_name
        self.policy_version = policy_version
        self.model_version = model_version
        self.rng = rng or np.random.default_rng()

    def decide(
        self,
        *,
        user_id: str,
        context: NudgeContext,
        now: datetime | None = None,
        nudges_today: int = 0,
        nudges_last_7d: int = 0,
        disabled_arms: Iterable[int] = (),
        recently_dismissed_arms: Iterable[int] = (),
    ) -> NudgeDecision:
        """
        Select and render a nudge decision.

        Flow:
        1. Evaluate hard safety and eligibility rules.
        2. Return no_nudge immediately when a hard safety rule blocks nudges.
        3. Otherwise call the epsilon-mixed LinUCB policy.
        4. Render deterministic message text.
        """

        now = now or datetime.now(timezone.utc)

        eligible_arm_ids, safety_reasons = get_eligible_arms(
            context,
            now=now,
            nudges_today=nudges_today,
            nudges_last_7d=nudges_last_7d,
            disabled_arms=disabled_arms,
            recently_dismissed_arms=recently_dismissed_arms,
            config=self.safety_config,
        )

        no_nudge_id = int(NudgeArm.NO_NUDGE)

        # A hard safety block returns only no_nudge and does not call LinUCB.
        if eligible_arm_ids == [no_nudge_id]:
            selected_arm_id = no_nudge_id
            selected_arm_name = ARM_NAMES[selected_arm_id]
            propensity = 1.0
            action_scores = {
                str(selected_arm_id): 0.0,
            }
            reason_codes = list(safety_reasons)

        else:
            eligible_arm_names = [
                ARM_NAMES[arm_id]
                for arm_id in eligible_arm_ids
            ]

            bandit_result = self.engine.select_arm(
                context=context.to_vector(),
                eligible_arms=eligible_arm_names,
                rng=self.rng,
            )

            selected_arm_name = str(
                bandit_result["selected_arm"]
            )

            selected_arm_id = next(
                arm_id
                for arm_id, arm_name in ARM_NAMES.items()
                if arm_name == selected_arm_name
            )

            propensity = float(
                bandit_result["propensity"]
            )

            action_scores = {
                str(arm_id): float(
                    bandit_result["scores"][arm_name][
                        "ucb_score"
                    ]
                )
                for arm_id, arm_name in ARM_NAMES.items()
                if arm_name in bandit_result["scores"]
            }

            reason_codes = list(safety_reasons)

        rendered_nudge = render_nudge(
            arm_id=selected_arm_id,
            context=context,
            reason_codes=reason_codes,
        )

        return NudgeDecision(
            decision_id=self.decision_id_factory(),
            user_id=user_id,
            selected_arm_id=selected_arm_id,
            selected_arm_name=selected_arm_name,
            propensity=propensity,
            eligible_arms=eligible_arm_ids,
            action_scores=action_scores,
            context_snapshot=context.to_snapshot(),
            nudge=rendered_nudge,
            reason_codes=rendered_nudge.reason_codes,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            model_version=self.model_version,
            feature_schema_version=context.schema_version,
            created_at=now,
        )