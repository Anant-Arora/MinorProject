from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class NudgeContext(BaseModel):
    """
    Validated, versioned context for the Adaptive Nudge Engine.

    The model contains nine business features. The feature vector contains
    one additional intercept, so LinUCB receives ten dimensions.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"

    spending_velocity: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description=(
            "Recent spending divided by a historical spending baseline."
        ),
    )

    anomaly_count: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of detected anomalies in the current window.",
    )

    subscription_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Recurring subscription spending divided by tracked spending."
        ),
    )

    net_cashflow_health: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Normalized cashflow health. "
            "-1 means weak and 1 means strong."
        ),
    )

    budget_utilization: float = Field(
        ...,
        ge=0.0,
        le=2.0,
        description="Current tracked spending divided by the budget.",
    )

    days_to_month_end: int = Field(
        ...,
        ge=0,
        le=31,
        description="Number of days remaining in the current month.",
    )

    price_creep_count: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of detected price-creep cases.",
    )

    days_since_last_nudge: int = Field(
        ...,
        ge=0,
        description="Days since the user's last nudge.",
    )

    nudges_last_7d: int = Field(
        ...,
        ge=0,
        le=7,
        description="Number of nudges shown during the last seven days.",
    )

    @field_validator(
        "spending_velocity",
        "subscription_ratio",
        "net_cashflow_health",
        "budget_utilization",
        mode="before",
    )
    @classmethod
    def validate_finite_float(cls, value: float) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError("Context values must be finite.")

        return value

    def to_vector(self) -> np.ndarray:
        """
        Return the fixed numerical vector used by LinUCB.

        Dimension:
            1 intercept + 9 normalized business features = 10.
        """

        vector = np.array(
            [
                1.0,
                self.spending_velocity / 10.0,
                min(self.anomaly_count, 100) / 100.0,
                self.subscription_ratio,
                (self.net_cashflow_health + 1.0) / 2.0,
                min(self.budget_utilization, 2.0) / 2.0,
                self.days_to_month_end / 31.0,
                min(self.price_creep_count, 100) / 100.0,
                min(self.days_since_last_nudge, 30) / 30.0,
                self.nudges_last_7d / 7.0,
            ],
            dtype=np.float64,
        )

        if vector.shape != (10,):
            raise ValueError(
                f"Expected context vector shape (10,), got {vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Context vector contains non-finite values."
            )

        return vector

    def to_snapshot(self) -> dict:
        """
        Return a JSON-safe representation for decision logging.
        """

        return {
            "schema_version": self.schema_version,
            "spending_velocity": self.spending_velocity,
            "anomaly_count": self.anomaly_count,
            "subscription_ratio": self.subscription_ratio,
            "net_cashflow_health": self.net_cashflow_health,
            "budget_utilization": self.budget_utilization,
            "days_to_month_end": self.days_to_month_end,
            "price_creep_count": self.price_creep_count,
            "days_since_last_nudge": self.days_since_last_nudge,
            "nudges_last_7d": self.nudges_last_7d,
            "vector": self.to_vector().tolist(),
        }