from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(
        String(128),
        primary_key=True,
        index=True,
    )

    email = Column(
        String(320),
        unique=True,
        index=True,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
    )

    user_id = Column(
        String(128),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Keep the legacy column name "date" because upload.py may use it.
    # Change to Date only after confirming your database migration.
    date = Column(
        String,
        nullable=False,
    )

    description = Column(
        Text,
        nullable=False,
    )

    amount = Column(
        Numeric(14, 2),
        nullable=False,
    )

    merchant_clean = Column(
        String(255),
        nullable=True,
    )

    predicted_subscription = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    predicted_price_creep = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    predicted_anomaly = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_transactions_user_date",
            "user_id",
            "date",
        ),
    )


class NudgeDecision(Base):
    __tablename__ = "nudge_decisions"

    id = Column(
        String(36),
        primary_key=True,
        index=True,
    )

    user_id = Column(
        String(128),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_event_id = Column(
        String(128),
        nullable=True,
        index=True,
    )

    selected_arm_id = Column(
        Integer,
        nullable=False,
    )

    selected_arm_name = Column(
        String(100),
        nullable=False,
    )

    propensity = Column(
        Float,
        nullable=False,
    )

    context_snapshot = Column(
        JSON,
        nullable=False,
    )

    eligible_arms = Column(
        JSON,
        nullable=False,
    )

    action_scores = Column(
        JSON,
        nullable=False,
    )

    reason_codes = Column(
        JSON,
        nullable=False,
    )

    policy_name = Column(
        String(100),
        nullable=False,
    )

    policy_version = Column(
        String(50),
        nullable=False,
    )

    model_version = Column(
        String(50),
        nullable=False,
    )

    feature_schema_version = Column(
        String(50),
        nullable=False,
    )

    reward_status = Column(
        String(30),
        nullable=False,
        default="pending",
        server_default="pending",
    )

    reward_due_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class NudgeFeedbackEvent(Base):
    __tablename__ = "nudge_feedback_events"

    id = Column(
        String(100),
        primary_key=True,
        index=True,
    )

    decision_id = Column(
        String(36),
        ForeignKey("nudge_decisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    user_id = Column(
        String(128),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    event_type = Column(
        String(50),
        nullable=False,
    )

    event_value = Column(
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "decision_id",
            "event_type",
            name="uq_nudge_decision_event_type",
        ),
    )


class NudgeReward(Base):
    __tablename__ = "nudge_rewards"

    decision_id = Column(
        String(36),
        ForeignKey("nudge_decisions.id", ondelete="CASCADE"),
        primary_key=True,
    )

    reward = Column(
        Float,
        nullable=False,
    )

    reward_components = Column(
        JSON,
        nullable=False,
    )

    evaluation_window_days = Column(
        Integer,
        nullable=False,
        default=7,
        server_default="7",
    )

    resolved_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class BanditState(Base):
    __tablename__ = "bandit_state"

    arm_id = Column(
        Integer,
        primary_key=True,
    )

    arm_name = Column(
        String(100),
        unique=True,
        nullable=False,
    )

    dimension = Column(
        Integer,
        nullable=False,
    )

    matrix_a = Column(
        JSON,
        nullable=False,
    )

    vector_b = Column(
        JSON,
        nullable=False,
    )

    pulls = Column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    alpha = Column(
        Float,
        nullable=False,
    )

    l2 = Column(
        Float,
        nullable=False,
    )

    epsilon = Column(
        Float,
        nullable=False,
    )

    policy_version = Column(
        String(50),
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )