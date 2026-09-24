"""
Since FinSight has no real users yet, we validate the Adaptive Nudge
Engine the same way the project report describes: build a simulator
with a KNOWN ground-truth best action per situation, run the bandit
against it, and prove it learns to find that best action over time —
compared against a static baseline and a pure-random baseline.
"""

import numpy as np
from context import build_context_vector, FEATURE_DIM
from eligibility import get_eligible_arms
from linucb import DisjointLinUCB

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

rng = np.random.default_rng(42)


def simulate_situation():
    situation_type = rng.choice(["overspending", "subscription_issue", "on_track"])

    if situation_type == "overspending":
        raw_signals = {
            "spending_velocity": rng.uniform(1.5, 3.0),
            "anomaly_count": rng.integers(0, 2),
            "subscription_ratio": rng.uniform(0.1, 0.3),
            "net_cashflow_health": rng.uniform(-1.0, -0.3),
            "budget_utilization": rng.uniform(0.85, 1.3),
            "days_to_month_end": rng.integers(1, 15),
            "price_creep_count": 0,
            "days_since_last_nudge": rng.integers(1, 10),
        }
        evidence = {"budget_threshold_crossed": True, "cashflow_risk": True}
        true_best_arm = "budget_threshold"

    elif situation_type == "subscription_issue":
        raw_signals = {
            "spending_velocity": rng.uniform(-0.5, 0.5),
            "anomaly_count": 0,
            "subscription_ratio": rng.uniform(0.4, 0.7),
            "net_cashflow_health": rng.uniform(-0.2, 0.5),
            "budget_utilization": rng.uniform(0.4, 0.8),
            "days_to_month_end": rng.integers(5, 25),
            "price_creep_count": rng.integers(0, 2),
            "days_since_last_nudge": rng.integers(1, 15),
        }
        evidence = {"has_flagged_subscription": True}
        true_best_arm = "subscription_warning"

    else:
        raw_signals = {
            "spending_velocity": rng.uniform(-1.0, 0.3),
            "anomaly_count": 0,
            "subscription_ratio": rng.uniform(0.1, 0.3),
            "net_cashflow_health": rng.uniform(0.3, 1.0),
            "budget_utilization": rng.uniform(0.2, 0.6),
            "days_to_month_end": rng.integers(10, 30),
            "price_creep_count": 0,
            "days_since_last_nudge": rng.integers(5, 20),
        }
        evidence = {"goal_progress_positive": True, "cashflow_risk": False}
        true_best_arm = "savings_milestone"

    context = build_context_vector(raw_signals)
    eligible = get_eligible_arms(evidence)
    return context, eligible, true_best_arm


def simulate_reward(chosen_arm: str, true_best_arm: str) -> float:
    if chosen_arm == true_best_arm:
        base_reward = 0.7
    elif chosen_arm == "no_nudge":
        base_reward = 0.0
    else:
        base_reward = -0.3

    noise = rng.normal(0, 0.15)
    return float(np.clip(base_reward + noise, -1.0, 1.0))


def run_simulation(policy_type: str, num_rounds: int = 1000):
    if policy_type == "linucb":
        bandit = DisjointLinUCB(ARM_NAMES, FEATURE_DIM, alpha=0.5, epsilon=0.1)

    cumulative_rewards = []
    total_reward = 0.0
    correct_choices = 0

    for round_num in range(num_rounds):
        context, eligible, true_best_arm = simulate_situation()

        if policy_type == "linucb":
            decision = bandit.select_arm(context, eligible, rng)
            chosen_arm = decision["selected_arm"]
        elif policy_type == "random":
            chosen_arm = rng.choice(eligible)
        elif policy_type == "static":
            chosen_arm = "budget_threshold" if "budget_threshold" in eligible else "no_nudge"

        reward = simulate_reward(chosen_arm, true_best_arm)
        total_reward += reward
        cumulative_rewards.append(total_reward)

        if chosen_arm == true_best_arm:
            correct_choices += 1

        if policy_type == "linucb":
            bandit.update(chosen_arm, context, reward)

    accuracy = correct_choices / num_rounds
    return cumulative_rewards, accuracy


if __name__ == "__main__":
    NUM_ROUNDS = 1000

    linucb_rewards, linucb_accuracy = run_simulation("linucb", NUM_ROUNDS)
    random_rewards, random_accuracy = run_simulation("random", NUM_ROUNDS)
    static_rewards, static_accuracy = run_simulation("static", NUM_ROUNDS)

    print("=" * 60)
    print(f"SIMULATION RESULTS OVER {NUM_ROUNDS} ROUNDS")
    print("=" * 60)
    print(f"{'Policy':<15} {'Final Cumulative Reward':>25} {'Correct-Arm Rate':>20}")
    print(f"{'LinUCB':<15} {linucb_rewards[-1]:>25.2f} {linucb_accuracy:>19.1%}")
    print(f"{'Random':<15} {random_rewards[-1]:>25.2f} {random_accuracy:>19.1%}")
    print(f"{'Static':<15} {static_rewards[-1]:>25.2f} {static_accuracy:>19.1%}")

    print("\n--- Does LinUCB actually improve over time? ---")
    early_rewards = linucb_rewards[99] - linucb_rewards[0]
    late_rewards = linucb_rewards[-1] - linucb_rewards[-100]
    print(f"Avg reward, first 100 rounds: {early_rewards/100:.3f}")
    print(f"Avg reward, last 100 rounds:  {late_rewards/100:.3f}")