import numpy as np
from datetime import datetime, timezone
import json
import threading
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
POLICY_NAME = "disjoint_linucb"
POLICY_VERSION = "v1"


class LinUCBArm:
    def __init__(self, feature_dim: int, lambda_reg: float = 1.0):
        self.feature_dim = feature_dim
        self.A = lambda_reg * np.eye(feature_dim)
        self.b = np.zeros(feature_dim)
        self.pull_count = 0

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def score(self, x: np.ndarray, alpha: float) -> tuple:
        theta = self.theta()
        mean_estimate = float(x @ theta)
        A_inv_x = np.linalg.solve(self.A, x)
        uncertainty = float(alpha * np.sqrt(max(x @ A_inv_x, 0.0)))
        return mean_estimate + uncertainty, mean_estimate, uncertainty

    def update(self, x: np.ndarray, reward: float):
        self.A += np.outer(x, x)
        self.b += reward * x
        self.pull_count += 1


class DisjointLinUCB:
    def __init__(self, arm_names: list, feature_dim: int, alpha: float = 0.5, epsilon: float = 0.1):
        self.arm_names = list(arm_names)
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.epsilon = epsilon
        self.arms = {name: LinUCBArm(feature_dim) for name in arm_names}
        self._lock = threading.RLock()

    def select_arm(self, context: np.ndarray, eligible_arms: list, rng: np.random.Generator) -> dict:
        if "no_nudge" not in eligible_arms:
            eligible_arms = eligible_arms + ["no_nudge"]

        scores = {}
        for arm_name in eligible_arms:
            arm = self.arms[arm_name]
            ucb, mean_est, uncertainty = arm.score(context, self.alpha)
            scores[arm_name] = {
                "ucb_score": ucb,
                "mean_estimate": mean_est,
                "uncertainty_bonus": uncertainty,
            }

        is_exploring = rng.random() < self.epsilon

        if is_exploring:
            selected_arm = rng.choice(eligible_arms)
            propensity = self.epsilon / len(eligible_arms)
        else:
            selected_arm = max(eligible_arms, key=lambda a: scores[a]["ucb_score"])
            propensity = (1 - self.epsilon) + (self.epsilon / len(eligible_arms))

        return {
            "selected_arm": selected_arm,
            "eligible_arms": eligible_arms,
            "scores": scores,
            "propensity": propensity,
            "was_exploration": bool(is_exploring),
            "policy_name": POLICY_NAME,
            "policy_version": POLICY_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def update(self, arm_name: str, context: np.ndarray, reward: float):
        if not (-1.0 <= reward <= 1.0):
            raise ValueError(f"Reward must be in [-1, 1], got {reward}")
        with self._lock:
            self.arms[arm_name].update(context, reward)

    def save_state(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            state = {
                "policy_name": POLICY_NAME,
                "policy_version": POLICY_VERSION,
                "feature_dim": self.feature_dim,
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "arm_names": self.arm_names,
                "A": {name: arm.A.tolist() for name, arm in self.arms.items()},
                "b": {name: arm.b.tolist() for name, arm in self.arms.items()},
                "pull_counts": {name: arm.pull_count for name, arm in self.arms.items()},
            }

        output_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load_state(cls, path: str) -> "DisjointLinUCB":
        input_path = Path(path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Bandit state file not found: {input_path}")

        state = json.loads(input_path.read_text(encoding="utf-8"))

        if state["policy_version"] != POLICY_VERSION:
            raise ValueError(
                f"Saved policy version ({state['policy_version']}) does not match "
                f"current code's policy version ({POLICY_VERSION}). "
                f"Do not load state trained under a different schema/version."
            )

        engine = cls(
            arm_names=state["arm_names"],
            feature_dim=state["feature_dim"],
            alpha=state["alpha"],
            epsilon=state["epsilon"],
        )

        for name in engine.arm_names:
            arm = engine.arms[name]
            arm.A = np.array(state["A"][name], dtype=np.float64)
            arm.b = np.array(state["b"][name], dtype=np.float64)
            arm.pull_count = state["pull_counts"][name]

            if arm.A.shape != (engine.feature_dim, engine.feature_dim):
                raise ValueError(f"Invalid A shape for arm '{name}': {arm.A.shape}")
            if arm.b.shape != (engine.feature_dim,):
                raise ValueError(f"Invalid b shape for arm '{name}': {arm.b.shape}")

        return engine


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    bandit = DisjointLinUCB(
        arm_names=["no_nudge", "subscription_warning", "savings_milestone"],
        feature_dim=3,
        alpha=0.5,
        epsilon=0.1,
    )
    context = np.array([1.0, 0.8, 0.2])
    decision = bandit.select_arm(context, ["no_nudge", "subscription_warning"], rng)
    print("Decision:", decision)

    bandit.update(decision["selected_arm"], context, reward=0.4)
    print("Updated pull count:", bandit.arms[decision["selected_arm"]].pull_count)   
# Backward-compatible name used by decision_service.py.
# The actual implementation remains DisjointLinUCB.
AdaptiveNudgeEngine = DisjointLinUCB