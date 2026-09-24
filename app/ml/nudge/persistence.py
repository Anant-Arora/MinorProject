from pathlib import Path

from app.ml.nudge.linucb import AdaptiveNudgeEngine


MODEL_STATE_PATH = Path("data/nudge_bandit_state.json")


def load_engine() -> AdaptiveNudgeEngine:
    if MODEL_STATE_PATH.exists():
        return AdaptiveNudgeEngine.load_state(
            MODEL_STATE_PATH
        )

    return AdaptiveNudgeEngine(
        alpha=0.8,
        l2=1.0,
        epsilon=0.10,
    )