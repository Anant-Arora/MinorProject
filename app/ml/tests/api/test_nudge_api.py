from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_nudge_health():
    response = client.get("/api/nudge/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["policy_name"] == "linucb"


def test_decide_nudge():
    response = client.post(
        "/api/nudge/decide",
        json={
            "user_id": "test_user",
            "trigger": "manual_test",
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["user_id"] == "test_user"
    assert "decision_id" in data
    assert "selected_arm_name" in data
    assert "nudge" in data
    assert "message" in data["nudge"]
    assert 0.0 < data["propensity"] <= 1.0