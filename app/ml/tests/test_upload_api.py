from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


# File location:
# finsight-backend/app/ml/tests/test_upload_api.py
#
# parents[0] -> app/ml/tests
# parents[1] -> app/ml
# parents[2] -> app
# parents[3] -> finsight-backend
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "real_sample.csv"


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200, response.text

    data = response.json()

    assert isinstance(data, dict)
    assert data.get("status") == "healthy"


def test_upload_statement_api(client: TestClient):
    assert SAMPLE_CSV_PATH.is_file(), (
        f"Sample CSV not found at: {SAMPLE_CSV_PATH}"
    )

    with SAMPLE_CSV_PATH.open("rb") as file:
        response = client.post(
            "/api/upload-statement",
            files={
                "file": (
                    SAMPLE_CSV_PATH.name,
                    file,
                    "text/csv",
                )
            },
        )

    assert response.status_code == 200, (
        f"API failed with status {response.status_code}: "
        f"{response.text}"
    )

    data = response.json()

    assert data.get("status") == "success"
    assert "summary" in data
    assert "transactions" in data

    assert data["summary"]["total_transactions_processed"] > 0
    assert isinstance(data["transactions"], list)