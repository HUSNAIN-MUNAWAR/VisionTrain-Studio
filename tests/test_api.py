from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint_reports_trainer_mapping() -> None:
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["trainer_mapping_ok"] is True
    assert data["supported_model_count"] > 0
