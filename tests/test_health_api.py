
from fastapi.testclient import TestClient

from health_api import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/health")

    assert response.status_code == 200


def test_health_endpoint_returns_ok_status():
    response = client.get("/health")

    assert response.json() == {"status": "ok"}


def test_health_endpoint_returns_json():
    response = client.get("/health")

    assert response.headers["content-type"].startswith(
        "application/json"
    )


def test_unknown_endpoint_returns_404():
    response = client.get("/unknown")

    assert response.status_code == 404


def test_health_endpoint_supports_get_only():
    response = client.post("/health")

    assert response.status_code == 405
