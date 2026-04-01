from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_score_endpoint_returns_stable_json() -> None:
    response = client.get("/score", params={"value": 7})
    assert response.status_code == 200
    assert response.json() == {"value": 7, "double": 14, "is_even": 0}
