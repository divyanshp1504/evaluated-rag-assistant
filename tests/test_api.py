from fastapi.testclient import TestClient

from rag.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ask():
    r = client.post("/ask", json={"question": "What is the main way to prevent the flu?"})
    body = r.json()
    assert r.status_code == 200 and not body["refused"] and "vaccine" in body["answer"]


def test_validation_error():
    assert client.post("/ask", json={"question": ""}).status_code == 422
