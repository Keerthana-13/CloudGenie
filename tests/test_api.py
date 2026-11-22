import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def client():
    from src.api import app
    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_health_and_chat_page(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Cloud Genie support" in client.get("/").text


def test_ticket_endpoint_returns_decision_and_audit_trail(client):
    r = client.post("/v1/tickets", json={"channel": "chat", "body": "How do I paginate through very large result sets?"})
    j = r.json()
    assert r.status_code == 200 and j["disposition"] in {"auto_responded", "escalated"}
    assert {d["stage"] for d in j["decisions"]} >= {"classification", "retrieval", "routing"}


def test_prometheus_metrics_exposed(client):
    client.post("/v1/tickets", json={"body": "my deployment keeps rolling back"})
    assert "cloudserve_tickets_total" in client.get("/metrics").text


def test_kill_switch_requires_token_and_works(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret-for-tests")
    assert client.post("/admin/kill-switch", json={"enabled": False}).status_code == 403
    assert client.post("/admin/kill-switch", json={"enabled": False}, headers={"X-Admin-Token": "s3cret-for-tests"}).status_code == 200
    try:
        j = client.post("/v1/tickets", json={"body": "How do I paginate through very large result sets?"}).json()
        assert j["disposition"] == "escalated" and j["rule"] == "kill_switch"
    finally:
        client.post("/admin/kill-switch", json={"enabled": True}, headers={"X-Admin-Token": "s3cret-for-tests"})
    assert client.get("/health").json()["kill_switch_engaged"] is False
