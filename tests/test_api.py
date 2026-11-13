"""End-to-end tests through the HTTP API with mock upstreams."""

import json

from fastapi.testclient import TestClient

from relay.app import create_app

from .conftest import chat, make_settings


def test_rejects_missing_or_bad_key(client):
    assert chat(client, key="wrong").status_code == 401
    r = client.post("/v1/chat/completions", json={"model": "default", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"
    assert r.headers["x-request-id"]


def test_completion_is_openai_shaped(client):
    r = chat(client, "ping")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert "You said: ping" in body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert r.headers["x-relay-provider"] == "primary"
    assert r.headers["x-relay-cache"] == "miss"


def test_second_identical_request_hits_cache(client, providers):
    chat(client, "cache me")
    r = chat(client, "cache me")
    assert r.headers["x-relay-cache"] == "exact"
    assert providers["primary"].calls == 1

    r = chat(client, "CACHE me!")
    assert r.headers["x-relay-cache"] == "semantic"
    assert providers["primary"].calls == 1


def test_sampling_requests_bypass_cache(client, providers):
    chat(client, "creative", temperature=0.9)
    r = chat(client, "creative", temperature=0.9)
    assert r.headers["x-relay-cache"] == "bypass"
    assert providers["primary"].calls == 2


def test_falls_back_when_primary_is_down(client, providers):
    providers["primary"].failure_rate = 1.0
    r = chat(client, "are you there")
    assert r.status_code == 200
    assert r.headers["x-relay-provider"] == "backup"
    assert r.headers["x-relay-attempts"] == "3"  # primary + 1 retry, then backup


def test_502_when_all_upstreams_fail(client, providers):
    for p in providers.values():
        p.failure_rate = 1.0
    r = chat(client, "anyone?")
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "upstream_unavailable"


def test_circuit_opens_and_is_visible_in_admin_and_metrics(client, providers):
    providers["primary"].failure_rate = 1.0
    for i in range(3):
        chat(client, f"q{i}")
    status = client.get("/admin/providers", headers={"Authorization": "Bearer k-acme"}).json()
    assert status["primary"]["state"] == "open"
    assert status["backup"]["state"] == "closed"
    metrics = client.get("/metrics").text
    assert 'relay_circuit_open{provider="primary"} 1' in metrics
    assert 'relay_upstream_calls_total{provider="primary",outcome="short_circuited"}' in metrics


def test_admin_requires_admin_tenant(client):
    assert client.get("/admin/providers", headers={"Authorization": "Bearer k-tiny"}).status_code == 403


def test_rate_limit_returns_429_with_retry_after(client):
    assert chat(client, "a", key="k-tiny").status_code == 200
    assert chat(client, "b", key="k-tiny").status_code == 200
    r = chat(client, "c", key="k-tiny")
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1


def test_model_allow_list(client):
    assert chat(client, key="k-tiny", model="primary/big").status_code == 403
    assert chat(client, model="primary/big").status_code == 200


def test_unknown_model_is_404(client):
    assert chat(client, model="does-not-exist").status_code == 404


def test_budget_exhausted_returns_402(client):
    r = chat(client, key="k-broke")
    assert r.status_code == 402
    assert r.json()["error"]["code"] == "budget_exceeded"


def test_pii_is_redacted_before_reaching_upstream(client):
    r = chat(client, "my email is bob@corp.io")
    content = r.json()["choices"][0]["message"]["content"]
    assert "bob@corp.io" not in content and "[EMAIL]" in content
    assert 'relay_pii_redactions_total{kind="EMAIL"} 1.0' in client.get("/metrics").text


def test_usage_tracks_cost_and_cache_savings(client):
    chat(client, "spend some money " * 20)
    chat(client, "spend some money " * 20)
    usage = client.get("/v1/usage", headers={"Authorization": "Bearer k-acme"}).json()
    assert usage["requests"] == 2
    assert usage["cost_usd"] > 0
    assert usage["cache_saved_usd"] == usage["cost_usd"]
    assert usage["cache_hit_rate"] == 0.5
    assert usage["by_model"][0]["model"] == "big"


def _read_sse(response):
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def test_streaming_emits_openai_chunks_and_records_usage(client):
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "default", "stream": True, "messages": [{"role": "user", "content": "stream please"}]},
        headers={"Authorization": "Bearer k-acme"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _read_sse(r)

    assert events[-1] == "[DONE]"
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1])
    assert "You said: stream please" in text
    final = events[-2]
    assert final["choices"][0]["finish_reason"] == "stop" and final["usage"]["completion_tokens"] > 0

    # The streamed answer was cached, and a non-streaming request can reuse it.
    assert chat(client, "stream please").headers["x-relay-cache"] == "exact"
    usage = client.get("/v1/usage", headers={"Authorization": "Bearer k-acme"}).json()
    assert usage["requests"] == 2


def test_streaming_falls_back_and_502s_like_non_streaming(client, providers):
    providers["primary"].failure_rate = 1.0
    body = {"model": "default", "stream": True, "messages": [{"role": "user", "content": "x"}]}
    headers = {"Authorization": "Bearer k-acme"}
    with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
        assert r.headers["x-relay-provider"] == "backup"
        _read_sse(r)
    providers["backup"].failure_rate = 1.0
    body["messages"][0]["content"] = "y"
    assert client.post("/v1/chat/completions", json=body, headers=headers).status_code == 502


def test_ledger_persists_across_restarts(tmp_path, providers):
    settings = make_settings(db_path=str(tmp_path / "usage.db"))
    with TestClient(create_app(settings, providers)) as c:
        chat(c, "persist me")
    with TestClient(create_app(settings, providers)) as c:
        assert c.get("/v1/usage", headers={"Authorization": "Bearer k-acme"}).json()["requests"] == 1
