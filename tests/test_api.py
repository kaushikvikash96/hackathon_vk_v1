"""API contract and concurrency.

The harness sends up to three simultaneous /query requests and treats a
malformed or missing ``answer`` as zero, so both properties are tested here
rather than discovered on evaluation day.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import app  # noqa: E402
from src.schemas import QueryResponse  # noqa: E402

VALIDATE_SCHEMA = (
    Path(__file__).resolve().parents[1] / "Participant_Package" / "validate.json"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_datasets_loaded(client):
    assert client.get("/health").json()["datasets_loaded"] is True


def test_query_returns_the_required_shape(client):
    response = client.post("/query", json={"question": "How many RBA decision records are there?"})
    assert response.status_code == 200
    payload = response.json()

    assert isinstance(payload["answer"], str) and payload["answer"].strip()
    assert isinstance(payload["steps"], int) and payload["steps"] >= 0
    assert isinstance(payload["tool_trace"], list)
    for entry in payload["tool_trace"]:
        assert set(entry) >= {"tool", "args", "result"}
        assert isinstance(entry["args"], dict)


def test_query_response_satisfies_validate_json(client):
    """Field-for-field against the organizer's published JSON Schema."""
    schema = json.loads(VALIDATE_SCHEMA.read_text(encoding="utf-8"))
    payload = client.post("/query", json={"question": "What is the highest cash-rate target?"}).json()

    for field in schema["required"]:
        assert field in payload, f"missing required field {field}"
    assert len(payload["answer"]) >= schema["properties"]["answer"]["minLength"]
    assert payload["steps"] >= schema["properties"]["steps"]["minimum"]


def test_empty_question_is_rejected_not_crashed(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_malformed_body_is_rejected(client):
    assert client.post("/query", json={"prompt": "wrong field"}).status_code == 422


def test_nonsense_question_still_returns_an_answer(client):
    """Never return an empty answer - the rules require a stated limitation."""
    payload = client.post("/query", json={"question": "asdfghjkl qwerty zxcv?"}).json()
    assert payload["answer"].strip()


def test_three_concurrent_queries_stay_independent(client):
    questions = [
        "What is the lowest cash-rate target in the RBA dataset, when did it first take effect, and how many decision records show that rate?",
        "Excluding Tabcorp, which ticker has the highest average daily volume over the full sample?",
        "What are the dimensions and common date range of the ASX dataset?",
    ]

    def ask(question: str) -> dict:
        return client.post("/query", json={"question": question}).json()

    with ThreadPoolExecutor(max_workers=3) as pool:
        payloads = list(pool.map(ask, questions))

    answers = [p["answer"] for p in payloads]
    assert all(answers), "a concurrent request returned an empty answer"
    assert len(set(answers)) == 3, "responses were mixed between concurrent requests"

    assert "0.1" in answers[0] and "2020-11-04" in answers[0]
    assert "AMP.AX" in answers[1]
    assert "18" in answers[2] and "1,774" in answers[2]


def test_response_model_rejects_an_empty_answer():
    with pytest.raises(Exception):
        QueryResponse(answer="", steps=1, tool_trace=[])
