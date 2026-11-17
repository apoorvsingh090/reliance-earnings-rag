"""Milestone 9 tests: API input bounds."""

from __future__ import annotations

from fastapi.testclient import TestClient

import main
from main import _clamp_k


def test_clamp_k_keeps_sane_values():
    assert _clamp_k(8) == 8
    assert _clamp_k(1) == 1
    assert _clamp_k(50) == 50


def test_clamp_k_bounds_abuse():
    assert _clamp_k(0) == 1
    assert _clamp_k(-5) == 1
    assert _clamp_k(1_000_000) == 50


def test_search_endpoint_clamps_k():
    # Wiring: an abusive k must not reach retrieval unbounded.
    client = TestClient(main.app)
    r = client.get("/search", params={"q": "revenue", "k": 1_000_000, "method": "bm25"})
    assert r.status_code == 200
    assert len(r.json()["hits"]) <= 50
    r = client.get("/search", params={"q": "revenue", "k": -3, "method": "bm25"})
    assert r.status_code == 200
    assert len(r.json()["hits"]) >= 1
