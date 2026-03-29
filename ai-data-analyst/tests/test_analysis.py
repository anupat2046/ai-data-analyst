"""
Tests for the /api/analysis/* and /api/chat/* endpoints.

Covers:
  1. test_auto_eda         — full EDA returns stats + charts + insights
  2. test_correlation      — correlation heatmap + top pairs
  3. test_distribution     — distribution charts for a numeric column
  4. test_anomaly          — Isolation Forest result shape
  5. test_clustering       — K-Means result shape
  6. test_timeseries       — time-series analysis on date+revenue columns
  7. test_chat_ask         — /api/chat/ask with mocked Gemini
  8. test_analysis_file_not_found — 404 on missing file
"""
from fastapi.testclient import TestClient


# ── 1. Auto EDA ───────────────────────────────────────────────────────────────

def test_auto_eda(test_client: TestClient, uploaded_file: str):
    response = test_client.post(
        "/api/analysis/auto-eda",
        json={"filename": uploaded_file},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["analysis_type"] == "auto_eda"
    assert "results" in data
    assert "charts" in data
    assert "insights" in data
    assert "timestamp" in data

    results = data["results"]
    assert "summary_stats" in results
    assert "missing_analysis" in results
    # Numeric columns exist → at least one chart generated
    assert len(data["charts"]) > 0


# ── 2. Correlation ────────────────────────────────────────────────────────────

def test_correlation(test_client: TestClient, uploaded_file: str):
    response = test_client.post(
        "/api/analysis/correlation",
        json={"filename": uploaded_file},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["analysis_type"] == "correlation"
    assert len(data["charts"]) >= 1           # heatmap
    assert "top_correlations" in data["results"]
    assert isinstance(data["results"]["top_correlations"], list)


# ── 3. Distribution ───────────────────────────────────────────────────────────

def test_distribution(test_client: TestClient, uploaded_file: str):
    response = test_client.post(
        "/api/analysis/distribution",
        json={"filename": uploaded_file, "column": "revenue"},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["analysis_type"] == "distribution"
    assert data["results"]["column"] == "revenue"
    assert "stats" in data["results"]
    assert len(data["charts"]) > 0


def test_distribution_invalid_column(test_client: TestClient, uploaded_file: str):
    response = test_client.post(
        "/api/analysis/distribution",
        json={"filename": uploaded_file, "column": "nonexistent_col"},
    )
    assert response.status_code == 400


# ── 4. Anomaly detection ──────────────────────────────────────────────────────

def test_anomaly(test_client: TestClient, uploaded_file: str):
    response = test_client.post(
        "/api/analysis/anomaly",
        json={
            "filename": uploaded_file,
            "task": "anomaly_detection",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert "total_rows" in data
    assert "anomaly_count" in data
    assert "anomaly_indices" in data
    assert data["total_rows"] > 0
    # sample.csv has 3 extreme outliers — expect at least 1 flagged
    assert data["anomaly_count"] >= 1


# ── 5. Clustering ─────────────────────────────────────────────────────────────

def test_clustering(test_client: TestClient, uploaded_file: str):
    response = test_client.post(
        "/api/analysis/clustering",
        json={
            "filename": uploaded_file,
            "task": "clustering",
            "target_column": "3",   # request k=3 clusters
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert "n_clusters" in data
    assert "cluster_sizes" in data
    assert "silhouette_score" in data
    assert data["n_clusters"] == 3


# ── 6. Time-series ────────────────────────────────────────────────────────────

def test_timeseries(test_client: TestClient, uploaded_file: str):
    response = test_client.post(
        "/api/analysis/timeseries",
        json={
            "filename": uploaded_file,
            "date_col": "date",
            "value_col": "revenue",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert "charts" in data
    assert len(data["charts"]) > 0


def test_timeseries_invalid_column(test_client: TestClient, uploaded_file: str):
    response = test_client.post(
        "/api/analysis/timeseries",
        json={
            "filename": uploaded_file,
            "date_col": "no_such_date",
            "value_col": "revenue",
        },
    )
    assert response.status_code == 400


# ── 7. Chat /ask with mocked Gemini ──────────────────────────────────────────

def test_chat_ask(test_client: TestClient, uploaded_file: str, mock_gemini):
    response = test_client.post(
        "/api/chat/ask",
        json={
            "filename": uploaded_file,
            "question": "Which category has the highest revenue?",
            "conversation_history": [],
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert "answer" in data
    assert len(data["answer"]) > 0
    # Gemini mock should have been called exactly once
    mock_gemini.return_value.models.generate_content.assert_called_once()


def test_chat_ask_no_api_key(test_client: TestClient, uploaded_file: str, monkeypatch):
    """When GEMINI_API_KEY is empty the endpoint must return 503."""
    from app.config import settings
    monkeypatch.setattr(settings, "gemini_api_key", "")

    response = test_client.post(
        "/api/chat/ask",
        json={
            "filename": uploaded_file,
            "question": "What is the total revenue?",
        },
    )
    assert response.status_code == 503


# ── 8. 404 on missing file ────────────────────────────────────────────────────

def test_analysis_file_not_found(test_client: TestClient):
    response = test_client.post(
        "/api/analysis/auto-eda",
        json={"filename": "does_not_exist.csv"},
    )
    assert response.status_code == 404
