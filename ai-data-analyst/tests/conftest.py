"""
Shared pytest fixtures for the AI Data Analyst test suite.
"""
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure a clean upload dir for tests so they don't collide with dev uploads
_TEST_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "ai_analyst_test_uploads")

# Patch the upload_dir BEFORE app is imported so Settings picks it up
os.environ.setdefault("UPLOAD_DIR", _TEST_UPLOAD_DIR)

from app.main import app  # noqa: E402 — must come after env patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """FastAPI TestClient shared across the entire test session."""
    os.makedirs(_TEST_UPLOAD_DIR, exist_ok=True)
    with TestClient(app) as client:
        yield client
    # Cleanup uploads created during tests
    shutil.rmtree(_TEST_UPLOAD_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_csv_path() -> Path:
    """Absolute path to the bundled sample CSV file."""
    p = Path(__file__).parent / "test_data" / "sample.csv"
    assert p.exists(), f"sample.csv not found at {p}"
    return p


@pytest.fixture()
def mock_gemini():
    """
    Patch google.genai.Client so no real API calls are made.

    The mock returns a simple response object whose .text attribute
    carries a canned answer string.
    """
    fake_text = (
        "Based on your dataset, revenue is highest in the Electronics category. "
        "No further action required."
    )
    mock_response = MagicMock()
    mock_response.text = fake_text

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client_instance) as mock_cls:
        yield mock_cls


@pytest.fixture()
def uploaded_file(test_client: TestClient, sample_csv_path: Path) -> str:
    """
    Upload sample.csv via the API and return the server-assigned filename.

    The fixture tears down by deleting the file after the test.
    """
    with open(sample_csv_path, "rb") as fh:
        response = test_client.post(
            "/api/upload/file",
            files={"file": ("sample.csv", fh, "text/csv")},
        )
    assert response.status_code == 200, response.text
    filename = response.json()["filename"]
    yield filename
    # Best-effort cleanup
    test_client.delete(f"/api/upload/{filename}")
