"""
Tests for the /api/upload/* endpoints.

Covers:
  1. test_upload_csv              — happy-path CSV upload
  2. test_upload_invalid_extension — rejected file type returns 400
  3. test_upload_too_large         — file that exceeds size limit returns 413
  4. test_get_preview              — first 20 rows returned correctly
  5. test_auto_clean               — cleaning report + cleaned file created
  6. test_get_columns              — column metadata endpoint
  7. test_delete_file              — file deleted, subsequent preview returns 404
"""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── 1. Upload CSV ─────────────────────────────────────────────────────────────

def test_upload_csv(test_client: TestClient, sample_csv_path: Path):
    with open(sample_csv_path, "rb") as fh:
        response = test_client.post(
            "/api/upload/file",
            files={"file": ("sample.csv", fh, "text/csv")},
        )

    assert response.status_code == 200, response.text
    data = response.json()

    # Schema fields from FileUploadResponse
    assert "filename" in data
    assert "rows" in data
    assert "columns" in data
    assert "column_names" in data
    assert "dtypes" in data
    assert "missing_values" in data
    assert "preview" in data

    # Basic content checks against sample.csv
    assert data["rows"] > 0
    assert data["columns"] == 9  # id,date,product,category,quantity,price,revenue,customer_id,country
    assert "revenue" in data["column_names"]
    assert isinstance(data["preview"], list)
    assert len(data["preview"]) <= 5

    # Clean up
    test_client.delete(f"/api/upload/{data['filename']}")


# ── 2. Invalid extension ──────────────────────────────────────────────────────

def test_upload_invalid_extension(test_client: TestClient):
    response = test_client.post(
        "/api/upload/file",
        files={"file": ("data.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"].lower()


# ── 3. File too large ─────────────────────────────────────────────────────────

def test_upload_too_large(test_client: TestClient):
    from app.config import settings

    # Create an in-memory file 1 byte larger than the limit
    oversized = b"x" * (settings.max_file_size_bytes + 1)
    response = test_client.post(
        "/api/upload/file",
        files={"file": ("big.csv", io.BytesIO(oversized), "text/csv")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()


# ── 4. Preview ────────────────────────────────────────────────────────────────

def test_get_preview(test_client: TestClient, uploaded_file: str):
    response = test_client.get(f"/api/upload/preview/{uploaded_file}")

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["filename"] == uploaded_file
    assert "total_rows" in data
    assert "preview_rows" in data
    assert "data" in data

    assert data["total_rows"] > 0
    assert data["preview_rows"] <= 20
    assert isinstance(data["data"], list)
    # Each row should contain the expected columns
    if data["data"]:
        first_row = data["data"][0]
        assert "revenue" in first_row


# ── 5. Auto-clean ─────────────────────────────────────────────────────────────

def test_auto_clean(test_client: TestClient, uploaded_file: str):
    response = test_client.post(
        "/api/upload/clean",
        json={"filename": uploaded_file, "options": {}},
    )

    assert response.status_code == 200, response.text
    data = response.json()

    assert "original_filename" in data
    assert "cleaned_filename" in data
    assert "cleaning_report" in data
    assert "overview" in data

    assert data["original_filename"] == uploaded_file
    assert data["cleaned_filename"].endswith("_cleaned.csv")

    report = data["cleaning_report"]
    # Cleaning report structure: {before: {shape, ...}, after: {shape, ...}, actions: [...]}
    assert isinstance(report, dict)
    assert "before" in report
    assert "after" in report
    assert "actions" in report
    # sample.csv has duplicates + missing values → at least one action expected
    assert len(report["actions"]) >= 1
    # Row count should have decreased (duplicates removed)
    assert report["after"]["shape"][0] < report["before"]["shape"][0]

    # Clean up both files
    test_client.delete(f"/api/upload/{data['cleaned_filename']}")


# ── 6. Column metadata ────────────────────────────────────────────────────────

def test_get_columns(test_client: TestClient, uploaded_file: str):
    response = test_client.get(f"/api/upload/columns/{uploaded_file}")

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["filename"] == uploaded_file
    assert "column_names" in data
    assert "dtypes" in data
    assert "missing_values" in data
    assert "column_types" in data

    col_names = data["column_names"]
    assert "quantity" in col_names
    assert "revenue" in col_names
    assert "category" in col_names

    # Semantic types should be present for each column
    col_types = data["column_types"]
    assert isinstance(col_types, dict)
    assert len(col_types) == len(col_names)


# ── 7. Delete file ────────────────────────────────────────────────────────────

def test_delete_file(test_client: TestClient, sample_csv_path: Path):
    # Upload a fresh file specifically for this test
    with open(sample_csv_path, "rb") as fh:
        up = test_client.post(
            "/api/upload/file",
            files={"file": ("to_delete.csv", fh, "text/csv")},
        )
    assert up.status_code == 200
    filename = up.json()["filename"]

    # Delete it
    del_response = test_client.delete(f"/api/upload/{filename}")
    assert del_response.status_code == 200
    assert "deleted" in del_response.json()["detail"].lower()

    # Subsequent preview should return 404
    preview = test_client.get(f"/api/upload/preview/{filename}")
    assert preview.status_code == 404
