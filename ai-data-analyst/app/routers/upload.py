import os
import logging
from typing import Any

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.models.schemas import FileUploadResponse
from app.services.data_processor import DataProcessor

logger = logging.getLogger(__name__)
router = APIRouter()
processor = DataProcessor()


# ── Inline request/response bodies used only in this router ──────────────────

class CleanRequest(BaseModel):
    filename: str
    options: dict = {}


class CleanResponse(BaseModel):
    original_filename: str
    cleaned_filename: str
    cleaning_report: dict[str, Any]
    overview: dict[str, Any]


class ColumnsResponse(BaseModel):
    filename: str
    column_names: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, Any]
    column_types: dict[str, str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_path(filename: str) -> str:
    """Return the full upload path and raise 404 if the file doesn't exist."""
    path = os.path.join(settings.upload_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return path


def _safe_filename(original: str) -> str:
    """Strip path components from a filename."""
    return os.path.basename(original).replace(" ", "_")


def _overview_to_response(filename: str, overview: dict) -> FileUploadResponse:
    """Map DataProcessor.get_overview() dict → FileUploadResponse."""
    shape = overview.get("shape", {})
    missing_raw = overview.get("missing_values", {})
    # Flatten missing_values: accept both {col: int} and {col: {count, percent}}
    missing_flat: dict[str, int] = {}
    for col, val in missing_raw.items():
        if isinstance(val, dict):
            missing_flat[col] = int(val.get("count", 0))
        else:
            missing_flat[col] = int(val)

    return FileUploadResponse(
        filename=filename,
        rows=shape.get("rows", 0) if isinstance(shape, dict) else (shape[0] if shape else 0),
        columns=shape.get("columns", 0) if isinstance(shape, dict) else (shape[1] if shape else 0),
        column_names=overview.get("column_names", []),
        dtypes=overview.get("dtypes", {}),
        missing_values=missing_flat,
        preview=overview.get("preview", []),
    )


# ── POST /file ────────────────────────────────────────────────────────────────

@router.post(
    "/file",
    response_model=FileUploadResponse,
    summary="Upload a data file",
    description=(
        "Upload a CSV, XLSX, or JSON file (max "
        f"{settings.max_file_size_mb} MB). "
        "Returns an overview of the dataset including shape, column types, "
        "missing values, and a 5-row preview."
    ),
)
async def upload_file(file: UploadFile = File(...)) -> FileUploadResponse:
    # ── Validate extension ────────────────────────────────────────────────
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in settings.get_allowed_extensions():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Extension '.{ext}' is not allowed. "
                f"Accepted: {settings.allowed_extensions}"
            ),
        )

    # ── Read & validate size ──────────────────────────────────────────────
    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size ({len(content) / 1024 / 1024:.1f} MB) exceeds "
                f"the maximum allowed {settings.max_file_size_mb} MB."
            ),
        )

    # ── Persist to uploads/ ───────────────────────────────────────────────
    safe_name = _safe_filename(file.filename or f"upload.{ext}")
    dest_path = os.path.join(settings.upload_dir, safe_name)

    # Avoid overwriting: append _1, _2, … if file already exists
    base, dot_ext = os.path.splitext(safe_name)
    counter = 1
    while os.path.exists(dest_path):
        safe_name = f"{base}_{counter}{dot_ext}"
        dest_path = os.path.join(settings.upload_dir, safe_name)
        counter += 1

    async with aiofiles.open(dest_path, "wb") as f:
        await f.write(content)

    logger.info("Uploaded '%s' → '%s' (%d bytes)", file.filename, dest_path, len(content))

    # ── Parse & return overview ───────────────────────────────────────────
    try:
        df = processor.load_file(dest_path)
        overview = processor.get_overview(df)
        return _overview_to_response(safe_name, overview)
    except Exception as exc:
        # Remove the saved file so broken uploads don't litter disk
        os.remove(dest_path)
        logger.error("Failed to parse uploaded file: %s", exc)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")


# ── POST /clean ───────────────────────────────────────────────────────────────

@router.post(
    "/clean",
    response_model=CleanResponse,
    summary="Auto-clean an uploaded file",
    description=(
        "Load a previously uploaded file, run automatic cleaning "
        "(imputation, deduplication, column standardisation, date parsing), "
        "and save a new `<name>_cleaned.<ext>` version. "
        "Returns the cleaning report and an overview of the cleaned dataset."
    ),
)
async def clean_file(body: CleanRequest) -> CleanResponse:
    src_path = _resolve_path(body.filename)

    try:
        df_raw = processor.load_file(src_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not load file: {exc}")

    try:
        df_clean, report = processor.auto_clean(df_raw, options=body.options)
    except Exception as exc:
        logger.error("auto_clean failed for '%s': %s", body.filename, exc)
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {exc}")

    # Save cleaned file as CSV regardless of original format
    base = os.path.splitext(body.filename)[0]
    cleaned_name = f"{base}_cleaned.csv"
    cleaned_path = os.path.join(settings.upload_dir, cleaned_name)
    try:
        df_clean.to_csv(cleaned_path, index=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save cleaned file: {exc}")

    logger.info("Cleaned '%s' → '%s'", body.filename, cleaned_name)

    overview = processor.get_overview(df_clean)
    return CleanResponse(
        original_filename=body.filename,
        cleaned_filename=cleaned_name,
        cleaning_report=report,
        overview=overview,
    )


# ── GET /preview/{filename} ───────────────────────────────────────────────────

@router.get(
    "/preview/{filename}",
    summary="Preview first 20 rows",
    description="Return the first 20 rows of an uploaded file as a JSON array.",
)
async def preview_file(filename: str) -> dict:
    path = _resolve_path(filename)
    try:
        df = processor.load_file(path)
        rows = df.head(20).replace({float("nan"): None}).to_dict(orient="records")
        return {
            "filename": filename,
            "total_rows": len(df),
            "preview_rows": len(rows),
            "data": rows,
        }
    except Exception as exc:
        logger.error("preview_file failed for '%s': %s", filename, exc)
        raise HTTPException(status_code=500, detail=f"Could not read file: {exc}")


# ── GET /columns/{filename} ───────────────────────────────────────────────────

@router.get(
    "/columns/{filename}",
    response_model=ColumnsResponse,
    summary="Get column metadata",
    description=(
        "Return column names, data types, missing-value counts, "
        "and semantic type classification for each column."
    ),
)
async def get_columns(filename: str) -> ColumnsResponse:
    path = _resolve_path(filename)
    try:
        df = processor.load_file(path)
        overview = processor.get_overview(df)
        col_types = processor.detect_column_types(df)
    except Exception as exc:
        logger.error("get_columns failed for '%s': %s", filename, exc)
        raise HTTPException(status_code=500, detail=f"Could not analyse columns: {exc}")

    return ColumnsResponse(
        filename=filename,
        column_names=overview.get("column_names", []),
        dtypes=overview.get("dtypes", {}),
        missing_values=overview.get("missing_values", {}),
        column_types=col_types,
    )


# ── DELETE /{filename} ────────────────────────────────────────────────────────

@router.delete(
    "/{filename}",
    summary="Delete an uploaded file",
    description="Permanently delete a file from the uploads directory.",
)
async def delete_file(filename: str) -> dict:
    path = _resolve_path(filename)
    try:
        os.remove(path)
        logger.info("Deleted '%s'", path)
        return {"detail": f"File '{filename}' deleted successfully."}
    except Exception as exc:
        logger.error("delete_file failed for '%s': %s", filename, exc)
        raise HTTPException(status_code=500, detail=f"Could not delete file: {exc}")
