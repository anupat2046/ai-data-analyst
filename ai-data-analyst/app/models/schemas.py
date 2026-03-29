from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class AnalysisType(str, Enum):
    summary = "summary"
    correlation = "correlation"
    distribution = "distribution"
    anomaly = "anomaly"
    custom = "custom"


class MLTask(str, Enum):
    anomaly_detection = "anomaly_detection"
    clustering = "clustering"
    classification = "classification"


# ── Upload ────────────────────────────────────────────────────────────────────

class FileUploadResponse(BaseModel):
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    preview: list[dict[str, Any]]


# ── Analysis ──────────────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    analysis_type: str
    results: dict[str, Any]
    charts: list[str]
    insights: list[str]
    timestamp: datetime


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    filename: str
    question: str
    conversation_history: list[dict[str, str]] = []


class ChatResponse(BaseModel):
    answer: str
    code_executed: str | None = None
    charts: list[str] = []
    data_preview: list[dict[str, Any]] | None = None


# ── ML ────────────────────────────────────────────────────────────────────────

class MLRequest(BaseModel):
    filename: str
    task: MLTask
    target_column: str | None = None
    features: list[str] | None = None


class MLResponse(BaseModel):
    task: str
    model_name: str
    results: dict[str, Any]
    metrics: dict[str, Any]
    charts: list[str]
    insights: list[str]
