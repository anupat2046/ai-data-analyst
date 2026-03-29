import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.models.schemas import (
    AnalysisResponse,
    MLRequest,
    MLResponse,
)
from app.services.data_processor import DataProcessor
from app.services.eda_engine import EDAEngine
from app.services.llm_service import LLMService
from app.services.ml_engine import MLEngine

logger = logging.getLogger(__name__)
router = APIRouter()

processor = DataProcessor()
eda = EDAEngine()
ml = MLEngine()


# ── Inline request models used only in this router ───────────────────────────

class FilenameRequest(BaseModel):
    filename: str


class DistributionRequest(BaseModel):
    filename: str
    column: str


class BivariateRequest(BaseModel):
    filename: str
    col1: str
    col2: str


class TimeSeriesRequest(BaseModel):
    filename: str
    date_col: str
    value_col: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_path(filename: str) -> str:
    path = os.path.join(settings.upload_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return path


def _load(filename: str):
    """Load file → DataFrame, raise 404 / 422 on failure."""
    path = _resolve_path(filename)
    try:
        return processor.load_file(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not load file: {exc}")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── POST /auto-eda ────────────────────────────────────────────────────────────

@router.post(
    "/auto-eda",
    response_model=AnalysisResponse,
    summary="Run automatic EDA",
    description=(
        "Perform a full automatic Exploratory Data Analysis on the uploaded file. "
        "Returns summary statistics, missing-value breakdown, correlation heatmap, "
        "distribution charts for every numeric column, bar charts for categorical "
        "columns, and a list of auto-generated text insights."
    ),
)
async def auto_eda(body: FilenameRequest) -> AnalysisResponse:
    df = _load(body.filename)
    try:
        result = eda.auto_eda(df)
    except Exception as exc:
        logger.error("auto_eda failed for '%s': %s", body.filename, exc)
        raise HTTPException(status_code=500, detail=f"EDA failed: {exc}")

    # Flatten all base64 chart strings into a single list
    charts: list[str] = []
    if result.get("correlation_matrix"):
        charts.append(result["correlation_matrix"])
    for item in result.get("distribution_charts", []):
        charts.extend(item.get("charts", []))
    for item in result.get("categorical_charts", []):
        charts.extend(item.get("charts", []))

    return AnalysisResponse(
        analysis_type="auto_eda",
        results={
            "summary_stats": result.get("summary_stats", {}),
            "missing_analysis": result.get("missing_analysis", {}),
        },
        charts=charts,
        insights=result.get("insights", []),
        timestamp=_now(),
    )


# ── POST /correlation ─────────────────────────────────────────────────────────

@router.post(
    "/correlation",
    response_model=AnalysisResponse,
    summary="Correlation analysis",
    description=(
        "Compute pairwise Pearson correlations for all numeric columns. "
        "Returns an annotated heatmap, the top 10 highly-correlated pairs, "
        "and narrative insights."
    ),
)
async def correlation(body: FilenameRequest) -> AnalysisResponse:
    df = _load(body.filename)
    try:
        result = eda.correlation_analysis(df)
    except Exception as exc:
        logger.error("correlation failed for '%s': %s", body.filename, exc)
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {exc}")

    charts = [result["chart"]] if result.get("chart") else []
    return AnalysisResponse(
        analysis_type="correlation",
        results={"top_correlations": result.get("top_correlations", [])},
        charts=charts,
        insights=result.get("insights", []),
        timestamp=_now(),
    )


# ── POST /distribution ────────────────────────────────────────────────────────

@router.post(
    "/distribution",
    response_model=AnalysisResponse,
    summary="Distribution analysis for a single column",
    description=(
        "Generate histogram + KDE, box plot, and Q-Q plot for the specified "
        "numeric column. Includes descriptive statistics and a normality test "
        "(Shapiro-Wilk for n < 5000, Kolmogorov-Smirnov otherwise)."
    ),
)
async def distribution(body: DistributionRequest) -> AnalysisResponse:
    df = _load(body.filename)
    if body.column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{body.column}' not found. "
                   f"Available columns: {df.columns.tolist()}",
        )
    try:
        result = eda.distribution_analysis(df, body.column)
    except Exception as exc:
        logger.error("distribution failed for column '%s': %s", body.column, exc)
        raise HTTPException(status_code=500, detail=f"Distribution analysis failed: {exc}")

    return AnalysisResponse(
        analysis_type="distribution",
        results={"column": body.column, "stats": result.get("stats", {})},
        charts=result.get("charts", []),
        insights=result.get("insights", []),
        timestamp=_now(),
    )


# ── POST /anomaly ─────────────────────────────────────────────────────────────

@router.post(
    "/anomaly",
    summary="Anomaly detection (Isolation Forest)",
    description=(
        "Run Isolation Forest anomaly detection on the specified features "
        "(defaults to all numeric columns). Returns anomaly count, indices, "
        "a 2-D PCA scatter chart, and the top 10 most anomalous rows."
    ),
)
async def anomaly(body: MLRequest) -> dict[str, Any]:
    if body.task != "anomaly_detection":
        raise HTTPException(
            status_code=400,
            detail="Set task='anomaly_detection' for this endpoint.",
        )
    df = _load(body.filename)
    result = ml.anomaly_detection(df, features=body.features)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


# ── POST /clustering ──────────────────────────────────────────────────────────

@router.post(
    "/clustering",
    summary="K-Means clustering",
    description=(
        "Cluster the dataset using K-Means. If `n_clusters` is omitted, the "
        "optimal k is chosen automatically via the silhouette score (k = 2–10). "
        "Returns cluster sizes, centres, silhouette score, and three charts: "
        "elbow curve, 2-D PCA scatter, and cluster profile."
    ),
)
async def clustering(body: MLRequest) -> dict[str, Any]:
    if body.task != "clustering":
        raise HTTPException(
            status_code=400,
            detail="Set task='clustering' for this endpoint.",
        )
    df = _load(body.filename)

    # MLRequest uses target_column to carry n_clusters for clustering
    n_clusters: int | None = None
    if body.target_column and body.target_column.isdigit():
        n_clusters = int(body.target_column)

    result = ml.clustering(df, features=body.features, n_clusters=n_clusters)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


# ── POST /classification ──────────────────────────────────────────────────────

@router.post(
    "/classification",
    summary="Classification (Logistic Regression / Random Forest / Gradient Boosting)",
    description=(
        "Train and compare three classifiers on the dataset. "
        "`target_column` is required. `features` defaults to all numeric columns "
        "except the target. Returns accuracy per model, the best model's confusion "
        "matrix, feature importance chart, and (for binary tasks) an ROC curve."
    ),
)
async def classification(body: MLRequest) -> dict[str, Any]:
    if body.task != "classification":
        raise HTTPException(
            status_code=400,
            detail="Set task='classification' for this endpoint.",
        )
    if not body.target_column:
        raise HTTPException(
            status_code=400,
            detail="'target_column' is required for classification.",
        )
    df = _load(body.filename)
    result = ml.classification_report(
        df, target=body.target_column, features=body.features, filename=body.filename
    )
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


# ── POST /report ──────────────────────────────────────────────────────────────

@router.post(
    "/report",
    summary="Generate AI executive report",
    description=(
        "Run auto EDA and anomaly detection, then send the combined results to "
        "Gemini to produce a markdown-formatted executive report with summary, "
        "key metrics, insights, anomalies, and actionable recommendations."
    ),
)
async def generate_report(body: FilenameRequest) -> dict[str, Any]:
    df = _load(body.filename)

    # 1. EDA
    try:
        eda_result = eda.auto_eda(df)
    except Exception as exc:
        logger.error("EDA step failed in report generation: %s", exc)
        eda_result = {}

    # 2. Anomaly detection
    try:
        anomaly_result = ml.anomaly_detection(df)
    except Exception as exc:
        logger.error("Anomaly step failed in report generation: %s", exc)
        anomaly_result = {}

    # 3. LLM report
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured. Set it in your .env file.",
        )
    try:
        llm = LLMService(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
        overview = processor.get_overview(df)
        combined_results = {
            "eda_summary": {
                k: v for k, v in eda_result.items()
                if k not in ("correlation_matrix", "distribution_charts", "categorical_charts")
            },
            "anomaly_summary": {
                k: v for k, v in anomaly_result.items()
                if k not in ("charts", "sample_anomalies")
            },
        }
        report_md = llm.generate_report(df_info=overview, analysis_results=combined_results)
    except Exception as exc:
        logger.error("LLM report generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Report generation failed: {exc}")

    return {
        "filename": body.filename,
        "report": report_md,
        "timestamp": _now().isoformat(),
    }


# ── POST /bivariate ───────────────────────────────────────────────────────────

@router.post(
    "/bivariate",
    summary="Bivariate analysis between two columns",
    description=(
        "Analyse the relationship between two columns. "
        "Numeric–numeric: scatter + regression. "
        "Numeric–categorical: grouped box plot. "
        "Categorical–categorical: cross-tabulation heatmap."
    ),
)
async def bivariate(body: BivariateRequest) -> dict[str, Any]:
    df = _load(body.filename)
    for col in (body.col1, body.col2):
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' not found. Available: {df.columns.tolist()}",
            )
    try:
        result = eda.bivariate_analysis(df, body.col1, body.col2)
    except Exception as exc:
        logger.error("bivariate failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Bivariate analysis failed: {exc}")
    return result


# ── POST /timeseries ──────────────────────────────────────────────────────────

@router.post(
    "/timeseries",
    summary="Time-series analysis",
    description=(
        "Analyse a time-series column. Produces a trend + moving-average line chart, "
        "monthly and weekly aggregation bar/line charts, and (when enough data is "
        "available) a seasonal decomposition chart."
    ),
)
async def timeseries(body: TimeSeriesRequest) -> dict[str, Any]:
    df = _load(body.filename)
    for col in (body.date_col, body.value_col):
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' not found. Available: {df.columns.tolist()}",
            )
    try:
        result = eda.time_series_analysis(df, body.date_col, body.value_col)
    except Exception as exc:
        logger.error("timeseries failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Time-series analysis failed: {exc}")
    return result
