import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.models.schemas import ChatRequest, ChatResponse
from app.services.data_processor import DataProcessor
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter()

processor = DataProcessor()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_path(filename: str) -> str:
    path = os.path.join(settings.upload_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return path


def _get_llm() -> LLMService:
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY is not configured. "
                "Add it to your .env file and restart the server."
            ),
        )
    return LLMService(api_key=settings.gemini_api_key, model=settings.gemini_model)


def _load_overview(filename: str) -> dict:
    """Load file and return its overview dict for use as LLM context."""
    path = _resolve_path(filename)
    try:
        df = processor.load_file(path)
        return processor.get_overview(df)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read file '{filename}': {exc}",
        )


# ── Inline request model ──────────────────────────────────────────────────────

class SuggestRequest(BaseModel):
    filename: str


# ── POST /ask ─────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="Ask AI a question about the dataset",
    description=(
        "Send a natural-language question about an uploaded file to Gemini. "
        "The dataset's shape, column types, statistics, and a 5-row preview are "
        "automatically included as context. Supports both Thai and English. "
        "If the response contains a Python code block it is extracted and returned "
        "separately in `code_executed`."
    ),
)
async def ask(body: ChatRequest) -> ChatResponse:
    overview = _load_overview(body.filename)
    llm = _get_llm()

    try:
        result = llm.analyze_data(
            df_info=overview,
            question=body.question,
            conversation_history=body.conversation_history,
        )
    except RuntimeError as exc:
        # Surface LLM-layer errors (rate-limit, invalid key, …) clearly
        status = 429 if "limit" in str(exc).lower() else 502
        raise HTTPException(status_code=status, detail=str(exc))
    except Exception as exc:
        logger.error("LLM ask failed for '%s': %s", body.filename, exc)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")

    return ChatResponse(
        answer=result.get("answer", ""),
        code_executed=result.get("suggested_code"),
        charts=[],          # placeholder — future: run code_executed & capture charts
        data_preview=None,
    )


# ── POST /suggest ─────────────────────────────────────────────────────────────

@router.post(
    "/suggest",
    summary="Suggest analysis questions for a dataset",
    description=(
        "Ask Gemini to recommend 5–8 analyses or questions that would be most "
        "valuable for the uploaded dataset. Returns a list of objects with "
        "`type`, `description`, and `reason` fields. "
        "Also includes generic follow-up questions derived from column names."
    ),
)
async def suggest(body: SuggestRequest) -> dict:
    overview = _load_overview(body.filename)
    llm = _get_llm()

    # 1. AI-generated suggestions
    try:
        suggestions = llm.suggest_analysis(df_info=overview)
    except RuntimeError as exc:
        status = 429 if "limit" in str(exc).lower() else 502
        raise HTTPException(status_code=status, detail=str(exc))
    except Exception as exc:
        logger.error("suggest failed for '%s': %s", body.filename, exc)
        raise HTTPException(status_code=500, detail=f"Suggest failed: {exc}")

    # 2. Generic follow-up questions from column names
    cols = overview.get("column_names", [])
    follow_ups: list[str] = [
        "What are the main trends in this dataset?",
        "Are there any data quality issues I should address?",
    ]
    if cols:
        follow_ups.append(f"What does the distribution of '{cols[0]}' look like?")
    if len(cols) >= 2:
        follow_ups.append(
            f"Is there a significant relationship between '{cols[0]}' and '{cols[1]}'?"
        )
    follow_ups.append("Which columns are most important for predictive modelling?")

    return {
        "filename": body.filename,
        "suggestions": suggestions,
        "follow_up_questions": follow_ups,
    }


# ── POST /explain-anomalies ───────────────────────────────────────────────────

@router.post(
    "/explain-anomalies",
    summary="Get AI explanation for anomaly detection results",
    description=(
        "Pass in the result from `POST /api/analysis/anomaly` together with "
        "the filename, and Gemini will explain why the flagged rows are anomalous, "
        "suggest possible real-world causes, and recommend actions to take."
    ),
)
async def explain_anomalies(body: dict) -> dict:
    filename: str = body.get("filename", "")
    anomaly_data: dict = body.get("anomaly_data", {})

    if not filename:
        raise HTTPException(status_code=400, detail="'filename' is required.")
    if not anomaly_data:
        raise HTTPException(
            status_code=400,
            detail="'anomaly_data' is required. Pass the result from /api/analysis/anomaly.",
        )

    overview = _load_overview(filename)
    llm = _get_llm()

    try:
        explanation = llm.explain_anomalies(
            anomaly_data=anomaly_data, df_info=overview
        )
    except RuntimeError as exc:
        status = 429 if "limit" in str(exc).lower() else 502
        raise HTTPException(status_code=status, detail=str(exc))
    except Exception as exc:
        logger.error("explain_anomalies failed for '%s': %s", filename, exc)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")

    return {"filename": filename, "explanation": explanation}
