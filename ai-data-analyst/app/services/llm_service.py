import json
import logging
import re
import time
from datetime import date
from typing import Any

from google import genai
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)

# ── Rate-limit constants (Gemini free tier) ───────────────────────────────────
_RPM_DELAY = 6          # seconds between consecutive calls  (10 RPM ≈ 1 call / 6s)
_DAILY_LIMIT = 250
_RATE_LIMIT_WAIT = 65   # seconds to pause after a 429 response
_MAX_RETRIES = 3
_MAX_CONTEXT_CHARS = 12_000   # truncate data context if too long

_SYSTEM_INSTRUCTION = """You are a senior data analyst assistant. You analyze datasets \
and provide clear, actionable insights. When asked about data:
1. Explain findings in simple language
2. Provide specific numbers and percentages
3. Suggest business actions based on insights
4. If the user asks in Thai, respond in Thai
5. Format responses with clear sections using markdown

You will receive data context including column info, statistics, and sample data. \
Use this to answer questions accurately."""


class LLMService:
    """Thin wrapper around the Google Gemini API with rate-limit management."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is empty. Set it in your .env file and never hardcode it."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

        # Daily request counter — resets when the calendar date changes
        self._request_count: int = 0
        self._count_date: date = date.today()
        self._last_call_ts: float = 0.0

        logger.info("LLMService initialised — model=%s", self.model)

    # ------------------------------------------------------------------ #
    # 2. analyze_data                                                      #
    # ------------------------------------------------------------------ #

    def analyze_data(
        self,
        df_info: dict,
        question: str,
        conversation_history: list[dict] = [],
    ) -> dict:
        """Ask Gemini a question about a dataset.

        Args:
            df_info: dict produced by DataProcessor.get_overview() or similar.
            question: User's natural-language question.
            conversation_history: Prior turns [{role, content}, ...] (optional).

        Returns:
            {answer, suggested_code, follow_up_questions}
        """
        logger.info("analyze_data — question=%r", question[:80])
        context = self._build_data_context(df_info)

        history_text = ""
        if conversation_history:
            turns = []
            for turn in conversation_history[-6:]:   # keep last 6 turns to stay within limits
                role = turn.get("role", "user").capitalize()
                turns.append(f"**{role}:** {turn.get('content', '')}")
            history_text = "\n\n---\n**Conversation history:**\n" + "\n\n".join(turns)

        prompt = (
            f"{_SYSTEM_INSTRUCTION}\n\n"
            f"---\n**Dataset context:**\n{context}"
            f"{history_text}\n\n"
            f"---\n**Question:** {question}"
        )

        raw = self._call_api(prompt)

        answer = raw
        suggested_code: str | None = None

        # Extract fenced code blocks
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        if code_blocks:
            suggested_code = "\n\n".join(b.strip() for b in code_blocks)
            # Remove code blocks from the answer text
            answer = re.sub(r"```(?:python)?.*?```", "", raw, flags=re.DOTALL).strip()

        follow_ups = self._extract_follow_ups(raw) or self._generate_follow_ups(df_info)

        return {
            "answer": answer,
            "suggested_code": suggested_code,
            "follow_up_questions": follow_ups,
        }

    # ------------------------------------------------------------------ #
    # 3. generate_report                                                   #
    # ------------------------------------------------------------------ #

    def generate_report(self, df_info: dict, analysis_results: dict) -> str:
        """Generate a markdown executive report from overview + analysis results."""
        logger.info("generate_report called")
        context = self._build_data_context(df_info)

        # Summarise analysis results — avoid sending huge nested dicts
        results_summary = self._summarise_analysis(analysis_results)

        prompt = (
            f"{_SYSTEM_INSTRUCTION}\n\n"
            "You are creating a professional **executive data analysis report**.\n\n"
            f"---\n**Dataset overview:**\n{context}\n\n"
            f"---\n**Analysis results:**\n{results_summary}\n\n"
            "---\n**Task:** Write a concise executive report in markdown with these exact sections:\n"
            "## Executive Summary\n"
            "- 3–5 bullet points covering the most important findings\n\n"
            "## Key Metrics\n"
            "- Important numbers, ranges, and distributions\n\n"
            "## Insights & Findings\n"
            "- Detailed observations with specific values\n\n"
            "## Anomalies & Risks\n"
            "- Any unusual patterns or data quality issues\n\n"
            "## Recommendations\n"
            "- 3–5 concrete, actionable next steps\n"
        )

        return self._call_api(prompt)

    # ------------------------------------------------------------------ #
    # 4. explain_anomalies                                                 #
    # ------------------------------------------------------------------ #

    def explain_anomalies(self, anomaly_data: dict, df_info: dict) -> str:
        """Return a markdown explanation of anomaly detection results."""
        logger.info("explain_anomalies called — count=%s", anomaly_data.get("anomaly_count"))
        context = self._build_data_context(df_info)

        anomaly_count = anomaly_data.get("anomaly_count", "?")
        anomaly_pct = anomaly_data.get("anomaly_percentage", "?")
        features_used = anomaly_data.get("features_used", [])
        sample = anomaly_data.get("sample_anomalies", [])

        sample_text = ""
        if sample:
            sample_text = (
                "\n**Sample anomalous rows (top 10):**\n"
                + json.dumps(sample[:10], indent=2, default=str)
            )

        prompt = (
            f"{_SYSTEM_INSTRUCTION}\n\n"
            f"---\n**Dataset context:**\n{context}\n\n"
            "---\n**Anomaly detection results (Isolation Forest):**\n"
            f"- Total anomalies found: {anomaly_count} ({anomaly_pct}% of data)\n"
            f"- Features analysed: {', '.join(features_used)}\n"
            f"{sample_text}\n\n"
            "---\n**Task:** Write a markdown explanation covering:\n"
            "1. Why these rows are likely anomalies (reference specific column values)\n"
            "2. Possible real-world causes\n"
            "3. Recommended actions (investigate, remove, flag, etc.)\n"
            "4. Any data quality concerns\n"
        )

        return self._call_api(prompt)

    # ------------------------------------------------------------------ #
    # 5. suggest_analysis                                                  #
    # ------------------------------------------------------------------ #

    def suggest_analysis(self, df_info: dict) -> list[dict]:
        """Return a list of suggested analyses for the dataset.

        Each item: {type, description, reason}
        """
        logger.info("suggest_analysis called")
        context = self._build_data_context(df_info)

        prompt = (
            f"{_SYSTEM_INSTRUCTION}\n\n"
            f"---\n**Dataset context:**\n{context}\n\n"
            "---\n**Task:** Based on this dataset, suggest 5–8 analyses that would be most valuable.\n"
            "Return a **JSON array only** (no markdown fences, no extra text) where each item has:\n"
            '  {"type": "<analysis_type>", "description": "<one sentence>", "reason": "<why useful>"}\n\n'
            "Valid analysis types: summary, correlation, distribution, anomaly, clustering, "
            "classification, time_series, categorical, bivariate, custom"
        )

        raw = self._call_api(prompt)

        # Strip optional markdown fences
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            suggestions = json.loads(clean)
            if isinstance(suggestions, list):
                return suggestions
            raise ValueError("Response is not a JSON array.")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("suggest_analysis — JSON parse failed (%s), returning raw text", exc)
            return [{"type": "custom", "description": raw, "reason": "Parsed from raw LLM response"}]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _call_api(self, prompt: str) -> str:
        """Call Gemini with retry logic and rate-limit awareness."""
        self._reset_daily_counter_if_new_day()

        if self._request_count >= _DAILY_LIMIT:
            msg = (
                f"Daily request limit reached ({_DAILY_LIMIT} requests). "
                "The counter resets at midnight."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if self._request_count >= _DAILY_LIMIT * 0.9:
            logger.warning(
                "Approaching daily limit — %d / %d requests used today.",
                self._request_count, _DAILY_LIMIT,
            )

        for attempt in range(1, _MAX_RETRIES + 1):
            self._throttle()
            try:
                logger.debug("Calling Gemini API (attempt %d/%d)", attempt, _MAX_RETRIES)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                self._request_count += 1
                self._last_call_ts = time.monotonic()
                logger.info(
                    "API call success — request #%d today", self._request_count
                )
                return response.text or ""

            except genai_errors.ClientError as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if status == 429:
                    logger.warning(
                        "Rate limit hit (429) on attempt %d. Waiting %ds…",
                        attempt, _RATE_LIMIT_WAIT,
                    )
                    time.sleep(_RATE_LIMIT_WAIT)
                    continue
                if status in (401, 403):
                    raise RuntimeError(
                        "Invalid or unauthorised Gemini API key. "
                        "Check GEMINI_API_KEY in your .env file."
                    ) from exc
                logger.error("ClientError on attempt %d: %s", attempt, exc)
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(f"Gemini API error after {_MAX_RETRIES} attempts: {exc}") from exc
                time.sleep(2 ** attempt)

            except genai_errors.ServerError as exc:
                logger.error("ServerError on attempt %d: %s", attempt, exc)
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(f"Gemini server error after {_MAX_RETRIES} attempts: {exc}") from exc
                time.sleep(2 ** attempt)

            except Exception as exc:
                logger.error("Unexpected API error on attempt %d: %s", attempt, exc)
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(f"Unexpected error calling Gemini: {exc}") from exc
                time.sleep(2 ** attempt)

        raise RuntimeError("All retry attempts exhausted.")  # should not reach

    def _throttle(self) -> None:
        """Ensure at least _RPM_DELAY seconds between consecutive API calls."""
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < _RPM_DELAY:
            wait = _RPM_DELAY - elapsed
            logger.debug("Throttling — sleeping %.1fs to respect rate limit", wait)
            time.sleep(wait)

    def _reset_daily_counter_if_new_day(self) -> None:
        today = date.today()
        if today != self._count_date:
            logger.info(
                "New day detected — resetting daily counter (was %d).", self._request_count
            )
            self._request_count = 0
            self._count_date = today

    def _build_data_context(self, df_info: dict) -> str:
        """Convert a df_info dict into a concise text block for the prompt."""
        lines: list[str] = []

        shape = df_info.get("shape", {})
        if isinstance(shape, dict):
            lines.append(f"- Shape: {shape.get('rows', '?')} rows × {shape.get('columns', '?')} columns")
        elif isinstance(shape, (list, tuple)) and len(shape) == 2:
            lines.append(f"- Shape: {shape[0]} rows × {shape[1]} columns")

        cols = df_info.get("column_names", df_info.get("columns", []))
        if cols:
            lines.append(f"- Columns: {', '.join(cols)}")

        dtypes = df_info.get("dtypes", {})
        if dtypes:
            dtype_str = ", ".join(f"{c}: {t}" for c, t in list(dtypes.items())[:20])
            lines.append(f"- Data types: {dtype_str}")

        missing = df_info.get("missing_values", {})
        if missing:
            flagged = {
                c: v for c, v in missing.items()
                if (isinstance(v, dict) and v.get("count", 0) > 0)
                or (isinstance(v, (int, float)) and v > 0)
            }
            if flagged:
                miss_str = ", ".join(
                    f"{c}: {v['count']} ({v['percent']}%)" if isinstance(v, dict) else f"{c}: {v}"
                    for c, v in list(flagged.items())[:10]
                )
                lines.append(f"- Missing values: {miss_str}")

        stats = df_info.get("numeric_stats", df_info.get("summary_stats", {}))
        if stats:
            stats_json = json.dumps(stats, default=str)[:3000]
            lines.append(f"- Numeric statistics:\n{stats_json}")

        preview = df_info.get("preview", [])
        if preview:
            preview_json = json.dumps(preview[:5], default=str)[:2000]
            lines.append(f"- Sample rows (first 5):\n{preview_json}")

        context = "\n".join(lines)

        # Truncate at a clean line boundary to avoid sending broken JSON/markdown mid-string
        if len(context) > _MAX_CONTEXT_CHARS:
            cutoff = context.rfind("\n", 0, _MAX_CONTEXT_CHARS)
            if cutoff == -1:
                cutoff = _MAX_CONTEXT_CHARS
            context = context[:cutoff] + "\n[… context truncated at line boundary to fit token limit]"
            logger.warning("Data context truncated to ~%d chars", cutoff)

        return context

    def _summarise_analysis(self, results: dict) -> str:
        """Flatten analysis results dict to a short text for the report prompt."""
        parts: list[str] = []
        skip_keys = {"charts", "sample_anomalies"}

        for key, val in results.items():
            if key in skip_keys:
                continue
            if isinstance(val, dict):
                snippet = json.dumps(val, default=str)[:800]
            elif isinstance(val, list):
                snippet = json.dumps(val[:5], default=str)[:400]
            else:
                snippet = str(val)[:400]
            parts.append(f"**{key}:** {snippet}")

        summary = "\n".join(parts)
        if len(summary) > 6000:
            cutoff = summary.rfind("\n", 0, 6000)
            if cutoff == -1:
                cutoff = 6000
            summary = summary[:cutoff] + "\n[… truncated at line boundary]"
        return summary

    @staticmethod
    def _extract_follow_ups(text: str) -> list[str]:
        """Pull out any numbered/bulleted follow-up questions from the response."""
        pattern = r"(?:follow[- ]up|next|you (?:might|could|may) (?:also )?(?:ask|want)).*?(?:\n|$)"
        block = re.search(pattern, text, re.IGNORECASE)
        if not block:
            return []
        questions = re.findall(r"[-*\d.]+\s+(.+\?)", text[block.start():])
        return [q.strip() for q in questions[:4]]

    @staticmethod
    def _generate_follow_ups(df_info: dict) -> list[str]:
        """Fallback: produce generic follow-up questions from column names."""
        cols = df_info.get("column_names", df_info.get("columns", []))
        questions: list[str] = ["What are the main trends in this dataset?"]
        if cols:
            questions.append(f"Are there any outliers in the '{cols[0]}' column?")
        if len(cols) >= 2:
            questions.append(f"Is there a relationship between '{cols[0]}' and '{cols[1]}'?")
        questions.append("What actions should be taken based on these findings?")
        return questions[:4]
