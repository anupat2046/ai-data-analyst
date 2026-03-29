import asyncio
import logging
import os
import time

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import analysis, chat, upload

# Files older than this are auto-deleted by the background janitor
_FILE_MAX_AGE_SECONDS = 3 * 60 * 60  # 3 hours
_JANITOR_INTERVAL_SECONDS = 30 * 60  # run every 30 minutes

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Data Analyst API",
    description=(
        "Upload CSV/Excel data and get AI-powered analysis, "
        "anomaly detection, clustering, and natural language insights."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Streamlit frontend on any port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router,   prefix="/api/upload",   tags=["Upload"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(chat.router,     prefix="/api/chat",     tags=["Chat"])

# ── Background janitor ────────────────────────────────────────────────────────
async def _janitor() -> None:
    """Periodically delete uploads and saved models older than _FILE_MAX_AGE_SECONDS."""
    while True:
        await asyncio.sleep(_JANITOR_INTERVAL_SECONDS)
        now = time.time()
        removed = 0
        try:
            for fname in os.listdir(settings.upload_dir):
                fpath = os.path.join(settings.upload_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                age = now - os.path.getmtime(fpath)
                if age > _FILE_MAX_AGE_SECONDS:
                    os.remove(fpath)
                    removed += 1
                    logger.debug("Janitor removed: %s (age=%.0fs)", fname, age)
        except Exception as exc:
            logger.warning("Janitor error: %s", exc)
        if removed:
            logger.info("Janitor cleaned %d file(s) from '%s'", removed, settings.upload_dir)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("  AI Data Analyst API  v1.0.0  starting up")
    logger.info("  Upload dir : %s", os.path.abspath(settings.upload_dir))
    logger.info("  Gemini model: %s", settings.gemini_model)
    logger.info("  API key set : %s", bool(settings.gemini_api_key))
    logger.info("  Docs        : http://%s:%d/docs", settings.host, settings.port)
    logger.info("  File janitor: every %dm, max age %dh",
                _JANITOR_INTERVAL_SECONDS // 60, _FILE_MAX_AGE_SECONDS // 3600)
    logger.info("=" * 60)
    asyncio.create_task(_janitor())

# ── Health check ──────────────────────────────────────────────────────────────
@app.get(
    "/",
    summary="Health check",
    description="Returns service status and version. Used by Docker and load-balancers.",
    tags=["Health"],
)
async def health_check() -> dict:
    return {
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs",
        "upload_dir": settings.upload_dir,
        "gemini_model": settings.gemini_model,
    }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )
