#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — run FastAPI + Streamlit together in a single container.
# Used when both services share one Docker container (development / demo).
# In production prefer docker-compose, which runs them as separate services.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
API_PORT="${PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
UPLOAD_DIR="${UPLOAD_DIR:-uploads}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "========================================================"
echo "  AI Data Analyst — starting services"
echo "  FastAPI   → http://${HOST}:${API_PORT}/docs"
echo "  Streamlit → http://${HOST}:${STREAMLIT_PORT}"
echo "========================================================"

# Ensure upload directory exists
mkdir -p "${UPLOAD_DIR}"

# ── Start FastAPI in the background ──────────────────────────────────────────
uvicorn app.main:app \
    --host "${HOST}" \
    --port "${API_PORT}" \
    --log-level "${LOG_LEVEL}" &
API_PID=$!
echo "[start.sh] FastAPI started (PID ${API_PID})"

# Wait until the API is accepting connections before starting Streamlit
echo "[start.sh] Waiting for API to become ready…"
MAX_WAIT=30
WAITED=0
until python -c "import urllib.request; urllib.request.urlopen('http://localhost:${API_PORT}/')" > /dev/null 2>&1; do
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        echo "[start.sh] ERROR: API did not start within ${MAX_WAIT}s. Aborting."
        kill "${API_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done
echo "[start.sh] API is ready (waited ${WAITED}s)."

# ── Start Streamlit in the foreground ─────────────────────────────────────────
streamlit run streamlit_app/app.py \
    --server.port="${STREAMLIT_PORT}" \
    --server.address="${HOST}" \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false &
STREAMLIT_PID=$!
echo "[start.sh] Streamlit started (PID ${STREAMLIT_PID})"

# ── Graceful shutdown on SIGTERM / SIGINT ─────────────────────────────────────
_shutdown() {
    echo "[start.sh] Shutting down…"
    kill "${STREAMLIT_PID}" 2>/dev/null || true
    kill "${API_PID}"       2>/dev/null || true
    wait "${STREAMLIT_PID}" 2>/dev/null || true
    wait "${API_PID}"       2>/dev/null || true
    echo "[start.sh] All processes stopped."
}
trap _shutdown SIGTERM SIGINT

# Keep the script alive; exit if either process dies
wait -n "${API_PID}" "${STREAMLIT_PID}"
EXIT_CODE=$?
echo "[start.sh] A process exited (code ${EXIT_CODE}). Shutting down remaining services."
_shutdown
exit "${EXIT_CODE}"
