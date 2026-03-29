"""
streamlit_app/app.py
────────────────────
AI Data Analyst — Streamlit frontend.
Connects to FastAPI backend running on localhost:8000.
"""

import base64
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
# Reads API_URL env var when running inside Docker; falls back to localhost for dev.
BASE_URL = os.getenv("API_URL", "http://localhost:8000")
API = f"{BASE_URL}/api"

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _b64_image(b64: str) -> bytes:
    return base64.b64decode(b64)


def _show_charts(charts: list[str], ncols: int = 2) -> None:
    """Render a list of base64 PNG strings in a grid."""
    if not charts:
        return
    cols = st.columns(ncols)
    for i, c in enumerate(charts):
        if c:
            cols[i % ncols].image(_b64_image(c), use_container_width=True)


def _api_error(resp: requests.Response, label: str = "Request") -> None:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    if "quota" in detail.lower() or "limit" in detail.lower() or resp.status_code == 429:
        st.warning(
            f"⚠️ Gemini Free Tier quota reached (250 req/day). "
            "Wait until midnight UTC or upgrade your API plan."
        )
    else:
        st.error(f"❌ {label} failed ({resp.status_code}): {detail}")


def _load_demo() -> None:
    """Upload sample.csv, auto-clean it, and set it as the active session file."""
    demo_path = Path(__file__).parent.parent / "tests" / "test_data" / "sample.csv"
    if not demo_path.exists():
        st.error("Demo file not found. Make sure `tests/test_data/sample.csv` exists.")
        return

    with st.spinner("Loading demo dataset…"):
        # 1. Upload
        try:
            up_resp = requests.post(
                f"{API}/upload/file",
                files={"file": ("sample.csv", demo_path.read_bytes(), "text/csv")},
                timeout=60,
            )
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Is the FastAPI server running on port 8000?")
            return
        if up_resp.status_code != 200:
            _api_error(up_resp, "Demo Upload")
            return
        meta = up_resp.json()
        raw_filename = meta["filename"]

        # 2. Auto-clean
        clean_resp = requests.post(
            f"{API}/upload/clean",
            json={"filename": raw_filename, "options": {}},
            timeout=60,
        )
        if clean_resp.status_code == 200:
            clean_data = clean_resp.json()
            cleaned_filename = clean_data["cleaned_filename"]
            overview = clean_data.get("overview", {})
            shape = overview.get("shape", {})
            rows = shape.get("rows", 0) if isinstance(shape, dict) else (shape[0] if shape else 0)
            cols_n = shape.get("columns", 0) if isinstance(shape, dict) else (shape[1] if shape else 0)
            missing_raw = overview.get("missing_values", {})
            missing_flat = {
                col: int(val.get("count", 0)) if isinstance(val, dict) else int(val)
                for col, val in missing_raw.items()
            }
            active_filename = cleaned_filename
            active_meta = {
                "filename": cleaned_filename,
                "rows": rows,
                "columns": cols_n,
                "column_names": overview.get("column_names", []),
                "dtypes": overview.get("dtypes", {}),
                "missing_values": missing_flat,
                "preview": overview.get("preview", []),
            }
        else:
            # Cleaning failed — fall back to raw upload
            active_filename = raw_filename
            active_meta = meta

        st.session_state["filename"] = active_filename
        st.session_state["file_meta"] = active_meta
        st.session_state["chat_history"] = []
        st.session_state["suggestions"] = []

    st.success("✅ Demo dataset loaded & cleaned! (100 retail transactions)")
    st.rerun()


def _post(endpoint: str, payload: dict, label: str = "Request") -> dict | None:
    try:
        resp = requests.post(f"{API}{endpoint}", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        _api_error(resp, label)
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is the FastAPI server running on port 8000?")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "filename": None,
    "file_meta": None,
    "chat_history": [],
    "suggestions": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#2E5090;margin-bottom:0'>🤖 AI Data Analyst</h2>"
        "<p style='color:#888;font-size:13px;margin-top:4px'>Powered by Gemini</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    uploaded = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "json"],
        help="CSV, Excel, or JSON — max 50 MB",
    )

    if uploaded:
        if st.button("Upload & Analyse", type="primary", use_container_width=True):
            with st.spinner("Uploading…"):
                try:
                    resp = requests.post(
                        f"{API}/upload/file",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        meta = resp.json()
                        st.session_state["filename"] = meta["filename"]
                        st.session_state["file_meta"] = meta
                        st.session_state["chat_history"] = []
                        st.session_state["suggestions"] = []
                        st.success(f"✅ Uploaded: {meta['filename']}")
                    else:
                        _api_error(resp, "Upload")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend on port 8000.")

    st.markdown(
        "<div style='text-align:center;color:#aaa;font-size:12px;margin:4px 0'>— or —</div>",
        unsafe_allow_html=True,
    )
    if st.button(
        "🎯 No data? Try our demo dataset\n(100 retail transactions)",
        use_container_width=True,
        help="Loads a sample retail dataset, auto-cleans it, and takes you straight to the overview.",
    ):
        _load_demo()

    # File info card
    if st.session_state["file_meta"]:
        m = st.session_state["file_meta"]
        st.markdown("---")
        st.markdown(f"**📄 {m['filename']}**")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{m['rows']:,}")
        c2.metric("Columns", m["columns"])

    st.divider()

    page = st.radio(
        "Navigate",
        ["Data Overview", "Auto EDA", "ML Analysis", "AI Chat", "Report"],
        format_func=lambda p: {
            "Data Overview": "📋  Data Overview",
            "Auto EDA":      "📊  Auto EDA",
            "ML Analysis":   "🤖  ML Analysis",
            "AI Chat":       "💬  AI Chat",
            "Report":        "📝  Report",
        }[p],
    )

# ── Guard: require uploaded file ─────────────────────────────────────────────
filename = st.session_state["filename"]

if not filename:
    st.markdown(
        "<br><h2 style='text-align:center;color:#2E5090'>🤖 AI Data Analyst</h2>"
        "<p style='text-align:center;color:#666'>"
        "Upload a CSV, Excel, or JSON file in the sidebar to get started.</p>",
        unsafe_allow_html=True,
    )
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if st.button(
            "🎯 Try Demo Dataset  (100 retail transactions)",
            type="primary",
            use_container_width=True,
        ):
            _load_demo()
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "Data Overview":
    st.header("📋 Data Overview")
    meta = st.session_state["file_meta"]

    # KPI cards
    missing_vals = meta.get("missing_values", {})
    total_cells = meta["rows"] * meta["columns"]
    total_missing = sum(missing_vals.values())
    missing_pct = round(total_missing / total_cells * 100, 1) if total_cells else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{meta['rows']:,}")
    k2.metric("Columns", meta["columns"])
    k3.metric("Missing %", f"{missing_pct}%")
    k4.metric("Total Missing", f"{total_missing:,}")

    st.divider()

    # Column info table
    with st.expander("📌 Column Details", expanded=True):
        col_resp = None
        try:
            col_resp = requests.get(f"{API}/upload/columns/{filename}", timeout=30)
        except Exception:
            pass

        if col_resp and col_resp.status_code == 200:
            col_data = col_resp.json()
            rows = []
            for col in col_data["column_names"]:
                mv = col_data["missing_values"].get(col, {})
                missing_count = mv.get("count", mv) if isinstance(mv, dict) else mv
                rows.append({
                    "Column": col,
                    "Type": col_data["dtypes"].get(col, ""),
                    "Semantic": col_data["column_types"].get(col, ""),
                    "Missing": missing_count,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            # Fallback: build from meta
            rows = []
            for col in meta.get("column_names", []):
                rows.append({
                    "Column": col,
                    "Type": meta["dtypes"].get(col, ""),
                    "Missing": missing_vals.get(col, 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Data preview
    with st.expander("🔍 Data Preview (first 20 rows)", expanded=True):
        try:
            prev_resp = requests.get(f"{API}/upload/preview/{filename}", timeout=30)
            if prev_resp.status_code == 200:
                data = prev_resp.json().get("data", meta.get("preview", []))
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.dataframe(pd.DataFrame(meta.get("preview", [])), use_container_width=True)
        except Exception:
            st.dataframe(pd.DataFrame(meta.get("preview", [])), use_container_width=True)

    # Auto clean
    st.divider()
    st.subheader("🧹 Auto Clean Data")
    st.caption(
        "Standardises column names, imputes missing values (median/mode), "
        "removes duplicates, and parses date columns."
    )

    with st.expander("⚙️ Cleaning options"):
        drop_thresh = st.slider(
            "Drop column if missing > X%", 10, 90, 50, step=5
        ) / 100

    if st.button("Auto Clean Data", type="primary"):
        with st.spinner("Cleaning…"):
            result = _post(
                "/upload/clean",
                {"filename": filename, "options": {"drop_missing_threshold": drop_thresh}},
                "Clean",
            )
        if result:
            st.success(f"✅ Cleaned file saved as **{result['cleaned_filename']}**")
            report = result.get("cleaning_report", {})
            st.markdown(f"**Actions taken:** {len(report.get('actions', []))}")
            before = report.get("before", {})
            after  = report.get("after", {})
            b1, b2, b3 = st.columns(3)
            b1.metric("Rows before → after",
                       f"{before.get('shape', [0])[0]:,} → {after.get('shape', [0])[0]:,}")
            b2.metric("Missing before → after",
                       f"{before.get('missing_total', 0)} → {after.get('missing_total', 0)}")
            b3.metric("Duplicates removed",
                       before.get("duplicates", 0) - after.get("duplicates", 0))
            with st.expander("Full cleaning report"):
                st.json(report)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Auto EDA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Auto EDA":
    st.header("📊 Auto Exploratory Data Analysis")

    if st.button("Run Auto EDA", type="primary"):
        with st.spinner("Running EDA — this may take a moment…"):
            result = _post("/analysis/auto-eda", {"filename": filename}, "Auto EDA")

        if result:
            st.session_state["eda_result"] = result

    result = st.session_state.get("eda_result")
    if not result:
        st.info("Click **Run Auto EDA** to start.")
        st.stop()

    # Insights
    insights = result.get("insights", [])
    if insights:
        st.subheader("💡 Insights")
        for ins in insights:
            st.markdown(f"- {ins}")

    charts: list[str] = result.get("charts", [])
    if not charts:
        st.warning("No charts were generated.")
        st.stop()

    # Split charts: first = correlation heatmap, rest = distributions + categorical
    st.subheader("🔗 Correlation Heatmap")
    if charts:
        st.image(_b64_image(charts[0]), use_container_width=True)

    remaining = charts[1:]
    if remaining:
        st.subheader("📈 Distribution & Category Charts")
        _show_charts(remaining, ncols=2)

    # Summary stats
    stats = result.get("results", {}).get("summary_stats", {})
    if stats:
        with st.expander("📐 Numeric Summary Statistics"):
            st.dataframe(pd.DataFrame(stats), use_container_width=True)

    missing = result.get("results", {}).get("missing_analysis", {})
    if missing:
        with st.expander("❓ Missing Value Analysis"):
            rows = [
                {"Column": c, "Count": v.get("count", 0), "Percent (%)": v.get("percent", 0)}
                for c, v in missing.items()
                if isinstance(v, dict)
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ML Analysis":
    st.header("🤖 Machine Learning Analysis")

    # Get column info for feature selectors
    col_names: list[str] = st.session_state["file_meta"].get("column_names", [])
    num_cols: list[str] = [
        c for c, t in st.session_state["file_meta"].get("dtypes", {}).items()
        if t in ("integer", "float", "int64", "float64")
    ] or col_names

    tab_anomaly, tab_cluster, tab_clf = st.tabs(
        ["🚨 Anomaly Detection", "🔵 Clustering", "🎯 Classification"]
    )

    # ── Tab 1: Anomaly ────────────────────────────────────────────────────────
    with tab_anomaly:
        st.subheader("Anomaly Detection — Isolation Forest")
        features_a = st.multiselect(
            "Select features (leave empty = all numeric)",
            options=num_cols,
            key="anomaly_features",
        )
        if st.button("Detect Anomalies", type="primary", key="btn_anomaly"):
            payload: dict = {"filename": filename, "task": "anomaly_detection"}
            if features_a:
                payload["features"] = features_a
            with st.spinner("Detecting anomalies…"):
                result = _post("/analysis/anomaly", payload, "Anomaly Detection")
            if result:
                st.session_state["anomaly_result"] = result

        result = st.session_state.get("anomaly_result")
        if result:
            total = result.get("total_rows", result.get("anomaly_count", 0))
            count = result.get("anomaly_count", 0)
            pct   = result.get("anomaly_percentage", 0)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Rows", f"{total:,}")
            m2.metric("Anomalies Found", f"{count:,}")
            m3.metric("Anomaly Rate", f"{pct}%")

            for ins in result.get("insights", []):
                st.markdown(f"> {ins}")

            charts_a = result.get("charts", [])
            if charts_a:
                st.image(_b64_image(charts_a[0]), use_container_width=True)

            samples = result.get("sample_anomalies", [])
            if samples:
                with st.expander(f"Top {len(samples)} Anomalous Rows"):
                    st.dataframe(pd.DataFrame(samples), use_container_width=True)

    # ── Tab 2: Clustering ─────────────────────────────────────────────────────
    with tab_cluster:
        st.subheader("K-Means Clustering")
        features_c = st.multiselect(
            "Select features (leave empty = all numeric)",
            options=num_cols,
            key="cluster_features",
        )
        auto_k = st.checkbox("Auto-select optimal k", value=True)
        n_clusters = None
        if not auto_k:
            n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)

        if st.button("Run Clustering", type="primary", key="btn_cluster"):
            payload = {
                "filename": filename,
                "task": "clustering",
                "target_column": str(n_clusters) if n_clusters else None,
            }
            if features_c:
                payload["features"] = features_c
            with st.spinner("Clustering… this may take a moment."):
                result = _post("/analysis/clustering", payload, "Clustering")
            if result:
                st.session_state["cluster_result"] = result

        result = st.session_state.get("cluster_result")
        if result:
            m1, m2 = st.columns(2)
            m1.metric("Clusters", result.get("n_clusters", "-"))
            m2.metric("Silhouette Score", round(result.get("silhouette_score", 0), 3))

            sizes = result.get("cluster_sizes", {})
            if sizes:
                st.bar_chart(pd.DataFrame.from_dict(sizes, orient="index", columns=["count"]))

            for ins in result.get("insights", []):
                st.markdown(f"> {ins}")

            _show_charts(result.get("charts", []), ncols=2)

            centers = result.get("cluster_centers")
            if centers:
                with st.expander("Cluster Centres"):
                    st.dataframe(pd.DataFrame(centers), use_container_width=True)

    # ── Tab 3: Classification ─────────────────────────────────────────────────
    with tab_clf:
        st.subheader("Classification — Compare 3 Models")
        target = st.selectbox("Target column", options=col_names, key="clf_target")
        feat_opts = [c for c in col_names if c != target]
        features_f = st.multiselect(
            "Select features (leave empty = all numeric except target)",
            options=feat_opts,
            key="clf_features",
        )

        if st.button("Train Models", type="primary", key="btn_clf"):
            payload = {
                "filename": filename,
                "task": "classification",
                "target_column": target,
            }
            if features_f:
                payload["features"] = features_f
            with st.spinner("Training models… (Logistic Regression, Random Forest, Gradient Boosting)"):
                result = _post("/analysis/classification", payload, "Classification")
            if result:
                st.session_state["clf_result"] = result

        result = st.session_state.get("clf_result")
        if result:
            m1, m2 = st.columns(2)
            m1.metric("Best Model", result.get("best_model", "-"))
            m2.metric("Accuracy", f"{result.get('accuracy', 0):.2%}")

            all_acc = result.get("all_model_accuracies", {})
            if all_acc:
                st.bar_chart(pd.DataFrame.from_dict(all_acc, orient="index", columns=["accuracy"]))

            for ins in result.get("insights", []):
                st.markdown(f"> {ins}")

            _show_charts(result.get("charts", []), ncols=2)

            cr = result.get("classification_report", {})
            if cr:
                with st.expander("Full Classification Report"):
                    rows = []
                    for cls, metrics in cr.items():
                        if isinstance(metrics, dict):
                            rows.append({"class": cls, **{k: round(v, 4) for k, v in metrics.items()}})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AI Chat
# ═════════════════════════════════════════════════════════════════════════════
elif page == "AI Chat":
    st.header("💬 AI Chat")
    st.caption("Ask anything about your dataset — Thai or English.")

    # Load suggestions once per file
    if not st.session_state["suggestions"]:
        try:
            resp = requests.post(
                f"{API}/chat/suggest",
                json={"filename": filename},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["suggestions"] = (
                    [s.get("description", "") for s in data.get("suggestions", [])]
                    + data.get("follow_up_questions", [])
                )
        except Exception:
            pass

    # Suggested questions
    suggestions = [s for s in st.session_state["suggestions"] if s]
    if suggestions:
        with st.expander("💡 Suggested questions"):
            cols = st.columns(2)
            for i, q in enumerate(suggestions[:8]):
                if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state["_pending_question"] = q

    # Render conversation history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("code"):
                with st.expander("💻 Code"):
                    st.code(msg["code"], language="python")
            for chart_b64 in msg.get("charts", []):
                st.image(_b64_image(chart_b64), use_container_width=True)

    # Handle suggested-question button click (runs on next rerender)
    pending = st.session_state.pop("_pending_question", None)
    question = st.chat_input("Ask a question about your dataset…") or pending

    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp_data = _post(
                    "/chat/ask",
                    {
                        "filename": filename,
                        "question": question,
                        "conversation_history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state["chat_history"][:-1]
                        ],
                    },
                    "Chat",
                )

            if resp_data:
                answer  = resp_data.get("answer", "")
                code    = resp_data.get("code_executed")
                charts  = resp_data.get("charts", [])

                st.markdown(answer)
                if code:
                    with st.expander("💻 Code"):
                        st.code(code, language="python")
                for chart_b64 in charts:
                    st.image(_b64_image(chart_b64), use_container_width=True)

                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "code": code,
                    "charts": charts,
                })

    # Clear chat button
    if st.session_state["chat_history"]:
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Report
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Report":
    st.header("📝 AI Executive Report")
    st.caption(
        "Runs Auto EDA + Anomaly Detection, then sends results to Gemini "
        "to generate a structured executive report."
    )

    st.info(
        "ℹ️ **Gemini Free Tier** — 250 requests/day, 10 req/min. "
        "Report generation uses ~3 requests. If quota is exceeded, "
        "a warning will appear below.",
        icon="ℹ️",
    )

    if st.button("Generate AI Report", type="primary"):
        with st.spinner("Analysing data and generating report (may take 15–30 s)…"):
            result = _post("/analysis/report", {"filename": filename}, "Report")
        if result:
            st.session_state["report_result"] = result

    result = st.session_state.get("report_result")
    if not result:
        st.info("Click **Generate AI Report** to start.")
        st.stop()

    report_md: str = result.get("report", "")
    ts: str        = result.get("timestamp", "")

    if ts:
        st.caption(f"Generated at {ts}")

    if report_md:
        st.markdown(report_md)
        st.divider()

        # Download button
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=report_md.encode("utf-8"),
            file_name=f"report_{filename.rsplit('.', 1)[0]}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.warning("Report content is empty — check the backend logs.")
