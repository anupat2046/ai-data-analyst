# 🤖 AI Data Analyst

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_API-Free-4285F4?style=flat&logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-18%20passed-brightgreen?style=flat&logo=pytest)

---

## Overview

**AI Data Analyst** is a full-stack, AI-powered data analytics web application.
Upload a CSV, Excel, or JSON file — the platform automatically runs Exploratory Data Analysis, Machine Learning, and natural-language AI Q&A powered by **Google Gemini** (free tier).

```
Upload CSV/XLSX/JSON  →  Auto EDA  →  ML Analysis  →  AI Chat  →  Executive Report
```

- **Backend**: FastAPI (async, production-ready)
- **Frontend**: Streamlit (5-page interactive UI)
- **AI**: Google Gemini API (gemini-2.5-flash, free at aistudio.google.com)
- **ML**: scikit-learn (Isolation Forest, KMeans, Logistic Regression, Random Forest, Gradient Boosting)
- **Deployment**: Docker + docker-compose, single command start

---

## Features

| Feature | Description |
|---------|-------------|
| 📊 **Auto EDA** | Summary statistics, missing values, correlation heatmap, distribution charts, categorical bar charts |
| 🔍 **Anomaly Detection** | Isolation Forest with PCA scatter visualization and top anomalous rows |
| 📈 **Clustering** | KMeans with auto-k selection (silhouette score), elbow curve, cluster profiles |
| 🏷️ **Classification** | Train & compare Logistic Regression, Random Forest, Gradient Boosting — confusion matrix + ROC curve |
| 📉 **Time-Series Analysis** | Trend + moving average, monthly/weekly aggregation, seasonal decomposition |
| 🔗 **Bivariate Analysis** | Scatter + regression, grouped box plots, cross-tabulation heatmaps |
| 🤖 **AI Chat** | Ask questions about your data in Thai or English, Gemini answers with context |
| 💡 **Analysis Suggestions** | AI recommends the most valuable analyses for your specific dataset |
| 📝 **Executive Report** | Markdown report with summary, key metrics, anomalies, and actionable recommendations |
| 🧹 **Auto Data Cleaning** | Imputation, deduplication, column standardisation, date parsing |
| ⏱️ **Auto File Janitor** | Files older than 3 hours are automatically deleted in the background |

---

## Architecture

```
┌─────────────────────────────────────┐
│          Streamlit Frontend         │
│  (5 pages: Upload / EDA / ML /      │
│   Chat / Report)   :8501            │
└──────────────────┬──────────────────┘
                   │ HTTP REST
┌──────────────────▼──────────────────┐
│          FastAPI Backend            │
│  /api/upload   /api/analysis        │
│  /api/chat     /docs (Swagger)      │
│                :8000                │
└────┬───────────┬────────────┬───────┘
     │           │            │
┌────▼───┐  ┌───▼────┐  ┌────▼────┐
│ pandas │  │sklearn │  │ Gemini  │
│  EDA   │  │  ML    │  │  API    │
└────────┘  └────────┘  └─────────┘
```

---

## Screenshots

| Page | Description |
|------|-------------|
| ![Upload](docs/screenshots/upload.png) | **Data Upload** — drag-and-drop CSV/Excel/JSON, instant overview |
| ![EDA](docs/screenshots/eda.png) | **Auto EDA** — correlation heatmap, distributions, categorical charts |
| ![ML](docs/screenshots/ml.png) | **ML Analysis** — anomaly detection, clustering, classification |
| ![Chat](docs/screenshots/ai.png) | **AI Chat** — Thai/English natural language Q&A about your data |
| ![Report](docs/screenshots/report.png) | **Executive Report** — AI-generated markdown summary |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.x | Interactive UI, session state, file upload |
| **Backend** | FastAPI + Uvicorn | Async REST API, OpenAPI docs |
| **Data** | pandas + NumPy | Data loading, cleaning, EDA |
| **ML** | scikit-learn | Isolation Forest, KMeans, classifiers, PCA |
| **Stats** | SciPy + statsmodels | Normality tests, seasonal decomposition |
| **Charts** | Matplotlib + Seaborn | Base64-encoded chart images |
| **AI/LLM** | Google Gemini API | Natural language Q&A, report generation |
| **Validation** | Pydantic v2 | Request/response schemas, settings |
| **Persistence** | joblib | Model serialization to disk |
| **Containers** | Docker + Compose | Multi-stage build, non-root user |
| **Testing** | pytest + httpx | 18 integration tests, mocked Gemini |

---

## Quick Start

### Prerequisites

- Python 3.11+
- A free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/anupat2046/ai-data-analyst.git
cd ai-data-analyst
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your API key:

```env
GEMINI_API_KEY=your_key_here
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the backend

```bash
uvicorn app.main:app --reload
# API running at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### 5. Start the frontend

```bash
streamlit run streamlit_app/app.py
# UI running at http://localhost:8501
```

---

## Docker (Recommended)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI + Swagger | http://localhost:8000/docs |

> Set `GEMINI_API_KEY` in your `.env` file before running Docker.

---

## API Documentation

Interactive Swagger UI is available at **http://localhost:8000/docs** when the server is running.

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/file` | Upload CSV / XLSX / JSON (max 50 MB) |
| `POST` | `/api/upload/clean` | Auto-clean an uploaded file |
| `GET` | `/api/upload/preview/{filename}` | First 20 rows as JSON |
| `GET` | `/api/upload/columns/{filename}` | Column names, dtypes, missing values |
| `DELETE` | `/api/upload/{filename}` | Delete an uploaded file |
| `POST` | `/api/analysis/auto-eda` | Full automatic EDA |
| `POST` | `/api/analysis/correlation` | Pearson correlation heatmap |
| `POST` | `/api/analysis/distribution` | Distribution + normality test for one column |
| `POST` | `/api/analysis/anomaly` | Isolation Forest anomaly detection |
| `POST` | `/api/analysis/clustering` | KMeans clustering with auto-k |
| `POST` | `/api/analysis/classification` | Multi-model classification comparison |
| `POST` | `/api/analysis/timeseries` | Time-series trend + decomposition |
| `POST` | `/api/analysis/bivariate` | Bivariate relationship analysis |
| `POST` | `/api/analysis/report` | AI executive report (Gemini) |
| `POST` | `/api/chat/ask` | Natural language Q&A about a dataset |
| `POST` | `/api/chat/suggest` | AI-recommended analyses for a dataset |
| `POST` | `/api/chat/explain-anomalies` | AI explanation of anomaly results |
| `GET` | `/` | Health check |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key *(required for AI features)* | `""` |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `HOST` | API server host | `0.0.0.0` |
| `PORT` | API server port | `8000` |
| `MAX_FILE_SIZE_MB` | Maximum upload file size | `50` |
| `ALLOWED_EXTENSIONS` | Comma-separated allowed file types | `csv,xlsx,json` |
| `UPLOAD_DIR` | Directory for uploaded files | `uploads` |

---

## Running Tests

```bash
pytest tests/ -v
```

```
18 passed in 6.82s
```

The test suite covers all upload and analysis endpoints with an isolated temp upload directory and a mocked Gemini client (no API key required to run tests).

---

## Project Structure

```
ai-data-analyst/
├── app/
│   ├── main.py                  # FastAPI app, CORS, background janitor
│   ├── config.py                # Pydantic settings, env vars
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── routers/
│   │   ├── upload.py            # File upload, clean, preview, columns, delete
│   │   ├── analysis.py          # EDA, correlation, distribution, anomaly, ML, report
│   │   └── chat.py              # AI Q&A, suggestions, anomaly explanation
│   ├── services/
│   │   ├── data_processor.py    # Load, overview, auto-clean
│   │   ├── eda_engine.py        # Auto EDA, correlation, distribution, time-series
│   │   ├── ml_engine.py         # Anomaly detection, clustering, classification
│   │   └── llm_service.py       # Gemini API client, rate limiting, retries
│   └── utils/
│       └── chart_generator.py   # Matplotlib chart helpers (base64 output)
├── streamlit_app/
│   └── app.py                   # 5-page Streamlit frontend
├── tests/
│   ├── conftest.py              # Fixtures: test_client, mock_gemini, uploaded_file
│   ├── test_upload.py           # 7 upload endpoint tests
│   ├── test_analysis.py         # 11 analysis + chat tests
│   └── test_data/
│       └── sample.csv           # 100-row test dataset
├── docs/
│   └── screenshots/             # UI screenshots (add your own)
├── Dockerfile                   # Multi-stage build, non-root user
├── docker-compose.yml           # API + frontend services
├── requirements.txt             # Python dependencies
├── start.sh                     # Convenience start script
└── .env.example                 # Environment variable template
```
---

## Author

**Anupat Suttilert**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Anupat_Suttilert-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/anupat-suttilert)
[![GitHub](https://img.shields.io/badge/GitHub-anupat2046-181717?style=flat&logo=github&logoColor=white)](https://github.com/anupat2046)
[![Website](https://img.shields.io/badge/Website-anupatsuttilert.com-4CAF50?style=flat&logo=google-chrome&logoColor=white)](https://anupatsuttilert.com)

---

*Built with FastAPI + Streamlit + Google Gemini*
