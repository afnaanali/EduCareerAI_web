import os
import sys
from pathlib import Path

# Prevent OpenMP / MKL thread deadlocks in Windows web worker processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cross-version compatibility for unpickling Scikit-Learn pipelines and models
try:
    import sklearn._loss._loss as _cy_loss
    sys.modules["sklearn.ensemble._hist_gradient_boosting._loss"] = _cy_loss
    sys.modules["_loss"] = _cy_loss
except Exception:
    pass

try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.database import init_db
from api.routers.auth import router as auth_router
from api.routers.career import router as career_router
from api.routers.roadmap import router as roadmap_router
from api.routers.resume import router as resume_router
from api.routers.chat import router as chat_router
from api.routers.dl_lab import router as dl_lab_router
from api.routers.interview import router as interview_router

# Initialize SQLite tables
init_db()

app = FastAPI(
    title="EduCareerAI Unified Intelligence API",
    description="High-performance backend API serving ML Career Rankings, ATS Resume Studio, RAG Assistant, Deep Learning Neural Lab, and AI Interview Coach.",
    version="2.0.0",
)

# CORS Middleware to allow requests from Vercel frontend and localhost
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8501",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "*",  # Allow all origins for Vercel dynamic preview and production URLs
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(auth_router)
app.include_router(career_router)
app.include_router(roadmap_router)
app.include_router(resume_router)
app.include_router(chat_router)
app.include_router(dl_lab_router)
app.include_router(interview_router)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "EduCareerAI Unified Intelligence API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health_check():
    """
    Lightweight health-check endpoint.
    Used by UptimeRobot, cron jobs, or Render liveness probes to keep the container warm 24/7.
    """
    return {
        "status": "healthy",
        "uptime": "active",
        "timestamp": os.environ.get("RENDER_INSTANCE_ID", "local"),
    }


@app.get("/api/info")
def system_info():
    return {
        "models": {
            "career_ranking": "HistGradientBoostingClassifier (Scikit-Learn)",
            "course_recommendation": "MultiOutputClassifier (Scikit-Learn)",
            "career_ann": "8-Skill Artificial Neural Network",
            "digit_cnn": "MNIST Handwritten Digit Vision CNN",
            "sentiment_lstm": "Bidirectional LSTM Sentiment Analyzer",
        },
        "database": "SQLite (data/educareer_ai.db)",
        "framework": "FastAPI + Uvicorn",
    }
