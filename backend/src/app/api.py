"""
FastAPI server for Career Path Recommender System.
Exposes ML models (Career Predictor, University Recommender, Job Recommender) as REST endpoints.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

# Load .env before anything else so all os.getenv calls below pick it up
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  

from src.data.loader import load_config, load_raw_data
from src.data.preprocessing import clean_text
from src.features.build_features import FeatureBuilder
from src.models.career_predictor import CareerPredictor
from src.models.university_recommender import UniversityRecommender
from src.models.career_recommender import CareerRecommender
from src.auth.auth import hash_password, verify_password, create_access_token
from src.db.mongo import get_db, close_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS — read allowed origins from env (comma-separated).
# Falls back to localhost dev servers so local dev works without .env.
# Example: ALLOWED_ORIGINS=https://myapp.com,https://www.myapp.com
# ---------------------------------------------------------------------------
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:8080,http://localhost:3000",
)
ALLOWED_ORIGINS: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Global model state — populated during lifespan startup
# ---------------------------------------------------------------------------
career_predictor: Optional[CareerPredictor] = None
university_recommender: Optional[UniversityRecommender] = None
job_recommender: Optional[CareerRecommender] = None
job_df: Any = None          # pd.DataFrame — typed as Any to avoid import at module scope
_models_ready: bool = False  # False until all models are loaded; health endpoint uses this


@asynccontextmanager
async def lifespan(application: FastAPI):
    """App lifespan: initialize all ML models and DB on startup, clean up on shutdown."""
    global career_predictor, university_recommender, job_recommender, job_df, _models_ready

    logger.info("Starting Career Path Recommender API…")
    try:
        import numpy as np
        import pandas as pd

        # ── Load config & datasets ──────────────────────────────────────────
        config = load_config()
        datasets = load_raw_data(config)

        # ── Career Predictor ────────────────────────────────────────────────
        logger.info("Loading Career Predictor…")
        career_predictor = CareerPredictor()
        career_predictor.load_or_train(datasets["student_reco"], verbose=False)

        # ── University Recommender ──────────────────────────────────────────
        logger.info("Loading University Recommender…")
        feature_builder = FeatureBuilder()
        university_recommender = UniversityRecommender(
            feature_builder=feature_builder,
            indian_df=datasets["indian_colleges"],
            world_df=datasets["world_universities"],
            student_df=datasets.get("student_reco"),
            train_ranker=True,
        )

        # ── Job Recommender ─────────────────────────────────────────────────
        logger.info("Loading Job Recommender…")
        cache_dir = Path("models/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        job_df_path = cache_dir / "job_df.parquet"
        job_emb_path = cache_dir / "job_embeddings.npy"

        if job_df_path.exists() and job_emb_path.exists():
            job_df = pd.read_parquet(job_df_path)
            job_embeddings = np.load(job_emb_path)
        else:
            job_df = (
                datasets["job_descriptions"]
                .drop_duplicates(subset=["Job Title"])
                .reset_index(drop=True)
                .copy()
            )
            job_df["job_idx"] = job_df.index
            job_df["content"] = (
                job_df[["Job Title", "skills", "Job Description", "Responsibilities"]]
                .fillna("")
                .agg(" ".join, axis=1)
            )
            job_feature_builder = FeatureBuilder()
            job_embeddings = job_feature_builder.encode(
                job_df["content"].tolist(), batch_size=128
            )
            job_df.to_parquet(job_df_path, index=False)
            np.save(job_emb_path, job_embeddings)

        job_feature_builder = FeatureBuilder()
        job_recommender = CareerRecommender(
            job_df=job_df,
            embedding_matrix=job_embeddings,
            feature_builder=job_feature_builder,
        )

        _models_ready = True
        logger.info("All systems initialized successfully.")

    except Exception as exc:
        logger.error("Startup failed: %s", exc, exc_info=True)
        raise

    yield  # ── application runs ────────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down — closing MongoDB connection…")
    close_db()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Career Path Recommender API",
    description="AI-powered career guidance system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Request/Response Models ====================

class StudentProfileRequest(BaseModel):
    """Request model for student profile assessment."""
    mathematics_score: int = Field(..., ge=0, le=100)
    science_score: int = Field(..., ge=0, le=100)
    language_arts_score: int = Field(..., ge=0, le=100)
    social_studies_score: int = Field(..., ge=0, le=100)
    logical_reasoning: int = Field(..., ge=0, le=100)
    creativity: int = Field(..., ge=0, le=100)
    communication: int = Field(..., ge=0, le=100)
    leadership: int = Field(..., ge=0, le=100)
    social_skills: int = Field(..., ge=0, le=100)
    skills_text: str = Field(default="")
    preferred_location: str = Field(default="")


class CareerPredictionResponse(BaseModel):
    """Response model for career predictions."""
    career_field: str
    confidence: float
    # Each element is [field_name, confidence_score] — using List[List] for
    # JSON-safe serialization (Pydantic cannot serialize bare Python tuples).
    alternatives: List[List[Any]] = []


class UniversityRecommendation(BaseModel):
    """Single university recommendation."""
    name: str
    country: str
    state: Optional[str] = None
    district: Optional[str] = None
    website: str
    score: float


class UniversitiesResponse(BaseModel):
    """Response model for university recommendations."""
    universities: List[UniversityRecommendation]
    total: int


class JobRecommendation(BaseModel):
    """Single job recommendation."""
    title: str
    score: float


class JobsResponse(BaseModel):
    """Response model for job recommendations."""
    jobs: List[JobRecommendation]
    total: int


class RecommendationResponse(BaseModel):
    """Complete recommendation response."""
    career: CareerPredictionResponse
    universities: UniversitiesResponse
    jobs: JobsResponse


# ── Auth Models ───────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(...)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: str = Field(...)
    password: str = Field(...)


class AuthResponse(BaseModel):
    token: str
    userId: str
    name: str


@app.get("/")
async def root():
    """Root endpoint - API status."""
    return {
        "status": "online",
        "message": "Career Path Recommender API is running",
        "version": "1.0.0"
    }


@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendations(profile: StudentProfileRequest):
    """
    Get complete recommendations (career, universities, jobs) based on student profile.
    
    Args:
        profile: StudentProfileRequest with academic and skill scores
    
    Returns:
        RecommendationResponse with career, university, and job recommendations
    """
    if not all([career_predictor, university_recommender, job_recommender, job_df]):
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        # Convert request to dict format expected by models
        user_profile = {
            "Mathematics_Score": profile.mathematics_score,
            "Science_Score": profile.science_score,
            "Language_Arts_Score": profile.language_arts_score,
            "Social_Studies_Score": profile.social_studies_score,
            "Logical_Reasoning": profile.logical_reasoning,
            "Creativity": profile.creativity,
            "Communication": profile.communication,
            "Leadership": profile.leadership,
            "Social_Skills": profile.social_skills,
            "skills_text": profile.skills_text,
        }
        
        # ==================== Career Prediction ====================
        predictions = career_predictor.predict_top_k(
            user_profile, k=3, skills_text=profile.skills_text or ""
        )
        top_career, confidence = predictions[0]
        alternatives = [[field, float(conf)] for field, conf in predictions[1:]]

        career_response = CareerPredictionResponse(
            career_field=top_career,
            confidence=float(confidence),
            alternatives=alternatives,
        )
        
        # ==================== University Recommendations ====================
        query = profile.skills_text or top_career
        query = clean_text(query)
        
        universities_df = university_recommender.recommend(
            query=top_career,
            country=profile.preferred_location if profile.preferred_location else None,
            top_k=10,
            skills_text=profile.skills_text,
        )
        
        universities_list = []
        if not universities_df.empty:
            for _, row in universities_df.iterrows():
                universities_list.append(
                    UniversityRecommendation(
                        name=row.get("name", ""),
                        country=row.get("country", ""),
                        state=row.get("State", None),
                        district=row.get("District", None),
                        website=row.get("Website", ""),
                        score=float(row.get("score", 0.0)),
                    )
                )
        
        universities_response = UniversitiesResponse(
            universities=universities_list,
            total=len(universities_list)
        )
        
        # ==================== Job Recommendations ====================
        job_query = profile.skills_text or top_career
        job_query = clean_text(job_query)
        
        jobs_df = job_recommender.recommend(job_query, top_k=10)
        
        jobs_list = []
        seen = set()
        if not jobs_df.empty:
            for _, row in jobs_df.iterrows():
                job_idx = int(row.get("job_idx", -1))
                if job_idx >= 0 and job_idx < len(job_df):
                    title = job_df.loc[job_idx, "Job Title"]
                    if title not in seen:
                        seen.add(title)
                        jobs_list.append(
                            JobRecommendation(
                                title=title,
                                score=float(row.get("score", 0.0))
                            )
                        )
        
        jobs_response = JobsResponse(
            jobs=jobs_list[:5],  # Top 5 jobs
            total=len(jobs_list)
        )
        
        return RecommendationResponse(
            career=career_response,
            universities=universities_response,
            jobs=jobs_response
        )
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/career-predict", response_model=CareerPredictionResponse)
async def predict_career(profile: StudentProfileRequest):
    """Predict career field based on student profile."""
    if not career_predictor:
        raise HTTPException(status_code=503, detail="Career predictor not initialized")
    
    try:
        user_profile = {
            "Mathematics_Score": profile.mathematics_score,
            "Science_Score": profile.science_score,
            "Language_Arts_Score": profile.language_arts_score,
            "Social_Studies_Score": profile.social_studies_score,
            "Logical_Reasoning": profile.logical_reasoning,
            "Creativity": profile.creativity,
            "Communication": profile.communication,
            "Leadership": profile.leadership,
            "Social_Skills": profile.social_skills,
        }
        
        skills_text = getattr(profile, "skills_text", "") or ""
        predictions = career_predictor.predict_top_k(
            user_profile, k=3, skills_text=skills_text
        )
        top_career, confidence = predictions[0]
        alternatives = [[field, float(conf)] for field, conf in predictions[1:]]

        return CareerPredictionResponse(
            career_field=top_career,
            confidence=float(confidence),
            alternatives=alternatives,
        )
    except Exception as e:
        logger.error(f"Career prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/universities", response_model=UniversitiesResponse)
async def recommend_universities(
    query: str,
    country: Optional[str] = None,
    top_k: int = 10,
    skills_text: str = ""
):
    """Recommend universities based on career interests and location."""
    if not university_recommender:
        raise HTTPException(status_code=503, detail="University recommender not initialized")
    
    try:
        query = clean_text(query)
        
        universities_df = university_recommender.recommend(
            query=query,
            country=country,
            top_k=top_k,
            skills_text=skills_text,
        )
        
        universities_list = []
        if not universities_df.empty:
            for _, row in universities_df.iterrows():
                universities_list.append(
                    UniversityRecommendation(
                        name=row.get("name", ""),
                        country=row.get("country", ""),
                        state=row.get("State", None),
                        district=row.get("District", None),
                        website=row.get("Website", ""),
                        score=float(row.get("score", 0.0)),
                    )
                )
        
        return UniversitiesResponse(
            universities=universities_list,
            total=len(universities_list)
        )
    except Exception as e:
        logger.error(f"University recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs", response_model=JobsResponse)
async def recommend_jobs(query: str, top_k: int = 10):
    """Recommend jobs based on skills and interests."""
    if not job_recommender or job_df is None:
        raise HTTPException(status_code=503, detail="Job recommender not initialized")
    
    try:
        query = clean_text(query)
        
        jobs_df = job_recommender.recommend(query, top_k=top_k)
        
        jobs_list = []
        seen = set()
        if not jobs_df.empty:
            for _, row in jobs_df.iterrows():
                job_idx = int(row.get("job_idx", -1))
                if job_idx >= 0 and job_idx < len(job_df):
                    title = job_df.loc[job_idx, "Job Title"]
                    if title not in seen:
                        seen.add(title)
                        jobs_list.append(
                            JobRecommendation(
                                title=title,
                                score=float(row.get("score", 0.0))
                            )
                        )
        
        return JobsResponse(
            jobs=jobs_list[:top_k],
            total=len(jobs_list)
        )
    except Exception as e:
        logger.error(f"Job recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint. Returns 503 while models are still initializing."""
    ready = _models_ready and all([career_predictor, university_recommender, job_recommender])
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "starting", "models_ready": False},
        )
    return {"status": "healthy", "models_ready": True}


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=AuthResponse, tags=["auth"])
async def register(body: RegisterRequest):
    """Register a new user. Returns a JWT token."""
    try:
        db = get_db()
        users = db["users"]

        existing = users.find_one({"email": body.email.lower()})
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

        hashed = hash_password(body.password)
        result = users.insert_one({
            "name": body.name,
            "email": body.email.lower(),
            "password_hash": hashed,
            "created_at": datetime.now(timezone.utc),
        })
        user_id = str(result.inserted_id)
        token = create_access_token(user_id, body.email.lower())
        return AuthResponse(token=token, userId=user_id, name=body.name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Register error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/auth/login", response_model=AuthResponse, tags=["auth"])
async def login(body: LoginRequest):
    """Login with email and password. Returns a JWT token."""
    try:
        db = get_db()
        users = db["users"]

        user = users.find_one({"email": body.email.lower()})
        if not user or not verify_password(body.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user_id = str(user["_id"])
        token = create_access_token(user_id, body.email.lower())
        return AuthResponse(token=token, userId=user_id, name=user.get("name", ""))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
