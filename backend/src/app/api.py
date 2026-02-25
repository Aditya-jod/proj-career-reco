"""
FastAPI server for Career Path Recommender System.
Exposes ML models (Career Predictor, University Recommender, Job Recommender) as REST endpoints.
"""

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

# Load .env before anything else so all os.getenv calls below pick it up
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from src.data.loader import load_config, load_raw_data
from src.data.preprocessing import clean_text
from src.features.build_features import FeatureBuilder
from src.models.sbert_career_classifier import SBERTCareerClassifier
from src.models.university_recommender import UniversityRecommender
from src.models.career_recommender import CareerRecommender
from src.services.career_service import CareerService
from src.auth.user_service import AuthService, UserRepository
from src.auth.auth import get_current_user
from src.db.mongo import get_db, close_db
from src.db.career_repository import (
    get_all_careers,
    get_career,
    get_career_metadata,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limiter middleware (simple sliding-window per-IP)
# ---------------------------------------------------------------------------

class _RateLimitState:
    """Thread-safe in-memory rate-limit store keyed by (ip, path)."""

    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        hits = self._hits[key]
        # Remove expired timestamps
        self._hits[key] = [t for t in hits if now - t < self.window]
        if len(self._hits[key]) >= self.max_calls:
            return False
        self._hits[key].append(now)
        return True


_login_limiter = _RateLimitState(max_calls=10, window_seconds=60)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply rate‐limiting to /auth/login only."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/auth/login" and request.method == "POST":
            ip = request.client.host if request.client else "unknown"
            if not _login_limiter.is_allowed(ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many login attempts. Try again later."},
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:8080,http://localhost:3000",
)
ALLOWED_ORIGINS: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Global model state — assigned during startup lifespan, used by route handlers.
# ---------------------------------------------------------------------------
career_predictor: Optional[CareerService] = None
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
        if datasets is None:
            raise RuntimeError("Failed to load datasets — check config.yaml paths.")

        # ── Shared Sentence-BERT encoder (university, job, and SBERT career classifier)
        # Loaded once; the same model instance is reused across all components.
        logger.info("Loading Sentence-BERT encoder…")
        feature_builder = FeatureBuilder()

        # ── Supervised SBERT Career Classifier (load trained classifier) ────
        logger.info("Loading supervised SBERT career classifier…")
        sbert_clf = SBERTCareerClassifier(encoder=feature_builder)
        sbert_clf.load()   # loads trained LogisticRegression from disk

        # ── CareerService wraps the classifier ──────────────────────────────
        career_predictor = CareerService(classifier=sbert_clf)

        # ── University Recommender ──────────────────────────────────────────
        logger.info("Loading University Recommender…")
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
            job_feature_builder = feature_builder   # reuse the shared encoder
            job_embeddings = job_feature_builder.encode(
                job_df["content"].tolist(), batch_size=128
            )
            job_df.to_parquet(job_df_path, index=False)
            np.save(job_emb_path, job_embeddings)

        job_recommender = CareerRecommender(
            job_df=job_df,
            embedding_matrix=job_embeddings,
            feature_builder=feature_builder,   # reuse the shared encoder
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
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
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

    @field_validator("state", "district", mode="before")
    @classmethod
    def coerce_nan_to_none(cls, v: object) -> Optional[str]:
        """Convert pandas NaN (float) and blank strings to None."""
        if v is None:
            return None
        if isinstance(v, float):          # NaN comes through as float from pandas
            return None
        text = str(v).strip()
        return text if text else None


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
    career_metadata: Optional["CareerMetadataDetail"] = None


class CareerMetadataDetail(BaseModel):
    """Dynamic career metadata served from MongoDB."""
    career_id: str
    title: str
    salary_display: str = ""
    growth_description: str = ""
    growth_rate: str = ""
    skills: List[str] = []
    pathway: List[dict] = []


class AlternativeMetadata(BaseModel):
    """Metadata for an alternative career."""
    career_id: str
    title: str
    salary_display: str = ""
    growth_description: str = ""
    growth_rate: str = ""
    skills: List[str] = []
    pathway: List[dict] = []


class EnrichedRecommendationResponse(BaseModel):
    """Recommendation response with full career metadata."""
    career: CareerPredictionResponse
    universities: UniversitiesResponse
    jobs: JobsResponse
    career_metadata: Optional[CareerMetadataDetail] = None
    alternatives_metadata: List[AlternativeMetadata] = []


class SuggestionsResponse(BaseModel):
    """Dynamic suggestions for the assessment form."""
    interests: List[str] = []
    skills: List[str] = []
    hobbies: List[str] = []
    academic_streams: List[str] = []


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
        "version": "2.0.0"
    }


@app.post("/api/recommend", response_model=EnrichedRecommendationResponse)
async def get_recommendations(
    profile: StudentProfileRequest,
    _user: dict = Depends(get_current_user),
):
    """
    Get complete recommendations (career, universities, jobs) based on student profile.
    
    Args:
        profile: StudentProfileRequest with academic and skill scores
    
    Returns:
        RecommendationResponse with career, university, and job recommendations
    """
    if not all([career_predictor, university_recommender, job_recommender]) or job_df is None:
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
        assert career_predictor is not None, "Models not loaded"
        assert university_recommender is not None, "Models not loaded"
        assert job_recommender is not None, "Models not loaded"

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
        # Build a rich natural-language query for Sentence-BERT.
        # Do NOT apply clean_text() here — NLTK stopword-stripping hurts SBERT.
        # Combine skills_text with the predicted career so the semantic search
        # finds jobs that match both the user's skills AND the career field.
        job_query_parts = [p for p in [profile.skills_text, top_career] if p]
        job_query = ", ".join(job_query_parts)
        
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

        # ==================== Career Metadata from MongoDB ====================
        primary_meta = get_career_metadata(top_career)
        meta_detail = None
        if primary_meta:
            meta_detail = CareerMetadataDetail(
                career_id=primary_meta.get("career_id", top_career),
                title=primary_meta.get("title", top_career),
                salary_display=primary_meta.get("salary_display", ""),
                growth_description=primary_meta.get("growth_description", ""),
                growth_rate=primary_meta.get("growth_rate", ""),
                skills=primary_meta.get("skills", []),
                pathway=primary_meta.get("pathway", []),
            )

        alt_meta_list = []
        for alt_field, _ in predictions[1:]:
            alt_doc = get_career_metadata(str(alt_field))
            if alt_doc:
                alt_meta_list.append(AlternativeMetadata(
                    career_id=alt_doc.get("career_id", str(alt_field)),
                    title=alt_doc.get("title", str(alt_field)),
                    salary_display=alt_doc.get("salary_display", ""),
                    growth_description=alt_doc.get("growth_description", ""),
                    growth_rate=alt_doc.get("growth_rate", ""),
                    skills=alt_doc.get("skills", []),
                    pathway=alt_doc.get("pathway", []),
                ))

        return EnrichedRecommendationResponse(
            career=career_response,
            universities=universities_response,
            jobs=jobs_response,
            career_metadata=meta_detail,
            alternatives_metadata=alt_meta_list,
        )
        
    except Exception as e:
        logger.error("Recommendation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")


@app.post("/api/career-predict", response_model=CareerPredictionResponse)
async def predict_career(
    profile: StudentProfileRequest,
    _user: dict = Depends(get_current_user),
):
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
        logger.error("Career prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to predict career")


@app.post("/api/universities", response_model=UniversitiesResponse)
async def recommend_universities(
    query: str,
    country: Optional[str] = None,
    top_k: int = 10,
    skills_text: str = "",
    _user: dict = Depends(get_current_user),
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
        logger.error("University recommendation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate university recommendations")


@app.post("/api/jobs", response_model=JobsResponse)
async def recommend_jobs(
    query: str,
    top_k: int = 10,
    _user: dict = Depends(get_current_user),
):
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
        logger.error("Job recommendation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate job recommendations")


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
        service = AuthService(UserRepository(get_db()))
        result = service.register(body.name, body.email, body.password)
        return AuthResponse(token=result.token, userId=result.user_id, name=result.name)
    except ValueError as e:
        status = 409 if "already registered" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e))
    except Exception as e:
        logger.error("Register error: %s", e)
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/auth/login", response_model=AuthResponse, tags=["auth"])
async def login(body: LoginRequest):
    """Login with email and password. Returns a JWT token."""
    try:
        service = AuthService(UserRepository(get_db()))
        result = service.login(body.email, body.password)
        return AuthResponse(token=result.token, userId=result.user_id, name=result.name)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error("Login error: %s", e)
        raise HTTPException(status_code=500, detail="Login failed")


# ── Career Metadata Endpoints ────────────────────────────────────────────────

@app.get("/api/careers", response_model=List[CareerMetadataDetail], tags=["careers"])
async def list_careers():
    """Return all career fields with their metadata (salary, skills, growth, pathway)."""
    docs = get_all_careers()
    return [
        CareerMetadataDetail(
            career_id=d.get("career_id", ""),
            title=d.get("title", ""),
            salary_display=d.get("salary_display", ""),
            growth_description=d.get("growth_description", ""),
            growth_rate=d.get("growth_rate", ""),
            skills=d.get("skills", []),
            pathway=d.get("pathway", []),
        )
        for d in docs
    ]


@app.get("/api/careers/{career_id}", response_model=CareerMetadataDetail, tags=["careers"])
async def get_career_detail(career_id: str):
    """Return metadata for a single career field."""
    doc = get_career(career_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Career '{career_id}' not found")
    return CareerMetadataDetail(
        career_id=doc.get("career_id", career_id),
        title=doc.get("title", career_id),
        salary_display=doc.get("salary_display", ""),
        growth_description=doc.get("growth_description", ""),
        growth_rate=doc.get("growth_rate", ""),
        skills=doc.get("skills", []),
        pathway=doc.get("pathway", []),
    )


@app.get("/api/suggestions", response_model=SuggestionsResponse, tags=["suggestions"])
async def get_suggestions():
    """Return dynamic suggestion lists for the assessment form (interests, skills, hobbies, streams)."""
    db = get_db()
    doc = db["suggestions"].find_one({"doc_id": "main"}, {"_id": 0})
    if not doc:
        return SuggestionsResponse()
    return SuggestionsResponse(
        interests=doc.get("interests", []),
        skills=doc.get("skills", []),
        hobbies=doc.get("hobbies", []),
        academic_streams=doc.get("academic_streams", []),
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
