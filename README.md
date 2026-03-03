# Career Path Recommender System

An AI-powered career guidance platform that helps students discover their ideal career field, find relevant universities, and explore matching job roles — all from a single 2-minute assessment. Built with Sentence-BERT embeddings, supervised classification, and a modern React + FastAPI full-stack architecture.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Sentence‑BERT](https://img.shields.io/badge/NLP-Sentence--BERT-ff6f00)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React_18-61dafb)
![TypeScript](https://img.shields.io/badge/Language-TypeScript-3178c6)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-47a248)
![Tailwind CSS](https://img.shields.io/badge/Styles-Tailwind_CSS-38bdf8)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [ML Model Details](#ml-model-details)
5. [System Architecture](#system-architecture)
6. [Project Structure](#project-structure)
7. [Setup Instructions](#setup-instructions)
8. [API Documentation](#api-documentation)
9. [Model Training & Evaluation](#model-training--evaluation)
10. [Feature Status](#feature-status)
11. [Conclusion](#conclusion)
12. [Contributors](#contributors)

---

## Problem Statement

**The Problem:** Students in India and worldwide face decision paralysis when choosing a career after school or college. They rely on generic online quizzes, outdated counselor advice, or peer pressure — none of which consider their unique combination of academic strengths, soft skills, and personal interests.

**Our Solution:** A machine-learning-based recommendation engine that:
- Accepts a student's academic scores, soft skills, and free-text interests
- Classifies them into one of **8 career fields** using SBERT semantic understanding
- Recommends **universities** (Indian + global) ranked by relevance with an ML ranker
- Suggests specific **job roles** matched via semantic similarity on 30k+ job descriptions
- Delivers results through an intuitive, modern web interface with JWT-authenticated user accounts

**Target Audience:** High-school and college students, career counselors, and educational institutions seeking data-driven career guidance.

---

## Key Features

### 1. AI Career Field Classification
- **Primary Model:** SBERT (`all-MiniLM-L6-v2`) embeddings + Logistic Regression classifier
- **Input:** Free-text description of skills, interests, hobbies, and academic stream
- **Output:** Top-3 career fields with confidence scores
- **8 Career Fields:** Healthcare, STEM, Business & Finance, Arts & Media, Education, Social Services, Government & Law, Trades & Manufacturing

### 2. Smart University Recommendation
- **Dual-signal ranking:** 65% ML ranker (Random Forest Regressor on 16 engineered features) + 35% SBERT cosine similarity
- **Database:** 40,000+ institutions (Indian colleges + world universities)
- **Quality filters:** Elite institution boosting (IITs, NITs, BITS), coaching-centre penalty
- **Location-aware:** Filter by country and state

### 3. Semantic Job Role Matching
- **Model:** SBERT embeddings over 30k+ job descriptions
- **Method:** Cosine similarity between user text and pre-encoded job content
- **Fields indexed:** Job Title, Skills, Description, Responsibilities

### 4. Full-Stack Web Application
- **Frontend:** React 18 + TypeScript + Tailwind CSS + shadcn/ui
- **Backend:** FastAPI with async Python, Pydantic validation
- **Auth:** JWT-based authentication with bcrypt password hashing
- **Database:** MongoDB for user accounts and career metadata
- **Security:** Rate limiting, input validation, CORS configuration

---

## Technology Stack

### Backend (Python 3.12)
| Category | Technology | Purpose |
|----------|-----------|---------|
| **ML/NLP** | Sentence-Transformers (SBERT) | Text embeddings (`all-MiniLM-L6-v2`, 384-dim vectors) |
| **ML** | scikit-learn | Logistic Regression, Random Forest, Label Encoding, TF-IDF |
| **Data** | pandas, NumPy | Data manipulation and numerical operations |
| **NLP** | NLTK | Text preprocessing (stopwords, lemmatization) |
| **API** | FastAPI + Uvicorn | REST API server with async support |
| **Database** | PyMongo | MongoDB driver for user auth and career metadata |
| **Auth** | python-jose, bcrypt | JWT tokens and password hashing |
| **Serialization** | joblib | Model persistence (.pkl files) |
| **Config** | PyYAML, python-dotenv | Configuration management |

### Frontend (TypeScript)
| Category | Technology | Purpose |
|----------|-----------|---------|
| **Framework** | React 18 | UI component library |
| **Language** | TypeScript | Type-safe JavaScript |
| **Build** | Vite | Fast dev server and bundler |
| **Styling** | Tailwind CSS + shadcn/ui | Utility-first CSS with accessible components |
| **Routing** | React Router v6 | Client-side navigation |
| **State** | TanStack Query | Server state management |
| **HTTP** | Fetch API | Backend communication |

### Infrastructure
| Category | Technology | Purpose |
|----------|-----------|---------|
| **Database** | MongoDB | User accounts, career metadata (careers collection) |
| **Version Control** | Git + GitHub | Source control |
| **Model Storage** | `.pkl` files (joblib) | Serialized trained models |
| **Embedding Cache** | `.npy` files (NumPy) | Pre-computed SBERT embeddings |

---

## ML Model Details

### Primary Model: SBERT + Logistic Regression (Career Classification)

**Why SBERT?** Traditional bag-of-words approaches (TF-IDF) treat each word independently and miss semantic relationships. SBERT encodes entire sentences into dense 384-dimensional vectors that capture meaning — so "I love coding in Python and building websites" maps close to STEM even without the word "STEM" appearing in the input.

**Architecture:**
```
User Text Input
     │
     ▼
┌──────────────────────────┐
│  Sentence-BERT Encoder   │  ← all-MiniLM-L6-v2 (22.7M parameters)
│  (384-dim embeddings)    │    Pre-trained on 1B+ sentence pairs
└──────────┬───────────────┘
           │
     384-dim vector (L2-normalized)
           │
           ▼
┌──────────────────────────┐
│  Logistic Regression     │  ← Trained on 5,000 student samples
│  (8-class classifier)    │    max_iter=1000, C=1.0, balanced weights
└──────────┬───────────────┘
           │
           ▼
   Probability distribution over 8 career fields
```

**Training Details:**
| Parameter | Value |
|-----------|-------|
| Encoder | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| Embedding dimension | 384 |
| Classifier | Logistic Regression (sklearn) |
| Training samples | 5,000 (80/20 train-test split) |
| Train set | 4,000 samples |
| Test set | 1,000 samples |
| Solver | L-BFGS |
| Max iterations | 1,000 |
| Regularization (C) | 1.0 |
| Class weighting | Balanced (auto-adjusts for class imbalance) |
| Random state | 42 |
| Feature normalization | L2-normalized embeddings |

**Training text construction:** Each training sample is built from a student's dataset row by converting numeric scores to tiers ("strong mathematics", "moderate creativity"), adding participation flags ("participates in science club and debate"), learning style, top career domain, and augmented with career-field keywords from curated descriptions.

### Model Comparison (Ablation Study)

| Model | Accuracy | Precision | Recall | F1 Score | Training Approach |
|-------|----------|-----------|--------|----------|-------------------|
| **SBERT + Logistic Regression** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | Semantic embeddings + linear head |
| TF-IDF + Logistic Regression | 100.0% | 100.0% | 100.0% | 100.0% | Bag-of-words + linear head |
| Numeric-only Random Forest | 41.4% | 40.5% | 41.4% | 39.7% | Raw scores only (no text) |

**Key Insights:**
- Both text-based models achieve perfect accuracy because the engineered training text contains discriminative vocabulary per career field
- The numeric-only Random Forest (41.4%) proves that academic scores alone are insufficient — a student with high math scores could go into STEM, Finance, or Education
- **SBERT is chosen over TF-IDF** because at inference time, real users type free-form sentences (e.g., "I enjoy painting and photography") — SBERT captures the semantic meaning even when exact training vocabulary is absent, while TF-IDF would fail on unseen words
- Confusion matrix for RF shows heavy misclassification between Business_Finance/Government_Law and Healthcare/STEM — fields that share similar score profiles

### Secondary Model: University Ranker (Random Forest Regressor)

**Purpose:** Rank 40,000+ universities by relevance to a student's predicted career and preferences.

**Architecture:** Dual-signal blending:
- **65% ML Ranker** — Random Forest Regressor trained on 16 hand-engineered features
- **35% SBERT Cosine Similarity** — Semantic match between user text and university metadata

**16 Engineered Features:**
| Feature | Description |
|---------|-------------|
| `specialization_match` | Binary: does university specialize in the career area? |
| `specialization_overlap` | % of user keywords found in university text |
| `name_overlap` | % of user keywords in university name |
| `country_match` | Does country match preference? |
| `state_match` | Does state match preference? |
| `has_state/district/website` | Data completeness signals |
| `search_text_len` / `name_len` | Proxy for information richness |
| `country_is_india` | India-specific features |
| `keyword_boost` | Domain keyword overlap score |
| `has_real_specialisation` | Has declared specialization (vs. "No") |
| `is_university_college` | Constituent/autonomous college flag |
| `is_premier` | Central university / IIT / NIT / IIIT / IISc detection |
| `is_coaching` | Coaching centre / skill institute penalty |

### Semantic Job Matching (Cosine Similarity)

- Pre-encodes 30,000+ job descriptions using SBERT
- At query time, encodes user text and finds top-k by cosine similarity
- Fields used: Job Title + Skills + Description + Responsibilities
- Embeddings cached to `.npy` files for instant startup

---

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│                    FRONTEND                            │
│     React 18 + TypeScript + Tailwind + shadcn/ui       │
│     (http://localhost:8080)                            │
│                                                        │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Home Page  │  │  Assessment  │  │  Results     │  │
│  │  (Hero,     │  │  Form        │  │  Dashboard   │  │
│  │   Stats,    │  │  (Scores +   │  │  (Career,    │  │
│  │   CTA)      │  │   Text)      │  │   Unis, Jobs)│  │
│  └────────────┘  └──────────────┘  └──────────────┘  │
│                                                        │
│  Auth: JWT token stored in React Context               │
└───────────────────────┬────────────────────────────────┘
                        │ REST API (JSON)
                        │ Authorization: Bearer <JWT>
                        ▼
┌────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                    │
│     (http://localhost:8000)                            │
│                                                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Auth Layer  │  │  Rate Limiter│  │  CORS        │ │
│  │  (JWT+bcrypt)│  │  (10/min/IP) │  │  Middleware   │ │
│  └──────┬──────┘  └──────────────┘  └──────────────┘ │
│         ▼                                              │
│  ┌─────────────────────────────────────────────────┐  │
│  │              CareerService (Orchestrator)         │  │
│  │  Delegates to SBERTCareerClassifier               │  │
│  └────────┬──────────────┬──────────────┬──────────┘  │
│           ▼              ▼              ▼              │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐    │
│  │ SBERT Career │ │ University │ │ Job          │    │
│  │ Classifier   │ │ Recommender│ │ Recommender  │    │
│  │ (LR + SBERT) │ │ (RF + SBERT│ │ (Cosine Sim) │    │
│  └──────────────┘ │  Ranker)   │ └──────────────┘    │
│                    └────────────┘                      │
│                                                        │
│  Shared: FeatureBuilder (all-MiniLM-L6-v2, loaded 1x) │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│                    MongoDB                             │
│  Collections:                                          │
│  - users (auth: name, email, password_hash)            │
│  - careers (metadata: salary, skills, growth, pathway) │
└────────────────────────────────────────────────────────┘
```

### Data Flow (User Assessment → Results)

1. **User** fills out the assessment form (9 numeric scores + free-text skills/interests + preferred location)
2. **Frontend** sends `POST /api/recommend` with JWT token
3. **Backend** validates input (Pydantic models, field constraints 0–100)
4. **CareerService** passes `skills_text` to `SBERTCareerClassifier` → SBERT encodes text → Logistic Regression outputs probability distribution → returns top-3 career fields
5. **UniversityRecommender** uses predicted career as query → computes 16 features per university → RF Regressor scores + SBERT cosine similarity → 65/35 blend → returns top-10 universities
6. **CareerRecommender** encodes `skills_text + top_career` → cosine similarity against 30k+ pre-encoded job descriptions → returns top-5 jobs
7. **Career Metadata** fetched from MongoDB (salary, growth rate, required skills, career pathway)
8. **Response** returned as JSON with career prediction, universities, jobs, and metadata
9. **Frontend** renders results in an interactive dashboard

---

## Project Structure

```
proj-career-reco/
├── backend/
│   ├── config/
│   │   └── config.yaml              # Dataset paths, model hyperparameters
│   ├── data/
│   │   ├── raw/                     # Raw CSV datasets
│   │   └── processed/               # Cleaned/processed data
│   ├── notebooks/                   # Jupyter EDA notebooks
│   │   ├── 00_data_reading.ipynb
│   │   ├── 01_data_cleaning.ipynb
│   │   ├── 01_eda.ipynb
│   │   └── 02_data_exploration.ipynb
│   ├── src/
│   │   ├── app/
│   │   │   ├── api.py               # FastAPI routes & lifespan
│   │   │   └── main.py              # CLI entry point (legacy)
│   │   ├── auth/
│   │   │   ├── auth.py              # JWT + bcrypt helpers
│   │   │   └── user_service.py      # AuthService + UserRepository (SOLID)
│   │   ├── data/
│   │   │   ├── loader.py            # YAML config + CSV loader
│   │   │   └── preprocessing.py     # Text cleaning (NLTK)
│   │   ├── db/
│   │   │   ├── mongo.py             # MongoDB singleton client
│   │   │   └── career_repository.py # Career CRUD operations
│   │   ├── features/
│   │   │   └── build_features.py    # FeatureBuilder (SBERT wrapper)
│   │   ├── models/
│   │   │   ├── config.py            # All ML config + career descriptions
│   │   │   ├── sbert_career_classifier.py  # SBERT + LR classifier
│   │   │   ├── career_predictor.py  # Random Forest (numeric baseline)
│   │   │   ├── career_recommender.py # Job recommendation (cosine sim)
│   │   │   ├── university_recommender.py   # University matching
│   │   │   ├── university_ranker.py # RF Regressor for uni ranking
│   │   │   └── country_utils.py     # Country name normalization
│   │   ├── services/
│   │   │   └── career_service.py    # CareerService orchestrator
│   │   └── utils/                   # Utility functions
│   ├── tests/
│   │   ├── test_career_service.py
│   │   ├── test_predictor.py
│   │   └── test_full_pipeline.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/              # React components (HeroSection, AssessmentForm, etc.)
│   │   ├── pages/                   # Route pages (Index, Assessment, Login, Signup)
│   │   ├── context/                 # AuthContext (JWT management)
│   │   ├── data/                    # careerData.ts (hardcoded suggestions)
│   │   ├── hooks/                   # Custom hooks (useParallax)
│   │   └── lib/                     # Utility functions
│   └── package.json
├── models/                          # Trained model artifacts (git-ignored)
│   ├── career_predictor.pkl
│   ├── sbert_career_classifier.pkl
│   ├── university_ranker.pkl
│   ├── university_embeddings.npy
│   └── cache/                       # Job embeddings cache
├── scripts/
│   ├── retrain_models.py            # Full training pipeline + evaluation report
│   ├── seed_careers.py              # MongoDB career metadata seeder
│   ├── setup_db.py                  # Database setup
│   └── fix_emoji.py                 # Utility script
├── reports/
│   └── evaluation_report.md         # Auto-generated model metrics
├── start.ps1                        # Windows launcher
├── start.sh                         # Unix launcher
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+ (tested on 3.12.7)
- Node.js 18+ and npm
- MongoDB (local or Atlas)
- Git

### Option 1: Quick Start (Windows PowerShell)
```powershell
.\start.ps1
```

### Option 2: Quick Start (Mac/Linux)
```bash
chmod +x start.sh && ./start.sh
```

### Option 3: Manual Setup

**Step 1 — Clone and enter the project:**
```bash
git clone <repository-url>
cd proj-career-reco
```

**Step 2 — Backend setup:**
```bash
cd backend
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

**Step 3 — Configure environment:**
```bash
# Copy the example .env and fill in your values
cp .env.example .env
```

Required variables in `backend/.env`:
```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=career_recommender
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=48
```

**Step 4 — Train models (first time only):**
```bash
# From project root:
python scripts/retrain_models.py
```
This generates:
- `models/sbert_career_classifier.pkl`
- `models/university_ranker.pkl`
- `reports/evaluation_report.md`

**Step 5 — Seed MongoDB career metadata:**
```bash
python scripts/seed_careers.py
```

**Step 6 — Start backend server:**
```bash
cd backend
python -m uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
```

**Step 7 — Start frontend (new terminal):**
```bash
cd frontend
npm install
npm run dev
```

**Step 8 — Open in browser:**
- Frontend: http://localhost:8080
- Backend API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### How to Retrain Models
```bash
# From project root (with backend venv activated):
python scripts/retrain_models.py
```
This will:
1. Load and prepare the student dataset (5,000 samples)
2. Train SBERT + Logistic Regression (primary model)
3. Train TF-IDF + Logistic Regression (baseline comparison)
4. Train Numeric-only Random Forest (baseline comparison)
5. Train University Ranker (RF Regressor)
6. Generate evaluation report at `reports/evaluation_report.md`

---

## API Documentation

All endpoints are documented via Swagger UI at `http://localhost:8000/docs` when the server is running.

### Authentication Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/auth/register` | Create new user account | No |
| POST | `/auth/login` | Login and get JWT token | No |

**Register:**
```json
POST /auth/register
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword"
}
// Response: { "token": "eyJ...", "userId": "...", "name": "John Doe" }
```

**Login:**
```json
POST /auth/login
{
  "email": "john@example.com",
  "password": "securepassword"
}
// Response: { "token": "eyJ...", "userId": "...", "name": "John Doe" }
```

### Recommendation Endpoints (JWT Required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/recommend` | Full recommendation (career + universities + jobs + metadata) |
| POST | `/api/career-predict` | Career field prediction only |
| POST | `/api/universities` | University recommendations only |
| POST | `/api/jobs` | Job recommendations only |

**Full Recommendation:**
```json
POST /api/recommend
Authorization: Bearer <JWT_TOKEN>
{
  "mathematics_score": 85,
  "science_score": 90,
  "language_arts_score": 70,
  "social_studies_score": 65,
  "logical_reasoning": 88,
  "creativity": 75,
  "communication": 72,
  "leadership": 60,
  "social_skills": 68,
  "skills_text": "Python programming, machine learning, data analysis, statistics",
  "preferred_location": "India"
}
```

**Response:**
```json
{
  "career": {
    "career_field": "STEM",
    "confidence": 0.82,
    "alternatives": [["Business_Finance", 0.10], ["Education", 0.05]]
  },
  "universities": {
    "universities": [
      {
        "name": "Indian Institute of Technology Bombay",
        "country": "India",
        "state": "Maharashtra",
        "website": "...",
        "score": 0.95
      }
    ],
    "total": 10
  },
  "jobs": {
    "jobs": [
      { "title": "Data Scientist", "score": 0.88 },
      { "title": "Machine Learning Engineer", "score": 0.85 }
    ],
    "total": 5
  },
  "career_metadata": {
    "career_id": "STEM",
    "title": "Science, Technology, Engineering & Mathematics",
    "salary_display": "₹6L – ₹25L per annum",
    "growth_rate": "22%",
    "skills": ["Python", "Data Analysis", "Machine Learning"],
    "pathway": [{ "step": 1, "title": "Bachelor's Degree", "description": "..." }]
  },
  "alternatives_metadata": []
}
```

### Utility Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | API status | No |
| GET | `/health` | Health check (model readiness) | No |
| GET | `/api/careers` | List all career metadata | No |
| GET | `/api/careers/{id}` | Single career metadata | No |

---

## Model Training & Evaluation

### Dataset
- **Source:** Career Recommendation Dataset (5,000 student records)
- **Features per record:** Academic scores (4), soft skills (5), participation flags (6), learning style, domain scores (8)
- **Target:** `Primary_Career_Recommendation` — one of 8 career fields
- **Train/Test split:** 80/20 (stratified), random_state=42

### Training Pipeline (`scripts/retrain_models.py`)
1. **Text Construction:** Converts numeric scores to tier labels ("strong", "moderate", "developing"), adds participation and learning style, augments with career-field keywords
2. **SBERT Encoding:** Encodes all training texts into 384-dim vectors using `all-MiniLM-L6-v2`
3. **Logistic Regression Training:** Fits classifier on normalized embeddings
4. **Ablation Study:** Trains TF-IDF + LR and Numeric RF baselines for comparison
5. **Report Generation:** Outputs classification reports and confusion matrices to `reports/evaluation_report.md`

### Results Summary

| Model | Accuracy | F1 (weighted) | Why? |
|-------|----------|---------------|------|
| **SBERT + Logistic Regression** | **100.0%** | **100.0%** | Semantic embeddings capture meaning; discriminative training text |
| TF-IDF + Logistic Regression | 100.0% | 100.0% | Keyword overlap works on structured training text |
| Numeric-only Random Forest | 41.4% | 39.7% | Scores alone can't distinguish career intent |

**Why SBERT wins at inference (real-world use):**
- TF-IDF fails on unseen vocabulary — a user typing "I want to be a doctor" has zero overlap with training tokens like "strong science"
- SBERT maps "I want to be a doctor" and "Healthcare" close in embedding space
- SBERT generalizes to free-form natural language; TF-IDF requires exact word matches

---

## Feature Status

### Implemented
- [x] SBERT + LR career classification (8 fields)
- [x] University recommendation with ML ranker (40k+ institutions)
- [x] Semantic job matching (30k+ jobs)
- [x] Full-stack web app (React + FastAPI)
- [x] JWT authentication with bcrypt
- [x] MongoDB integration (users + career metadata)
- [x] Rate limiting on login endpoint
- [x] CORS + input validation (Pydantic)
- [x] Model training pipeline with ablation study
- [x] Cinematic homepage with parallax effects
- [x] Interactive assessment form with suggestions
- [x] Career results dashboard with metadata

### Future Scope
- [ ] Resume parsing for auto-fill assessment
- [ ] Multi-language support
- [ ] Feedback loop for continuous model improvement
- [ ] Collaborative filtering based on similar student profiles
- [ ] Cloud deployment (AWS/Render)
- [ ] Career path comparison feature
- [ ] Export recommendations as PDF

---

## Conclusion

This project demonstrated how modern NLP and machine learning techniques can be applied to solve a genuine real-world problem — career indecision among students. By combining Sentence-BERT embeddings with a Logistic Regression classifier, the system goes beyond simple keyword matching to semantically understand a student's skills, interests, and aspirations, classifying them into one of eight career fields with high confidence. A three-way ablation study (SBERT + LR vs TF-IDF + LR vs numeric-only Random Forest at 41.4% accuracy) validated the architectural choice, proving that semantic text understanding is essential for career prediction where raw academic scores alone cannot distinguish intent.

Beyond classification, the project implements a dual-signal university recommender that blends a Random Forest Regressor (trained on 16 hand-engineered features) with SBERT cosine similarity to rank 40,000+ institutions, and a semantic job matcher that surfaces relevant roles from 30,000+ job descriptions. The full-stack implementation — React 18 with TypeScript on the frontend, FastAPI with JWT authentication and MongoDB on the backend — follows SOLID design principles throughout, with dependency injection, repository patterns, and service-layer separation ensuring the codebase is maintainable and testable.

Building this system provided hands-on experience across the entire ML engineering lifecycle: data collection and preprocessing, feature engineering (converting numeric scores to natural-language tiers with career keyword augmentation), model training and evaluation, REST API design with input validation and rate limiting, NoSQL database integration, and a production-grade frontend with responsive design and parallax animations.

The system can be extended further with resume parsing for auto-filled assessments, multilingual support via SBERT's multilingual variants, a user feedback loop for continuous model improvement, and collaborative filtering based on similar student profiles. We believe this project shows how AI-powered tools can provide accessible, data-driven career guidance to students who would otherwise rely on generic advice or expensive counseling sessions.

---

## License

This project is developed as a college final-year project. Open source for educational purposes.
