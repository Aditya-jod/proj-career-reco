# Frontend-Backend Setup & Connection Guide

Complete guide to set up and run the Career Path Recommender System with the new FastAPI backend.

## Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 18+** or **Bun** (for frontend)
- **Git** (already in use)

## Quick Start

### 1. Backend Setup (Python)

#### Step 1.1: Install Backend Dependencies

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download NLP models (first time only)
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

#### Step 1.2: Configure Environment (Optional)

Create a `.env` file in the `backend/` directory:

```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=career_recommender
API_PORT=8000
APP_ENV=development
```

Or copy from example:
```bash
cp .env.example .env
```

#### Step 1.3: Start the Backend API Server

```bash
# From backend directory with activated venv
python -m uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

Visit `http://localhost:8000/docs` to see Swagger API documentation.

---

### 2. Frontend Setup (React + Vite)

#### Step 2.1: Install Frontend Dependencies

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install
# OR with bun:
bun install
```

#### Step 2.2: Configure Environment

Create a `.env` file in the `frontend/` directory:

```env
VITE_API_URL=http://localhost:8000
VITE_ENV=development
```

Or copy from example:
```bash
cp .env.example .env
```

#### Step 2.3: Start the Development Server

```bash
# From frontend directory
npm run dev
# OR with bun:
bun run dev
```

**Expected output:**
```
    Local:     http://localhost:8080/
    press q + enter to stop
```

Visit `http://localhost:8080` in your browser.

---

## How Frontend Connects to Backend

### Architecture

```
Frontend (React)
         ↓
[careerData.ts] ← API calls with Fetch
         ↓
Backend FastAPI Server (http://localhost:8000)
         ↓
ML Models (Career Predictor, University Recommender, Job Recommender)
```

### Key Connection Points

1. **API Base URL** configured in `frontend/.env`:
     ```typescript
     const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"
     ```

2. **API Endpoints** used:
     - `POST /api/recommend` — Get complete recommendations (main endpoint)
     - `POST /api/career-predict` — Predict career field only
     - `POST /api/universities` — Get university recommendations
     - `POST /api/jobs` — Get job recommendations
     - `GET /health` — Health check

3. **Data Flow**:
     ```
     User fills form → AssessmentForm.tsx
             ↓
     Form data → StudentProfile object
             ↓
     careerData.ts getRecommendations() function
             ↓
     Sends POST to /api/recommend
             ↓
     Backend processes with ML models
             ↓
     Returns RecommendationResult JSON
             ↓
     Frontend renders CareerResults component
     ```

---

## Verify Everything Works

### Check Backend is Running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
    "status": "healthy",
    "models_ready": true
}
```

### Check Frontend Can Reach Backend

Open browser DevTools (F12) → Console and run:

```javascript
fetch('http://localhost:8000/health')
    .then(r => r.json())
    .then(d => console.log('Backend is running:', d))
    .catch(e => console.error('Backend error:', e))
```

### Test Full Flow

1. Visit http://localhost:8080
2. Fill the assessment form
3. Click "Get Recommendations"
4. Watch browser DevTools → Network tab
5. Verify `POST /api/recommend` request succeeds (200 status)
6. See recommendations displayed

---


---

## Troubleshooting

### Backend Won't Start

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

**Error**: `Address already in use` (port 8000 taken)

**Solution**: Run on different port:
```bash
python -m uvicorn src.app.api:app --port 8001
# Then update frontend .env: VITE_API_URL=http://localhost:8001
```

### Frontend Can't Connect to Backend

**Error**: `Failed to get recommendations` / CORS error

**Checklist**:
- Backend is running on `http://localhost:8000`
- Backend `/health` endpoint returns 200
- `frontend/.env` has correct `VITE_API_URL`
- Browser Network tab shows POST request to backend

**Solution**: Open DevTools → Console → Run:
```javascript
fetch('http://localhost:8000/health').then(r => r.json()).catch(e => console.error(e))
```

### Models Not Training

**Error**: `ValueError: Insufficient data to train university ranker`

**Solution**: Ensure all CSV data files are properly loaded:
```bash
python -c "from src.data.loader import load_config, load_raw_data; c=load_config(); d=load_raw_data(c); print(list(d.keys()))"
```

---

## Production Build

### Build Frontend

```bash
cd frontend
npm run build
```

Creates `dist/` folder ready for deployment.

### Deploy Backend

Use services like Render.com, Heroku, AWS, etc.

Update frontend `.env` to production API URL.

---

## Security Checklist

- `node_modules/` in `.gitignore`
- `venv/` in `.gitignore`
- `.env` files in `.gitignore` (only `.env.example` committed)
- `models/cache/` excluded (cache rebuilt on startup)
- `__pycache__/` excluded
- CORS enabled only for trusted origins (update for production)

---

## Additional Commands

```bash
# Frontend
npm run dev        # Start dev server
npm run build      # Build for production
npm run preview    # Preview production build
npm run lint       # Lint code
npm run test       # Run tests

# Backend
python src/app/main.py              # Run CLI version
python -m uvicorn src.app.api:app   # Run API server
pytest                              # Run tests (if configured)
```

---

## Next Steps

1. Backend API running → http://localhost:8000
2. Frontend connected → http://localhost:8080
3. MongoDB integration (optional for user sessions)
4. User authentication
5. Deploy to cloud

---

