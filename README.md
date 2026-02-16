# Career Path Recommender System

A AI recommendation engine that helps students discover their ideal career path, find relevant universities, and explore specific job roles based on their academic scores, soft skills, and personal interests. 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/AI-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-green)
![NLTK](https://img.shields.io/badge/NLP-NLTK-yellowgreen)
![SpaCy](https://img.shields.io/badge/NLP-SpaCy-lightgrey)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)
![React](https://img.shields.io/badge/Frontend-React-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![NumPy](https://img.shields.io/badge/Math-NumPy-blueviolet)
![Git](https://img.shields.io/badge/Version%20Control-Git-red)
![Render](https://img.shields.io/badge/Deployment-Render.com-brightgreen)

## Key Features

### 1. Intelligent Career Prediction
- **Model:** RandomForestClassifier.
- **Inputs:** Academic + soft-skill scores.
- **Output:** Top career fields plus fallback options when confidence < 50%.

### 2. University Recommendation (Semantic)
- **Model:** Sentence-BERT embeddings + cosine similarity.
- **Cache:** Embeddings stored once, then reused for fast lookup.
- **Scope:** Filters by country/state/city for location-aware results.

### 3. Job Role Recommendation (NLP)
- **Model:** Sentence-BERT embeddings over 30k job descriptions.
- **Logic:** Content-based matching on user-entered skills/interests.
- **Override:** Interests can influence the final career choice.

---

## Architecture Overview

1. **Classification Phase (Random Forest)**
   - Input: academic and soft-skill scores.
   - Model: RandomForestClassifier trained on 9k+ labeled student profiles.
   - Output: top-k career fields with confidence metrics; low-confidence results trigger a fallback list for user choice.

2. **Semantic University Matching (Sentence-BERT)**
   - Encode university descriptions and field tags using `all-MiniLM-L6-v2`.
   - Store embeddings on disk to avoid recomputation; load into memory on startup.
   - Filter by country/state/city, then rank via cosine similarity to the chosen career field vector.

3. **Semantic Job Retrieval (Sentence-BERT)**
   - Concatenate job title, skills, description, and responsibilities into a `content` field.
   - Encode 30k+ jobs once; reuse embeddings for fast similarity search.
   - Encode user free-text interests at runtime and retrieve top-k roles based on cosine scores.

4. **MongoDB Persistence (Roadmap)**
   - Collections planned for `users`, `sessions`, and `saved_plans`.
   - Store inputs/outputs per session to enable history, analytics, and personalized recommendations.
   - Connection handled through `src/db/mongo.py`, reading `MONGODB_URI` / `MONGODB_DB_NAME` from environment variables.

5. **Frontend + API Integration (✅ LIVE)**
   - Backend: FastAPI server (`src/app/api.py`) exposing all ML models via REST endpoints.
   - Frontend: React + Vite + Tailwind for interactive forms and result visualizations.
   - Connection: Fetch API calls from frontend to backend with CORS support.
   - Environment-based configuration for easy switching between local and production servers.

---

## 🚀 **Quick Start**

### **Option 1: Automatic (PowerShell on Windows)**
```bash
.\start.ps1
```
This opens two terminals: one for backend, one for frontend.

### **Option 2: Automatic (Bash on Mac/Linux)**
```bash
chmod +x start.sh
./start.sh
```

### **Option 3: Manual Setup**

**Terminal 1 - Start Backend:**
```bash
cd backend
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.app.api:app --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm install
# or: bun install
npm run dev
# or: bun run dev
```

**Visit in browser:**
- Frontend: http://localhost:8080
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📚 **Documentation**

- **SETUP.md** — Detailed setup guide with troubleshooting
- **IMPLEMENTATION_SUMMARY.md** — Technical details on backend-frontend connection
- **Backend API Docs** — Swagger UI at http://localhost:8000/docs (when backend running)

---

## Architecture Diagram

```
┌─────────────────────────────────┐
│   React Frontend                │
│   (http://localhost:8080)       │
│   - Assessment Form             │
│   - Results Display             │
└──────────────┬──────────────────┘
               │
        (Fetch API calls)
               │
               ▼
┌─────────────────────────────────┐
│   FastAPI Backend               │
│   (http://localhost:8000)       │
│   - /api/recommend              │
│   - /api/career-predict         │
│   - /api/universities           │
│   - /api/jobs                   │
│                                 │
│   ML Models:                    │
│   - RandomForest Classifier     │
│   - Sentence-BERT Embeddings    │
│   - Career/University/Job       │
│     Recommenders                │
└─────────────────────────────────┘
       ▲
       │
    (Predictions & Recommendations)
```

---

## 📁 **Project Structure**

```
proj-career-reco/
├── backend/                    # Python/ML backend
│   ├── src/
│   │   ├── app/
│   │   │   ├── main.py        # CLI version
│   │   │   └── api.py         # FastAPI server (NEW)
│   │   ├── models/            # ML models
│   │   ├── features/          # Feature engineering
│   │   ├── data/              # Data loading
│   │   └── db/                # Database connections
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment template
│   └── .gitignore
│
├── frontend/                   # React/TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── AssessmentForm.tsx
│   │   │   ├── CareerResults.tsx
│   │   │   └── HeroSection.tsx
│   │   ├── pages/
│   │   │   └── Index.tsx       # Main page (UPDATED)
│   │   ├── data/
│   │   │   └── careerData.ts   # API integration (NEW)
│   │   └── App.tsx
│   ├── package.json
│   ├── vite.config.ts
│   ├── .env.example            # Environment template
│   └── .gitignore
│
├── models/                     # Trained ML models & cache
├── SETUP.md                    # Detailed setup guide (NEW)
├── IMPLEMENTATION_SUMMARY.md   # Tech details (NEW)
├── start.ps1                   # Windows quick start (NEW)
├── start.sh                    # Mac/Linux quick start (NEW)
└── README.md                   # This file

```

---

## 🔌 **Backend-Frontend Connection**

The system is now fully integrated:

1. **Frontend submits assessment form** with user interests and skills
2. **Frontend calls backend API** (`POST /api/recommend`)
3. **Backend runs ML models**:
   - CareerPredictor → predicts career field
   - UniversityRecommender → finds matching universities
   - CareerRecommender → matches job roles
4. **Backend returns recommendations** as JSON
5. **Frontend displays results** beautifully

All communication is via REST API with full error handling and fallbacks.

---

## 🛠️ **Configuration**

### Backend Environment Variables (`.env`)
```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=career_recommender
API_HOST=0.0.0.0
API_PORT=8000
APP_ENV=development
```

### Frontend Environment Variables (`.env`)
```env
VITE_API_URL=http://localhost:8000
VITE_ENV=development
```

See `.env.example` files in both folders for template.

---

## 🧪 **Testing the Setup**

**Verify backend:**
```bash
curl http://localhost:8000/health
# Response: {"status": "healthy", "models_ready": true}
```

**Check API documentation:**
Visit http://localhost:8000/docs when backend is running (Swagger UI)

**Test frontend-backend connection:**
Open browser DevTools console and run:
```javascript
fetch('http://localhost:8000/health').then(r => r.json()).then(d => console.log(d))
```

---

## 📚 **Available Commands**

### Backend
```bash
# Development
python -m uvicorn src.app.api:app --reload

# Run CLI version (legacy)
python src/app/main.py

# Check health
curl http://localhost:8000/health
```

### Frontend
```bash
# Development
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint

# With bun
bun run dev
bun run build
```

---

## 🎯 **Next Steps**

- [ ] MongoDB local setup for user sessions
- [ ] Authentication & user accounts
- [ ] Save recommendation history
- [ ] Feedback mechanism for model improvement
- [ ] Deploy to Render.com or cloud provider
- [ ] Enhance assessment form with score input
- [ ] Add career comparison feature

---

## 🤝 **Contributing**

Contributions welcome! Please:
1. Create a feature branch
2. Make your changes
3. Test locally
4. Commit with clear messages
5. Push and create a Pull Request

---

## 📄 **License**

This project is open source. See LICENSE file for details.

---

## 👨‍💻 **Author**

Built with ❤️ as an AI-powered career guidance system.

For questions or issues, check SETUP.md or refer to IMPLEMENTATION_SUMMARY.md for technical details.
````