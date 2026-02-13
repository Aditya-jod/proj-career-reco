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

5. **Frontend + API Integration (Roadmap)**
   - Backend: refactor CLI into FastAPI endpoints (`/auth`, `/recommend`, `/history`).
   - Frontend: React + Vite + Tailwind for interactive forms, dashboards, and result visualizations.
   - Deployment: both services hosted on Render with environment-based configuration and CORS-enabled communication.
---


## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Aditya-jod/proj-career-reco.git
    cd proj-career-reco
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\Activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLP models**:
    ```bash
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
    ```

5.  **Run the application**:
    ```bash
    python src/app/main.py
    ```