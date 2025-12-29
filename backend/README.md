# Career Path Recommender – Backend

This backend powers the AI-driven recommendation engine for the Career Path Recommender System. It provides RESTful APIs for career prediction, university recommendations, and job role suggestions based on user input.

## Features

- **Career Prediction:** Uses a Random Forest Classifier to predict broad career fields from academic scores and soft skills.
- **University Recommendation:** Maps predicted career fields to relevant university courses using rule-based logic and a curated database.
- **Job Role Recommendation:** Employs TF-IDF vectorization and NLP to match user interests/skills to real-world job titles.
- **Alternative Options:** If prediction confidence is low, returns top 3 career fields for user selection.

## Tech Stack

- Python 3.8+
- FastAPI (API framework)
- Scikit-Learn (ML)
- Pandas, NumPy (Data processing)
- NLTK, SpaCy (NLP)
- Joblib (Model serialization)

## Project Structure

```text
backend/
├── src/
│   ├── app/
│   │   └── main.py               # FastAPI app entry point
│   ├── data/
│   │   ├── augmentation.py       # Data augmentation logic
│   │   └── loader.py             # Data loading utilities
│   ├── features/
│   │   └── build_features.py     # TF-IDF vectorization logic
│   └── models/
│       ├── career_predictor.py   # Random Forest logic
│       ├── recommender.py        # Job search logic (TF-IDF)
│       └── university_recommender.py # University search logic
├── data/                         # Raw datasets (CSV files)
├── models/                       # Saved AI models (.pkl)
├── notebooks/                    # Jupyter notebooks for data cleaning & exploration
├── requirements.txt              # Python dependencies
└── README.md                     # Backend documentation
```

## Setup & Usage

1. **Create and activate a virtual environment:**
	```bash
	python -m venv venv
	# Windows
	.\venv\Scripts\activate
	# Mac/Linux
	source venv/bin/activate
	```

2. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

3. **Download NLP models:**
	```bash
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
	```

4. **Run the API server:**
	```bash
	uvicorn src.app.main:app --reload
	```

5. **API Documentation:**
	- Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger UI.

## API Endpoints (Examples)

- `POST /predict-career` – Predicts broad career field from user data
- `POST /recommend-universities` – Returns university list for a given career field
- `POST /recommend-jobs` – Suggests job roles based on user interests/skills

## Notes

- All ML models and data files should be placed in the respective `models/` and `data/` folders.
- For development and testing, use the provided Jupyter notebooks in `notebooks/`.

---

For full project details, see the root `README.md`.
