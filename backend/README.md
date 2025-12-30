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
├── config/
│   └── config.yaml               # Configuration file
├── data/
│   ├── raw/                      # Raw datasets (CSV files)
│   └── processed/                # Processed datasets
├── models/
│   ├── career_predictor.pkl      # Saved Random Forest model
│   └── saved_job_recommender/    # Saved job recommender models
├── notebooks/
│   ├── 01_data_cleaning.ipynb    # Data cleaning notebook
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   └── 02_data_exploration.ipynb # Data exploration notebook
├── src/
│   ├── app/
│   │   └── main.py               # FastAPI app entry point
│   ├── data/
│   │   ├── augmentation.py       # Data augmentation logic
│   │   ├── loader.py             # Data loading utilities
│   │   ├── preprocessing.py      # Data preprocessing
│   │   └── __init__.py
│   ├── features/
│   │   ├── build_features.py     # TF-IDF vectorization logic
│   │   ├── nlp_utils.py          # NLP helper functions
│   │   └── __init__.py
│   ├── models/
│   │   ├── career_predictor.py   # Random Forest logic
│   │   ├── recommender.py        # Job search logic (TF-IDF)
│   │   ├── university_recommender.py # University search logic
│   │   ├── clustering.py         # Clustering logic
│   │   └── __init__.py
│   ├── scripts/                  # Utility scripts (empty)
│   ├── utils/
│   │   └── __init__.py           # Utility functions
│   └── __init__.py
├── tests/
│   ├── test_full_pipeline.py     # End-to-end pipeline tests
│   └── test_predictor.py         # Career predictor tests
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # Backend documentation
```

## Machine Learning Pipeline Workflow

The backend ML workflow consists of several key stages, each designed to ensure robust, interpretable, and accurate recommendations:

### 1. Data Collection & Exploration
- Gathered 5,000+ student profiles with academic scores, soft skills, and interests.
- Performed EDA (Exploratory Data Analysis) to understand distributions, spot outliers, and identify important features.

### 2. Data Preprocessing & Feature Engineering
- Cleaned data (handled missing values, removed duplicates, standardized formats).
- Engineered new features (e.g., composite skill scores, encoded categorical variables).
- Normalized/standardized numerical features for model compatibility.
- For job role recommendations, applied NLP preprocessing (tokenization, stopword removal, lemmatization).

### 3. Model Selection: Why Random Forest?
- Random Forest Classifier chosen for career field prediction because:
	- Handles both numerical and categorical data well.
	- Robust to overfitting (ensemble averaging).
	- Provides feature importance for interpretability.
	- Captures non-linear relationships and complex interactions.
	- Outperformed other models in initial experiments.

### 4. Model Training & Evaluation
- Split data into training and validation sets.
- Trained Random Forest on engineered features.
- Tuned hyperparameters (number of trees, max depth, etc.).
- Evaluated using accuracy, precision, recall, and F1-score.
- If model confidence < 50%, system returns top 3 career fields for user choice.

### 5. Content-Based Filtering for Job Roles
- Used TF-IDF vectorization to convert job descriptions and user interests into vectors.
- Applied cosine similarity to match user input to relevant job titles.
- Leveraged NLTK and SpaCy for advanced text processing.

### 6. Rule-Based Mapping for University Recommendations
- Mapped predicted career fields to university courses using dictionaries and keyword matching.
- Ensured recommendations are diverse and relevant.

### 7. Model Serialization & Deployment
- Saved trained models using Joblib for fast loading in production.
- Exposed prediction endpoints via FastAPI for frontend integration.

---

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

## API Endpoints (Examples)

- `POST /predict-career` – Predicts broad career field from user data
- `POST /recommend-universities` – Returns university list for a given career field
- `POST /recommend-jobs` – Suggests job roles based on user interests/skills

## Notes

- All ML models and data files should be placed in the respective `models/` and `data/` folders.
- For development and testing, use the provided Jupyter notebooks in `notebooks/`.

---

For full project details, see the root `README.md`.
