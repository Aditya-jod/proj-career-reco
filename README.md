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

### 1. Intelligent Career Prediction (Machine Learning)
*   **Algorithm:** Random Forest Classifier.
*   **Input:** Academic scores (Math, Science, etc.) and Soft Skills (Leadership, Creativity, etc.).
*   **Logic:** Analyzes patterns from 5,000+ student profiles to predict the most suitable **Broad Career Field** (e.g., STEM, Healthcare, Business).
*   **Smart Feature:** Includes an **"Alternative Options"** system. If the AI is less than 50% confident, it presents the top 3 choices and lets the user decide.

### 2. University Recommendation (Rule-Based Filtering)
*   **Logic:** Maps the predicted career field to specific university courses.
*   **Smart Mapping:** Handles specific job titles (e.g., "Field Engineer" -> "STEM") to ensure relevant results.
*   **Database:** Searches through a database of Indian and International universities.
*   **Diversity:** Uses randomized sampling to show a diverse mix of colleges across different states/regions, avoiding bias towards a single location.

### 3. Job Role Recommendation (NLP & Content-Based Filtering)
*   **Algorithm:** TF-IDF Vectorization + Cosine Similarity.
*   **Dataset:** Trained on a sample of **30,000 real-world job descriptions**.
*   **Logic:** Takes user-defined interests (e.g., "Python, Drawing") and finds the most mathematically similar job titles from the database.
*   **Feature:** Supports specific skill queries (e.g., searching "Lawyer" will override a "STEM" prediction to show legal jobs).

---

## Technical Architecture

The system uses a **Hybrid Recommendation Approach**:

1.  **Phase 1 (Classification):**
    *   *Input:* Numerical Scores.
    *   *Model:* Random Forest.
    *   *Output:* Broad Category (e.g., "Healthcare").

2.  **Phase 2 (Rule-Based Mapping):**
    *   *Input:* Broad Category + Location.
    *   *Logic:* Keyword Matching & Dictionary Mapping.
    *   *Output:* List of Universities.

3.  **Phase 3 (Content-Based Filtering):**
    *   *Input:* User Keywords (Skills/Interests).
    *   *Model:* TF-IDF (Term Frequency-Inverse Document Frequency).
    *   *Output:* Specific Job Titles (e.g., "Neurosurgeon", "Python Developer").

---

## Project Structure

```text
proj-career-reco/
├── backend/
│   ├── config/
│   │   └── config.yaml                   # Configuration file
│   ├── data/
│   │   ├── raw/                          # Raw datasets (CSV files)
│   │   └── processed/                    # Processed datasets
│   ├── models/
│   │   ├── career_predictor.pkl          # Saved Random Forest model
│   │   └── saved_job_recommender/        # Saved job recommender models
│   ├── notebooks/
│   │   ├── 01_data_cleaning.ipynb        # Data cleaning notebook
│   │   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   │   └── 02_data_exploration.ipynb     # Data exploration notebook
│   ├── src/
│   │   ├── app/
│   │   │   └── main.py                   # FastAPI app entry point
│   │   ├── data/
│   │   │   ├── augmentation.py           # Data augmentation logic
│   │   │   ├── loader.py                 # Data loading utilities
│   │   │   ├── preprocessing.py          # Data preprocessing
│   │   │   └── __init__.py
│   │   ├── features/
│   │   │   ├── build_features.py         # TF-IDF vectorization logic
│   │   │   ├── nlp_utils.py              # NLP helper functions
│   │   │   └── __init__.py
│   │   ├── models/
│   │   │   ├── career_predictor.py       # Random Forest logic
│   │   │   ├── recommender.py            # Job search logic (TF-IDF)
│   │   │   ├── university_recommender.py # University search logic
│   │   │   ├── clustering.py             # Clustering logic
│   │   │   └── __init__.py
│   │   ├── scripts/                      # Utility scripts (empty)
│   │   ├── utils/
│   │   │   └── __init__.py               # Utility functions
│   │   └── __init__.py
│   ├── tests/
│   │   ├── test_full_pipeline.py         # End-to-end pipeline tests
│   │   └── test_predictor.py             # Career predictor tests
│   ├── requirements.txt                  # Python dependencies
│   ├── .gitignore                        # Git ignore file
│   └── README.md                         # Backend documentation
├── frontend/
│   └── .gitkeep                          # Placeholder for frontend code (React, Vue, etc.)
└── README.md                             # Project documentation (root)
```


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