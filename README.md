# AI Career Path Recommender System

A hybrid AI recommendation engine that helps students discover their ideal career path, find relevant universities, and explore specific job roles based on their academic scores, soft skills, and personal interests.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/AI-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-green)

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
├── data/                   # Raw datasets (CSV files)
├── models/                 # Saved AI models (.pkl)
├── notebooks/              # Jupyter notebooks for data cleaning & exploration
├── src/
│   ├── app/
│   │   └── main.py         # Main application entry point
│   ├── data/
│   │   ├── augmentation.py # Data augmentation logic
│   │   └── loader.py       # Data loading utilities
│   ├── features/
│   │   └── build_features.py # TF-IDF vectorization logic
│   └── models/
│       ├── career_predictor.py      # Random Forest Logic
│       ├── recommender.py           # Job Search Logic (TF-IDF)
│       └── university_recommender.py # University Search Logic
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

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