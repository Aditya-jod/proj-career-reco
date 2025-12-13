# Career Path Recommendation System

## Goal
Build an intelligent system that recommends suitable career paths to students based on their academic performance, interests, hobbies, and skills.

## Scope
*   **Focus**: Career guidance starting from Class 12 to professional careers.
*   **Recommendations**: Career domains (e.g., Data Science, Mechanical Engineering, UI/UX, Finance, Healthcare).
*   **Suggestions**: Suitable job roles and required skills.
*   **Alignment**: Recommendations aligned with current job market trends.

## Data Used
1.  **Student/Career Path Dataset**:
    *   Academic performance (GPA)
    *   Skills (coding, communication, problem-solving, teamwork, etc.)
    *   Interests, extracurricular activities, projects, internships
2.  **Career Path Dataset**:
    *   Career domains and required skills
3.  **Job Description Dataset**:
    *   Used to understand market-demanded skills and trends.

## Methods
*   **Content-based recommendation**: Matching student profiles to careers.
*   **Skill and interest matching**: Core logic for relevance.
*   **Text Understanding**: TF-IDF or embeddings.
*   **Matching**: Cosine similarity.
*   **Clustering (Optional)**: K-Means for grouping similar careers.

## Output
*   Top recommended career paths
*   Relevant job roles
*   Skill gap analysis
*   Learning roadmap for the chosen career

## Datasets

The system utilizes three key datasets:

1.  **Student Dataset**: Contains skills, interests, career goals, academic scores, and personality traits.
2.  **Career Path Dataset**: Contains career categories, required skills, typical responsibilities, growth potential, salary range, and industry type.
3.  **Job Description Dataset**: A large dataset (approx. 1.6M rows) containing job titles, roles, descriptions, skills, experience, location, and salary ranges. This helps map careers to current market demands.

## Technologies & Techniques

### NLP (Natural Language Processing)
*   **Text Cleaning**: Lowercasing, removing special characters and stop words.
*   **Lemmatization**: Reducing words to their base form.
*   **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency).

### Machine Learning
*   **Content-Based Filtering**: Matching student profiles with similar job descriptions using Cosine Similarity.
*   **Clustering**: Grouping similar careers using K-Means.
*   **Hybrid Model**: Combining skill similarity, interest matching, and cluster similarity.

## Model Output

The system provides:
*   Top 5 Career Recommendations
*   Match Score
*   Missing Skills Analysis
*   Suggested Learning Roadmap
*   Recommended Courses
*   Real Job Titles & Salary Insights

## Project Structure

```
proj-career-reco/
├── config/             # Configuration files
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for EDA and prototyping
├── src/                # Source code
│   ├── app/            # Main application logic
│   ├── data/           # Data loading and preprocessing
│   ├── features/       # Feature engineering (NLP, Embeddings)
│   ├── models/         # ML models (Recommender, Clustering)
│   └── utils/          # Helper functions
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
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