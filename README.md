# Career Path Recommendation System

## Overview

This project aims to solve a major real-world problem: millions of students have no clarity about what career aligns with their interests, skills, strengths, and long-term goals. Students often choose careers based on random advice, outdated information, or by copying someone else — which leads to dissatisfaction, frustration, and unemployment.

The **Career Path Recommendation System** is built to guide students toward the right profession using data-driven insights, machine learning, and NLP analysis of real job market data. It provides personalized career recommendations, required skills, learning roadmap, job trends, and matching job titles.

## Core Objectives

1.  **Analyze Student Profile**:
    *   Interests & Skills
    *   Academic background
    *   Personality traits
    *   Goals
2.  **Analyze Job Descriptions**:
    *   Extract required skills, experience level, and responsibilities.
    *   Identify role categories.
3.  **Intelligent Matching**:
    *   Match students with the most suitable career paths using ML.
4.  **Guidance**:
    *   Provide a step-by-step learning roadmap.
    *   Suggest courses, skills to learn, and future opportunities.

## Datasets

The system utilizes three key datasets:

1.  **Student Dataset**: Contains skills, interests, career goals, academic scores, and personality traits.
2.  **Career Path Dataset**: Contains career categories, required skills, typical responsibilities, growth potential, salary range, and industry type.
3.  **Job Description Dataset**: A large dataset (approx. 1.6M rows) containing job titles, roles, descriptions, skills, experience, location, and salary ranges. This helps map careers to current market demands.

## Technologies & Techniques

### NLP (Natural Language Processing)
*   **Text Cleaning**: Lowercasing, removing special characters and stop words.
*   **Lemmatization**: Reducing words to their base form.
*   **Skill Extraction**: Identifying key skills from text.
*   **Vectorization**: TF-IDF and Embeddings (Sentence Transformers/Word2Vec/SpaCy).

### Machine Learning
*   **Content-Based Filtering**: Matching student profiles with similar job descriptions using Cosine Similarity.
*   **Clustering**: Grouping similar careers using K-Means to recommend based on clusters.
*   **Hybrid Model**: Combining skill similarity, interest matching, personality fit, and JD similarity for a robust recommendation score.

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