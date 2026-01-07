"""Configuration constants for Career Predictor model."""

# Model storage path
MODEL_PATH = "models/career_predictor.pkl"

# Feature columns required for prediction
FEATURE_COLUMNS = [
    "Mathematics_Score",
    "Science_Score",
    "Language_Arts_Score",
    "Social_Studies_Score",
    "Logical_Reasoning",
    "Creativity",
    "Communication",
    "Leadership",
    "Social_Skills",
]

# Target column in training data
TARGET_COLUMN = "Primary_Career_Recommendation"

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42

# Train-test split ratio
TEST_SIZE = 0.2
TRAIN_RANDOM_STATE = 42
