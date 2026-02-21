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
RF_N_ESTIMATORS = 300
RF_RANDOM_STATE = 42
RF_MAX_DEPTH = None          # let trees grow fully
RF_MIN_SAMPLES_SPLIT = 5
RF_CLASS_WEIGHT = "balanced" # compensate for any class imbalance

# Train-test split ratio
TEST_SIZE = 0.2
TRAIN_RANDOM_STATE = 42
