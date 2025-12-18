import pandas as pd 
import numpy as np
import joblib
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class CareerPredictor:
    def __init__(self, model_path="models/career_predictor.pkl"):
        self.model_path = model_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trianed = False

        self.feature_columns = [
            'Mathematics_Score', 
            'Science_Score', 
            'Language_Arts_Score', 
            'Social_Studies_Score', 
            'Logical_Reasoning', 
            'Creativity', 
            'Communication', 
            'Leadership', 
            'Social_Skills'
        ] 

    def _clean_label(self, label):
        """Cleans the target label by removing brackets, quotes, and extra spaces."""
        if isinstance(label, str):
            clean = re.sub(r"[\[\]\"']", "", label)
            return clean.strip()
        return label

    def train(self, df, verbose=True):
        if verbose:
            print("\n" + "="*40)
            print("   TRAINING CAREER PREDICTOR MODEL")
            print("="*40)
            print("Preparing training data...")

        X = df[self.feature_columns]
        y = df['Primary_Career_Recommendation'].apply(self._clean_label)
        
        if verbose:
            print(f"Unique classes found: {y.unique()}")

        y_encoded = self.label_encoder.fit_transform(y)
        # Data split (80% Training, 20% Testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        if verbose:
            print("Fitting Random Forest Classifier...")
        self.model.fit(X_train, y_train)
        self.is_trianed = True

        if verbose:
            print("\n--- Model Evaluation Results ---")
            accuracy = self.model.score(X_test, y_test)
            print(f"Accuracy: {accuracy:.2%}")

            y_pred = self.model.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        self._save_model()

    def predict(self, user_input):
        if not self.is_trianed:
            if os.path.exists(self.model_path):
                self._load_model()
            else: 
                raise Exception("Model is not trained yet. Please train the model before prediction.")
            
        input_df = pd.DataFrame([user_input]) 
        input_df = input_df[self.feature_columns]

        pred_idx = self.model.predict(input_df)[0]
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

        probs = self.model.predict_proba(input_df)[0]
        confidence = np.max(probs)

        return pred_label, confidence

    def predict_top_k(self, user_input, k=3):
        """Returns the top k predictions with their probabilities."""
        if not self.is_trianed:
            if os.path.exists(self.model_path):
                self._load_model()
            else: 
                raise Exception("Model is not trained yet.")
            
        input_df = pd.DataFrame([user_input]) 
        input_df = input_df[self.feature_columns]

        probs = self.model.predict_proba(input_df)[0]
        top_k_indices = np.argsort(probs)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            label = self.label_encoder.inverse_transform([idx])[0]
            prob = probs[idx]
            results.append((label, prob))
            
        return results
    
    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'encoder': self.label_encoder,
            'features': self.feature_columns
        }, self.model_path)
        print(f"Model saved successfully to {self.model_path}")

    def _load_model(self):
        print(f"Loading saved model from {self.model_path}...")
        data = joblib.load(self.model_path)
        self.model = data['model']
        self.label_encoder = data['encoder']
        self.feature_columns = data['features']
        self.is_trianed = True
        print("Model loaded successfully.")
