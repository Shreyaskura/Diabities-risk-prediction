"""
Diabetes Risk Prediction - Prediction & Inference Service
Course Outcome:
- CO6: Basic ML engineering practices including model packaging and a prediction-serving interface.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel, Field, field_validator
from preprocessing import (
    load_preprocessors,
    transform_features,
    FEATURE_COLUMNS,
    ZERO_AS_MISSING_COLUMNS
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = os.path.join(PARENT_DIR, 'backend', 'models')
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = os.path.join(os.getcwd(), 'backend', 'models')


# ==============================================================================
# CO6: Pydantic Clinical Input Schema Validation
# ==============================================================================

class PatientFeatures(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=25, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, le=500, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: float = Field(..., ge=0, le=250, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, le=120, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, le=1000, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, le=90, description="Body Mass Index (kg/m^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=3.5, description="Diabetes pedigree genetic score")
    Age: int = Field(..., ge=1, le=130, description="Age in years")

    @field_validator('Glucose', 'BloodPressure', 'BMI')
    @classmethod
    def check_non_negative(cls, v):
        if v < 0:
            raise ValueError("Clinical measurement cannot be negative")
        return v


class ModelInferenceService:
    """
    CO6: Prediction-Serving Interface.
    Loads packaged models and scalers into memory for low-latency inference.
    """
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.lr_model = None
        self.rf_model = None
        self.imputer = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized joblib artifacts."""
        lr_path = os.path.join(self.models_dir, 'logistic_regression.pkl')
        rf_path = os.path.join(self.models_dir, 'random_forest.pkl')
        
        if not os.path.exists(lr_path) or not os.path.exists(rf_path):
            raise FileNotFoundError(
                f"Model artifacts not found in {self.models_dir}. Please run train.py first."
            )
            
        self.lr_model = joblib.load(lr_path)
        self.rf_model = joblib.load(rf_path)
        self.imputer, self.scaler = load_preprocessors(self.models_dir)
        print("Inference service initialized: Logistic Regression, Random Forest, Scaler & Imputer loaded.")

    def predict(self, patient_data: dict) -> dict:
        """
        CO6: Preprocesses patient input and runs dual-model inference.
        Returns prediction class, risk probability, and qualitative risk category.
        """
        # Convert dictionary to DataFrame with standard feature columns
        df_input = pd.DataFrame([{col: patient_data[col] for col in FEATURE_COLUMNS}])
        
        # Replace 0 with NaN for features where 0 represents missing clinical measurement
        for col in ZERO_AS_MISSING_COLUMNS:
            df_input[col] = df_input[col].replace(0, np.nan)
            
        # CO2: Preprocess consistently with training pipeline
        X_imputed, X_scaled = transform_features(df_input, self.imputer, self.scaler)
        
        # CO2: Logistic Regression Inference (uses scaled features)
        lr_pred = int(self.lr_model.predict(X_scaled)[0])
        lr_prob = float(self.lr_model.predict_proba(X_scaled)[0][1])
        
        # CO3: Random Forest Inference (uses unscaled imputed features)
        rf_pred = int(self.rf_model.predict(X_imputed)[0])
        rf_prob = float(self.rf_model.predict_proba(X_imputed)[0][1])
        
        def categorize_risk(prob: float) -> str:
            if prob < 0.35:
                return "Low Risk"
            elif prob < 0.65:
                return "Moderate Risk"
            else:
                return "High Risk"

        return {
            "logistic_regression": {
                "prediction": lr_pred,
                "probability": round(lr_prob, 4),
                "risk_label": "Diabetic Risk" if lr_pred == 1 else "Non-Diabetic",
                "risk_category": categorize_risk(lr_prob)
            },
            "random_forest": {
                "prediction": rf_pred,
                "probability": round(rf_prob, 4),
                "risk_label": "Diabetic Risk" if rf_pred == 1 else "Non-Diabetic",
                "risk_category": categorize_risk(rf_prob)
            },
            "consensus": {
                "agreement": lr_pred == rf_pred,
                "average_probability": round((lr_prob + rf_prob) / 2.0, 4),
                "overall_status": "Diabetic Risk" if (lr_pred == 1 or rf_pred == 1) else "Low Risk"
            },
            "disclaimer": "This application is an educational machine-learning risk prediction system and is not a medical diagnostic tool."
        }
