"""
Diabetes Risk Prediction - Preprocessing & Pipeline Module
Course Outcomes Covered:
- CO1: End-to-end Machine Learning Lifecycle (Data Collection -> Cleaning -> Transformation -> Splits)
- CO2: Logistic Regression with Feature Engineering and Regularization (Scaling, Imputation, Leakage Prevention)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# CO1: Feature definitions and physiological validity configuration
FEATURE_COLUMNS = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

TARGET_COLUMN = 'Outcome'

# Columns where a value of 0 is physiologically impossible and indicates a missing value
ZERO_AS_MISSING_COLUMNS = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]


# ==============================================================================
# CO1: End-to-end Machine Learning Lifecycle - Data Loading & Inspection
# ==============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    CO1: Load raw dataset from CSV file.
    Validates existence and schema conformity.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at expected path: {filepath}")
    
    df = pd.read_csv(filepath)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    actual = set(df.columns)
    
    if not required.issubset(actual):
        missing = required - actual
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    CO1: Inspect dataset dimensions, statistics, missing values, and zero counts.
    Returns a structured dictionary of data health insights.
    """
    zero_counts = {}
    for col in ZERO_AS_MISSING_COLUMNS:
        count = int((df[col] == 0).sum())
        pct = float((count / len(df)) * 100)
        zero_counts[col] = {"count": count, "percentage": round(pct, 2)}

    class_dist = df[TARGET_COLUMN].value_counts().to_dict()
    
    summary = {
        "total_records": len(df),
        "total_features": len(FEATURE_COLUMNS),
        "class_distribution": {
            "negative_0": int(class_dist.get(0, 0)),
            "positive_1": int(class_dist.get(1, 0)),
            "positive_ratio": round(float(class_dist.get(1, 0) / len(df)), 4)
        },
        "zero_as_missing_counts": zero_counts,
        "is_null_counts": df.isnull().sum().to_dict()
    }
    return summary


# ==============================================================================
# CO1 & CO2: Data Cleaning, Missing Value Handling & Leakage Prevention
# ==============================================================================

def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    CO1 & CO2: Convert physiologically impossible zeros into NaN representations.
    Pregnancies = 0 is valid and therefore preserved.
    """
    df_clean = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        df_clean[col] = df_clean[col].replace(0, np.nan)
    return df_clean


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    CO1: Stratified train-test split to preserve outcome class distribution.
    Crucial for imbalanced medical data (~65% negative, ~35% positive).
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


# ==============================================================================
# CO2: Feature Engineering, Imputation & Standardization
# ==============================================================================

def fit_preprocessors(X_train: pd.DataFrame):
    """
    CO2: Fit SimpleImputer and StandardScaler strictly on X_train.
    Prevents data leakage: test/production statistics must never influence training.
    - Imputer: Median strategy is robust against skewed medical distributions (e.g. Insulin).
    - Scaler: StandardScaler transforms features to mean=0, std=1 (essential for Logistic Regression & PCA).
    """
    # 1. Imputer fitted solely on training partition
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    
    # Impute training values
    X_train_imputed = imputer.transform(X_train)
    
    # 2. Scaler fitted solely on imputed training partition
    scaler = StandardScaler()
    scaler.fit(X_train_imputed)
    
    return imputer, scaler


def transform_features(X, imputer: SimpleImputer, scaler: StandardScaler):
    """
    CO2: Transform arbitrary feature inputs using pre-fitted imputer and scaler.
    Maintains feature names to ensure compatibility with estimators.
    Returns:
    - X_imputed: Unscaled data as DataFrame (ideal for Random Forest tree splits)
    - X_scaled: Standardized data as DataFrame (required for Logistic Regression & PCA)
    """
    if isinstance(X, pd.DataFrame):
        X_df = X[FEATURE_COLUMNS]
    else:
        X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        
    X_imputed = imputer.transform(X_df)
    X_scaled = scaler.transform(X_imputed)
    
    df_imputed = pd.DataFrame(X_imputed, columns=FEATURE_COLUMNS)
    df_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    
    return df_imputed, df_scaled


def save_preprocessors(imputer: SimpleImputer, scaler: StandardScaler, output_dir: str):
    """
    CO6: Model Packaging - Persist fitted preprocessor artifacts via joblib.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(imputer, os.path.join(output_dir, 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"Preprocessors saved successfully in: {output_dir}")


def load_preprocessors(model_dir: str) -> tuple[SimpleImputer, StandardScaler]:
    """
    CO6: Model Packaging - Load preprocessors for production serving.
    """
    imputer_path = os.path.join(model_dir, 'imputer.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(imputer_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Preprocessor files missing in {model_dir}")
        
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    return imputer, scaler


if __name__ == '__main__':
    # Quick standalone test of preprocessing lifecycle
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'diabetes.csv')
    
    df = load_data(dataset_path)
    report = inspect_data(df)
    print("CO1 Inspection Summary:", report)
    
    df_clean = replace_invalid_zeros(df)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    imputer, scaler = fit_preprocessors(X_train)
    
    X_train_imp, X_train_scl = transform_features(X_train, imputer, scaler)
    X_test_imp, X_test_scl = transform_features(X_test, imputer, scaler)
    
    print(f"X_train scaled shape: {X_train_scl.shape}, X_test scaled shape: {X_test_scl.shape}")
    print("Preprocessing successfully verified.")
