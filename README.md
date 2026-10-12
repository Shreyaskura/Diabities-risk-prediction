# Diabetes Risk Prediction Using Logistic Regression and Random Forest with Cross-Validation

## 1. Project Title
**Diabetes Risk Prediction Using Logistic Regression and Random Forest with Cross-Validation**  
*An Academic Machine Learning Project aligned with Course Outcomes CO1 through CO6.*

---

## 2. Abstract
Diabetes mellitus is a chronic metabolic disorder characterized by elevated levels of blood glucose, leading over time to serious damage to the heart, blood vessels, eyes, kidneys, and nerves. Early detection significantly mitigates clinical complications. This project develops an end-to-end, reproducible Machine Learning pipeline utilizing the Pima Indians Diabetes Database to predict diabetes onset risk. We evaluate and compare two supervised learning algorithms: an L2-regularized **Logistic Regression** baseline and a non-linear **Random Forest Classifier**. Rigorous evaluation is conducted via **5-Fold Stratified Cross-Validation** alongside an isolated 80/20 holdout test split. The trained models are packaged and deployed through a high-performance **FastAPI** REST backend and paired with an interactive **React (Vite)** clinical decision-support interface featuring live dual-model probability comparison, feature importance insights, and academic diagnostic visualizations.

---

## 3. Introduction
Machine Learning in healthcare presents unique requirements: statistical rigor, data integrity, interpretability, and robust validation against overfitting. This academic project demonstrates the complete engineering and analytical lifecycle of a clinical decision-support model. Rather than treating machine learning models as black boxes, this project details:
- Clinical data preprocessing and missing value imputation (handling physiologically impossible zero readings).
- Linear modeling with feature standardization and regularized log-odds estimation.
- Ensemble decision trees capable of capturing non-linear interactions.
- Principal Component Analysis (PCA) for orthogonal feature variance decomposition.
- Stratified cross-validation for unbiased performance evaluation.
- Production-grade microservice architecture with API contracts and responsive client visualization.

---

## 4. Problem Statement
Diagnosing diabetes risk at an early, asymptomatic stage is challenging in routine clinical practice due to multi-factorial risk vectors (glucose tolerance, insulin resistance, adiposity index, genetic predisposition, and age). The objective is to design, train, evaluate, and deploy an academic-grade machine learning system that accurately stratifies patients into risk tiers, provides consensus probabilities from complementary algorithms, and maintains clinical interpretability without synthetic or hardcoded metrics.

---

## 5. Objectives
1. **CO1 - Full ML Lifecycle:** Implement an end-to-end pipeline covering data ingestion, physiological cleaning, feature scaling, model training, holdout evaluation, and REST deployment.
2. **CO2 - Logistic Regression:** Train an interpretable linear classifier with L2 regularization and feature standardization, analyzing odds ratios and feature weights.
3. **CO3 - Random Forest:** Train an ensemble of bagged decision trees to model complex non-linear feature interactions and extract Gini impurity feature importances.
4. **CO4 - Dimensionality Analysis:** Perform Principal Component Analysis (PCA) to assess variance distribution, cumulative scree behavior, and evaluate dimensionality reduction trade-offs against clinical interpretability.
5. **CO5 - Validation & Optimization:** Execute 5-Fold Stratified Cross-Validation and Grid Search hyperparameter optimization, computing authentic Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
6. **CO6 - Serving & Packaging:** Package serialized artifacts (`joblib`) and serve predictions through a documented FastAPI backend and an interactive React dashboard.

---

## 6. Technologies Used

### Machine Learning & Data Science
- **Language:** Python 3.14
- **Data Manipulation:** `pandas` (v2.2+), `NumPy` (v2.0+)
- **Machine Learning:** `scikit-learn` (v1.5+)
- **Visualization:** `matplotlib` (v3.9+), `seaborn` (v0.13+)
- **Model Packaging:** `joblib` (v1.4+)

### Backend API
- **Framework:** `FastAPI` (v0.115+)
- **ASGI Server:** `Uvicorn` (v0.34+)
- **Data Validation:** `Pydantic` (v2.10+)

### Frontend Interface
- **Framework:** React 18 with Vite
- **Styling:** Custom Modern CSS (CSS Grid, Flexbox, Glassmorphism, Theme Variables)
- **Icons:** `lucide-react`
- **Charts:** Static academic image galleries + dynamic visual metric gauges

---

## 7. Dataset

- **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) - Pima Indians Diabetes Database.
- **Instance Count:** 768 patient records (614 training instances, 154 test instances; 80/20 stratified split).
- **Target Variable:** `Outcome` (Binary: `0` = Non-Diabetic, `1` = Diabetic). Class balance: ~65.1% negative (500), ~34.9% positive (268).
- **Feature Set:**
  1. `Pregnancies`: Number of times pregnant
  2. `Glucose`: Plasma glucose concentration at 2 hours in an oral glucose tolerance test (mg/dL)
  3. `BloodPressure`: Diastolic blood pressure (mm Hg)
  4. `SkinThickness`: Triceps skin fold thickness (mm)
  5. `Insulin`: 2-Hour serum insulin (mu U/ml)
  6. `BMI`: Body mass index (weight in kg / (height in m)²)
  7. `DiabetesPedigreeFunction`: Diabetes pedigree genetic score
  8. `Age`: Age in years

### Physiological Data Cleaning
In biological measurements, values of `0` for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are physiologically incompatible with life and represent missing observations. Our pipeline replaces these zeros with `NaN` and imputes them using **median imputation** fit strictly on training splits to prevent data leakage.

---

## 8. System Architecture

```
+-----------------------------------------------------------------------+
|                           User / Clinician                            |
+-----------------------------------------------------------------------+
                                   |
                                   v  HTTP / JSON
+-----------------------------------------------------------------------+
|                    React + Vite Frontend (Port 5173)                   |
|  - Dashboard (Metrics Overview, CO Mapping, System Health)            |
|  - Risk Prediction (Dual Model Inputs, Validation, Consensus)         |
|  - Model Performance (Holdout Metrics, CV Folds, 6 Diagnostic Plots)  |
|  - About Project (CO1-CO6 Curriculum Details, Architecture)           |
+-----------------------------------------------------------------------+
                                   |
                                   v  REST API (CORS Enabled)
+-----------------------------------------------------------------------+
|                     FastAPI Backend (Port 8000)                       |
|  - /health          : System health & artifact verification           |
|  - /predict         : Live dual-model prediction & consensus logic    |
|  - /metrics         : JSON store of verified model evaluation         |
|  - /plots/{filename}: Dynamic serving of matplotlib academic figures   |
+-----------------------------------------------------------------------+
         |                                           |
         v Loads serialized pipelines                v Reads cached stats
+-----------------------------------+       +---------------------------+
|          backend/models/          |       |         results/          |
|  - imputer.pkl                    |       |  - metrics.json           |
|  - scaler.pkl                     |       |  - confusion_matrix.png   |
|  - logistic_regression.pkl        |       |  - roc_curve.png          |
|  - random_forest.pkl              |       |  - model_comparison.png   |
+-----------------------------------+       |  - cv_scores.png          |
                                            |  - feature_importance.png |
                                            |  - pca_variance.png       |
                                            +---------------------------+
```

---

## 9. ML Workflow

```
[Raw CSV Dataset]
        │
        ▼
[Physiological Zero -> NaN Conversion] (Glucose, BP, Skin, Insulin, BMI)
        │
        ▼
[Stratified Train-Test Split] (80% Train: 614, 20% Test: 154, Stratify=Outcome)
        │
        ├─── Training Split ──────────────────────────────┐
        │                                                 ▼
        ▼                                      [SimpleImputer(median)]
[Median Calculation] (Fit on Train)                       │
        │                                                 ▼
        ├─── Scaled Branch (CO2 Logistic Regression) ── [StandardScaler]
        │                                                 │
        └─── Unscaled Branch (CO3 Random Forest) ─────────┘
        │
        ▼
[5-Fold Stratified K-Fold Cross-Validation + GridSearchCV Tuning]
        │
        ├──► Logistic Regression (C=0.05, solver='liblinear', penalty='l2')
        └──► Random Forest (n_estimators=50, max_depth=5, min_samples_split=5)
        │
        ▼
[Holdout Test Set Evaluation] (Authentic metrics: Acc, Prec, Rec, F1, ROC-AUC)
        │
        ▼
[Model Packaging & Serialization via Joblib]
        │
        ▼
[REST Serving via FastAPI & Visualization in React]
```

---

## 10. CO1 Implementation: End-to-End ML Lifecycle
- **Implementation File:** [`backend/preprocessing.py`](file:///c:/Users/K.Shreyas/OneDrive/Desktop/diabetes-risk-prediction/backend/preprocessing.py) and [`backend/train.py`](file:///c:/Users/K.Shreyas/OneDrive/Desktop/diabetes-risk-prediction/backend/train.py)
- **Key Methods:** `load_data()`, `handle_missing_values()`, `prepare_data_splits()`, `create_preprocessors()`
- Demonstrates data ingestion, exploratory statistics, missing data imputation, normalization, training, testing, artifact saving, and deployment serving.

---

## 11. CO2 Implementation: Logistic Regression
- **Implementation File:** [`backend/train.py`](file:///c:/Users/K.Shreyas/OneDrive/Desktop/diabetes-risk-prediction/backend/train.py) (`train_logistic_regression()`)
- **Regularization:** L2 Ridge penalty with inverse regularization strength parameter $C = 0.05$.
- **Feature Scaling:** Mandatory `StandardScaler` ($\mu = 0, \sigma = 1$) applied to prevent feature dominance.
- **Top Feature Coefficients:**
  - Glucose: **+0.8854** (strongest predictor of diabetes risk)
  - BMI: **+0.4927**
  - Pregnancies: **+0.2783**
  - Diabetes Pedigree Function: **+0.1926**
  - Age: **+0.1513**

---

## 12. CO3 Implementation: Random Forest
- **Implementation File:** [`backend/train.py`](file:///c:/Users/K.Shreyas/OneDrive/Desktop/diabetes-risk-prediction/backend/train.py) (`train_random_forest()`)
- **Model Parameters:** `n_estimators=50`, `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`.
- **Ensemble Characteristics:** Aggregates de-correlated decision trees trained on bootstrap samples with random feature sub-selection, reducing variance.
- **Gini Feature Importances:**
  - Glucose: **35.31%**
  - BMI: **18.29%**
  - Age: **12.77%**
  - Insulin: **10.00%**
  - Diabetes Pedigree Function: **7.94%**

---

## 13. CO4 Implementation: PCA Dimensionality Reduction
- **Implementation File:** [`backend/train.py`](file:///c:/Users/K.Shreyas/OneDrive/Desktop/diabetes-risk-prediction/backend/train.py) (`analyze_pca()`)
- **Analysis:** PCA computed across 8 standardized clinical variables.
- **Variance Decomposition:**
  - PC1: **29.43%** explained variance
  - PC2: **18.79%** explained variance
  - PC1 + PC2 Cumulative: **48.22%**
  - 6 Principal Components required to explain $\ge 85\%$ variance (82.64% at 5, 89.35% at 6).
- **Academic Finding:** While PCA projects the feature space orthogonally, clinical practitioners require direct biomarker interpretability (e.g., fasting glucose thresholds). Consequently, models retain original clinical features for transparent diagnostic deployment.

---

## 14. CO5 Implementation: Stratified K-Fold CV & Metrics
- **Implementation File:** [`backend/train.py`](file:///c:/Users/K.Shreyas/OneDrive/Desktop/diabetes-risk-prediction/backend/train.py) (`run_stratified_cv()`)
- **Strategy:** 5-Fold Stratified K-Fold preserving the ~35% positive class ratio across all folds.
- **Hyperparameter Tuning:** `GridSearchCV` exploring regularization parameters $C \in [0.001, 10.0]$ and tree configurations (`n_estimators`, `max_depth`, `min_samples_split`).
- **Comprehensive Evaluation:** Holdout Confusion Matrix, Precision-Recall trade-offs, and ROC Curves computed from actual scikit-learn runs without synthetic data.

---

## 15. CO6 Implementation: Model Packaging & API Serving
- **Packaging:** Serialized via `joblib` into `backend/models/`:
  - `imputer.pkl`
  - `scaler.pkl`
  - `logistic_regression.pkl`
  - `random_forest.pkl`
- **Inference Pipeline:** [`backend/prediction.py`](file:///c:/Users/K.Shreyas/OneDrive/Desktop/diabetes-risk-prediction/backend/prediction.py) loads models once on startup, validates patient features, performs imputation and scaling dynamically, outputs probabilities, assigns risk tiers (`Low`, `Moderate`, `High`), and determines consensus agreement.

---

## 16. How to Install

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Node.js 18+ and npm

### 1. Clone or Open the Repository
```bash
cd c:/Users/K.Shreyas/OneDrive/Desktop/diabetes-risk-prediction
```

### 2. Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 3. Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 17. How to Train
To re-train the models on the dataset, optimize hyperparameters, evaluate metrics, and regenerate all academic plots:
```bash
cd backend
python train.py
```
Outputs generated:
- Serialized models: `backend/models/*.pkl`
- Metrics report: `results/metrics.json`
- Academic visualizations: `results/*.png`

---

## 18. How to Run Backend
Start the FastAPI server via Uvicorn:
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

---

## 19. How to Run Frontend
In a separate terminal, launch the Vite development server:
```bash
cd frontend
npm run dev
```
Access the application at: `http://localhost:5173`

---

## 20. API Endpoints

| HTTP Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status, title, version, and documentation links |
| `GET` | `/health` | Verifies models, pipeline components, and results readiness |
| `GET` | `/metrics` | Returns full holdout and 5-fold CV metrics JSON |
| `GET` | `/plots/{plot_name}` | Serves diagnostic charts (`confusion_matrix`, `roc_curve`, etc.) |
| `POST` | `/predict` | Computes dual-model predictions and consensus risk for a patient |

### Sample `/predict` Request Body
```json
{
  "Pregnancies": 2,
  "Glucose": 145.0,
  "BloodPressure": 80.0,
  "SkinThickness": 32.0,
  "Insulin": 130.0,
  "BMI": 33.5,
  "DiabetesPedigreeFunction": 0.65,
  "Age": 45
}
```

### Sample `/predict` Response
```json
{
  "logistic_regression": {
    "prediction": 1,
    "probability": 0.6842,
    "risk_label": "Diabetic",
    "risk_category": "High Risk"
  },
  "random_forest": {
    "prediction": 1,
    "probability": 0.712,
    "risk_label": "Diabetic",
    "risk_category": "High Risk"
  },
  "consensus": {
    "agreement": true,
    "average_probability": 0.6981,
    "overall_status": "High Risk"
  },
  "disclaimer": "This application is an educational machine-learning risk prediction system and is not a medical diagnostic tool."
}
```

---

## 21. Results

### Authentic Holdout Evaluation (Test Set $N=154$)
| Metric | Logistic Regression | Random Forest | Better Performing Model |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 70.78% | **75.97%** | **Random Forest (+5.19%)** |
| **Precision** | 59.18% | **69.77%** | **Random Forest (+10.59%)** |
| **Recall** | 53.70% | **55.56%** | **Random Forest (+1.86%)** |
| **F1-Score** | 56.31% | **61.86%** | **Random Forest (+5.55%)** |
| **ROC-AUC** | 80.81% | **82.28%** | **Random Forest (+1.47%)** |

### 5-Fold Stratified Cross-Validation
| Metric | Logistic Regression (Mean $\pm$ Std) | Random Forest (Mean $\pm$ Std) |
| :--- | :--- | :--- |
| **CV Accuracy** | **77.34%** $\pm$ 2.15% | 76.82% $\pm$ 2.37% |
| **CV Precision** | **72.58%** $\pm$ 7.08% | 72.00% $\pm$ 5.68% |
| **CV Recall** | **58.22%** $\pm$ 4.89% | 55.60% $\pm$ 5.86% |
| **CV F1-Score** | **64.18%** $\pm$ 2.39% | 62.50% $\pm$ 4.35% |
| **CV ROC-AUC** | 83.56% $\pm$ 2.12% | **83.72%** $\pm$ 2.13% |

*Key Takeaway:* Random Forest achieved superior holdout accuracy (75.97%) and test precision (69.77%), significantly reducing false positives. Logistic Regression demonstrated strong stability across cross-validation folds. Both models reliably identified `Glucose` and `BMI` as the dominant risk factors.

---

## 22. Limitations
1. **Demographic Specificity:** The Pima Indians Diabetes Database is drawn exclusively from female patients of Pima Indian heritage aged $\ge 21$. Generalizability to diverse global populations and male cohorts is not guaranteed.
2. **Missing Feature Density:** Variables such as `Insulin` and `SkinThickness` exhibited significant zero values (~48.7% and ~29.6% respectively), requiring median imputation which, while robust, compresses natural biological variance.
3. **Dataset Volume:** With 768 total records, sample size limits the exploration of deep architectures without high variance.

---

## 23. Future Scope
1. **Expanded Multi-Ethnic Datasets:** Integrate larger federated clinical datasets (e.g., NHANES or UK Biobank) to improve external validity.
2. **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) and LIME to generate localized, patient-specific waterfall explanation plots for individual clinical consultations.
3. **Threshold Calibration:** Allow clinicians to dynamically adjust decision thresholds to optimize sensitivity (recall) in screening environments where missing a true positive carries acute clinical penalties.
4. **Automated Retraining Pipeline:** Implement MLOps pipelines (e.g., MLflow, GitHub Actions) for automated model drift detection and scheduled re-training.

---

## 24. Medical Disclaimer
> **IMPORTANT NOTICE:**  
> This software is an educational Machine Learning project developed strictly for academic, research, and technical demonstration purposes. It is **not** a certified medical diagnostic device and is **not** intended to replace clinical judgment, diagnosis, treatment, or advice from a certified healthcare professional. Real-world medical decisions should always be made in consultation with qualified medical practitioners.
