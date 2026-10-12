"""
Diabetes Risk Prediction - FastAPI Serving Interface
Course Outcome:
- CO6: Basic ML engineering practices including model packaging and a prediction-serving interface.
"""

import os
import json
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from prediction import PatientFeatures, ModelInferenceService
from preprocessing import FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')

# Initialize FastAPI application
app = FastAPI(
    title="Diabetes Risk Prediction System",
    description="Academic Machine Learning API implementing Course Outcomes CO1-CO6 with Logistic Regression and Random Forest",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite default ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount results directory to serve generated charts (Confusion Matrix, ROC Curve, etc.)
if os.path.exists(RESULTS_DIR):
    app.mount("/static/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Initialize Inference Service
inference_service = None

def get_inference_service():
    global inference_service
    if inference_service is None:
        try:
            inference_service = ModelInferenceService()
        except Exception as e:
            print(f"Warning: Could not load models immediately: {e}")
            return None
    return inference_service


# ==============================================================================
# CO6: Prediction-Serving Interface Endpoints
# ==============================================================================

@app.get("/", tags=["General"])
def root():
    """
    CO6: Root endpoint describing the ML system, models, and Course Outcomes.
    """
    return {
        "project_title": "Diabetes Risk Prediction Using Logistic Regression and Random Forest with Cross-Validation",
        "academic_context": "College Machine Learning Project (Course Outcomes CO1 - CO6)",
        "status": "online",
        "models_available": [
            "Logistic Regression (L1/L2 Regularized)",
            "Random Forest Classifier (Ensemble Tree-Based)"
        ],
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "features": "/features",
            "predict": "/predict (POST)",
            "docs": "/docs"
        },
        "disclaimer": "This application is an educational machine-learning risk prediction system and is not a medical diagnostic tool."
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    CO6: Health-check endpoint verifying model loading and system operational status.
    """
    service = get_inference_service()
    models_ready = service is not None and service.lr_model is not None and service.rf_model is not None
    metrics_ready = os.path.exists(METRICS_PATH)
    
    return {
        "status": "healthy" if (models_ready and metrics_ready) else "degraded",
        "models_loaded": models_ready,
        "metrics_computed": metrics_ready,
        "results_directory_available": os.path.exists(RESULTS_DIR)
    }


@app.get("/metrics", tags=["Evaluation"])
def get_model_metrics():
    """
    CO5 & CO6: Returns authentic measured performance metrics from train/test evaluation and 5-fold cross-validation.
    """
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics file not found. Ensure train.py has completed execution."
        )
    with open(METRICS_PATH, 'r') as f:
        raw_text = f.read().replace('Infinity', '1.0').replace('-Infinity', '0.0').replace('NaN', 'null')
        metrics_data = json.loads(raw_text)
    return metrics_data


@app.get("/features", tags=["Metadata"])
def get_features_metadata():
    """
    Returns clinical feature metadata and physiological boundary descriptions.
    """
    metadata_path = os.path.join(BASE_DIR, 'dataset', 'dataset_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {"features": FEATURE_COLUMNS}


@app.post("/predict", tags=["Inference"])
def predict_diabetes_risk(patient: PatientFeatures):
    """
    CO6: Prediction-Serving Interface endpoint.
    Accepts validated patient features and returns predictions and calibrated probabilities
    from both Logistic Regression and Random Forest models.
    """
    service = get_inference_service()
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference service not initialized. Check if models are trained and packaged."
        )
    
    try:
        results = service.predict(patient.model_dump())
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.get("/results/chart/{chart_name}", tags=["Visualizations"])
def get_chart_image(chart_name: str):
    """
    Serves generated academic plots (confusion_matrix.png, roc_curve.png, model_comparison.png, etc.)
    """
    allowed_charts = [
        "confusion_matrix.png",
        "roc_curve.png",
        "model_comparison.png",
        "cv_scores.png",
        "feature_importance.png",
        "pca_variance.png"
    ]
    if chart_name not in allowed_charts:
        raise HTTPException(status_code=400, detail="Invalid chart name requested")
        
    chart_path = os.path.join(RESULTS_DIR, chart_name)
    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail="Chart not found on disk")
        
    return FileResponse(chart_path, media_type="image/png")
