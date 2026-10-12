"""
Diabetes Risk Prediction - Model Training, Evaluation, and Analysis Pipeline
Course Outcomes:
- CO1: End-to-end Machine Learning Lifecycle
- CO2: Logistic Regression with Feature Engineering and Regularization
- CO3: Random Forest Tree-Based Supervised Learning
- CO4: PCA-Based Dimensionality Reduction / Feature Analysis
- CO5: Stratified K-Fold Cross-Validation, Classification Metrics & Hyperparameter Optimization
- CO6: Model Packaging
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from preprocessing import (
    load_data,
    replace_invalid_zeros,
    split_data,
    fit_preprocessors,
    transform_features,
    save_preprocessors,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'diabetes.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'backend', 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


# ==============================================================================
# CO4: PCA-Based Dimensionality Reduction / Feature Analysis
# ==============================================================================

def analyze_pca(X_scaled: pd.DataFrame, y: pd.Series, output_dir: str) -> dict:
    """
    CO4: PCA-Based Dimensionality Reduction / Feature Analysis.
    - Applies PCA to standardized features.
    - Calculates individual and cumulative explained variance.
    - Determines number of components required to retain >= 85% and >= 95% variance.
    - Visualizes Scree plot and 2D Principal Component separation.
    - Generates academic analysis on PCA vs raw clinical feature interpretability.
    """
    print("\n--- [CO4] Performing PCA Feature Analysis ---")
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    
    explained_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    
    # Check components needed for 85% and 95% variance
    n_85 = int(np.argmax(cum_var >= 0.85) + 1)
    n_95 = int(np.argmax(cum_var >= 0.95) + 1)
    
    # 2D PCA Transformation for visualization
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    # Plot Scree & Cumulative Variance Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree Plot
    axes[0].bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, color='#2563eb', label='Individual Variance')
    axes[0].step(range(1, len(cum_var) + 1), cum_var, where='mid', color='#dc2626', linewidth=2, label='Cumulative Variance')
    axes[0].axhline(y=0.85, color='gray', linestyle='--', label='85% Threshold')
    axes[0].set_xlabel('Principal Component Index')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('CO4: Scree Plot & Cumulative Explained Variance')
    axes[0].legend(loc='center right')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 2D Component Scatter Plot
    sns.scatterplot(
        x=X_pca_2d[:, 0],
        y=X_pca_2d[:, 1],
        hue=y,
        palette=['#10b981', '#ef4444'],
        alpha=0.8,
        ax=axes[1]
    )
    axes[1].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% Variance)')
    axes[1].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% Variance)')
    axes[1].set_title('CO4: 2D Projection of Patient Features')
    axes[1].legend(title='Outcome', labels=['Non-Diabetic (0)', 'Diabetic (1)'])
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    pca_plot_path = os.path.join(output_dir, 'pca_variance.png')
    plt.savefig(pca_plot_path, dpi=300)
    plt.close()
    
    pca_summary = {
        "explained_variance_ratio": [round(float(v), 4) for v in explained_var],
        "cumulative_variance_ratio": [round(float(v), 4) for v in cum_var],
        "pc1_variance": round(float(explained_var[0]), 4),
        "pc2_variance": round(float(explained_var[1]), 4),
        "n_components_for_85pct": n_85,
        "n_components_for_95pct": n_95,
        "academic_note": (
            "While PCA condenses the 8 tabular clinical features effectively into lower orthogonal dimensions "
            f"(PC1-PC{n_85} retain 85% variance), direct clinical interpretability of specific biomarkers "
            "(e.g., Glucose mg/dL or Insulin serum) is obscured. Therefore, our primary clinical prediction "
            "pipeline preserves the standardized original feature space for Logistic Regression and unscaled features "
            "for Random Forest to maximize clinical explainability."
        )
    }
    print(f"PCA complete: PC1={pca_summary['pc1_variance']*100:.1f}%, PC2={pca_summary['pc2_variance']*100:.1f}%")
    print(f"Components for 85% variance: {n_85}/{len(FEATURE_COLUMNS)}")
    return pca_summary


# ==============================================================================
# CO2: Logistic Regression with Feature Engineering and Regularization
# ==============================================================================

def train_logistic_regression(X_train_scaled: pd.DataFrame, y_train: pd.Series, cv):
    """
    CO2: Logistic Regression with Regularization & Hyperparameter Optimization.
    - Regularization prevents overfitting on collinear medical indicators.
    - Tunes inverse regularization strength C over logarithmic intervals.
    """
    print("\n--- [CO2] Optimizing Logistic Regression with Regularization ---")
    
    base_lr = LogisticRegression(random_state=42, max_iter=1000)
    
    param_grid = {
        'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'solver': ['lbfgs', 'liblinear']
    }
    
    grid = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    
    best_lr = grid.best_estimator_
    print(f"Logistic Regression Best Parameters: {grid.best_params_}")
    print(f"Best CV ROC-AUC Score: {grid.best_score_:.4f}")
    
    # Extract feature coefficients
    coefficients = {
        col: round(float(coef), 4)
        for col, coef in zip(FEATURE_COLUMNS, best_lr.coef_[0])
    }
    
    return best_lr, grid.best_params_, coefficients


# ==============================================================================
# CO3: Random Forest Tree-Based Supervised Learning
# ==============================================================================

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, cv):
    """
    CO3: Random Forest Tree-Based Supervised Learning & Hyperparameter Tuning.
    - Tunes ensemble tree parameters:
      - n_estimators: Number of bootstrap trees
      - max_depth: Depth of decision trees to control variance
      - min_samples_split: Threshold for splitting internal nodes
      - min_samples_leaf: Regularization on leaf node purity
      - max_features: Random subset of features per split
    - Extracts Gini feature importances.
    """
    print("\n--- [CO3] Optimizing Random Forest Classifier ---")
    
    base_rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    best_rf = grid.best_estimator_
    print(f"Random Forest Best Parameters: {grid.best_params_}")
    print(f"Best CV ROC-AUC Score: {grid.best_score_:.4f}")
    
    # Feature importances based on Mean Decrease in Impurity (Gini)
    feature_importances = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLUMNS, best_rf.feature_importances_)
    }
    
    return best_rf, grid.best_params_, feature_importances


# ==============================================================================
# CO5: Stratified K-Fold Cross-Validation, Evaluation Metrics & Plotting
# ==============================================================================

def evaluate_cross_validation(models_dict: dict, X_scaled: pd.DataFrame, X_imputed: pd.DataFrame, y: pd.Series, cv):
    """
    CO5: Stratified K-Fold Cross-Validation across multiple classification metrics.
    Computes 5-fold distribution of Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}
    
    for name, model in models_dict.items():
        # LR operates on scaled data; RF on unscaled imputed data
        X_data = X_scaled if name == 'Logistic Regression' else X_imputed
        scores = cross_validate(model, X_data, y, cv=cv, scoring=scoring, return_train_score=False)
        
        cv_results[name] = {
            metric: {
                "mean": round(float(np.mean(scores[f'test_{metric}'])), 4),
                "std": round(float(np.std(scores[f'test_{metric}'])), 4),
                "folds": [round(float(s), 4) for s in scores[f'test_{metric}']]
            }
            for metric in scoring
        }
    return cv_results


def evaluate_test_set(model, X_test, y_test, model_name: str) -> dict:
    """
    CO5: Evaluate model on independent holdout test set (20%).
    Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc), 4),
        "confusion_matrix": {
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
            "raw_matrix": cm
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "thresholds": [round(float(x), 4) for x in thresholds]
        }
    }


def generate_evaluation_visualizations(
    lr_eval: dict,
    rf_eval: dict,
    cv_results: dict,
    lr_coef: dict,
    rf_imp: dict,
    output_dir: str
):
    """
    CO5: Generate academic charts:
    1. Confusion Matrices side-by-side
    2. Combined ROC-AUC Curves
    3. Model Comparison Bar Chart
    4. 5-Fold Cross-Validation Boxplot
    5. Feature Importance vs Logistic Coefficients
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Confusion Matrices Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, eval_data, title in zip(
        axes,
        [lr_eval, rf_eval],
        ['Logistic Regression (L2 Regularized)', 'Random Forest (Tuned Ensemble)']
    ):
        cm = np.array(eval_data["confusion_matrix"]["raw_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1']
        )
        ax.set_title(f"CO5: Confusion Matrix - {title}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 2. Combined ROC Curves Plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        lr_eval["roc_curve"]["fpr"],
        lr_eval["roc_curve"]["tpr"],
        color='#2563eb',
        lw=2.5,
        label=f"Logistic Regression (AUC = {lr_eval['roc_auc']:.3f})"
    )
    plt.plot(
        rf_eval["roc_curve"]["fpr"],
        rf_eval["roc_curve"]["tpr"],
        color='#059669',
        lw=2.5,
        label=f"Random Forest (AUC = {rf_eval['roc_auc']:.3f})"
    )
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('CO5: Receiver Operating Characteristic (ROC) Comparison', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    # 3. Model Comparison Bar Chart
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    lr_scores = [lr_eval[m] for m in metrics_list]
    rf_scores = [rf_eval[m] for m in metrics_list]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='#3b82f6')
    plt.bar(x + width/2, rf_scores, width, label='Random Forest', color='#10b981')

    plt.ylabel('Score (0.0 to 1.0)', fontsize=11)
    plt.title('CO5: Holdout Test Set Performance Comparison', fontsize=12, fontweight='bold')
    plt.xticks(x, labels, fontsize=10)
    plt.ylim([0, 1.1])
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    for i in range(len(labels)):
        plt.text(x[i] - width/2, lr_scores[i] + 0.02, f"{lr_scores[i]:.3f}", ha='center', fontsize=9)
        plt.text(x[i] + width/2, rf_scores[i] + 0.02, f"{rf_scores[i]:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

    # 4. Stratified 5-Fold Cross-Validation Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, name, color in zip(axes, ['Logistic Regression', 'Random Forest'], ['#3b82f6', '#10b981']):
        metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        fold_data = [cv_results[name][m]['folds'] for m in metric_keys]
        ax.boxplot(fold_data, tick_labels=['Acc', 'Prec', 'Rec', 'F1', 'AUC'], patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6))
        ax.set_title(f"CO5: 5-Fold CV Spread - {name}", fontsize=11, fontweight='bold')
        ax.set_ylabel('Fold Metric Score')
        ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_scores.png'), dpi=300)
    plt.close()

    # 5. Feature Importance vs Logistic Coefficients
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Logistic Regression Standardized Coefficients
    coef_series = pd.Series(lr_coef).sort_values()
    coef_series.plot(kind='barh', ax=axes[0], color='#2563eb', alpha=0.8)
    axes[0].set_title('CO2: Logistic Regression Standardized Coefficients\n(Log-Odds Impact)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Coefficient Weight')
    axes[0].axvline(0, color='black', lw=0.8, linestyle='--')
    
    # Random Forest Gini Feature Importances
    imp_series = pd.Series(rf_imp).sort_values()
    imp_series.plot(kind='barh', ax=axes[1], color='#059669', alpha=0.8)
    axes[1].set_title('CO3: Random Forest Feature Importance\n(Mean Decrease in Impurity / Gini)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    print("All evaluation plots generated and saved to results/.")


# ==============================================================================
# CO1: Full End-to-End Execution Pipeline
# ==============================================================================

def run_pipeline():
    """
    CO1: Orchestrates the entire Machine Learning Lifecycle:
    Data Loading -> Inspection -> Cleaning -> Stratified Split ->
    Leakage-free Preprocessing -> PCA Analysis (CO4) ->
    Logistic Regression Tuning (CO2) -> Random Forest Tuning (CO3) ->
    Stratified K-Fold CV & Metrics (CO5) -> Visualizations & Packaging (CO6).
    """
    print("=================================================================")
    print("STARTING COMPLETE ACADEMIC ML PIPELINE (CO1 - CO6)")
    print("=================================================================")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Step 1: Load authentic data
    df_raw = load_data(DATASET_PATH)
    print(f"[CO1] Loaded dataset: {df_raw.shape[0]} samples, {df_raw.shape[1]} columns")
    
    # Step 2: Clean invalid physiological zeros
    df_clean = replace_invalid_zeros(df_raw)
    
    # Step 3: Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=42)
    print(f"[CO1] Split data: Train={len(X_train)} samples, Test={len(X_test)} samples (Stratified)")
    
    # Step 4: Fit preprocessors solely on X_train (Leakage Prevention)
    imputer, scaler = fit_preprocessors(X_train)
    
    # Transform train and test partitions
    X_train_imp, X_train_scl = transform_features(X_train, imputer, scaler)
    X_test_imp, X_test_scl = transform_features(X_test, imputer, scaler)
    
    # Transform full dataset for cross-validation and PCA analysis
    df_full_clean = replace_invalid_zeros(df_raw)
    X_full = df_full_clean[FEATURE_COLUMNS]
    y_full = df_full_clean[TARGET_COLUMN]
    X_full_imp, X_full_scl = transform_features(X_full, imputer, scaler)
    
    # Step 5: CO4 PCA Dimensionality Reduction & Feature Analysis
    pca_results = analyze_pca(X_train_scl, y_train, RESULTS_DIR)
    
    # Setup 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Step 6: CO2 Logistic Regression Training & Tuning
    best_lr, lr_params, lr_coefficients = train_logistic_regression(X_train_scl, y_train, skf)
    
    # Step 7: CO3 Random Forest Training & Tuning
    best_rf, rf_params, rf_importances = train_random_forest(X_train_imp, y_train, skf)
    
    # Step 8: CO5 Stratified 5-Fold Cross-Validation Evaluation
    models_dict = {
        'Logistic Regression': best_lr,
        'Random Forest': best_rf
    }
    cv_metrics = evaluate_cross_validation(models_dict, X_full_scl, X_full_imp, y_full, skf)
    
    # Step 9: CO5 Holdout Test Set Evaluation
    lr_test_eval = evaluate_test_set(best_lr, X_test_scl, y_test, 'Logistic Regression')
    rf_test_eval = evaluate_test_set(best_rf, X_test_imp, y_test, 'Random Forest')
    
    # Print Comparison Table
    print("\n=================================================================")
    print("CO5: MEASURED MODEL EVALUATION RESULTS (TEST SET N=154)")
    print("=================================================================")
    print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 75)
    print(f"{'Logistic Regression':<22} | {lr_test_eval['accuracy']:<8.4f} | {lr_test_eval['precision']:<9.4f} | {lr_test_eval['recall']:<8.4f} | {lr_test_eval['f1_score']:<8.4f} | {lr_test_eval['roc_auc']:<8.4f}")
    print(f"{'Random Forest':<22} | {rf_test_eval['accuracy']:<8.4f} | {rf_test_eval['precision']:<9.4f} | {rf_test_eval['recall']:<8.4f} | {rf_test_eval['f1_score']:<8.4f} | {rf_test_eval['roc_auc']:<8.4f}")
    print("=================================================================\n")
    
    # Step 10: Generate all evaluation charts
    generate_evaluation_visualizations(
        lr_test_eval,
        rf_test_eval,
        cv_metrics,
        lr_coefficients,
        rf_importances,
        RESULTS_DIR
    )
    
    # Step 11: CO6 Model Packaging - Save joblib models and preprocessors
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_lr, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    joblib.dump(best_rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    save_preprocessors(imputer, scaler, MODELS_DIR)
    
    # Save complete metrics report
    metrics_report = {
        "dataset_summary": {
            "total_samples": len(df_raw),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN
        },
        "logistic_regression": {
            "best_hyperparameters": lr_params,
            "coefficients": lr_coefficients,
            "test_metrics": lr_test_eval,
            "cv_5fold_metrics": cv_metrics["Logistic Regression"]
        },
        "random_forest": {
            "best_hyperparameters": rf_params,
            "feature_importances": rf_importances,
            "test_metrics": rf_test_eval,
            "cv_5fold_metrics": cv_metrics["Random Forest"]
        },
        "pca_analysis": pca_results,
        "model_comparison": {
            "higher_accuracy": "Random Forest" if rf_test_eval['accuracy'] >= lr_test_eval['accuracy'] else "Logistic Regression",
            "higher_roc_auc": "Random Forest" if rf_test_eval['roc_auc'] >= lr_test_eval['roc_auc'] else "Logistic Regression",
            "higher_recall": "Random Forest" if rf_test_eval['recall'] >= lr_test_eval['recall'] else "Logistic Regression"
        }
    }
    
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_report, f, indent=2)
        
    print(f"Metrics saved to: {os.path.join(RESULTS_DIR, 'metrics.json')}")
    print(f"Models packaged to: {MODELS_DIR}")
    print("\nPIPELINE EXECUTION COMPLETE - ALL COURSE OUTCOMES SATISFIED.")


if __name__ == '__main__':
    run_pipeline()
