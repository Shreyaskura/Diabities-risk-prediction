import React, { useState } from 'react';
import DisclaimerBanner from '../components/DisclaimerBanner';
import { 
  Database, 
  GitBranch, 
  Layers, 
  Cpu, 
  ChevronDown, 
  ChevronUp, 
  ExternalLink,
  ShieldCheck,
  FileCode
} from 'lucide-react';

export default function AboutPage() {
  const [methodologyOpen, setMethodologyOpen] = useState(false);

  return (
    <div className="about-page">
      {/* Compact Page Header */}
      <section className="compact-hero">
        <div className="compact-hero-content">
          <h1>DiaRisk ML System Architecture</h1>
          <p>
            An end-to-end clinical machine-learning pipeline evaluating supervised linear and ensemble models with leak-free cross-validation.
          </p>
        </div>
      </section>

      {/* 4 Compact Specification Cards */}
      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        {/* Dataset Specifications */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <Database size={16} color="#38bdf8" />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>NIDDK Clinical Cohort</h3>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
            768 patient records from the National Institute of Diabetes and Digestive and Kidney Diseases. 8 continuous and discrete physiological biomarkers.
          </p>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', display: 'flex', gap: '1rem' }}>
            <span>Split: 80% Train (614) / 20% Test (154)</span>
            <span>Class Balance: 65.1% / 34.9%</span>
          </div>
        </div>

        {/* Supervised Algorithms */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <GitBranch size={16} color="#818cf8" />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Supervised Models</h3>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
            L2-Regularized Logistic Regression baseline paired with a tuned Random Forest ensemble (50 depth-restricted bootstrap decision trees).
          </p>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', display: 'flex', gap: '1rem' }}>
            <span>LogReg: C=0.05, liblinear</span>
            <span>RF: max_depth=5, min_split=5</span>
          </div>
        </div>

        {/* Validation Methodology */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <ShieldCheck size={16} color="#10b981" />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Validation Methodology</h3>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
            Stratified 5-Fold Cross-Validation for subgroup stability alongside an isolated holdout test split. Median imputation computed strictly on training folds to prevent data leakage.
          </p>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', display: 'flex', gap: '1rem' }}>
            <span>5-Fold Stratification</span>
            <span>Zero-as-Missing Imputation</span>
          </div>
        </div>

        {/* Technology Stack */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <Cpu size={16} color="#f59e0b" />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Engineering Stack</h3>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
            FastAPI REST serving backend with Pydantic contract validation, React 18 client with Vite and Recharts, and Scikit-Learn serialized pipelines.
          </p>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', display: 'flex', gap: '1rem' }}>
            <span>FastAPI • Uvicorn</span>
            <span>React • Vite • Recharts</span>
          </div>
        </div>
      </div>

      {/* Collapsible Section: Technical Methodology */}
      <div className="accordion" style={{ marginBottom: '1.5rem' }}>
        <button
          type="button"
          className="accordion-trigger"
          onClick={() => setMethodologyOpen(!methodologyOpen)}
          aria-expanded={methodologyOpen}
          style={{ padding: '1rem 1.25rem' }}
        >
          <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
            Technical Methodology & Implementation Notes
          </span>
          {methodologyOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>

        {methodologyOpen && (
          <div className="accordion-content">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <h4 style={{ color: '#38bdf8', fontSize: '0.9rem', marginBottom: '0.35rem' }}>
                  1. Handling Physiological Missing Values & Leakage Prevention
                </h4>
                <p>
                  In the Pima Indians dataset, values of 0 for Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI represent unrecorded observations. Instead of dropping records or applying global imputation, our pipeline computes median statistics strictly within training splits and applies the fitted transformer to test data and API inference calls.
                </p>
              </div>

              <div>
                <h4 style={{ color: '#818cf8', fontSize: '0.9rem', marginBottom: '0.35rem' }}>
                  2. Dual Modeling Trade-Off: Interpretability vs. Discrimination
                </h4>
                <p>
                  Logistic Regression optimizes log-odds over standardized continuous features, offering direct clinical coefficients (Glucose: +0.885, BMI: +0.493). Random Forest relaxes linear boundaries, capturing high-order non-linear interactions across glucose and age to yield higher holdout precision (69.77% vs 59.18%) and ROC-AUC (82.28% vs 80.81%).
                </p>
              </div>

              <div>
                <h4 style={{ color: '#f59e0b', fontSize: '0.9rem', marginBottom: '0.35rem' }}>
                  3. Principal Component Analysis (PCA) Variance Trade-Off
                </h4>
                <p>
                  PCA decomposes the 8 correlated variables into orthogonal dimensions, where PC1 and PC2 explain 29.4% and 18.8% of variance respectively. While 6 components preserve &ge;85% variance, clinical deployment retains authentic biological dimensions so clinicians can interpret natural units (mg/dL, mm Hg, BMI).
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <DisclaimerBanner />
    </div>
  );
}
