import React, { useState } from 'react';
import { predictRisk } from '../services/api';
import DisclaimerBanner from '../components/DisclaimerBanner';
import { 
  Stethoscope, 
  CheckCircle2, 
  AlertOctagon, 
  ChevronDown, 
  ChevronUp, 
  Loader2, 
  Sparkles,
  Info,
  TrendingUp
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  LabelList
} from 'recharts';

const PRESETS = {
  exampleA: {
    name: 'Example A (Normal Range)',
    values: {
      Pregnancies: 1,
      Glucose: 85,
      BloodPressure: 66,
      SkinThickness: 29,
      Insulin: 90,
      BMI: 23.5,
      DiabetesPedigreeFunction: 0.35,
      Age: 25,
    },
  },
  exampleB: {
    name: 'Example B (Moderate Range)',
    values: {
      Pregnancies: 3,
      Glucose: 128,
      BloodPressure: 78,
      SkinThickness: 32,
      Insulin: 140,
      BMI: 29.8,
      DiabetesPedigreeFunction: 0.52,
      Age: 42,
    },
  },
  exampleC: {
    name: 'Example C (Elevated Range)',
    values: {
      Pregnancies: 6,
      Glucose: 175,
      BloodPressure: 88,
      SkinThickness: 38,
      Insulin: 220,
      BMI: 36.4,
      DiabetesPedigreeFunction: 0.85,
      Age: 54,
    },
  },
};

const FIELD_CONFIGS = [
  { key: 'Pregnancies', label: 'Pregnancies', unit: 'Count', min: 0, max: 20, step: 1 },
  { key: 'Glucose', label: 'Plasma Glucose', unit: 'mg/dL (2h OGTT)', min: 0, max: 400, step: 1 },
  { key: 'BloodPressure', label: 'Diastolic BP', unit: 'mm Hg', min: 0, max: 200, step: 1 },
  { key: 'SkinThickness', label: 'Skin Fold Thickness', unit: 'mm (Triceps)', min: 0, max: 99, step: 1 },
  { key: 'Insulin', label: '2-Hour Serum Insulin', unit: 'mu U/ml', min: 0, max: 900, step: 1 },
  { key: 'BMI', label: 'Body Mass Index', unit: 'kg/m²', min: 10, max: 70, step: 0.1 },
  { key: 'DiabetesPedigreeFunction', label: 'Pedigree Genetic Score', unit: 'Index (0-3)', min: 0.01, max: 3.0, step: 0.001 },
  { key: 'Age', label: 'Patient Age', unit: 'Years', min: 18, max: 120, step: 1 },
];

export default function PredictionPage() {
  const [formData, setFormData] = useState(PRESETS.exampleA.values);
  const [activePreset, setActivePreset] = useState('exampleA');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showExplanation, setShowExplanation] = useState(false);

  const handleInputChange = (field, value) => {
    setActivePreset(null);
    setFormData((prev) => ({
      ...prev,
      [field]: value === '' ? '' : Number(value),
    }));
  };

  const loadPreset = (presetKey) => {
    setActivePreset(presetKey);
    setFormData(PRESETS[presetKey].values);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;

    setLoading(true);
    setError(null);

    try {
      const res = await predictRisk(formData);
      setPrediction(res);
    } catch (err) {
      setError(err.message || 'Prediction request failed. Check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="prediction-page">
      {/* Compact Page Header */}
      <section className="compact-hero">
        <div className="compact-hero-content">
          <h1>Clinical Risk Prediction</h1>
          <p>
            Evaluate diagnostic biomarkers through dual trained models to estimate statistical diabetes risk probability.
          </p>
        </div>
      </section>

      {/* Preset Illustrative Selector */}
      <div className="preset-strip" role="region" aria-label="Sample input presets">
        <span className="preset-label">Sample Presets:</span>
        <button
          type="button"
          className={`preset-chip ${activePreset === 'exampleA' ? 'active' : ''}`}
          onClick={() => loadPreset('exampleA')}
        >
          Example A (Normal)
        </button>
        <button
          type="button"
          className={`preset-chip ${activePreset === 'exampleB' ? 'active' : ''}`}
          onClick={() => loadPreset('exampleB')}
        >
          Example B (Moderate)
        </button>
        <button
          type="button"
          className={`preset-chip ${activePreset === 'exampleC' ? 'active' : ''}`}
          onClick={() => loadPreset('exampleC')}
        >
          Example C (Elevated)
        </button>
        <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', marginLeft: '0.25rem' }}>
          *Illustrative sample profiles for demonstration; not personal clinical guarantees.
        </span>
      </div>

      {/* 2-Column Responsive Desktop Grid */}
      <div className="grid-2" style={{ alignItems: 'start' }}>
        {/* LEFT COLUMN: Input Form */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div className="section-header" style={{ marginBottom: '1.25rem' }}>
            <div>
              <h2>
                <Stethoscope size={18} color="#38bdf8" />
                Patient Measurements
              </h2>
              <p>Standardized diagnostic parameters from Pima Indian study</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} noValidate={false}>
            <div className="form-grid">
              {FIELD_CONFIGS.map((field) => (
                <div className="form-group" key={field.key}>
                  <label className="form-label" htmlFor={`input-${field.key}`}>
                    <span>{field.label}</span>
                    <span className="form-label-unit">{field.unit}</span>
                  </label>
                  <input
                    id={`input-${field.key}`}
                    type="number"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    required
                    className="form-input"
                    value={formData[field.key] ?? ''}
                    onChange={(e) => handleInputChange(field.key, e.target.value)}
                  />
                </div>
              ))}
            </div>

            <button
              type="submit"
              className="btn-primary"
              disabled={loading}
              style={{ width: '100%', padding: '0.75rem', fontSize: '0.92rem' }}
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" size={16} />
                  <span>Computing Dual-Model Inference...</span>
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  <span>Calculate Prediction</span>
                </>
              )}
            </button>
          </form>

          {error && (
            <div
              style={{
                marginTop: '1rem',
                padding: '0.75rem 1rem',
                background: 'rgba(244, 63, 94, 0.1)',
                border: '1px solid rgba(244, 63, 94, 0.3)',
                borderRadius: '8px',
                color: '#fb7185',
                fontSize: '0.82rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
              }}
            >
              <AlertOctagon size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: Prediction Results */}
        <div className="prediction-results-col">
          {prediction ? (
            <div className="results-wrapper">
              {/* Logistic Regression Result Card */}
              <div
                className={`result-model-card ${
                  prediction.logistic_regression.prediction === 1 ? 'high-risk' : 'low-risk'
                }`}
              >
                <div className="result-model-header">
                  <div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Logistic Regression</h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      L2 Regularized Linear Classifier (Scaled Space)
                    </span>
                  </div>
                  <span
                    className={`result-badge ${
                      prediction.logistic_regression.prediction === 1 ? 'high' : 'low'
                    }`}
                  >
                    {prediction.logistic_regression.prediction === 1 ? (
                      <AlertOctagon size={13} />
                    ) : (
                      <CheckCircle2 size={13} />
                    )}
                    {prediction.logistic_regression.risk_label}
                  </span>
                </div>

                <div className="result-prob-row">
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                    Calculated Probability
                  </span>
                  <span className="result-prob-val">
                    {(prediction.logistic_regression.probability * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="result-meter-track" aria-hidden="true">
                  <div
                    className={`result-meter-bar ${
                      prediction.logistic_regression.prediction === 1 ? 'high' : 'low'
                    }`}
                    style={{
                      width: `${Math.max(prediction.logistic_regression.probability * 100, 4)}%`,
                    }}
                  />
                </div>

                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.76rem',
                    color: 'var(--text-muted)',
                  }}
                >
                  <span>
                    Stratification Tier:{' '}
                    <strong style={{ color: 'var(--text-primary)' }}>
                      {prediction.logistic_regression.risk_category}
                    </strong>
                  </span>
                  <span>Decision Threshold: 0.50</span>
                </div>
              </div>

              {/* Random Forest Result Card */}
              <div
                className={`result-model-card ${
                  prediction.random_forest.prediction === 1 ? 'high-risk' : 'low-risk'
                }`}
              >
                <div className="result-model-header">
                  <div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Random Forest Classifier</h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Non-Linear Tree Ensemble (50 Depth-Tuned Trees)
                    </span>
                  </div>
                  <span
                    className={`result-badge ${
                      prediction.random_forest.prediction === 1 ? 'high' : 'low'
                    }`}
                  >
                    {prediction.random_forest.prediction === 1 ? (
                      <AlertOctagon size={13} />
                    ) : (
                      <CheckCircle2 size={13} />
                    )}
                    {prediction.random_forest.risk_label}
                  </span>
                </div>

                <div className="result-prob-row">
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                    Calculated Probability
                  </span>
                  <span className="result-prob-val">
                    {(prediction.random_forest.probability * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="result-meter-track" aria-hidden="true">
                  <div
                    className={`result-meter-bar ${
                      prediction.random_forest.prediction === 1 ? 'high' : 'low'
                    }`}
                    style={{
                      width: `${Math.max(prediction.random_forest.probability * 100, 4)}%`,
                    }}
                  />
                </div>

                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.76rem',
                    color: 'var(--text-muted)',
                  }}
                >
                  <span>
                    Stratification Tier:{' '}
                    <strong style={{ color: 'var(--text-primary)' }}>
                      {prediction.random_forest.risk_category}
                    </strong>
                  </span>
                  <span>Decision Threshold: 0.50</span>
                </div>
              </div>

              {/* Consensus Agreement Card */}
              <div className="consensus-card">
                <div
                  className={`consensus-icon ${
                    prediction.consensus.agreement
                      ? 'consensus-concordant'
                      : 'consensus-discordant'
                  }`}
                >
                  {prediction.consensus.agreement ? (
                    <CheckCircle2 size={18} />
                  ) : (
                    <Info size={18} />
                  )}
                </div>
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      fontSize: '0.88rem',
                      fontWeight: 600,
                      color: prediction.consensus.agreement ? '#34d399' : '#f59e0b',
                      marginBottom: '0.2rem',
                    }}
                  >
                    {prediction.consensus.agreement
                      ? 'Concordant Prediction (Both Models Agree)'
                      : 'Discordant Prediction (Models Differ on Threshold)'}
                  </div>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    Combined average risk probability:{' '}
                    <strong style={{ color: 'var(--text-primary)' }}>
                      {(prediction.consensus.average_probability * 100).toFixed(1)}%
                    </strong>
                    . Output is a statistical likelihood estimate, not a definitive diagnosis.
                  </p>
                </div>
              </div>

              {/* Live Inference Probability Chart as per Given Patient Data */}
              <div className="glass-panel" style={{ padding: '1.15rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <TrendingUp size={15} color="#38bdf8" />
                    Comparative Model Probabilities
                  </h4>
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                    Decision Cutoff: 50%
                  </span>
                </div>
                <div style={{ width: '100%', height: 130 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={[
                        {
                          model: 'LogReg',
                          probability: Number((prediction.logistic_regression.probability * 100).toFixed(1)),
                          fill: prediction.logistic_regression.prediction === 1 ? '#fb7185' : '#38bdf8',
                        },
                        {
                          model: 'RandForest',
                          probability: Number((prediction.random_forest.probability * 100).toFixed(1)),
                          fill: prediction.random_forest.prediction === 1 ? '#fb7185' : '#818cf8',
                        },
                      ]}
                      margin={{ top: 5, right: 40, left: 10, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" horizontal={false} />
                      <XAxis type="number" domain={[0, 100]} unit="%" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 10 }} />
                      <YAxis dataKey="model" type="category" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <ReferenceLine x={50} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: '50% Threshold', fill: '#f59e0b', fontSize: 9, position: 'insideTopRight' }} />
                      <Bar dataKey="probability" radius={[0, 4, 4, 0]} maxBarSize={20}>
                        <LabelList dataKey="probability" position="right" fill="#e2e8f0" fontSize={11} formatter={(v) => `${v}%`} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Optional Collapsible Explanation */}
              <div className="accordion">
                <button
                  type="button"
                  className="accordion-trigger"
                  onClick={() => setShowExplanation(!showExplanation)}
                  aria-expanded={showExplanation}
                >
                  <span>How were these predictions generated?</span>
                  {showExplanation ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
                {showExplanation && (
                  <div className="accordion-content">
                    <p style={{ marginBottom: '0.5rem' }}>
                      <strong>1. Logistic Regression:</strong> Features are standardized ($Z$-scores) using training mean and variance. Predictions reflect log-odds weights where Glucose (+0.885) and BMI (+0.493) exert strongest positive pressure.
                    </p>
                    <p style={{ marginBottom: '0.5rem' }}>
                      <strong>2. Random Forest:</strong> 50 depth-constrained bootstrap decision trees evaluate unscaled features and compute the proportion of positive tree votes.
                    </p>
                    <p>
                      <strong>3. Leak-Free Preprocessing:</strong> Biologically impossible zero measurements in glucose, blood pressure, skin, insulin, and BMI are imputed with training-set median values.
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="empty-state">
              <div className="empty-state-icon">
                <Stethoscope size={26} />
              </div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 600, marginBottom: '0.35rem', color: 'var(--text-primary)' }}>
                Awaiting Patient Parameters
              </h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', maxWidth: '300px', lineHeight: 1.5 }}>
                Fill out the diagnostic measurements or select a sample preset above, then click <strong>Calculate Prediction</strong>.
              </p>
            </div>
          )}
        </div>
      </div>

      <DisclaimerBanner />
    </div>
  );
}
