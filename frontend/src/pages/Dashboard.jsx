import React, { useState, useMemo } from 'react';
import MetricCard from '../components/MetricCard';
import DisclaimerBanner from '../components/DisclaimerBanner';
import { 
  Database, 
  Layers, 
  Activity, 
  Award, 
  ArrowRight, 
  TrendingUp, 
  Sparkles,
  BarChart2,
  Stethoscope
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  LabelList
} from 'recharts';

export default function Dashboard({ setActiveTab, metricsData }) {
  const [evalMode, setEvalMode] = useState('holdout'); // 'holdout' | 'cv'

  const dataset = metricsData?.dataset_summary;
  const lr = metricsData?.logistic_regression?.test_metrics || {
    accuracy: 0.7078,
    precision: 0.5918,
    recall: 0.5370,
    f1_score: 0.5631,
    roc_auc: 0.8081,
  };
  const rf = metricsData?.random_forest?.test_metrics || {
    accuracy: 0.7597,
    precision: 0.6977,
    recall: 0.5556,
    f1_score: 0.6186,
    roc_auc: 0.8228,
  };
  const lrCV = metricsData?.logistic_regression?.cv_5fold_metrics || {
    accuracy: { mean: 0.7734, std: 0.0215 },
    precision: { mean: 0.7258, std: 0.0708 },
    recall: { mean: 0.5822, std: 0.0489 },
    f1: { mean: 0.6418, std: 0.0239 },
    roc_auc: { mean: 0.8356, std: 0.0212 }
  };
  const rfCV = metricsData?.random_forest?.cv_5fold_metrics || {
    accuracy: { mean: 0.7682, std: 0.0237 },
    precision: { mean: 0.7200, std: 0.0568 },
    recall: { mean: 0.5560, std: 0.0586 },
    f1: { mean: 0.6250, std: 0.0435 },
    roc_auc: { mean: 0.8372, std: 0.0213 }
  };
  const rfImportance = metricsData?.random_forest?.feature_importances || {
    Glucose: 0.3531,
    BMI: 0.1829,
    Age: 0.1277,
    Insulin: 0.1000,
    DiabetesPedigreeFunction: 0.0794,
    Pregnancies: 0.0642,
    SkinThickness: 0.0492,
    BloodPressure: 0.0436,
  };

  // Prepare chart comparison data for Holdout Test vs 5-Fold CV evaluation
  const comparisonData = useMemo(() => {
    if (evalMode === 'cv') {
      return [
        {
          metric: 'Accuracy',
          'Logistic Regression': Number((lrCV.accuracy.mean * 100).toFixed(1)),
          'Random Forest': Number((rfCV.accuracy.mean * 100).toFixed(1)),
        },
        {
          metric: 'Precision',
          'Logistic Regression': Number((lrCV.precision.mean * 100).toFixed(1)),
          'Random Forest': Number((rfCV.precision.mean * 100).toFixed(1)),
        },
        {
          metric: 'Recall',
          'Logistic Regression': Number((lrCV.recall.mean * 100).toFixed(1)),
          'Random Forest': Number((rfCV.recall.mean * 100).toFixed(1)),
        },
        {
          metric: 'F1-Score',
          'Logistic Regression': Number((lrCV.f1.mean * 100).toFixed(1)),
          'Random Forest': Number((rfCV.f1.mean * 100).toFixed(1)),
        },
        {
          metric: 'ROC-AUC',
          'Logistic Regression': Number((lrCV.roc_auc.mean * 100).toFixed(1)),
          'Random Forest': Number((rfCV.roc_auc.mean * 100).toFixed(1)),
        },
      ];
    }

    return [
      {
        metric: 'Accuracy',
        'Logistic Regression': Number((lr.accuracy * 100).toFixed(1)),
        'Random Forest': Number((rf.accuracy * 100).toFixed(1)),
      },
      {
        metric: 'Precision',
        'Logistic Regression': Number((lr.precision * 100).toFixed(1)),
        'Random Forest': Number((rf.precision * 100).toFixed(1)),
      },
      {
        metric: 'Recall',
        'Logistic Regression': Number((lr.recall * 100).toFixed(1)),
        'Random Forest': Number((rf.recall * 100).toFixed(1)),
      },
      {
        metric: 'F1-Score',
        'Logistic Regression': Number((lr.f1_score * 100).toFixed(1)),
        'Random Forest': Number((rf.f1_score * 100).toFixed(1)),
      },
      {
        metric: 'ROC-AUC',
        'Logistic Regression': Number((lr.roc_auc * 100).toFixed(1)),
        'Random Forest': Number((rf.roc_auc * 100).toFixed(1)),
      },
    ];
  }, [evalMode, lr, rf, lrCV, rfCV]);

  // Derive top feature from actual feature importances
  const topFeature = useMemo(() => {
    if (!rfImportance) return null;
    const sorted = Object.entries(rfImportance).sort((a, b) => b[1] - a[1]);
    if (sorted.length === 0) return null;
    return { name: sorted[0][0], val: (sorted[0][1] * 100).toFixed(1) };
  }, [rfImportance]);

  // Derived ROC-AUC difference
  const rocDiff = useMemo(() => {
    if (!lr || !rf) return null;
    const diff = (rf.roc_auc - lr.roc_auc) * 100;
    return diff > 0 ? `+${diff.toFixed(2)}%` : `${diff.toFixed(2)}%`;
  }, [lr, rf]);

  const CustomChartTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const lrVal = payload.find((p) => p.dataKey === 'Logistic Regression')?.value;
      const rfVal = payload.find((p) => p.dataKey === 'Random Forest')?.value;
      const delta = rfVal !== undefined && lrVal !== undefined ? (rfVal - lrVal).toFixed(1) : null;
      return (
        <div className="custom-tooltip">
          <div className="custom-tooltip-title">{label} ({evalMode === 'cv' ? '5-Fold CV Mean' : 'Holdout Test'})</div>
          <div className="custom-tooltip-item">
            <span style={{ color: '#38bdf8' }}>Logistic Regression:</span>
            <strong>{lrVal}%</strong>
          </div>
          <div className="custom-tooltip-item">
            <span style={{ color: '#818cf8' }}>Random Forest:</span>
            <strong>{rfVal}%</strong>
          </div>
          {delta && (
            <div style={{ marginTop: '0.35rem', paddingTop: '0.35rem', borderTop: '1px solid rgba(255,255,255,0.08)', fontSize: '0.76rem', color: Number(delta) >= 0 ? '#34d399' : '#fb7185' }}>
              Random Forest Delta: {Number(delta) >= 0 ? `+${delta}%` : `${delta}%`}
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="dashboard-page">
      {/* A. COMPACT HERO SECTION */}
      <section className="compact-hero">
        <div className="compact-hero-content">
          <h1>Diabetes Risk Intelligence</h1>
          <p>Compare predictions from two supervised learning models trained on the Pima Indians diabetes cohort.</p>
        </div>
        <div className="compact-hero-actions">
          <button className="btn-primary" onClick={() => setActiveTab('predict')}>
            <Stethoscope size={16} />
            <span>Run Assessment</span>
            <ArrowRight size={14} />
          </button>
          <button className="btn-secondary" onClick={() => setActiveTab('performance')}>
            <BarChart2 size={16} />
            <span>Explore Analytics</span>
          </button>
        </div>
      </section>

      {/* B. COMPACT METRIC STRIP */}
      <section className="metric-strip" aria-label="Key Dataset & Model Metrics">
        <MetricCard
          label="Dataset Samples"
          value={dataset?.total_samples ?? '768'}
          subtext="NIDDK Pima Indian Cohort"
          icon={Database}
        />
        <MetricCard
          label="Input Features"
          value={dataset?.features?.length ?? '8'}
          subtext="Physiological Biomarkers"
          icon={Layers}
        />
        <MetricCard
          label="LogReg ROC-AUC"
          value={lr ? `${(lr.roc_auc * 100).toFixed(1)}%` : '80.8%'}
          subtext={`Accuracy: ${lr ? (lr.accuracy * 100).toFixed(1) : '70.8'}%`}
          icon={Activity}
        />
        <MetricCard
          label="Random Forest AUC"
          value={rf ? `${(rf.roc_auc * 100).toFixed(1)}%` : '82.3%'}
          subtext={`Accuracy: ${rf ? (rf.accuracy * 100).toFixed(1) : '76.0'}%`}
          icon={Award}
        />
      </section>

      {/* C. MODEL COMPARISON VISUALIZATION */}
      <section className="glass-panel chart-panel" style={{ marginBottom: '1.5rem' }}>
        <div className="chart-header">
          <div>
            <h2 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <TrendingUp size={18} color="#38bdf8" />
              Supervised Model Comparison ({evalMode === 'cv' ? 'Stratified 5-Fold CV' : 'Holdout Test N=154'})
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              {evalMode === 'cv' 
                ? 'Mean performance across 5 stratified training folds' 
                : 'Empirical evaluation on unseen held-out 20% test partition'}
            </p>
          </div>
          
          {/* Interactive Evaluation Mode Toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', background: 'rgba(255,255,255,0.03)', padding: '3px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <button
              type="button"
              className={`nav-link-btn ${evalMode === 'holdout' ? 'active' : ''}`}
              style={{ padding: '0.3rem 0.75rem', fontSize: '0.78rem' }}
              onClick={() => setEvalMode('holdout')}
            >
              Holdout Test (N=154)
            </button>
            <button
              type="button"
              className={`nav-link-btn ${evalMode === 'cv' ? 'active' : ''}`}
              style={{ padding: '0.3rem 0.75rem', fontSize: '0.78rem' }}
              onClick={() => setEvalMode('cv')}
            >
              5-Fold CV Spread
            </button>
          </div>
        </div>

        <div style={{ width: '100%', height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={comparisonData}
              margin={{ top: 25, right: 10, left: -15, bottom: 0 }}
              barGap={6}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" vertical={false} />
              <XAxis
                dataKey="metric"
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                axisLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
              />
              <YAxis
                domain={[0, 100]}
                unit="%"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
              />
              <Tooltip
                content={<CustomChartTooltip />}
                cursor={{ fill: 'rgba(255, 255, 255, 0.04)' }}
              />
              <Legend
                wrapperStyle={{ paddingTop: '10px', fontSize: '0.82rem' }}
                iconType="circle"
              />
              <Bar
                dataKey="Logistic Regression"
                fill="#38bdf8"
                radius={[4, 4, 0, 0]}
                maxBarSize={44}
              >
                <LabelList 
                  dataKey="Logistic Regression" 
                  position="top" 
                  fill="#7dd3fc" 
                  fontSize={11} 
                  formatter={(val) => `${val}%`} 
                />
              </Bar>
              <Bar
                dataKey="Random Forest"
                fill="#818cf8"
                radius={[4, 4, 0, 0]}
                maxBarSize={44}
              >
                <LabelList 
                  dataKey="Random Forest" 
                  position="top" 
                  fill="#c7d2fe" 
                  fontSize={11} 
                  formatter={(val) => `${val}%`} 
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* D. QUICK ACTIONS */}
      <section className="grid-2" style={{ marginBottom: '1.5rem' }}>
        <div className="glass-panel action-panel">
          <div className="action-panel-body">
            <h3>
              <Stethoscope size={18} color="#38bdf8" />
              New Risk Assessment
            </h3>
            <p>
              Submit patient biomarkers to receive side-by-side risk probabilities from both trained algorithms with concordance analysis.
            </p>
          </div>
          <button className="btn-primary" onClick={() => setActiveTab('predict')}>
            <span>Launch Predictor</span>
            <ArrowRight size={14} />
          </button>
        </div>

        <div className="glass-panel action-panel">
          <div className="action-panel-body">
            <h3>
              <BarChart2 size={18} color="#818cf8" />
              Explore Model Performance
            </h3>
            <p>
              Inspect empirical ROC curves, confusion matrices, 5-Fold Cross-Validation spread, and PCA dimensionality findings.
            </p>
          </div>
          <button className="btn-secondary" onClick={() => setActiveTab('performance')}>
            <span>View Full Analytics</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </section>

      {/* E. EVIDENCE-BASED INSIGHTS */}
      <section className="insights-strip" aria-label="Key Empirical Findings">
        <div className="insight-card">
          <Sparkles size={18} className="insight-icon" />
          <div className="insight-text">
            <h4>Highest Discrimination: Random Forest</h4>
            <p>
              Random Forest attained <strong>{rf ? (rf.roc_auc * 100).toFixed(1) : '82.3'}% ROC-AUC</strong> ({rocDiff || '+1.47%'} over Logistic Regression) with superior test precision (69.77%), reducing clinical false alarms.
            </p>
          </div>
        </div>

        <div className="insight-card">
          <Activity size={18} className="insight-icon" style={{ color: '#818cf8' }} />
          <div className="insight-text">
            <h4>Primary Risk Factor: {topFeature ? topFeature.name : 'Plasma Glucose'}</h4>
            <p>
              According to the trained Random Forest, <strong>{topFeature ? topFeature.name : 'Glucose'}</strong> drives split purity at <strong>{topFeature ? `${topFeature.val}%` : '35.3%'}</strong> Gini importance, followed closely by BMI (18.3%).
            </p>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <DisclaimerBanner />
    </div>
  );
}
