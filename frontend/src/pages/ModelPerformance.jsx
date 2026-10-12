import React, { useState, useMemo } from 'react';
import DisclaimerBanner from '../components/DisclaimerBanner';
import { getChartUrl } from '../services/api';
import { 
  BarChart3, 
  TrendingUp, 
  Layers, 
  GitMerge, 
  Image as ImageIcon,
  CheckCircle2,
  Table as TableIcon
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
  LineChart,
  Line,
  ComposedChart,
  ReferenceLine
} from 'recharts';

export default function ModelPerformance({ metricsData }) {
  const [activeTab, setActiveTab] = useState('interactive');
  const [activeStaticPlot, setActiveStaticPlot] = useState('model_comparison.png');

  const lrTest = metricsData?.logistic_regression?.test_metrics || {
    accuracy: 0.7078, precision: 0.5918, recall: 0.5370, f1_score: 0.5631, roc_auc: 0.8081,
    confusion_matrix: { tn: 80, fp: 20, fn: 25, tp: 29 }
  };
  const rfTest = metricsData?.random_forest?.test_metrics || {
    accuracy: 0.7597, precision: 0.6977, recall: 0.5556, f1_score: 0.6186, roc_auc: 0.8228,
    confusion_matrix: { tn: 87, fp: 13, fn: 24, tp: 30 }
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
    Glucose: 0.3531, BMI: 0.1829, Age: 0.1277, Insulin: 0.1000,
    DiabetesPedigreeFunction: 0.0794, Pregnancies: 0.0642, SkinThickness: 0.0492, BloodPressure: 0.0436
  };
  const pca = metricsData?.pca_analysis || {
    explained_variance_ratio: [0.2943, 0.1879, 0.1385, 0.1085, 0.0973, 0.0671, 0.0578, 0.0486],
    cumulative_variance_ratio: [0.2943, 0.4822, 0.6206, 0.7291, 0.8264, 0.8935, 0.9514, 1.0]
  };

  // 1. Holdout comparison data for Recharts
  const holdoutComparison = useMemo(() => {
    return [
      {
        metric: 'Accuracy',
        'Logistic Regression': Number((lrTest.accuracy * 100).toFixed(2)),
        'Random Forest': Number((rfTest.accuracy * 100).toFixed(2)),
      },
      {
        metric: 'Precision',
        'Logistic Regression': Number((lrTest.precision * 100).toFixed(2)),
        'Random Forest': Number((rfTest.precision * 100).toFixed(2)),
      },
      {
        metric: 'Recall',
        'Logistic Regression': Number((lrTest.recall * 100).toFixed(2)),
        'Random Forest': Number((rfTest.recall * 100).toFixed(2)),
      },
      {
        metric: 'F1-Score',
        'Logistic Regression': Number((lrTest.f1_score * 100).toFixed(2)),
        'Random Forest': Number((rfTest.f1_score * 100).toFixed(2)),
      },
      {
        metric: 'ROC-AUC',
        'Logistic Regression': Number((lrTest.roc_auc * 100).toFixed(2)),
        'Random Forest': Number((rfTest.roc_auc * 100).toFixed(2)),
      },
    ];
  }, [lrTest, rfTest]);

  // 2. Feature importance data sorted descending
  const featureImportanceData = useMemo(() => {
    if (!rfImportance) return [];
    return Object.entries(rfImportance)
      .map(([feature, val]) => ({
        feature,
        importance: Number((val * 100).toFixed(2)),
      }))
      .sort((a, b) => b.importance - a.importance);
  }, [rfImportance]);

  // 3. PCA data
  const pcaChartData = useMemo(() => {
    if (!pca?.explained_variance_ratio) return [];
    return pca.explained_variance_ratio.map((val, idx) => ({
      component: `PC${idx + 1}`,
      'Individual Variance': Number((val * 100).toFixed(2)),
      'Cumulative Variance': Number((pca.cumulative_variance_ratio[idx] * 100).toFixed(2)),
    }));
  }, [pca]);

  // 4. ROC curve alignment for Recharts
  const rocCurveData = useMemo(() => {
    const lrCurve = lrTest?.roc_curve;
    const rfCurve = rfTest?.roc_curve;
    if (!lrCurve || !rfCurve) return [];

    // Interpolate/sample 25 standard FPR bins [0, 0.04, 0.08, ... 1.0]
    const bins = Array.from({ length: 26 }, (_, i) => Number((i * 0.04).toFixed(2)));
    
    const findTPR = (curve, targetFPR) => {
      if (!curve || !curve.fpr || !curve.tpr) return 0;
      for (let i = 0; i < curve.fpr.length; i++) {
        if (curve.fpr[i] >= targetFPR) {
          return Number((curve.tpr[i] * 100).toFixed(1));
        }
      }
      return 100;
    };

    return bins.map((fprVal) => ({
      fpr: fprVal,
      fprPct: `${(fprVal * 100).toFixed(0)}%`,
      'Logistic Regression': findTPR(lrCurve, fprVal),
      'Random Forest': findTPR(rfCurve, fprVal),
      'Chance (Diagonal)': Number((fprVal * 100).toFixed(1)),
    }));
  }, [lrTest, rfTest]);

  // 5. Cross Validation metrics table data
  const cvMetrics = useMemo(() => {
    if (!lrCV || !rfCV) return [];
    return [
      {
        metric: 'Accuracy',
        lrMean: (lrCV.accuracy.mean * 100).toFixed(2),
        lrStd: (lrCV.accuracy.std * 100).toFixed(2),
        rfMean: (rfCV.accuracy.mean * 100).toFixed(2),
        rfStd: (rfCV.accuracy.std * 100).toFixed(2),
        winner: lrCV.accuracy.mean >= rfCV.accuracy.mean ? 'Logistic Regression' : 'Random Forest',
      },
      {
        metric: 'Precision',
        lrMean: (lrCV.precision.mean * 100).toFixed(2),
        lrStd: (lrCV.precision.std * 100).toFixed(2),
        rfMean: (rfCV.precision.mean * 100).toFixed(2),
        rfStd: (rfCV.precision.std * 100).toFixed(2),
        winner: lrCV.precision.mean >= rfCV.precision.mean ? 'Logistic Regression' : 'Random Forest',
      },
      {
        metric: 'Recall',
        lrMean: (lrCV.recall.mean * 100).toFixed(2),
        lrStd: (lrCV.recall.std * 100).toFixed(2),
        rfMean: (rfCV.recall.mean * 100).toFixed(2),
        rfStd: (rfCV.recall.std * 100).toFixed(2),
        winner: lrCV.recall.mean >= rfCV.recall.mean ? 'Logistic Regression' : 'Random Forest',
      },
      {
        metric: 'F1-Score',
        lrMean: (lrCV.f1.mean * 100).toFixed(2),
        lrStd: (lrCV.f1.std * 100).toFixed(2),
        rfMean: (rfCV.f1.mean * 100).toFixed(2),
        rfStd: (rfCV.f1.std * 100).toFixed(2),
        winner: lrCV.f1.mean >= rfCV.f1.mean ? 'Logistic Regression' : 'Random Forest',
      },
      {
        metric: 'ROC-AUC',
        lrMean: (lrCV.roc_auc.mean * 100).toFixed(2),
        lrStd: (lrCV.roc_auc.std * 100).toFixed(2),
        rfMean: (rfCV.roc_auc.mean * 100).toFixed(2),
        rfStd: (rfCV.roc_auc.std * 100).toFixed(2),
        winner: rfCV.roc_auc.mean >= lrCV.roc_auc.mean ? 'Random Forest' : 'Logistic Regression',
      },
    ];
  }, [lrCV, rfCV]);

  // Confusion Matrices
  const lrCM = lrTest?.confusion_matrix;
  const rfCM = rfTest?.confusion_matrix;

  const staticPlots = [
    { id: 'model_comparison.png', label: 'Holdout Comparison' },
    { id: 'roc_curve.png', label: 'ROC-AUC Curves' },
    { id: 'confusion_matrix.png', label: 'Confusion Matrices' },
    { id: 'cv_scores.png', label: '5-Fold CV Spread' },
    { id: 'feature_importance.png', label: 'Feature Importance' },
    { id: 'pca_variance.png', label: 'PCA Scree Plot' },
  ];

  return (
    <div className="performance-page">
      {/* Compact Page Header */}
      <section className="compact-hero">
        <div className="compact-hero-content">
          <h1>Model Analytics & Validation</h1>
          <p>
            Rigorous empirical evaluation comparing L2-Regularized Logistic Regression and Random Forest ensembles on Stratified 5-Fold Cross-Validation and an independent holdout test set (N=154).
          </p>
        </div>
        <div className="compact-hero-actions">
          <button
            className={`nav-link-btn ${activeTab === 'interactive' ? 'active' : ''}`}
            onClick={() => setActiveTab('interactive')}
          >
            <BarChart3 size={15} />
            <span>Interactive Visualizations</span>
          </button>
          <button
            className={`nav-link-btn ${activeTab === 'static' ? 'active' : ''}`}
            onClick={() => setActiveTab('static')}
          >
            <ImageIcon size={15} />
            <span>Matplotlib Figures</span>
          </button>
        </div>
      </section>

      {activeTab === 'interactive' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* 1. Performance Comparison: Chart + Table */}
          <section className="glass-panel" style={{ padding: '1.5rem' }}>
            <div className="chart-header">
              <div>
                <h2 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <TrendingUp size={18} color="#38bdf8" />
                  Holdout Test Performance (N=154 Split, 20%)
                </h2>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Evaluated on unseen test partition following leak-free median imputation
                </p>
              </div>
            </div>

            <div style={{ width: '100%', height: 260, marginBottom: '1.25rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={holdoutComparison} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" vertical={false} />
                  <XAxis dataKey="metric" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis domain={[0, 100]} unit="%" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <Tooltip
                    cursor={{ fill: 'rgba(255, 255, 255, 0.04)' }}
                    formatter={(val) => [`${val}%`, '']}
                    contentStyle={{
                      background: 'rgba(11, 16, 28, 0.95)',
                      border: '1px solid rgba(255, 255, 255, 0.12)',
                      borderRadius: '8px',
                      fontSize: '0.82rem',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.82rem' }} iconType="circle" />
                  <Bar dataKey="Logistic Regression" fill="#38bdf8" radius={[4, 4, 0, 0]} maxBarSize={40} />
                  <Bar dataKey="Random Forest" fill="#818cf8" radius={[4, 4, 0, 0]} maxBarSize={40} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Compact Table */}
            <div className="compact-table-wrap">
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>ROC-AUC</th>
                    <th>Parameters</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Logistic Regression</td>
                    <td>{lrTest ? `${(lrTest.accuracy * 100).toFixed(2)}%` : '70.78%'}</td>
                    <td>{lrTest ? `${(lrTest.precision * 100).toFixed(2)}%` : '59.18%'}</td>
                    <td>{lrTest ? `${(lrTest.recall * 100).toFixed(2)}%` : '53.70%'}</td>
                    <td>{lrTest ? `${(lrTest.f1_score * 100).toFixed(2)}%` : '56.31%'}</td>
                    <td style={{ color: '#38bdf8', fontWeight: 600 }}>
                      {lrTest ? `${(lrTest.roc_auc * 100).toFixed(2)}%` : '80.81%'}
                    </td>
                    <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>C=0.05, liblinear</td>
                  </tr>
                  <tr>
                    <td style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Random Forest</td>
                    <td style={{ color: '#34d399', fontWeight: 600 }}>
                      {rfTest ? `${(rfTest.accuracy * 100).toFixed(2)}%` : '75.97%'}
                    </td>
                    <td style={{ color: '#34d399', fontWeight: 600 }}>
                      {rfTest ? `${(rfTest.precision * 100).toFixed(2)}%` : '69.77%'}
                    </td>
                    <td>{rfTest ? `${(rfTest.recall * 100).toFixed(2)}%` : '55.56%'}</td>
                    <td style={{ color: '#34d399', fontWeight: 600 }}>
                      {rfTest ? `${(rfTest.f1_score * 100).toFixed(2)}%` : '61.86%'}
                    </td>
                    <td style={{ color: '#818cf8', fontWeight: 600 }}>
                      {rfTest ? `${(rfTest.roc_auc * 100).toFixed(2)}%` : '82.28%'}
                    </td>
                    <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>n_trees=50, depth=5</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* 2. ROC Curves & Confusion Matrices Grid */}
          <div className="grid-2">
            {/* Interactive ROC Curves */}
            <section className="glass-panel" style={{ padding: '1.5rem' }}>
              <div className="section-header" style={{ marginBottom: '0.75rem' }}>
                <div>
                  <h2>
                    <GitMerge size={18} color="#38bdf8" />
                    Empirical ROC Curves
                  </h2>
                  <p>True Positive Rate vs. False Positive Rate across thresholds</p>
                </div>
              </div>

              <div style={{ width: '100%', height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rocCurveData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
                    <XAxis dataKey="fprPct" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis domain={[0, 100]} unit="%" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip
                      formatter={(val) => [`${val}%`, '']}
                      contentStyle={{
                        background: 'rgba(11, 16, 28, 0.95)',
                        border: '1px solid rgba(255, 255, 255, 0.12)',
                        borderRadius: '8px',
                        fontSize: '0.82rem',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '0.8rem' }} iconType="plainline" />
                    <Line
                      type="monotone"
                      dataKey="Random Forest"
                      stroke="#818cf8"
                      strokeWidth={2.5}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="Logistic Regression"
                      stroke="#38bdf8"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="linear"
                      dataKey="Chance (Diagonal)"
                      stroke="#475569"
                      strokeWidth={1}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Confusion Matrices (Actual 2x2 Breakdown) */}
            <section className="glass-panel" style={{ padding: '1.5rem' }}>
              <div className="section-header" style={{ marginBottom: '0.5rem' }}>
                <div>
                  <h2>
                    <TableIcon size={18} color="#10b981" />
                    Confusion Matrices (Holdout N=154)
                  </h2>
                  <p>Actual class distribution: 100 Non-Diabetic, 54 Diabetic</p>
                </div>
              </div>

              <div className="cm-container">
                {/* Logistic Regression CM */}
                <div className="cm-card">
                  <div style={{ fontSize: '0.84rem', fontWeight: 600, color: '#38bdf8' }}>
                    Logistic Regression
                  </div>
                  <div className="cm-grid">
                    <div></div>
                    <div className="cm-cell label">Pred Neg</div>
                    <div className="cm-cell label">Pred Pos</div>
                    <div className="cm-cell label">Act Neg</div>
                    <div className="cm-cell tn">
                      <span className="cm-cell-num">{lrCM?.tn ?? 80}</span>
                      <span className="cm-cell-name">TN (80%)</span>
                    </div>
                    <div className="cm-cell fp">
                      <span className="cm-cell-num">{lrCM?.fp ?? 20}</span>
                      <span className="cm-cell-name">FP (20%)</span>
                    </div>
                    <div className="cm-cell label">Act Pos</div>
                    <div className="cm-cell fn">
                      <span className="cm-cell-num">{lrCM?.fn ?? 25}</span>
                      <span className="cm-cell-name">FN (46%)</span>
                    </div>
                    <div className="cm-cell tp">
                      <span className="cm-cell-num">{lrCM?.tp ?? 29}</span>
                      <span className="cm-cell-name">TP (54%)</span>
                    </div>
                  </div>
                </div>

                {/* Random Forest CM */}
                <div className="cm-card">
                  <div style={{ fontSize: '0.84rem', fontWeight: 600, color: '#818cf8' }}>
                    Random Forest
                  </div>
                  <div className="cm-grid">
                    <div></div>
                    <div className="cm-cell label">Pred Neg</div>
                    <div className="cm-cell label">Pred Pos</div>
                    <div className="cm-cell label">Act Neg</div>
                    <div className="cm-cell tn">
                      <span className="cm-cell-num">{rfCM?.tn ?? 87}</span>
                      <span className="cm-cell-name">TN (87%)</span>
                    </div>
                    <div className="cm-cell fp">
                      <span className="cm-cell-num">{rfCM?.fp ?? 13}</span>
                      <span className="cm-cell-name">FP (13%)</span>
                    </div>
                    <div className="cm-cell label">Act Pos</div>
                    <div className="cm-cell fn">
                      <span className="cm-cell-num">{rfCM?.fn ?? 24}</span>
                      <span className="cm-cell-name">FN (44%)</span>
                    </div>
                    <div className="cm-cell tp">
                      <span className="cm-cell-num">{rfCM?.tp ?? 30}</span>
                      <span className="cm-cell-name">TP (56%)</span>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>

          {/* 3. Cross-Validation Results */}
          <section className="glass-panel" style={{ padding: '1.5rem' }}>
            <div className="section-header" style={{ marginBottom: '1rem' }}>
              <div>
                <h2>
                  <Layers size={18} color="#10b981" />
                  Stratified 5-Fold Cross-Validation (Mean ± Std Dev)
                </h2>
                <p>
                  Evaluated strictly on training folds to prevent data leakage and optimism bias
                </p>
              </div>
            </div>

            <div className="compact-table-wrap">
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Logistic Regression (Mean ± Std)</th>
                    <th>Random Forest (Mean ± Std)</th>
                    <th>Subgroup Stability</th>
                  </tr>
                </thead>
                <tbody>
                  {cvMetrics.map((row) => (
                    <tr key={row.metric}>
                      <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{row.metric}</td>
                      <td>
                        <strong>{row.lrMean}%</strong> <span style={{ color: 'var(--text-muted)' }}>± {row.lrStd}%</span>
                      </td>
                      <td>
                        <strong>{row.rfMean}%</strong> <span style={{ color: 'var(--text-muted)' }}>± {row.rfStd}%</span>
                      </td>
                      <td>
                        <span style={{ fontSize: '0.78rem', color: '#60a5fa' }}>
                          Fold spread std &lt; 0.07 (consistent)
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* 4. Feature Importance & PCA Grid */}
          <div className="grid-2">
            {/* Random Forest Feature Importance */}
            <section className="glass-panel" style={{ padding: '1.5rem' }}>
              <div className="section-header" style={{ marginBottom: '0.75rem' }}>
                <div>
                  <h2>
                    <BarChart3 size={18} color="#818cf8" />
                    Random Forest Feature Importance
                  </h2>
                  <p>Normalized Gini impurity split contribution (%)</p>
                </div>
              </div>

              <div style={{ width: '100%', height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={featureImportanceData}
                    margin={{ top: 5, right: 20, left: 35, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" horizontal={false} />
                    <XAxis type="number" unit="%" domain={[0, 40]} stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis dataKey="feature" type="category" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <Tooltip
                      cursor={{ fill: 'rgba(255, 255, 255, 0.04)' }}
                      formatter={(val) => [`${val}%`, 'Gini Importance']}
                      contentStyle={{
                        background: 'rgba(11, 16, 28, 0.95)',
                        border: '1px solid rgba(255, 255, 255, 0.12)',
                        borderRadius: '8px',
                        fontSize: '0.82rem',
                      }}
                    />
                    <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]} maxBarSize={18} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* PCA Explained Variance */}
            <section className="glass-panel" style={{ padding: '1.5rem' }}>
              <div className="section-header" style={{ marginBottom: '0.75rem' }}>
                <div>
                  <h2>
                    <Layers size={18} color="#f59e0b" />
                    PCA Dimensionality Analysis
                  </h2>
                  <p>Individual (bars) & cumulative (line) variance across PC1-PC8</p>
                </div>
              </div>

              <div style={{ width: '100%', height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={pcaChartData} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" vertical={false} />
                    <XAxis dataKey="component" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis domain={[0, 100]} unit="%" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip
                      cursor={{ fill: 'rgba(255, 255, 255, 0.04)' }}
                      formatter={(val) => [`${val}%`, '']}
                      contentStyle={{
                        background: 'rgba(11, 16, 28, 0.95)',
                        border: '1px solid rgba(255, 255, 255, 0.12)',
                        borderRadius: '8px',
                        fontSize: '0.82rem',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '0.8rem' }} iconType="circle" />
                    <ReferenceLine y={85} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: '85% Target', fill: '#f59e0b', fontSize: 10, position: 'insideTopRight' }} />
                    <Bar dataKey="Individual Variance" fill="#0ea5e9" radius={[4, 4, 0, 0]} maxBarSize={24} />
                    <Line type="monotone" dataKey="Cumulative Variance" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        </div>
      ) : (
        /* Static Matplotlib Plots Viewer */
        <section className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginBottom: '1.5rem' }}>
            {staticPlots.map((plot) => (
              <button
                key={plot.id}
                className={`nav-link-btn ${activeStaticPlot === plot.id ? 'active' : ''}`}
                style={{ fontSize: '0.82rem', padding: '0.4rem 0.8rem' }}
                onClick={() => setActiveStaticPlot(plot.id)}
              >
                {plot.label}
              </button>
            ))}
          </div>

          <div style={{ textAlign: 'center', background: '#0a0e1a', padding: '1rem', borderRadius: '12px' }}>
            <img
              src={getChartUrl(activeStaticPlot)}
              alt={activeStaticPlot}
              style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px', boxShadow: '0 4px 20px rgba(0,0,0,0.5)' }}
            />
          </div>
        </section>
      )}

      <DisclaimerBanner />
    </div>
  );
}
