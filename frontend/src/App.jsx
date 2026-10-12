import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import PredictionPage from './pages/PredictionPage';
import ModelPerformance from './pages/ModelPerformance';
import AboutPage from './pages/AboutPage';
import { checkHealth, getModelMetrics } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemStatus, setSystemStatus] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchStatusAndMetrics = useCallback(async (isInitial = false) => {
    try {
      const health = await checkHealth();
      setSystemStatus(health);

      if (health.status === 'healthy' && (!metricsData || isInitial)) {
        const metrics = await getModelMetrics();
        setMetricsData(metrics);
      }
    } catch (err) {
      console.warn('API sync notice:', err.message);
    } finally {
      if (isInitial) {
        setLoading(false);
      }
    }
  }, [metricsData]);

  useEffect(() => {
    fetchStatusAndMetrics(true);

    // Efficient 30s background health check
    const interval = setInterval(() => {
      fetchStatusAndMetrics(false);
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchStatusAndMetrics]);

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        systemStatus={systemStatus}
      />

      <main className="main-content" id="main-content">
        {activeTab === 'dashboard' && (
          <Dashboard
            setActiveTab={setActiveTab}
            metricsData={metricsData}
          />
        )}

        {activeTab === 'predict' && (
          <PredictionPage />
        )}

        {activeTab === 'performance' && (
          <ModelPerformance
            metricsData={metricsData}
          />
        )}

        {activeTab === 'about' && (
          <AboutPage />
        )}
      </main>

      <footer
        style={{
          borderTop: '1px solid var(--border-subtle)',
          padding: '1.25rem 1.5rem',
          textAlign: 'center',
          color: 'var(--text-dim)',
          fontSize: '0.78rem',
          background: 'rgba(9, 13, 22, 0.95)',
        }}
      >
        <div style={{ color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '0.2rem' }}>
          DiaRisk ML • Clinical Decision-Support Intelligence
        </div>
        <div>
          FastAPI REST Engine • React 18 & Recharts Client • Scikit-Learn Supervised Pipeline
        </div>
      </footer>
    </div>
  );
}
