import React from 'react';
import { Activity, LayoutDashboard, Stethoscope, BarChart3, Info } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, systemStatus }) {
  const navItems = [
    { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
    { id: 'predict', label: 'Risk Prediction', icon: Stethoscope },
    { id: 'performance', label: 'Model Analytics', icon: BarChart3 },
    { id: 'about', label: 'About', icon: Info },
  ];

  const getStatusConfig = () => {
    if (!systemStatus) {
      return { class: 'offline', text: 'Connecting...' };
    }
    if (systemStatus.status === 'healthy') {
      return { class: 'healthy', text: 'API Ready' };
    }
    if (systemStatus.status === 'degraded') {
      return { class: 'degraded', text: 'API Degraded' };
    }
    return { class: 'offline', text: 'API Offline' };
  };

  const statusConfig = getStatusConfig();

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <div className="nav-brand" onClick={() => setActiveTab('dashboard')} role="button" tabIndex={0}>
          <div className="nav-brand-icon">
            <Activity size={18} color="#ffffff" />
          </div>
          <div className="nav-brand-text">
            <h1>DiaRisk ML</h1>
            <span className="nav-brand-badge">Clinical ML</span>
          </div>
        </div>

        <nav className="nav-links" aria-label="Main Navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                className={`nav-link-btn ${isActive ? 'active' : ''}`}
                onClick={() => setActiveTab(item.id)}
                aria-current={isActive ? 'page' : undefined}
              >
                <Icon size={15} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="nav-status" title={`Backend status: ${statusConfig.text}`}>
          <span className={`status-dot ${statusConfig.class}`}></span>
          <span>{statusConfig.text}</span>
        </div>
      </div>
    </header>
  );
}
