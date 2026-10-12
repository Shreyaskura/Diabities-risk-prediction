import React from 'react';

export default function MetricCard({ label, value, subtext, icon: Icon }) {
  return (
    <div className="metric-card-compact">
      <div className="metric-card-top">
        <span className="metric-card-label">{label}</span>
        {Icon && (
          <div className="metric-card-icon">
            <Icon size={15} />
          </div>
        )}
      </div>
      <div className="metric-card-val tabular-nums">{value}</div>
      {subtext && <div className="metric-card-sub">{subtext}</div>}
    </div>
  );
}
