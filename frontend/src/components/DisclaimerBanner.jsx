import React from 'react';
import { ShieldAlert } from 'lucide-react';

export default function DisclaimerBanner() {
  return (
    <aside className="disclaimer-banner" role="note" aria-label="Educational Medical Disclaimer">
      <ShieldAlert size={16} style={{ flexShrink: 0, marginTop: '2px', color: '#f59e0b' }} />
      <div>
        <strong>Educational Decision-Support Notice:</strong> DiaRisk ML is an academic machine-learning model demonstration. Output probabilities reflect statistical training distributions on benchmark cohorts and do not constitute certified medical diagnosis or clinical treatment guidance. Consult licensed healthcare practitioners for medical evaluation.
      </div>
    </aside>
  );
}
