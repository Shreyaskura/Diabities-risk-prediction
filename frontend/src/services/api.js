const API_BASE_URL = 'http://localhost:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  } catch (err) {
    return { status: 'offline', error: err.message };
  }
}

export async function getModelMetrics() {
  const res = await fetch(`${API_BASE_URL}/metrics`);
  if (!res.ok) throw new Error('Failed to fetch model metrics');
  return await res.json();
}

export async function getFeaturesMetadata() {
  const res = await fetch(`${API_BASE_URL}/features`);
  if (!res.ok) throw new Error('Failed to fetch feature metadata');
  return await res.json();
}

export async function predictRisk(patientData) {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(patientData),
  });
  
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Prediction request failed');
  }
  return await res.json();
}

export function getChartUrl(chartName) {
  return `${API_BASE_URL}/results/chart/${chartName}?t=${Date.now()}`;
}
