const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5055;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8001';

app.use(cors());
app.use(express.json());

// In-memory / JSON persistence for student assessments
const DATA_FILE = path.join(__dirname, 'assessments_history.json');
let assessmentsHistory = [];

if (fs.existsSync(DATA_FILE)) {
  try {
    assessmentsHistory = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
  } catch (err) {
    assessmentsHistory = [];
  }
}

function saveHistory() {
  try {
    fs.writeFileSync(DATA_FILE, JSON.stringify(assessmentsHistory.slice(-100), null, 2));
  } catch (err) {
    console.error('Failed to save history file:', err);
  }
}

// Routes
app.get('/', (req, res) => res.json({ status: 'online', service: 'Mental Health Backend API' }));
app.get('/api/health', async (req, res) => {
  let mlStatus = 'offline';
  try {
    const response = await fetch(`${ML_SERVICE_URL}/`);
    if (response.ok) mlStatus = 'online';
  } catch (e) {
    mlStatus = 'unreachable';
  }

  res.json({
    status: 'online',
    backend: 'Express Node.js',
    timestamp: new Date().toISOString(),
    ml_service: {
      url: ML_SERVICE_URL,
      status: mlStatus
    }
  });
});

// Proxy prediction request to Python FastAPI ML service
app.post('/api/predict', async (req, res) => {
  try {
    const studentData = req.body;
    
    const mlResponse = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(studentData)
    });

    if (!mlResponse.ok) {
      const errText = await mlResponse.text();
      return res.status(mlResponse.status).json({ error: `ML Service Error: ${errText}` });
    }

    const predictionResult = await mlResponse.json();

    const record = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      student: studentData,
      result: predictionResult
    };

    assessmentsHistory.unshift(record);
    saveHistory();

    res.json(record);
  } catch (error) {
    console.error('Error contacting ML Service:', error);
    res.status(500).json({
      error: 'Failed to communicate with ML inference engine',
      details: error.message
    });
  }
});

// Fetch research metrics & benchmarks (Goals 1, 2, 3)
app.get('/api/benchmarks', async (req, res) => {
  try {
    const mlResponse = await fetch(`${ML_SERVICE_URL}/metrics`);
    if (mlResponse.ok) {
      const data = await mlResponse.json();
      return res.json(data);
    }
  } catch (err) {
    console.warn('ML Service metrics endpoint unavailable, falling back to local artifacts.');
  }

  // Fallback to reading artifacts/metrics.json directly if available
  const localMetricsPath = path.join(__dirname, '..', 'ml_service', 'artifacts', 'metrics.json');
  if (fs.existsSync(localMetricsPath)) {
    try {
      const cached = JSON.parse(fs.readFileSync(localMetricsPath, 'utf8'));
      return res.json(cached);
    } catch (e) {
      console.error('Failed reading local metrics file:', e);
    }
  }

  res.status(503).json({ error: 'Metrics are still generating or unavailable.' });
});

// Fetch assessment history
app.get('/api/history', (req, res) => {
  res.json({
    total: assessmentsHistory.length,
    history: assessmentsHistory.slice(0, 20)
  });
});

app.listen(PORT, () => {
  console.log(`Node Express Backend running on http://localhost:${PORT}`);
  console.log(`Forwarding ML requests to: ${ML_SERVICE_URL}`);
});
