import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List

app = FastAPI(
    title="Student Mental Health ML Microservice",
    description="Hybrid Feature Extraction & Stacking Ensemble Service for Student Mental Health Prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

# In-memory model cache
PIPELINE = None
SELECTOR = None
LABEL_ENCODER = None
STACKING_MODEL = None
METRICS_CACHE = None

def load_resources():
    global PIPELINE, SELECTOR, LABEL_ENCODER, STACKING_MODEL, METRICS_CACHE
    try:
        pipeline_path = os.path.join(MODELS_DIR, "feature_pipeline.joblib")
        selector_path = os.path.join(MODELS_DIR, "feature_selector.joblib")
        encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        model_path = os.path.join(MODELS_DIR, "stacking_ensemble_f3.joblib")

        if os.path.exists(pipeline_path):
            PIPELINE = joblib.load(pipeline_path)
        if os.path.exists(selector_path):
            SELECTOR = joblib.load(selector_path)
        if os.path.exists(encoder_path):
            LABEL_ENCODER = joblib.load(encoder_path)
        if os.path.exists(model_path):
            STACKING_MODEL = joblib.load(model_path)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                METRICS_CACHE = json.load(f)
        print("ML resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")

@app.on_event("startup")
def startup_event():
    load_resources()

class StudentAssessmentInput(BaseModel):
    Age: int = Field(20, ge=12, le=60)
    Gender: str = Field("Male", pattern="^(Male|Female|Other)$")
    Country: str = Field("USA")
    Academic_Level: str = Field("Undergraduate")
    Most_Used_Platform: str = Field("Instagram")
    Purpose_Of_Use: str = Field("Entertainment")
    Avg_Daily_Usage_Hours: float = Field(4.5, ge=0.0, le=24.0)
    Daily_Unlocks: int = Field(120, ge=0, le=500)
    Study_Hours: float = Field(4.0, ge=0.0, le=24.0)
    Physical_Activity_Hours: float = Field(1.5, ge=0.0, le=24.0)
    Sleep_Hours_Per_Night: float = Field(7.0, ge=0.0, le=24.0)

def generate_recommendations(stress_level: str, student_data: StudentAssessmentInput) -> List[str]:
    recs = []
    if stress_level in ["High", "Very High"]:
        recs.append("Critical: Consider booking a session with your university mental wellness counselor.")
    
    if student_data.Avg_Daily_Usage_Hours > 5.0:
        recs.append(f"Screen Time Alert: You spend {student_data.Avg_Daily_Usage_Hours} hrs/day on social media. Try setting daily app limits to reduce cognitive fatigue.")
        
    if student_data.Daily_Unlocks > 150:
        recs.append(f"Habit Notice: {student_data.Daily_Unlocks} daily phone unlocks indicates frequent interruption. Consider enabling 'Do Not Disturb' during study hours.")

    if student_data.Sleep_Hours_Per_Night < 6.5:
        recs.append(f"Sleep Deprivation: {student_data.Sleep_Hours_Per_Night} hrs/night is below the recommended 7-9 hours. Avoid screens 45 minutes before bedtime.")

    if student_data.Physical_Activity_Hours < 1.0:
        recs.append("Movement: Incorporate at least 30 minutes of aerobic exercise, walking, or yoga daily to naturally lower cortisol levels.")

    if not recs:
        recs.append("Great balance! Maintain your current sleep, study, and social media habits.")

    return recs

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Student Mental Health Prediction Engine",
        "models_ready": STACKING_MODEL is not None
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metrics")
def get_all_metrics():
    if not METRICS_CACHE:
        load_resources()
    if not METRICS_CACHE:
        raise HTTPException(status_code=404, detail="Metrics not generated yet. Run train_and_evaluate.py first.")
    return METRICS_CACHE

@app.get("/metrics/boosting")
def get_boosting_metrics():
    data = get_all_metrics()
    return data.get("objective_1_boosting", [])

@app.get("/metrics/stacking")
def get_stacking_metrics():
    data = get_all_metrics()
    return data.get("objective_2_stacking", [])

@app.get("/metrics/literature")
def get_literature_metrics():
    data = get_all_metrics()
    return data.get("objective_3_literature", [])

@app.get("/metrics/confusion-matrix")
def get_confusion_matrix():
    data = get_all_metrics()
    return data.get("confusion_matrix", {})

@app.post("/predict")
def predict_mental_health(input_data: StudentAssessmentInput):
    global PIPELINE, SELECTOR, LABEL_ENCODER, STACKING_MODEL
    if STACKING_MODEL is None or PIPELINE is None:
        load_resources()
    if STACKING_MODEL is None or PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model is still training or not available.")

    # Convert to dataframe
    df_single = pd.DataFrame([input_data.dict()])
    
    # Transform using f3 hybrid mode
    f3_raw = PIPELINE.transform(df_single, mode='f3')
    
    # Feature selector
    if SELECTOR is not None:
        f3_selected = SELECTOR.transform(f3_raw)
    else:
        f3_selected = f3_raw

    # Prediction
    pred_idx = STACKING_MODEL.predict(f3_selected)[0]
    predicted_label = LABEL_ENCODER.inverse_transform([pred_idx])[0]

    # Probabilities
    probs = {}
    if hasattr(STACKING_MODEL, "predict_proba"):
        prob_values = STACKING_MODEL.predict_proba(f3_selected)[0]
        for idx, cls_name in enumerate(LABEL_ENCODER.classes_):
            probs[cls_name] = round(float(prob_values[idx]), 4)
        confidence = round(float(np.max(prob_values)) * 100, 2)
    else:
        confidence = 95.0

    recommendations = generate_recommendations(predicted_label, input_data)

    return {
        "prediction": predicted_label,
        "confidence_percentage": confidence,
        "class_probabilities": probs,
        "recommendations": recommendations,
        "model_used": "Multi-Boosting Stacking Ensemble on Hybrid Features (f3)"
    }
