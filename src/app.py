import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
from src.logging_config import setup_logging
import logging
from src.monitoring import REQUEST_COUNT, REQUEST_LATENCY
from fastapi import Request
import time
from prometheus_client import generate_latest
from fastapi.responses import Response

setup_logging()
logger = logging.getLogger(__name__)

model = mlflow.sklearn.load_model(model_uri="exported_model")

app = FastAPI(title="Heart Disease Prediction API")

class PatientInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    REQUEST_LATENCY.observe(time.time() - start_time)
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/")
def health_check():
    return "API is running"

@app.post("/predict")
def predict(data: PatientInput):
    logger.info(f"Prediction request received: {data}")

    df = pd.DataFrame([data.dict()])
    prediction = int(model.predict(df)[0])

    try:
        proba = model.predict_proba(df)[0]
        confidence = float(proba[prediction])
    except Exception as e:
        print(f"DEBUG: Probability failed with error: {e}") # This will show in your Docker logs
        confidence = None

    logger.info(f"Prediction={prediction}, confidence={confidence}")
    return {"prediction": prediction, "confidence": confidence}