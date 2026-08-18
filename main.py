"""
FastAPI Application Entrypoint for IEEE-CIS Fraud Detection ML Microservice.
"""

import os
import sys
import json
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure training directory is available for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'training'))

from feature_engineering import IEEEFeaturePipeline
from explain import TreeSHAPExplainer
from app.routes.predict import router as predict_router
from app.routes.model import router as model_router
from app.routes.health import router as health_router

ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')


def load_artifacts_into_state(target_state):
    """Loads ML model, pipeline, features, and explainers into target state."""
    model_path = os.path.join(ARTIFACTS_DIR, 'lightgbm_active.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(ARTIFACTS_DIR, 'lightgbm_v1.pkl')
        
    pipeline_path = os.path.join(ARTIFACTS_DIR, 'feature_pipeline.pkl')
    features_path = os.path.join(ARTIFACTS_DIR, 'selected_features.json')
    meta_path = os.path.join(ARTIFACTS_DIR, 'model_meta.json')
    metrics_path = os.path.join(ARTIFACTS_DIR, 'metrics.json')

    if os.path.exists(model_path) and os.path.exists(pipeline_path):
        target_state.model = joblib.load(model_path)
        target_state.pipeline = IEEEFeaturePipeline.load(pipeline_path)
        
        with open(features_path, 'r') as f:
            target_state.features = json.load(f)
        with open(meta_path, 'r') as f:
            target_state.meta = json.load(f)
        with open(metrics_path, 'r') as f:
            target_state.metrics = json.load(f)
            
        target_state.explainer = TreeSHAPExplainer(target_state.model, target_state.features)
        print(f"-> Active Model Loaded: {target_state.meta.get('version')} ({len(target_state.features)} features)", flush=True)
    else:
        print("WARNING: Artifacts not found. Initializing empty state.", flush=True)
        target_state.model = None
        target_state.pipeline = None
        target_state.features = []
        target_state.meta = {}
        target_state.metrics = {}
        target_state.explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads ML models, feature engineering pipeline, and SHAP explainer on startup."""
    print("=== [FastAPI] Initializing ML Microservice Lifespan ===", flush=True)
    load_artifacts_into_state(app.state)
    yield
    print("=== [FastAPI] Shutting down ML Microservice ===", flush=True)


app = FastAPI(
    title="IEEE-CIS Fraud Detection ML Microservice",
    description="High-throughput LightGBM binary classifier and TreeSHAP explainer for credit card / e-commerce fraud detection.",
    version="1.0.0",
    lifespan=lifespan
)

# Eagerly initialize state so non-lifespan test runners have immediate access
load_artifacts_into_state(app.state)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(model_router)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
